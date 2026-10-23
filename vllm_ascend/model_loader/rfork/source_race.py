# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Coordinate RFork loading and race compatible leases against mounted weights."""

import hashlib
import time
from collections.abc import Iterator
from contextlib import contextmanager

import torch
from vllm.logger import logger

from vllm_ascend.model_loader.rfork.types import RForkLifecycleState


class RForkLoadCoordinator:
    def __init__(self, vllm_config, *, is_draft: bool) -> None:
        # Read the active TP group after the drafter's get_tp_context has selected it.
        from vllm.distributed.parallel_state import get_ep_group, get_pcp_group, get_pp_group, get_tp_group

        parallel = vllm_config.parallel_config
        tp_group = get_tp_group()
        groups = [tp_group]
        pcp_size = getattr(parallel, "prefill_context_parallel_size", 1)
        if pcp_size > 1:
            groups.append(get_pcp_group())
        dp_size = getattr(parallel, "data_parallel_size", 1)
        if (
            getattr(parallel, "enable_expert_parallel", False)
            and getattr(parallel, "is_moe_model", None) is not False
            and dp_size > 1
        ):
            ep_group = get_ep_group()
            if ep_group.world_size != dp_size * pcp_size * tp_group.world_size:
                raise ValueError("Async RFork requires the draft and target EP participants to match")
            groups = [ep_group]
        model_config = getattr(vllm_config, "model_config", None)
        has_engram_layers = any(
            getattr(getattr(model_config, name, None), "engram_layer_ids", None)
            for name in ("hf_text_config", "hf_config")
        )
        if getattr(vllm_config, "engram_config", None) is not None and has_engram_layers:
            from vllm.distributed import get_engram_dp_group

            # Worker initialization already created the node-local DP x PCP sharing group.
            engram_group = get_engram_dp_group()
            if engram_group is not None:
                groups.append(engram_group)
        # A drafter is loaded only on PP-last; its siblings cannot join a PP collective.
        if not is_draft and getattr(parallel, "pipeline_parallel_size", 1) > 1:
            groups.append(get_pp_group())
        self.groups = [group.cpu_group for group in groups if group.world_size > 1]

    def flags(self, values: list[bool], *, all_ranks: bool = True) -> list[bool]:
        flags = torch.tensor(values, dtype=torch.int32, device="cpu")
        operation = torch.distributed.ReduceOp.MIN if all_ranks else torch.distributed.ReduceOp.MAX
        for group in self.groups:
            torch.distributed.all_reduce(flags, op=operation, group=group)
        return [bool(value) for value in flags.tolist()]

    @contextmanager
    def phase(self, name: str) -> Iterator[None]:
        error: Exception | None = None
        try:
            yield
        except Exception as exc:
            error = exc
        if not self.flags([error is None])[0]:
            detail = f": {error}" if error is not None else ""
            raise RuntimeError(f"RFork {name} failed on a participating rank{detail}") from error

    def check_identity(self, identity: str) -> None:
        digest = torch.tensor(list(hashlib.sha256(identity.encode()).digest()), dtype=torch.int32, device="cpu")
        minimum, maximum = digest.clone(), digest.clone()
        for group in self.groups:
            torch.distributed.all_reduce(minimum, op=torch.distributed.ReduceOp.MIN, group=group)
            torch.distributed.all_reduce(maximum, op=torch.distributed.ReduceOp.MAX, group=group)
        if not torch.equal(minimum, maximum):
            raise ValueError("Async RFork ranks disagree on the logical model identity")


def race_sources(
    session,
    mount,
    coordinator: RForkLoadCoordinator,
    retry_interval_sec: float,
    *,
    mount_required_for_preparation: bool = False,
) -> str:
    """Select one source for all participants; the loader owns cleanup of unused leases."""
    seed_acquired_at = float("-inf") if session.seed_lease is not None else float("inf")
    acquisition_error: Exception | None = None
    reported_miss = False
    should_wait = False
    initial_lookup_complete = False
    log = logger.info if session.identity.tp_rank == 0 else logger.debug

    while True:
        has_seed = session.seed_lease is not None and session.state is RForkLifecycleState.LEASED
        mount_ready = (
            mount.done.is_set()
            and mount.error is None
            and mount.path is not None
            and mount.completed_at is not None
            and mount.completed_at <= mount.deadline
        )
        valid_state = (session.state is RForkLifecycleState.REGISTERED and session.seed_lease is None) or has_seed
        stop_event = getattr(session, "lease_release_stop_event", None)
        stopping = getattr(session, "_shutdown_in_progress", False) or (stop_event is not None and stop_event.is_set())
        ready_seed, ready_mount, healthy, within_deadline = coordinator.flags(
            [
                has_seed and seed_acquired_at <= mount.deadline,
                mount_ready,
                acquisition_error is None and valid_state and not stopping,
                mount.remaining() > 0,
            ]
        )
        if ready_seed and healthy:
            if reported_miss:
                log("RFork seed retry won fallback race.")
                log("RFork seed is available after retry, continue RFork transfer.")
            return "seed"
        # Models requiring mounted files still get one seed lookup within the deadline.
        if ready_mount and (
            not mount_required_for_preparation or initial_lookup_complete or not healthy or not within_deadline
        ):
            log("Async model mount won RFork fallback race.")
            return "mount"
        if not healthy:
            raise RuntimeError(
                "RFork seed acquisition cannot be retried on a participating rank"
            ) from acquisition_error
        if not within_deadline:
            raise TimeoutError("Timed out racing RFork seed acquisition with async model mount") from mount.error

        if should_wait:
            delay = min(retry_interval_sec, mount.remaining())
            if mount.done.is_set():
                time.sleep(delay)
            else:
                mount.done.wait(delay)
            should_wait = False
            continue

        if not has_seed:
            remaining = mount.remaining()
            if remaining > 0:
                try:
                    acquired = session.acquire_seed(timeout_sec=remaining)
                    if acquired:
                        seed_acquired_at = time.monotonic()
                    elif not reported_miss:
                        log("Initial RFork seed lookup failed; racing seed retry with async model mount fallback.")
                        reported_miss = True
                except Exception as exc:
                    acquisition_error = exc
        # Every rank checks newly available sources before the next retry interval.
        initial_lookup_complete = True
        should_wait = True
