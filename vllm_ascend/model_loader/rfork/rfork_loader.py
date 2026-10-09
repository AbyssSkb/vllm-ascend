#
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

import gc
import logging
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager
from copy import copy
from dataclasses import dataclass
from functools import wraps
from threading import get_ident
from typing import Any, cast

import torch
import torch.nn as nn
from torch.nn import Module
from torch.utils.hooks import RemovableHandle
from vllm.config import ModelConfig, VllmConfig
from vllm.config.load import LoadConfig
from vllm.distributed import get_tensor_model_parallel_rank
from vllm.distributed.parallel_state import get_ep_group, get_pp_group
from vllm.logger import logger
from vllm.model_executor.model_loader import register_model_loader
from vllm.model_executor.model_loader.base_loader import BaseModelLoader
from vllm.model_executor.model_loader.utils import (
    initialize_model,
    process_weights_after_loading,
)
from vllm.utils.torch_utils import set_default_torch_dtype

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.model_loader.rfork.config import RForkConfig
from vllm_ascend.model_loader.rfork.identity import (
    _resolve_sharded_dp_rank,
    build_compatibility_fingerprint,
)
from vllm_ascend.model_loader.rfork.safety import mutable_weights_bypass_reason
from vllm_ascend.model_loader.rfork.session import RForkSession
from vllm_ascend.model_loader.rfork.types import (
    RForkFallbackCleanupResult,
    RForkIdentity,
    RForkLifecycleState,
    RForkSeedServiceStartResult,
)
from vllm_ascend.utils import ACL_FORMAT_FRACTAL_NZ, ACL_FORMAT_ND, is_310p


class _RForkSeedUnavailable(RuntimeError):
    pass


FALLBACK_CLEANUP_MAX_ATTEMPTS = 2
FALLBACK_MEMORY_RECLAIM_PASSES = 4
RFORK_FALLBACK_EXCEPTIONS = (ImportError, OSError, RuntimeError, ValueError)
INITIAL_ASCEND_MOE_COUNTER = -1
POST_LOAD_TIMING_SAMPLE_LIMIT = 10


@dataclass
class _RForkProcessGlobalModelState:
    """Snapshot process-global model registries before an RFork model attempt."""

    static_forward_context: tuple[dict[Any, Any], dict[Any, Any]] | None
    static_all_moe_layers: tuple[list[Any], list[Any]] | None
    rope_cache: dict[Any, Any] | None
    ascend_moe_layers: tuple[list[Any], list[Any]] | None = None
    ascend_moe_counter: int = INITIAL_ASCEND_MOE_COUNTER
    ascend_rope_caches: tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None] | None = None
    dynamo_bytecode_hook_ids: frozenset[int] = frozenset()


def _is_rfork_summary_rank(session: RForkSession) -> bool:
    identity = getattr(session, "identity", None)
    return getattr(identity, "tp_rank", 0) == 0


def _rfork_model_kind(session: RForkSession) -> str:
    identity = getattr(session, "identity", None)
    return "draft" if getattr(identity, "is_draft_model", False) else "main"


def _log_rfork_load_summary(session: RForkSession, source: str, started_at: float) -> None:
    # Include synchronous startup, but not deferred promotion after loader return.
    log_summary = logger.info if _is_rfork_summary_rank(session) else logger.debug
    log_summary(
        "RFork %s model loading completed: source=%s, elapsed=%.2fs",
        _rfork_model_kind(session),
        source,
        time.perf_counter() - started_at,
    )


def _start_rfork_seed_service(
    session: RForkSession,
    model: Module,
    processed_layout: bool,
    exclude_blocks: list[tuple[int, int]],
    *,
    load_source: str,
    structural_digest: str | None = None,
) -> bool:
    try:
        result = session.start_seed_service(
            model, processed_layout, exclude_blocks, structural_digest=structural_digest
        )
    except Exception:
        logger.exception(
            "RFork %s model loaded from %s, but seed service startup raised; inference can continue.",
            _rfork_model_kind(session),
            load_source,
        )
        return False
    if result is RForkSeedServiceStartResult.DEFERRED:
        return False
    started = bool(result)
    if not started:
        logger.warning(
            "RFork %s model loaded from %s is ready, but seed service startup failed; inference can continue.",
            _rfork_model_kind(session),
            load_source,
        )
    return started


def _is_mtp_hf_config(hf_config: object | None) -> bool:
    if hf_config is None:
        return False

    model_type = getattr(hf_config, "model_type", None)
    if isinstance(model_type, str) and model_type.lower().endswith("_mtp"):
        return True

    architectures = getattr(hf_config, "architectures", None)
    if isinstance(architectures, str):
        architectures = [architectures]
    if not isinstance(architectures, (list, tuple)):
        return False

    return any(isinstance(architecture, str) and architecture.endswith("MTPModel") for architecture in architectures)


def _is_draft_model_config(model_config: object | None) -> bool:
    if model_config is None:
        return False
    if getattr(model_config, "runner_type", None) == "draft":
        return True

    return any(
        _is_mtp_hf_config(getattr(model_config, hf_config_attr, None))
        for hf_config_attr in ("hf_config", "hf_text_config")
    )


def _is_draft_model(vllm_config: VllmConfig, model_config: ModelConfig | None = None) -> bool:
    return (
        _is_draft_model_config(model_config)
        or _is_draft_model_config(getattr(vllm_config, "model_config", None))
        or _is_draft_model_config(getattr(vllm_config, "scheduler_config", None))
    )


def _get_rfork_session_attr(vllm_config: VllmConfig, model_config: ModelConfig) -> str:
    return "rfork_draft_session" if _is_draft_model(vllm_config, model_config) else "rfork_session"


def _get_ep_rank(vllm_config: VllmConfig) -> int | None:
    parallel_config = vllm_config.parallel_config
    if not parallel_config.enable_expert_parallel or getattr(parallel_config, "is_moe_model", None) is False:
        return None

    try:
        return get_ep_group().rank_in_group
    except AssertionError as e:
        raise RuntimeError("Expert parallelism is enabled, but the EP group is not initialized.") from e


def _get_pp_rank(vllm_config: VllmConfig) -> int | None:
    if getattr(vllm_config.parallel_config, "pipeline_parallel_size", 1) <= 1:
        return None

    try:
        return get_pp_group().rank_in_group
    except AssertionError as e:
        raise RuntimeError("Pipeline parallelism is enabled, but the PP group is not initialized.") from e


def _make_fallback_load_config(load_config: LoadConfig) -> LoadConfig:
    fallback_load_config = copy(load_config)
    fallback_load_config.load_format = "auto"
    fallback_load_config.model_loader_extra_config = {}
    return fallback_load_config


def _load_with_default_loader(
    vllm_config: VllmConfig,
    model_config: ModelConfig,
    load_config: LoadConfig,
    prefix: str,
) -> Module | None:
    from vllm.model_executor.model_loader import get_model

    return get_model(
        vllm_config=vllm_config,
        model_config=model_config,
        load_config=_make_fallback_load_config(load_config),
        prefix=prefix,
    )


def _get_dynamo_bytecode_hooks() -> dict[int, Any]:
    """Return Dynamo's loaded hook registry without importing Dynamo."""
    module = sys.modules.get("torch._dynamo.convert_frame")
    hooks = getattr(module, "_bytecode_hooks", None)
    return hooks if isinstance(hooks, dict) else {}


def _remove_discarded_compilation_hooks(
    stale_module_ids: set[int], snapshot: _RForkProcessGlobalModelState | None
) -> None:
    """Detach compiler callbacks owned by a discarded RFork model."""
    hooks = _get_dynamo_bytecode_hooks()
    removed_count = 0
    for hook_id, hook in list(hooks.items()):
        if snapshot is not None and hook_id in snapshot.dynamo_bytecode_hook_ids:
            continue
        owner = getattr(hook, "__self__", None)
        if not isinstance(owner, Module):
            continue
        if snapshot is None and id(owner) not in stale_module_ids:
            continue
        handle = getattr(owner, "_bytecode_hook_handle", None)
        if isinstance(handle, RemovableHandle) and handle.id == hook_id and handle.hooks_dict_ref() is hooks:
            handle.remove()
            removed_count += 1
    if removed_count:
        logger.info("RFork fallback removed %d discarded model compilation hooks.", removed_count)


def _snapshot_process_global_model_state(vllm_config: VllmConfig) -> _RForkProcessGlobalModelState:
    """Snapshot registries that model construction can mutate in the process."""
    static_forward_context: tuple[dict[Any, Any], dict[Any, Any]] | None = None
    static_all_moe_layers: tuple[list[Any], list[Any]] | None = None
    compilation_config = getattr(vllm_config, "compilation_config", None)
    if compilation_config is not None:
        forward_context = getattr(compilation_config, "static_forward_context", None)
        if isinstance(forward_context, dict):
            static_forward_context = (
                forward_context,
                dict(forward_context),
            )
        moe_layers = getattr(compilation_config, "static_all_moe_layers", None)
        if isinstance(moe_layers, list):
            static_all_moe_layers = (moe_layers, list(moe_layers))

    rope_cache: dict[Any, Any] | None = None
    try:
        from vllm.model_executor.layers.rotary_embedding import _ROPE_DICT

        if isinstance(_ROPE_DICT, dict):
            rope_cache = dict(_ROPE_DICT)
    except Exception as e:  # pragma: no cover - best-effort across vLLM versions
        logger.debug("RFork fallback: skip snapshotting _ROPE_DICT: %s", e)

    ascend_rope_caches: tuple[torch.Tensor | None, torch.Tensor | None, torch.Tensor | None] | None = None
    rope_ops_module = sys.modules.get("vllm_ascend.ops.rotary_embedding")
    if rope_ops_module is not None:
        ascend_rope_caches = (
            getattr(rope_ops_module, "_cos_sin_cache", None),
            getattr(rope_ops_module, "_cos_cache", None),
            getattr(rope_ops_module, "_sin_cache", None),
        )

    adaptor = getattr(sys.modules.get("vllm_ascend.eplb.adaptor.vllm_adaptor"), "VllmEplbAdaptor", None)
    registry = getattr(adaptor, "_registered_moe_layers", None)
    ascend_moe_layers = (registry, list(registry)) if isinstance(registry, list) else None
    moe_runner = getattr(sys.modules.get("vllm_ascend.ops.fused_moe.fused_moe"), "AscendMoERunner", None)
    ascend_moe_counter = getattr(moe_runner, "moe_counter", INITIAL_ASCEND_MOE_COUNTER)
    return _RForkProcessGlobalModelState(
        static_forward_context,
        static_all_moe_layers,
        rope_cache,
        ascend_moe_layers,
        ascend_moe_counter,
        ascend_rope_caches,
        frozenset(_get_dynamo_bytecode_hooks()),
    )


def _reset_process_global_model_state(
    vllm_config: VllmConfig,
    model: Module | None = None,
    snapshot: _RForkProcessGlobalModelState | None = None,
) -> None:
    """Restore process-global registries to their pre-attempt state."""
    stale_module_ids = {id(module) for module in model.modules()} if model is not None else set()
    _remove_discarded_compilation_hooks(stale_module_ids, snapshot)

    adaptor = getattr(sys.modules.get("vllm_ascend.eplb.adaptor.vllm_adaptor"), "VllmEplbAdaptor", None)
    registry = getattr(adaptor, "_registered_moe_layers", None)
    if snapshot is not None and snapshot.ascend_moe_layers is not None:
        baseline_registry, baseline_layers = snapshot.ascend_moe_layers
        baseline_registry[:] = baseline_layers
        if adaptor is not None and registry is not baseline_registry:
            adaptor._registered_moe_layers = baseline_registry
    elif isinstance(registry, list):
        if snapshot is not None or not stale_module_ids:
            registry.clear()
        else:
            registry[:] = [layer for layer in registry if id(layer) not in stale_module_ids]

    if snapshot is not None:
        moe_runner = getattr(sys.modules.get("vllm_ascend.ops.fused_moe.fused_moe"), "AscendMoERunner", None)
        if moe_runner is not None:
            moe_runner.moe_counter = snapshot.ascend_moe_counter
    removed_names: set[Any] = set()
    compilation_config = getattr(vllm_config, "compilation_config", None)
    if compilation_config is not None:
        static_forward_context = getattr(compilation_config, "static_forward_context", None)
        if snapshot is not None and snapshot.static_forward_context is not None:
            baseline_context, baseline_context_values = snapshot.static_forward_context
            baseline_context.clear()
            baseline_context.update(baseline_context_values)
            if static_forward_context is not baseline_context:
                compilation_config.static_forward_context = baseline_context
        elif isinstance(static_forward_context, dict):
            for name, module in list(static_forward_context.items()):
                if stale_module_ids and id(module) in stale_module_ids:
                    removed_names.add(name)
                    del static_forward_context[name]
            if not stale_module_ids:
                static_forward_context.clear()

        static_all_moe_layers = getattr(compilation_config, "static_all_moe_layers", None)
        if snapshot is not None and snapshot.static_all_moe_layers is not None:
            baseline_moe_layers, baseline_moe_values = snapshot.static_all_moe_layers
            baseline_moe_layers[:] = baseline_moe_values
            if static_all_moe_layers is not baseline_moe_layers:
                compilation_config.static_all_moe_layers = baseline_moe_layers
        elif isinstance(static_all_moe_layers, list):
            if stale_module_ids:
                static_all_moe_layers[:] = [
                    layer
                    for layer in static_all_moe_layers
                    if id(layer) not in stale_module_ids and (not isinstance(layer, str) or layer not in removed_names)
                ]
            else:
                static_all_moe_layers.clear()

    try:
        from vllm.model_executor.layers.rotary_embedding import _ROPE_DICT

        if isinstance(_ROPE_DICT, dict):
            if snapshot is not None and snapshot.rope_cache is not None:
                _ROPE_DICT.clear()
                _ROPE_DICT.update(snapshot.rope_cache)
            else:
                _ROPE_DICT.clear()
    except Exception as e:  # pragma: no cover - best-effort across vLLM versions
        logger.debug("RFork fallback: skip resetting _ROPE_DICT: %s", e)

    rope_ops_module = cast(Any, sys.modules.get("vllm_ascend.ops.rotary_embedding"))
    if rope_ops_module is not None:
        if snapshot is not None and snapshot.ascend_rope_caches is not None:
            rope_ops_module._cos_sin_cache, rope_ops_module._cos_cache, rope_ops_module._sin_cache = (
                snapshot.ascend_rope_caches
            )
        else:
            rope_ops_module._cos_sin_cache = None
            rope_ops_module._cos_cache = None
            rope_ops_module._sin_cache = None


def _iter_ascend_moe_quant_methods(model: Module) -> Iterator[Any]:
    """Yield each quant method owned by an Ascend MoE runner once."""
    from vllm_ascend.ops.fused_moe.fused_moe import AscendMoERunner

    seen_quant_methods: set[int] = set()
    for module in model.modules():
        if not isinstance(module, AscendMoERunner):
            continue

        # AscendMoERunner exposes the routed experts' method via private _quant_method.
        quant_method = getattr(module, "_quant_method", None)
        if quant_method is None or id(quant_method) in seen_quant_methods:
            continue

        seen_quant_methods.add(id(quant_method))
        yield cast(Any, quant_method)


@contextmanager
def _rfork_pre_transfer_weight_processing(model: Module):
    """Use the unwrapped MoE post-load step so RFork pre-transfer skips shared-expert validation."""
    restored: list[tuple[Any, object]] = []
    for quant_method in _iter_ascend_moe_quant_methods(model):
        process_weights = getattr(quant_method, "process_weights_after_loading", None)
        original_process_weights = getattr(process_weights, "__wrapped__", None)
        if original_process_weights is None:
            continue

        restored.append((quant_method, process_weights))
        quant_method.process_weights_after_loading = original_process_weights

    try:
        yield
    finally:
        for quant_method, process_weights in restored:
            quant_method.process_weights_after_loading = process_weights


@contextmanager
def _rfork_allocate_nz_weights(session: RForkSession):
    """Allocate uninitialized NZ destinations; the seed supplies their final bytes."""
    import torch_npu

    original_cast = torch_npu.npu_format_cast
    storage_owner = getattr(torch_npu._C, "_tensor_construct_from_storage", None)
    owner_thread = get_ident()
    allocated = 0
    logical_bytes = 0
    allocation_host = 0.0
    first_allocation = None

    def allocate(self, acl_format, *args, **kwargs):
        nonlocal allocated, logical_bytes, allocation_host, first_allocation
        weight = self
        if (
            get_ident() != owner_thread
            or acl_format != ACL_FORMAT_FRACTAL_NZ
            or weight.dtype != torch.int8
            or weight.device.type != "npu"
            or len(weight.shape) < 2
            or not weight.is_contiguous()
            or args
            or kwargs
        ):
            return original_cast(weight, acl_format, *args, **kwargs)
        source_format = torch_npu.get_npu_format(weight)
        if source_format == ACL_FORMAT_FRACTAL_NZ:
            return weight
        weight_bytes = weight.numel() * weight.element_size()
        storage = weight.untyped_storage()
        if (
            source_format != ACL_FORMAT_ND
            or weight.storage_offset() != 0
            or storage.nbytes() != weight_bytes
            or storage_owner is None
        ):
            return original_cast(weight, acl_format, *args, **kwargs)
        try:
            owner = storage_owner(storage)
        except (RuntimeError, TypeError):
            return original_cast(weight, acl_format, *args, **kwargs)
        # Native format_cast allocates from the storage descriptor, not a reshaped view.
        if owner.shape != weight.shape or owner.stride() != weight.stride() or owner.dtype != weight.dtype:
            return original_cast(weight, acl_format, *args, **kwargs)
        started_at = time.perf_counter()
        result = torch_npu.empty_with_format(
            size=weight.shape, dtype=weight.dtype, device=weight.device, acl_format=acl_format
        )
        allocation_host += time.perf_counter() - started_at
        if torch_npu.get_npu_format(result) != ACL_FORMAT_FRACTAL_NZ:
            raise RuntimeError("RFork NZ destination allocation did not preserve format 29")
        allocated += 1
        logical_bytes += weight_bytes
        if first_allocation is None:
            first_allocation = {
                "shape": tuple(result.shape),
                "dtype": str(result.dtype),
                "format": torch_npu.get_npu_format(result),
                "storage_bytes": result.untyped_storage().nbytes(),
            }
        return result

    # ponytail: serial worker loading only; overlapping loader contexts need isolated dispatch.
    torch_npu.npu_format_cast = allocate
    try:
        yield
    finally:
        torch_npu.npu_format_cast = original_cast
        logger.debug(
            "RFork %s NZ destination allocations: tp_rank=%s allocated=%s logical_bytes=%s "
            "allocation_host=%.4fs first=%s",
            _rfork_model_kind(session),
            getattr(session.identity, "tp_rank", 0),
            allocated,
            logical_bytes,
            allocation_host,
            first_allocation,
        )


def _is_dynamic_eplb_enabled(vllm_config: VllmConfig) -> bool:
    parallel_config = getattr(vllm_config, "parallel_config", None)
    if bool(getattr(parallel_config, "enable_eplb", False)):
        return True

    eplb_config = get_ascend_config().eplb_config
    return eplb_config.dynamic_eplb or bool(eplb_config.expert_map_record_path)


@contextmanager
def _rfork_skip_unquantized_moe_post_load_processing(model: Module):
    """Suppress unquantized MoE post-load processing; dense layers still run theirs."""

    from vllm_ascend.ops.fused_moe.fused_moe import AscendUnquantizedFusedMoEMethod

    restored_methods: list[tuple[Any, object]] = []
    for quant_method in _iter_ascend_moe_quant_methods(model):
        if not isinstance(quant_method, AscendUnquantizedFusedMoEMethod):
            continue

        process_weights = getattr(quant_method, "process_weights_after_loading", None)
        if process_weights is None:
            continue

        restored_methods.append((quant_method, process_weights))
        quant_method.process_weights_after_loading = _noop_process_weights_after_loading  # type: ignore[method-assign]

    try:
        yield
    finally:
        for quant_method, process_weights in restored_methods:
            quant_method.process_weights_after_loading = process_weights


def _noop_process_weights_after_loading(*args: Any, **kwargs: Any) -> None:
    pass


def _refresh_rfork_flatquant_state(model: Module) -> None:
    """Refresh FlatQuant's existing host cache without rerunning layout conversion."""
    for module in model.modules():
        if not hasattr(module, "aclnn_clip_ratio"):
            continue
        clip_ratio = getattr(module, "clip_ratio", None)
        if not isinstance(clip_ratio, torch.Tensor) or clip_ratio.numel() != 1:
            raise ValueError("RFork FlatQuant runtime state requires a scalar clip_ratio tensor")
        # Refresh the host scalar once after transfer without rewriting device storage.
        module.aclnn_clip_ratio = clip_ratio.item()


@contextmanager
def _rfork_post_load_timing(model: Module, session: RForkSession):
    """Measure inclusive host hook time without adding device synchronization."""
    # Host time includes allocation, operator submission and implicit device waits.
    if not logger.isEnabledFor(logging.DEBUG):
        yield
        return

    modules = list(model.named_modules())
    module_names = {id(module): name or "<root>" for name, module in modules}
    missing = object()
    restored: list[tuple[Any, Any]] = []
    wrapped_owners: set[int] = set()
    records: list[tuple[str, str, str, float, bool]] = []
    active_hook = False
    linear_stages: dict[str, Any] | None = None

    def wrap_hook(owner: Any, stage: str, module_name: str) -> None:
        if owner is None or id(owner) in wrapped_owners:
            return
        original = getattr(owner, "process_weights_after_loading", None)
        if not callable(original):
            return
        implementation = getattr(owner, "quant_method" if stage == "quant" else "impl", None)
        implementation_name = type(owner if implementation is None else implementation).__qualname__

        @wraps(original)
        def timed_hook(*args: Any, **kwargs: Any):
            nonlocal active_hook, linear_stages
            # A module hook can call another wrapped hook; count it only in its outer owner.
            if active_hook:
                return original(*args, **kwargs)
            active_hook = True
            started_at = time.perf_counter()
            layer = args[0] if args else kwargs.get("layer")
            name = module_names.get(id(layer), module_name) if stage == "quant" else module_name
            stages = None
            if (
                stage == "quant"
                and implementation_name == "AscendW8A8DynamicLinearMethod"
                and name.rsplit(".", 1)[-1] == "wq_a"
                and linear_stages is None
                and isinstance(layer, Module)
            ):
                stages = linear_stages = {"module": name}
                previous_stages = vars(layer).get("_rfork_post_load_stages", missing)
                layer._rfork_post_load_stages = stages
            succeeded = False
            try:
                result = original(*args, **kwargs)
                succeeded = True
                return result
            finally:
                elapsed = time.perf_counter() - started_at
                records.append((stage, name, implementation_name, elapsed, succeeded))
                if stages is not None:
                    stages.update(host_s=elapsed, ok=succeeded)
                    if previous_stages is missing:
                        del layer._rfork_post_load_stages
                    else:
                        layer._rfork_post_load_stages = previous_stages
                active_hook = False

        restored.append((owner, vars(owner).get("process_weights_after_loading", missing)))
        wrapped_owners.add(id(owner))
        owner.process_weights_after_loading = timed_hook

    process_started_at = None
    try:
        for name, module in modules:
            wrap_hook(getattr(module, "quant_method", None), "quant", name or "<root>")
            wrap_hook(module, "attention", name or "<root>")
        process_started_at = time.perf_counter()
        yield
    finally:
        process_host = 0.0 if process_started_at is None else time.perf_counter() - process_started_at
        for owner, previous in reversed(restored):
            if previous is missing:
                del owner.process_weights_after_loading
            else:
                owner.process_weights_after_loading = previous
        groups: dict[tuple[str, str], dict[str, Any]] = {}
        for stage, _, implementation, elapsed, succeeded in records:
            group = groups.setdefault((stage, implementation), {"calls": 0, "host_s": 0.0, "failures": 0})
            group["calls"] += 1
            group["host_s"] += elapsed
            group["failures"] += not succeeded
        hook_host = sum(record[3] for record in records)
        slowest = [
            {"stage": stage, "module": name, "implementation": implementation, "host_s": elapsed, "ok": succeeded}
            for stage, name, implementation, elapsed, succeeded in sorted(
                records, key=lambda record: record[3], reverse=True
            )[:POST_LOAD_TIMING_SAMPLE_LIMIT]
        ]
        logger.debug(
            "RFork %s post-load hooks: tp_rank=%s process_host=%.4fs hook_host=%.4fs other_host=%.4fs "
            "calls=%d groups=%s slowest=%s",
            _rfork_model_kind(session),
            getattr(session.identity, "tp_rank", 0),
            process_host,
            hook_host,
            process_host - hook_host,
            len(records),
            groups,
            slowest,
        )
        if linear_stages is not None:
            logger.debug(
                "RFork %s first W8A8 linear stages: tp_rank=%s details=%s",
                _rfork_model_kind(session),
                getattr(session.identity, "tp_rank", 0),
                linear_stages,
            )


@register_model_loader("rfork")
class RForkModelLoader(BaseModelLoader):
    def __init__(self, load_config: LoadConfig):
        super().__init__(load_config)
        self.rfork_config = RForkConfig.from_extra_config(load_config.model_loader_extra_config)

        config = self.rfork_config
        logger.debug(
            "Initializing rfork with config: "
            "MODEL_URL=%s, MODEL_DEPLOY_STRATEGY_NAME=%s, "
            "SCHEDULER_URL=%s, SEED_TIMEOUT_SEC=%s, REQUEST_TIMEOUT_SEC=%s, "
            "SEED_BIND_HOST=%s, SEED_ADVERTISE_HOST=%s",
            config.model_url,
            config.model_deploy_strategy_name,
            config.planner_url,
            config.seed_timeout_sec,
            config.request_timeout_sec,
            config.seed_bind_host,
            config.seed_advertise_host,
        )

    def download_model(self, model_config: ModelConfig) -> None:
        raise NotImplementedError

    def load_weights(self, model: nn.Module, model_config: ModelConfig) -> None:
        raise NotImplementedError

    def _ensure_rfork_session(self, vllm_config: VllmConfig, model_config: ModelConfig) -> RForkSession:
        session_attr = _get_rfork_session_attr(vllm_config, model_config)
        # Store runtime sessions on the process-lifetime LoadConfig shared by both loaders.
        session = getattr(self.load_config, session_attr, None)
        if session is None:
            is_draft_model = _is_draft_model(vllm_config, model_config)
            global_rank = torch.distributed.get_rank()
            pp_rank = _get_pp_rank(vllm_config)
            ep_rank = _get_ep_rank(vllm_config)
            tp_rank = get_tensor_model_parallel_rank()
            compatibility_fingerprint = build_compatibility_fingerprint(
                vllm_config,
                model_config,
                model_url=self.rfork_config.model_url,
                model_deploy_strategy_name=self.rfork_config.model_deploy_strategy_name,
            )
            identity = RForkIdentity(
                tp_rank=tp_rank,
                global_rank=global_rank,
                is_draft_model=is_draft_model,
                pp_rank=pp_rank,
                ep_rank=ep_rank,
                compatibility_fingerprint=compatibility_fingerprint,
                sharded_dp_rank=_resolve_sharded_dp_rank(vllm_config),
            )
            session = RForkSession(self.rfork_config, identity)
            setattr(self.load_config, session_attr, session)
            if tp_rank == 0:
                logger.info(
                    "RFork %s session group initialized: model=%s, strategy=%s, planner=%s, "
                    "pp_rank=%s, ep_rank=%s, sharded_dp_rank=%s, fingerprint=%s",
                    "draft" if is_draft_model else "main",
                    self.rfork_config.model_url,
                    self.rfork_config.model_deploy_strategy_name,
                    self.rfork_config.planner_url,
                    pp_rank,
                    ep_rank,
                    identity.sharded_dp_rank,
                    compatibility_fingerprint,
                )
            else:
                logger.debug(
                    "RFork session initialized: model_kind=%s, tp_rank=%s, global_rank=%s, "
                    "pp_rank=%s, ep_rank=%s, sharded_dp_rank=%s, session_attr=%s, fingerprint=%s",
                    "draft" if is_draft_model else "main",
                    tp_rank,
                    global_rank,
                    pp_rank,
                    ep_rank,
                    identity.sharded_dp_rank,
                    session_attr,
                    compatibility_fingerprint,
                )
        return session

    def _get_target_registered_blocks(
        self,
        vllm_config: VllmConfig,
        model_config: ModelConfig,
    ) -> list[tuple[int, int]]:
        if not _is_draft_model(vllm_config, model_config):
            return []
        target_load_config = getattr(vllm_config, "load_config", None)
        target_session = getattr(target_load_config, "rfork_session", None)
        if target_session is None:
            target_session = getattr(self.load_config, "rfork_session", None)
        target_transfer_backend = getattr(target_session, "transfer_backend", None)
        if target_transfer_backend is None:
            return []
        # Snapshot under the backend lock while target storage stays alive during draft preparation.
        return target_transfer_backend.snapshot_registered_weight_blocks()

    def _requires_processed_layout_transfer(self, model_config: ModelConfig) -> bool:
        if getattr(model_config, "quantization", None) is not None:
            return True

        try:
            weight_nz_mode = getattr(get_ascend_config(), "weight_nz_mode", 0)
            if not isinstance(weight_nz_mode, bool) and int(weight_nz_mode) == 2:
                return True
        except (TypeError, ValueError, RuntimeError):
            # AscendConfig may be unavailable during early or CPU-only loader inspection.
            pass

        try:
            return is_310p()
        except (AttributeError, RuntimeError):
            return False

    def load_model(
        self,
        vllm_config: VllmConfig,
        model_config: ModelConfig,
        prefix: str = "",
    ) -> Module | None:
        load_started_at = time.perf_counter()
        device_config = vllm_config.device_config
        load_config = self.load_config
        load_device = device_config.device if load_config.device is None else load_config.device
        target_device = torch.device(load_device)

        with set_default_torch_dtype(model_config.dtype):
            model_init_started = False
            model: Module | None = None
            session: RForkSession | None = None
            model_state_snapshot: _RForkProcessGlobalModelState | None = None
            exclude_blocks: list[tuple[int, int]] = []
            processed_layout_transfer = self._requires_processed_layout_transfer(model_config)
            bypass_reason = mutable_weights_bypass_reason(vllm_config, model_config)
            if bypass_reason is None:
                eplb_config = get_ascend_config().eplb_config
                if getattr(eplb_config, "expert_map_path", None) is not None:
                    bypass_reason = "static expert placement (expert_map_path)"
                elif _is_dynamic_eplb_enabled(vllm_config):
                    bypass_reason = "dynamic EPLB"

            if bypass_reason is not None:
                logger.warning(
                    "RFork transfer is disabled when %s is enabled; using the default model loader.",
                    bypass_reason,
                )
                return _load_with_default_loader(vllm_config, model_config, self.load_config, prefix)

            try:
                # Keep session and TransferEngine initialization inside fallback-protected RFork loading.
                session = self._ensure_rfork_session(vllm_config, model_config)
                # Avoid re-registering target-model storage shared by draft workers.
                exclude_blocks = self._get_target_registered_blocks(vllm_config, model_config)
                model_state_snapshot = _snapshot_process_global_model_state(vllm_config)
                model_init_started = True
                model_init_start_time = time.perf_counter()
                with target_device:
                    model = initialize_model(
                        vllm_config=vllm_config,
                        model_config=model_config,
                        prefix=prefix,
                    )
                logger.debug(
                    "RFork %s model initialization took %.2f seconds",
                    _rfork_model_kind(session),
                    time.perf_counter() - model_init_start_time,
                )

                if exclude_blocks and session.can_reuse_shared_weights(
                    model, processed_layout_transfer, exclude_blocks
                ):
                    # Skip hooks because reprocessing can mutate or rebind target-shared storage.
                    model = model.eval()
                    _log_rfork_load_summary(session, "shared_target", load_started_at)
                    return model

                if processed_layout_transfer:
                    init_sync_started_at = time.perf_counter()
                    torch.npu.synchronize(target_device)
                    logger.debug(
                        "RFork %s post-load initialization drain: tp_rank=%s synchronize_wait=%.4fs",
                        _rfork_model_kind(session),
                        getattr(session.identity, "tp_rank", 0),
                        time.perf_counter() - init_sync_started_at,
                    )
                    layout_start_time = time.perf_counter()
                    logger.debug(
                        "RFork %s model uses post-load tensor layout transfer.",
                        _rfork_model_kind(session),
                    )
                    process_started_at = time.perf_counter()
                    with (
                        _rfork_pre_transfer_weight_processing(model),
                        _rfork_allocate_nz_weights(session),
                        _rfork_post_load_timing(model, session),
                    ):
                        process_weights_after_loading(model, model_config, target_device)
                    process_host = time.perf_counter() - process_started_at
                    # Complete async NPU layout conversion before exposing buffers.
                    sync_started_at = time.perf_counter()
                    torch.npu.synchronize()
                    sync_wait = time.perf_counter() - sync_started_at
                    logger.debug(
                        "RFork %s layout processing stages: process_host=%.4fs synchronize_wait=%.4fs total=%.4fs",
                        _rfork_model_kind(session),
                        process_host,
                        sync_wait,
                        time.perf_counter() - layout_start_time,
                    )
                    logger.debug(
                        "RFork %s model layout processing took %.2f seconds",
                        _rfork_model_kind(session),
                        time.perf_counter() - layout_start_time,
                    )

                weight_load_start_time = time.perf_counter()
                if not session.register_destination(model, processed_layout_transfer, exclude_blocks):
                    raise RuntimeError("destination registration failed.")

                acquire_seed_start_time = time.perf_counter()
                try:
                    acquired_seed = session.acquire_seed()
                finally:
                    logger.debug(
                        "RFork %s seed acquisition took %.2f seconds",
                        _rfork_model_kind(session),
                        time.perf_counter() - acquire_seed_start_time,
                    )
                if not acquired_seed:
                    raise _RForkSeedUnavailable("planner returned no compatible seed")

                if not session.transfer_from_seed(model, processed_layout_transfer):
                    raise RuntimeError("transfer failed.")
                logger.debug(
                    "RFork %s model registration and transfer took %.2f seconds",
                    _rfork_model_kind(session),
                    time.perf_counter() - weight_load_start_time,
                )

                if processed_layout_transfer:
                    _refresh_rfork_flatquant_state(model)
                else:
                    with _rfork_skip_unquantized_moe_post_load_processing(model):
                        process_weights_after_loading(model, model_config, target_device)

                # Advertise only after post-load and eval; the session owns failure cleanup.
                model = model.eval()
                structural_digest = session.log_transferred_model_layout(model, processed_layout_transfer)
                _start_rfork_seed_service(
                    session,
                    model,
                    processed_layout_transfer,
                    exclude_blocks,
                    load_source="transfer",
                    structural_digest=structural_digest,
                )
                _log_rfork_load_summary(session, "transfer", load_started_at)
                return model
            except _RForkSeedUnavailable as exc:
                if session is None:
                    raise RuntimeError("RFork seed acquisition failed without an active session") from exc
                fallback_source = "local"
                log_seed_miss = logger.info if _is_rfork_summary_rank(session) else logger.debug
                log_seed_miss(
                    "RFork %s seed acquisition was unsuccessful; loading locally.",
                    _rfork_model_kind(session),
                )
            except RFORK_FALLBACK_EXCEPTIONS as e:
                fallback_source = "fallback"
                logger.warning("RFork transfer failed: %s, clean up and fall back to default loader", e)

            cleanup_result: RForkFallbackCleanupResult | None = None
            if session is not None:
                for attempt in range(FALLBACK_CLEANUP_MAX_ATTEMPTS):
                    cleanup_result = session.prepare_for_fallback()
                    if cleanup_result.can_schedule_seed:
                        break
                    # Add backoff between cleanup attempts to avoid spinning on stuck resources.
                    if attempt + 1 < FALLBACK_CLEANUP_MAX_ATTEMPTS:
                        time.sleep(0.5)

            if model_init_started:
                _reset_process_global_model_state(vllm_config, model, model_state_snapshot)

            if cleanup_result is not None and not cleanup_result.can_schedule_seed:
                if session is not None and session.state is RForkLifecycleState.FINALIZED:
                    raise RuntimeError("RFork session has been finalized; model loading cannot resume after shutdown.")
                raise RuntimeError(
                    "RFork fallback aborted because seed service or registered memory cleanup failed; "
                    "tensor owners remain retained. Refusing to allocate a second model while old weights are pinned."
                )

            if model_init_started:
                model = None
                for _ in range(FALLBACK_MEMORY_RECLAIM_PASSES):
                    gc.collect()
                    torch.npu.empty_cache()

            model = _load_with_default_loader(vllm_config, model_config, self.load_config, prefix)

            # Advertise fallback only via an existing session; startup failure cleans its MR but keeps the model.
            if session is not None and cleanup_result is not None and cleanup_result.can_schedule_seed:
                _start_rfork_seed_service(
                    session,
                    model,
                    processed_layout_transfer,
                    exclude_blocks,
                    load_source="fallback",
                )
            if session is not None:
                _log_rfork_load_summary(session, fallback_source, load_started_at)
            return model
