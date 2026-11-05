# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
import threading
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from types import SimpleNamespace

import pytest
import torch


def test_non_engram_ranks_can_load_from_different_sources(rfork_helpers, monkeypatch):
    module = rfork_helpers.loader
    manager_started = threading.Event()
    release_manager = threading.Event()
    manager_calls = []
    transfers = []
    cleanups = []
    local_loads = []
    models = [torch.nn.Module(), torch.nn.Module()]
    local_model = torch.nn.Module()

    monkeypatch.setattr(module.managed_weights, "enabled", lambda: True)

    def mounted(config, model_config):
        manager_calls.append(config.rank)
        manager_started.set()
        assert release_manager.wait(timeout=5)
        return config, model_config

    monkeypatch.setattr(module.managed_weights, "mounted_configs", mounted)
    monkeypatch.setattr(module, "_rfork_bypass_reason", lambda *args: None)
    monkeypatch.setattr(module, "_snapshot_process_global_model_state", lambda *args: None)
    monkeypatch.setattr(module, "initialize_model", lambda **kwargs: models[kwargs["vllm_config"].rank])
    monkeypatch.setattr(module, "_prepare_rfork_model_state", lambda *args: None)
    monkeypatch.setattr(module, "load_local_weights", lambda *args: None)
    monkeypatch.setattr(module, "process_weights_after_loading", lambda *args: None)
    monkeypatch.setattr(module, "_rfork_pre_transfer_weight_processing", lambda *args: nullcontext())
    monkeypatch.setattr(module, "_refresh_rfork_flatquant_state", lambda *args: None)
    monkeypatch.setattr(module, "_validate_rfork_shared_experts", lambda *args: None)
    monkeypatch.setattr(module, "_schedule_seed_publication", lambda *args, **kwargs: None)
    monkeypatch.setattr(module, "_cleanup_failed_rfork_load", lambda session, *args: cleanups.append(session.rank))

    def load_local(config, *args):
        local_loads.append(config.rank)
        return local_model

    monkeypatch.setattr(module, "_load_with_default_loader", load_local)
    monkeypatch.setattr(
        torch, "npu", SimpleNamespace(synchronize=lambda: None, empty_cache=lambda: None), raising=False
    )

    def load_rank(rank):
        def transfer():
            transfers.append(rank)
            return True

        session = SimpleNamespace(
            rank=rank,
            loading=nullcontext,
            identity=SimpleNamespace(tp_rank=rank, global_rank=rank, is_draft_model=False),
            register_receiver_weights=lambda *args: True,
            acquire_seed_lease=lambda: rank == 0,
            read_weights_from_seed=transfer,
            validate_transferred_model_layout=lambda *args: None,
        )
        loader = module.RForkModelLoader.__new__(module.RForkModelLoader)
        loader.load_config = SimpleNamespace(device=None)
        loader._get_session = lambda *args: session
        model_config = SimpleNamespace(model="bootstrap", dtype=torch.float32)
        config = SimpleNamespace(
            rank=rank,
            device_config=SimpleNamespace(device="cpu"),
            model_config=model_config,
            engram_config=None,
            parallel_config=SimpleNamespace(world_size=2, data_parallel_size=1),
        )
        return loader.load_model(config, model_config)

    with ThreadPoolExecutor(max_workers=2) as pool:
        fallback = pool.submit(load_rank, 1)
        try:
            assert manager_started.wait(timeout=5)
            transferred = pool.submit(load_rank, 0)
            assert transferred.result(timeout=5) is models[0]
            assert not fallback.done()
        finally:
            release_manager.set()
        assert fallback.result(timeout=5) is local_model

    assert manager_calls == [1]
    assert transfers == [0]
    assert cleanups == [1]
    assert local_loads == [1]


@pytest.mark.parametrize("bypass", [False, True])
def test_manager_failure_aborts_all_ranks_before_default_load(rfork_helpers, monkeypatch, bypass):
    module = rfork_helpers.loader
    rank = threading.local()
    size = 4
    barrier = threading.Barrier(size, timeout=3)
    values = [None] * size
    cpu_group = object()
    default_loads = []
    cleanups = []

    def all_reduce(tensor, op, group):
        assert group is cpu_group
        values[rank.value] = tensor
        barrier.wait()
        result = max(int(value.item()) for value in values)
        barrier.wait()
        tensor.fill_(result)
        barrier.wait()

    monkeypatch.setattr(
        sys.modules["vllm.distributed.parallel_state"],
        "get_world_group",
        lambda: SimpleNamespace(cpu_group=cpu_group, world_size=size),
    )
    monkeypatch.setattr(torch.distributed, "all_reduce", all_reduce)
    monkeypatch.setattr(module.managed_weights, "enabled", lambda: True)

    def mounted(config, model_config):
        if rank.value == 2:
            raise RuntimeError("manager failed")
        return config, model_config

    monkeypatch.setattr(module.managed_weights, "mounted_configs", mounted)
    monkeypatch.setattr(module, "_rfork_bypass_reason", lambda *args: "test bypass" if bypass else None)
    monkeypatch.setattr(module, "_snapshot_process_global_model_state", lambda *args: None)
    monkeypatch.setattr(module, "_reset_process_global_model_state", lambda *args: None)
    monkeypatch.setattr(module, "initialize_model", lambda **kwargs: torch.nn.Module())
    monkeypatch.setattr(module, "_prepare_rfork_model_state", lambda *args: None)
    monkeypatch.setattr(module, "load_local_weights", lambda *args: None)
    monkeypatch.setattr(module, "release_local_weights", lambda *args: None)
    monkeypatch.setattr(module, "process_weights_after_loading", lambda *args: None)
    monkeypatch.setattr(module, "_rfork_pre_transfer_weight_processing", lambda *args: nullcontext())
    monkeypatch.setattr(module, "_load_with_default_loader", lambda *args: default_loads.append(rank.value))
    monkeypatch.setattr(
        torch, "npu", SimpleNamespace(synchronize=lambda: None, empty_cache=lambda: None), raising=False
    )

    def worker(value):
        rank.value = value
        session = SimpleNamespace(
            loading=nullcontext,
            identity=SimpleNamespace(tp_rank=value, global_rank=value, is_draft_model=False),
            register_receiver_weights=lambda *args: True,
            acquire_seed_lease=lambda: False,
            prepare_for_fallback=lambda: cleanups.append(value) or True,
        )
        loader = module.RForkModelLoader.__new__(module.RForkModelLoader)
        loader.load_config = SimpleNamespace(device=None)
        loader._get_session = lambda *args: session
        model_config = SimpleNamespace(model="bootstrap", dtype=torch.float32)
        config = SimpleNamespace(
            device_config=SimpleNamespace(device="cpu"),
            model_config=model_config,
            engram_config=SimpleNamespace(cpu_offload=False),
        )
        try:
            return loader.load_model(config, model_config)
        except Exception as exc:
            return exc

    with ThreadPoolExecutor(max_workers=size) as pool:
        results = list(pool.map(worker, range(size)))

    assert all(isinstance(result, module._RForkLoadAborted) for result in results), results
    assert default_loads == []
    assert sorted(cleanups) == ([] if bypass else list(range(size)))
