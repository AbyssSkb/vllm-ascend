# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import importlib.util
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch import nn

RFORK_ROOT = Path(__file__).resolve().parents[4] / "vllm_ascend/model_loader/rfork"


def _load_module(monkeypatch, module_name: str, file_name: str):
    spec = importlib.util.spec_from_file_location(module_name, RFORK_ROOT / file_name)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


def _stub(monkeypatch, module_name: str, **attributes):
    module = ModuleType(module_name)
    module.__dict__.update(attributes)
    monkeypatch.setitem(sys.modules, module_name, module)
    return module


@pytest.fixture
def tensor_runtime(monkeypatch):
    _stub(monkeypatch, "vllm", __path__=[])
    _stub(monkeypatch, "vllm.logger", logger=logging.getLogger("rfork-tensor-safety-test"))
    _stub(monkeypatch, "vllm.utils", __path__=[])
    _stub(
        monkeypatch,
        "vllm.utils.network_utils",
        get_ip=lambda: "127.0.0.1",
        get_open_port=lambda: 12345,
        join_host_port=lambda host, port: f"{host}:{port}",
    )
    for module_name in (
        "vllm_ascend",
        "vllm_ascend.model_loader",
        "vllm_ascend.model_loader.rfork",
    ):
        _stub(monkeypatch, module_name, __path__=[])

    @dataclass(frozen=True)
    class _SeedTransferInfo:
        session_id: str
        weights: dict
        shapes: dict | None = None

    _stub(monkeypatch, "vllm_ascend.model_loader.rfork.types", SeedTransferInfo=_SeedTransferInfo)
    _load_module(monkeypatch, "vllm_ascend.model_loader.rfork.manifest", "manifest.py")
    tensor_layout = _load_module(monkeypatch, "vllm_ascend.model_loader.rfork.tensor_layout", "tensor_layout.py")
    transfer_backend = _load_module(
        monkeypatch, "vllm_ascend.model_loader.rfork.transfer_backend", "transfer_backend.py"
    )
    return SimpleNamespace(
        tensor_layout=tensor_layout,
        transfer_backend=transfer_backend,
        RForkTransferBackend=transfer_backend.RForkTransferBackend,
        SeedTransferInfo=_SeedTransferInfo,
    )


class _CallableImpl:
    def __init__(self, value):
        self.weights = value
        self.cycle = []
        self.cycle.append(self.cycle)

    def __call__(self, value):
        return value


def test_collector_keeps_logical_aliases_when_larger_view_is_seen_later(tensor_runtime, monkeypatch):
    storage = torch.arange(8, dtype=torch.float32)
    model = nn.Module()
    model.impl = _CallableImpl(storage[:2])
    model.impl.large = storage[:5]
    monkeypatch.setattr(tensor_runtime.tensor_layout, "is_transferable_tensor", lambda _tensor: True)

    collected = tensor_runtime.tensor_layout.collect_processed_layout_tensors(model)

    assert [(name, tensor.numel()) for name, tensor in collected] == [
        ("impl.weights", 2),
        ("impl.large", 5),
    ]
    assert collected[0][1].data_ptr() == collected[1][1].data_ptr()


def test_collector_deduplicates_exact_same_logical_tensor(tensor_runtime, monkeypatch):
    tensor = torch.arange(4, dtype=torch.float32)
    model = nn.Module()
    model.weight = nn.Parameter(tensor)
    monkeypatch.setattr(tensor_runtime.tensor_layout, "is_transferable_tensor", lambda _tensor: True)

    collected = tensor_runtime.tensor_layout.collect_processed_layout_tensors(model)

    assert [(name, value.data_ptr()) for name, value in collected] == [("weight", tensor.data_ptr())]


def test_collector_rejects_conflicting_same_name_view(tensor_runtime, monkeypatch):
    storage = torch.arange(8, dtype=torch.float32)
    model = nn.Module()
    model.register_buffer("weight", storage[:2])
    model.__dict__["weight"] = storage[:5]
    monkeypatch.setattr(tensor_runtime.tensor_layout, "is_transferable_tensor", lambda _tensor: True)

    with pytest.raises(ValueError, match="one tensor ID label"):
        tensor_runtime.tensor_layout.collect_processed_layout_tensors(model)


@pytest.mark.parametrize("processed_layout", [False, True])
def test_collector_prefers_registered_tensor_and_keeps_distinct_views(tensor_runtime, monkeypatch, processed_layout):
    model = nn.Module()
    model.layer = nn.Module()
    model.layer.weight = nn.Parameter(torch.arange(4, dtype=torch.float32).reshape(2, 2))
    model.impl = SimpleNamespace(
        alias=model.layer.weight.data,
        transposed=model.layer.weight.data.t(),
        reshaped=model.layer.weight.data.view(4),
    )
    monkeypatch.setattr(tensor_runtime.tensor_layout, "is_transferable_tensor", lambda _tensor: True)

    collected = dict(tensor_runtime.tensor_layout.collect_transferable_tensors(model, processed_layout))

    assert set(collected) == {"layer.weight", "impl.transposed", "impl.reshaped"}
    assert collected["layer.weight"] is model.layer.weight
    assert collected["impl.transposed"].stride() != model.layer.weight.stride()
    assert collected["impl.reshaped"].shape != model.layer.weight.shape


@pytest.mark.parametrize("processed_layout", [False, True])
def test_collector_scans_shared_diamond_once(tensor_runtime, monkeypatch, processed_layout):
    accesses = []

    class Node:
        def __getattribute__(self, name):
            if name == "__dict__":
                accesses.append(id(self))
            return super().__getattribute__(name)

    node = Node()
    node.weight = torch.ones(2)
    for _ in range(12):
        parent = Node()
        parent.left = node
        parent.right = node
        node = parent
    node.loop = node
    model = nn.Module()
    model.impl = node
    monkeypatch.setattr(tensor_runtime.tensor_layout, "is_transferable_tensor", lambda _tensor: True)

    collected = tensor_runtime.tensor_layout.collect_transferable_tensors(model, processed_layout)

    assert [name for name, _ in collected] == ["impl" + ".left" * 12 + ".weight"]
    assert len(accesses) < 80


def test_collector_uses_stable_typed_container_keys(tensor_runtime, monkeypatch):
    monkeypatch.setattr(tensor_runtime.tensor_layout, "is_transferable_tensor", lambda _tensor: True)

    def collect(reverse):
        entries = [(1, torch.ones(1)), ("1", torch.ones(2)), ("a.b", torch.ones(3))]
        if reverse:
            entries.reverse()
        model = nn.Module()
        model.impl = SimpleNamespace(weights=dict(entries))
        return {
            name: tuple(tensor.shape)
            for name, tensor in tensor_runtime.tensor_layout.collect_transferable_tensors(model, True)
        }

    expected = {"impl.weights[1]": (1,), 'impl.weights["1"]': (2,), 'impl.weights["a.b"]': (3,)}
    assert collect(False) == collect(True) == expected


def test_collector_rejects_ambiguous_container_keys(tensor_runtime, monkeypatch):
    model = nn.Module()
    model.impl = {float("nan"): torch.ones(2), float("nan"): torch.ones(3)}
    monkeypatch.setattr(tensor_runtime.tensor_layout, "is_transferable_tensor", lambda _tensor: True)

    with pytest.raises(ValueError, match="one tensor ID label"):
        tensor_runtime.tensor_layout.collect_transferable_tensors(model, True)


def test_collector_rejects_gapped_tensor_layout(tensor_runtime, monkeypatch):
    model = nn.Module()
    model.impl = SimpleNamespace(weight=torch.arange(8)[::2])
    monkeypatch.setattr(tensor_runtime.tensor_layout, "is_transferable_tensor", lambda _tensor: True)

    with pytest.raises(ValueError, match="gapped or overlapping storage"):
        tensor_runtime.tensor_layout.collect_transferable_tensors(model, True)


def test_collector_scans_custom_callable_impl_and_skips_function(tensor_runtime, monkeypatch):
    storage = torch.arange(4, dtype=torch.float32)
    model = nn.Module()
    model.impl = _CallableImpl(storage)

    def function_impl():
        return None

    function_impl.weights = storage
    model.impl.function = function_impl
    monkeypatch.setattr(tensor_runtime.tensor_layout, "is_transferable_tensor", lambda _tensor: True)

    collected = tensor_runtime.tensor_layout.collect_checkpoint_layout_tensors(model)
    names = [name for name, _ in collected]

    assert "impl.weights" in names
    assert all(not name.startswith("impl.function") for name in names)


@pytest.mark.parametrize("processed_layout", [False, True])
def test_collector_excludes_scheduler_sized_topk_indices_buffer(tensor_runtime, monkeypatch, processed_layout):
    monkeypatch.setattr(tensor_runtime.tensor_layout, "is_transferable_tensor", lambda _tensor: True)

    def make_model(max_num_batched_tokens):
        model = nn.Module()
        model.weight = nn.Parameter(torch.ones(2))
        model.topk_indices_buffer = torch.empty(max_num_batched_tokens, 2048, dtype=torch.int32)
        model.indexer_op = nn.Module()
        model.indexer_op.impl = SimpleNamespace(
            packed_weight=torch.ones(3),
            topk_indices_buffer=model.topk_indices_buffer,
        )
        return model

    manifests = []
    for max_num_batched_tokens in (2048, 4096):
        collected = tensor_runtime.tensor_layout.collect_transferable_tensors(
            make_model(max_num_batched_tokens), processed_layout
        )
        assert all(name.rsplit(".", 1)[-1] != "topk_indices_buffer" for name, _ in collected)
        manifests.append([(name, tuple(tensor.shape)) for name, tensor in collected])

    assert manifests[0] == manifests[1]
    assert ("weight", (2,)) in manifests[0]
    assert ("indexer_op.impl.packed_weight", (3,)) in manifests[0]


def test_registration_merges_small_then_large_alias_ranges(tensor_runtime, monkeypatch):
    storage = torch.arange(8, dtype=torch.float32)
    small = storage[:2]
    large = storage[:5]
    registrations = []

    class _MemoryRegistration:
        def __init__(self, *values):
            self.values = values

    class _Result:
        def is_error(self):
            return False

    backend = tensor_runtime.RForkTransferBackend.__new__(tensor_runtime.RForkTransferBackend)
    backend.transfer_engine = SimpleNamespace(
        batch_register_memory_ex=lambda items: registrations.extend(item.values for item in items) or _Result()
    )
    backend._memory_registration_cls = _MemoryRegistration
    backend.registered_weight_blocks = []
    backend.registered_memory_addresses = []
    backend._registered_transferable_tensors = None
    backend._registered_transferable_storages = None

    monkeypatch.setattr(tensor_runtime.transfer_backend, "find_non_npu_state_tensors", lambda _model: [])
    monkeypatch.setattr(tensor_runtime.transfer_backend, "is_transferable_tensor", lambda _tensor: True)
    monkeypatch.setattr(
        tensor_runtime.transfer_backend,
        "collect_transferable_tensors",
        lambda *_args: [("small", small), ("large", large)],
    )
    monkeypatch.setattr(
        tensor_runtime.transfer_backend.torch,
        "npu",
        SimpleNamespace(
            memory=SimpleNamespace(
                memory_snapshot=lambda: [
                    {
                        "blocks": [
                            {
                                "address": storage.data_ptr(),
                                "size": storage.numel() * storage.element_size(),
                                "state": "active_allocated",
                            }
                        ]
                    }
                ]
            )
        ),
        raising=False,
    )

    assert backend.register_memory_region(object(), False)
    assert set(backend.weight_manifest) == {"small", "large"}
    assert backend.weight_manifest["small"][3] == (2,)
    assert backend.weight_manifest["large"][3] == (5,)
    assert registrations == [
        (
            small.data_ptr(),
            large.numel() * large.element_size(),
            storage.data_ptr(),
            storage.numel() * storage.element_size(),
        )
    ]


@pytest.mark.parametrize("successful_batches", [0, 1])
def test_registration_failure_tracks_failed_batch_and_storage_owners(tensor_runtime, monkeypatch, successful_batches):
    first = torch.arange(2, dtype=torch.float32)
    second = torch.arange(3, dtype=torch.float32)
    tensors = [("first", first), ("second", second)]
    registered_calls = []

    class _MemoryRegistration:
        def __init__(self, *values):
            self.values = values

    class _Result:
        def __init__(self, error, message):
            self.error = error
            self.message = message

        def is_error(self):
            return self.error

        def to_string(self):
            return self.message

    register_results = iter([_Result(False, "ok")] * successful_batches + [_Result(True, "partial failure")])

    def batch_register(items):
        registered_calls.append([item.values for item in items])
        return next(register_results)

    def batch_unregister(addresses):
        return _Result(True, "rollback unavailable")

    backend = tensor_runtime.RForkTransferBackend.__new__(tensor_runtime.RForkTransferBackend)
    backend.transfer_engine = SimpleNamespace(
        batch_register_memory_ex=batch_register,
        batch_unregister_memory=batch_unregister,
    )
    backend._memory_registration_cls = _MemoryRegistration
    backend.registered_weight_blocks = []
    backend.registered_memory_addresses = []
    backend._registered_transferable_tensors = None
    backend._registered_transferable_storages = None
    backend.weight_manifest = None
    backend.weight_shapes = None

    monkeypatch.setattr(tensor_runtime.transfer_backend, "find_non_npu_state_tensors", lambda _model: [])
    monkeypatch.setattr(tensor_runtime.transfer_backend, "is_transferable_tensor", lambda _tensor: True)
    monkeypatch.setattr(tensor_runtime.transfer_backend, "collect_transferable_tensors", lambda *_args: tensors)
    monkeypatch.setattr(tensor_runtime.transfer_backend, "MAX_MEMORY_REGISTRATION_BATCH_ITEMS", 1)
    monkeypatch.setattr(
        tensor_runtime.transfer_backend.torch,
        "npu",
        SimpleNamespace(
            memory=SimpleNamespace(
                memory_snapshot=lambda: [
                    {
                        "blocks": [
                            {
                                "address": first.data_ptr(),
                                "size": first.numel() * first.element_size(),
                                "state": "active_allocated",
                            },
                            {
                                "address": second.data_ptr(),
                                "size": second.numel() * second.element_size(),
                                "state": "active_allocated",
                            },
                        ]
                    }
                ]
            )
        ),
        raising=False,
    )

    assert not backend.register_memory_region(object(), False)
    assert len(registered_calls) == successful_batches + 1
    assert backend.registered_memory_addresses == [entry[0] for batch in registered_calls for entry in batch]
    assert backend._registered_transferable_tensors == tensors
    assert backend._registered_transferable_storages is not None
    assert all(owner is not None for owner in backend._registered_transferable_storages)
    first_storage_ptr = backend._registered_transferable_storages[0].data_ptr()
    first.data = torch.zeros_like(first)
    assert backend._registered_transferable_storages[0].data_ptr() == first_storage_ptr


def test_unregister_treats_explicit_not_found_as_idempotent(tensor_runtime, monkeypatch):
    not_found = object()
    calls = []

    class _Result:
        def __init__(self, error, code):
            self.error = error
            self.code = code

        def is_error(self):
            return self.error

        def get_code(self):
            return self.code

        def to_string(self):
            return "not found" if self.code is not None else "ok"

    def unregister(addresses):
        calls.append(list(addresses))
        if len(addresses) > 1:
            return _Result(True, not_found)
        if addresses == [10]:
            return _Result(True, not_found)
        return _Result(False, None)

    backend = tensor_runtime.RForkTransferBackend.__new__(tensor_runtime.RForkTransferBackend)
    backend.transfer_engine = SimpleNamespace(batch_unregister_memory=unregister)
    backend.registered_weight_blocks = [(10, 4), (20, 4)]
    backend.registered_memory_addresses = [10, 20]
    backend._registered_transferable_tensors = [("weight", torch.ones(1))]
    backend._not_found_error_code = not_found

    assert backend.unregister_memory_region()
    assert calls == [[10, 20], [10], [20]]
    assert backend.registered_memory_addresses == []
    assert backend._registered_transferable_tensors is None


def test_empty_transfer_succeeds_only_for_registered_all_shared_layout(tensor_runtime):
    backend = tensor_runtime.RForkTransferBackend.__new__(tensor_runtime.RForkTransferBackend)
    backend._registered_transferable_tensors = []
    backend._all_transferable_tensors_excluded = True
    backend.weight_manifest = {}
    backend.transfer_engine = SimpleNamespace()

    assert backend.read_weights_from_seed(object(), tensor_runtime.SeedTransferInfo("seed", {}, {}), True)
    assert not backend.read_weights_from_seed(
        object(), tensor_runtime.SeedTransferInfo("seed", {"weight": [1, 1, 1]}), True
    )
    backend._all_transferable_tensors_excluded = False
    assert not backend.read_weights_from_seed(object(), tensor_runtime.SeedTransferInfo("seed", {}, {}), True)


def test_unregister_does_not_infer_absence_from_error_text(tensor_runtime):
    backend = tensor_runtime.RForkTransferBackend.__new__(tensor_runtime.RForkTransferBackend)
    backend._not_found_error_code = object()
    backend.transfer_engine = SimpleNamespace(
        batch_unregister_memory=lambda addresses: SimpleNamespace(
            is_error=lambda: True, get_code=lambda: object(), to_string=lambda: "device context not found"
        )
    )
    backend.registered_weight_blocks = [(10, 4)]
    backend.registered_memory_addresses = [10]
    owners = [("weight", torch.ones(1))]
    backend._registered_transferable_tensors = owners
    assert not backend.unregister_memory_region()
    assert backend._registered_transferable_tensors is owners
    assert backend.registered_memory_addresses == [10]


def test_shared_reuse_requires_all_tensor_bytes_and_nonempty_state(tensor_runtime, monkeypatch):
    backend = tensor_runtime.RForkTransferBackend.__new__(tensor_runtime.RForkTransferBackend)
    weight = torch.arange(8, dtype=torch.float32)
    monkeypatch.setattr(tensor_runtime.transfer_backend, "find_non_npu_state_tensors", lambda model: [])
    monkeypatch.setattr(tensor_runtime.transfer_backend, "collect_transferable_tensors", lambda *args: [("w", weight)])
    address, size = weight.data_ptr(), weight.numel() * weight.element_size()
    assert backend.can_reuse_shared_weights(object(), False, [(address, size)])
    assert not backend.can_reuse_shared_weights(object(), False, [(address, size - 1)])
    monkeypatch.setattr(tensor_runtime.transfer_backend, "collect_transferable_tensors", lambda *args: [])
    assert not backend.can_reuse_shared_weights(object(), False, [(address, size)])


class _FakeYrEngine:
    def __init__(self, rpc_port=45678, initialize_error=False):
        self.rpc_port = rpc_port
        self.initialize_error = initialize_error
        self.initialize_calls: list[tuple[str, str, str]] = []
        self.finalize_calls = 0

    def initialize(self, endpoint, protocol, device_name):
        self.initialize_calls.append((endpoint, protocol, device_name))
        return SimpleNamespace(
            is_error=lambda: self.initialize_error,
            to_string=lambda: "mock initialize error",
        )

    def get_rpc_port(self):
        return self.rpc_port

    def finalize(self):
        self.finalize_calls += 1
        return SimpleNamespace(is_error=lambda: False)


def _install_yr_module_with_engine(monkeypatch, tensor_runtime, engine):
    transfer_backend = tensor_runtime.transfer_backend
    yr_module = ModuleType("yr")
    datasystem_module = ModuleType("yr.datasystem")
    datasystem_module.TransferEngine = lambda: engine  # type: ignore[attr-defined]
    datasystem_module.MemoryRegistration = object  # type: ignore[attr-defined]
    datasystem_module.ErrorCode = SimpleNamespace(kNotReady=1, kNotFound=2)  # type: ignore[attr-defined]
    yr_module.datasystem = datasystem_module  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "yr", yr_module)
    monkeypatch.setitem(sys.modules, "yr.datasystem", datasystem_module)
    monkeypatch.setattr(transfer_backend, "get_ip", lambda: "10.0.0.1")
    monkeypatch.setattr(
        transfer_backend,
        "torch",
        SimpleNamespace(npu=SimpleNamespace(current_device=lambda: 2)),
    )
    return engine


def test_initialize_transfer_engine_binds_dynamic_port_and_reads_rpc_port(tensor_runtime, monkeypatch):
    engine = _install_yr_module_with_engine(monkeypatch, tensor_runtime, _FakeYrEngine(rpc_port=45678))
    backend = tensor_runtime.RForkTransferBackend()

    assert engine.initialize_calls == [("10.0.0.1:0", "ascend", "npu:2")]
    assert backend.transfer_session_id == "10.0.0.1:45678"
    assert backend.transfer_engine is engine
    assert backend._is_initialized


@pytest.mark.parametrize("rpc_port", [0, -1, 65536, None])
def test_initialize_transfer_engine_rejects_invalid_rpc_port(tensor_runtime, monkeypatch, rpc_port):
    engine = _install_yr_module_with_engine(monkeypatch, tensor_runtime, _FakeYrEngine(rpc_port=rpc_port))
    with pytest.raises(RuntimeError, match="invalid RPC port"):
        tensor_runtime.RForkTransferBackend()

    assert engine.finalize_calls == 1


def test_initialize_transfer_engine_requires_get_rpc_port(tensor_runtime, monkeypatch):
    class _EngineWithoutRpcPort(_FakeYrEngine):
        get_rpc_port = None  # type: ignore[assignment]

    engine = _install_yr_module_with_engine(monkeypatch, tensor_runtime, _EngineWithoutRpcPort())
    with pytest.raises(RuntimeError, match="get_rpc_port"):
        tensor_runtime.RForkTransferBackend()

    assert engine.finalize_calls == 1


def test_initialize_transfer_engine_finalizes_engine_when_rpc_port_raises(tensor_runtime, monkeypatch):
    class _EngineWithRaisingRpcPort(_FakeYrEngine):
        def get_rpc_port(self):
            raise RuntimeError("native rpc port query failed")

    engine = _install_yr_module_with_engine(monkeypatch, tensor_runtime, _EngineWithRaisingRpcPort())
    with pytest.raises(RuntimeError, match="native rpc port query failed"):
        tensor_runtime.RForkTransferBackend()

    assert engine.finalize_calls == 1
