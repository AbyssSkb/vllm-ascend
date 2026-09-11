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
    small = storage[:2]
    large = storage[:5]
    collected: list[tuple[str, torch.Tensor]] = []
    seen_names: dict[str, int] = {}
    monkeypatch.setattr(tensor_runtime.tensor_layout, "is_transferable_tensor", lambda _tensor: True)

    with pytest.raises(ValueError, match="conflicting tensor entries"):
        tensor_runtime.tensor_layout._try_collect("weight", small, seen_names, collected)
        tensor_runtime.tensor_layout._try_collect("weight", large, seen_names, collected)


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
