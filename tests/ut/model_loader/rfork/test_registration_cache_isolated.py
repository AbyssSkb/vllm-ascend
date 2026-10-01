# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import ctypes
import hashlib
import logging
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import regex as re
import torch

from .rfork_test_support import _load_module


def test_tensor_collection_deduplicates_exact_impl_alias_but_keeps_distinct_view(tensor_runtime, monkeypatch):
    tensor_layout = tensor_runtime.tensor_layout
    weight = torch.nn.Parameter(torch.arange(4, dtype=torch.float32))
    model = torch.nn.Module()
    model.register_parameter("weight", weight)
    model.impl = SimpleNamespace(weight=weight, view=weight[:2])
    monkeypatch.setattr(tensor_layout, "is_transferable_tensor", lambda _tensor: True)

    collected = tensor_layout.collect_transferable_tensors(model, processed_layout=True)

    assert len(collected) == 2
    assert {tensor.numel() for _, tensor in collected} == {2, 4}
    assert all(re.fullmatch(r"[0-9a-f]{64}", name) for name, _ in collected)


@pytest.mark.parametrize("processed_layout", [False, True])
def test_collector_excludes_scheduler_sized_topk_indices_buffer(tensor_runtime, monkeypatch, processed_layout):
    monkeypatch.setattr(tensor_runtime.tensor_layout, "is_transferable_tensor", lambda _tensor: True)

    def make_model(max_num_batched_tokens):
        model = torch.nn.Module()
        model.weight = torch.nn.Parameter(torch.ones(2))
        model.topk_indices_buffer = torch.empty(max_num_batched_tokens, 2048, dtype=torch.int32)
        model.indexer_op = torch.nn.Module()
        model.indexer_op.impl = SimpleNamespace(
            packed_weight=torch.ones(3),
            topk_indices_buffer=model.topk_indices_buffer,
        )
        return model

    manifests = []
    for max_num_batched_tokens in (2048, 4096):
        model = make_model(max_num_batched_tokens)
        collected = tensor_runtime.tensor_layout.collect_transferable_tensors(model, processed_layout)
        assert all(tensor is not model.topk_indices_buffer for _, tensor in collected)
        manifests.append({name: tuple(tensor.shape) for name, tensor in collected})

    assert manifests[0] == manifests[1]
    assert set(manifests[0].values()) == {(2,), (3,)}


def test_layout_summary_is_one_bounded_info_record_with_fixed_digests(tensor_runtime, caplog):
    tensors = [(f"weight_{index}", torch.arange(4, dtype=torch.float32)) for index in range(6)]
    formats = {name: 29 for name, _ in tensors}

    with caplog.at_level(logging.INFO, logger=tensor_runtime.tensor_layout.logger.name):
        tensor_runtime.tensor_layout.log_tensor_layout_summary(
            tensors,
            stage="receiver_before_read",
            session_id="receiver-session",
            peer_session_id="seed-session",
            processed_layout=True,
            known_formats=formats,
        )

    records = [record.getMessage() for record in caplog.records if "RFork tensor layout summary" in record.getMessage()]
    assert len(records) == 1
    message = records[0]
    assert "tensors=6" in message
    assert "session=receiver-session peer_session=seed-session" in message
    assert len(re.findall(r"(?:semantic|physical)_digest=[0-9a-f]{64}", message)) == 2
    assert "weight_0" in message and "weight_2" in message
    assert "weight_3" not in message and "weight_5" not in message


def test_layout_summary_includes_npu_format_and_physical_size(tensor_runtime, monkeypatch, caplog):
    class _NPUTensorProxy:
        device = SimpleNamespace(type="npu")

        def __init__(self, tensor):
            self._tensor = tensor

        def __getattr__(self, name):
            return getattr(self._tensor, name)

    monkeypatch.setitem(
        sys.modules,
        "torch_npu",
        SimpleNamespace(
            get_npu_format=lambda tensor: 29,
            get_storage_size=lambda tensor: tensor.numel() + 8,
        ),
    )
    tensor = _NPUTensorProxy(torch.arange(4, dtype=torch.float32))

    with caplog.at_level(logging.INFO, logger=tensor_runtime.tensor_layout.logger.name):
        tensor_runtime.tensor_layout.log_tensor_layout_summary(
            [("weight", tensor)],
            stage="registered",
            session_id="seed-session",
            processed_layout=True,
        )

    message = next(
        record.getMessage() for record in caplog.records if "RFork tensor layout summary" in record.getMessage()
    )
    assert "physical_nonlogical_tensors=1" in message
    assert "formats={'29': 1}" in message
    assert "'npu_format': 29" in message
    assert "'npu_storage_numel': 12" in message


def test_post_load_layout_summary_is_observational(tensor_runtime, monkeypatch, caplog):
    tensor_layout = tensor_runtime.tensor_layout
    monkeypatch.setattr(tensor_layout, "is_tensor_on_transfer_device", lambda tensor: True)
    model = torch.nn.Module()
    model.weight = torch.nn.Parameter(torch.arange(4, dtype=torch.float32))
    original = model.weight.detach().clone()
    backend = tensor_runtime.RForkTransferBackend()
    backend.transfer_session_id = "receiver-session"

    with caplog.at_level(logging.INFO, logger=tensor_layout.logger.name):
        backend.log_model_layout_summary(
            model,
            False,
            stage="receiver_after_post_load",
            peer_session_id="seed-session",
        )

    message = next(record.getMessage() for record in caplog.records if "RFork tensor layout summary" in record.message)
    assert "stage=receiver_after_post_load" in message
    assert "session=receiver-session peer_session=seed-session" in message
    assert "tensors=1" in message
    torch.testing.assert_close(model.weight, original)


def test_post_load_layout_diagnostic_failure_does_not_escape(tensor_runtime, monkeypatch, caplog):
    transfer_backend = tensor_runtime.transfer_backend
    monkeypatch.setattr(
        transfer_backend,
        "collect_transferable_tensors",
        Mock(side_effect=RuntimeError("inspection failed")),
    )
    backend = tensor_runtime.RForkTransferBackend()

    with caplog.at_level(logging.INFO, logger=transfer_backend.logger.name):
        backend.log_model_layout_summary(object(), False, stage="receiver_after_post_load")

    assert "unavailable=RuntimeError:inspection failed" in caplog.text


@pytest.mark.parametrize("processed_layout", [False, True])
def test_bfs_expands_shared_metadata_once(tensor_runtime, monkeypatch, processed_layout):
    class CountingDict(dict):
        scans = 0

        def items(self):
            self.scans += 1
            return super().items()

    metadata = CountingDict(value=1)
    nodes = [metadata]
    for _ in range(12):
        metadata = CountingDict(left=metadata, right=metadata)
        nodes.append(metadata)
    model = torch.nn.Module()
    for index in range(8):
        layer = torch.nn.Module()
        layer.impl = SimpleNamespace(config=metadata, weight=torch.ones(2))
        model.add_module(f"layer_{index}", layer)
    monkeypatch.setattr(tensor_runtime.tensor_layout, "is_transferable_tensor", lambda _tensor: True)

    collected = tensor_runtime.tensor_layout.collect_transferable_tensors(model, processed_layout)

    assert {id(tensor) for _, tensor in collected} == {id(layer.impl.weight) for layer in model.children()}
    assert all(node.scans == 1 for node in nodes)


def test_bfs_keeps_object_scan_modes_separate(tensor_runtime, monkeypatch):
    model = torch.nn.Module()
    shared = [SimpleNamespace(weight=torch.ones(2))]
    model.metadata = shared
    model.impl = shared
    monkeypatch.setattr(tensor_runtime.tensor_layout, "is_transferable_tensor", lambda _tensor: True)

    collected = tensor_runtime.tensor_layout.collect_transferable_tensors(model, True)

    assert len(collected) == 1
    assert collected[0][1] is shared[0].weight


def test_bfs_ids_distinguish_flat_and_nested_keys(tensor_runtime, monkeypatch):
    model = torch.nn.Module()
    nested = torch.ones(2)
    flat = torch.ones(3)
    shared = {"weight": nested}
    model.impl = {"first": shared, "second": shared, "second.weight": flat}
    monkeypatch.setattr(tensor_runtime.tensor_layout, "is_transferable_tensor", lambda _tensor: True)

    collected = tensor_runtime.tensor_layout.collect_transferable_tensors(model, True)

    assert len(dict(collected)) == 2
    assert {id(tensor) for _, tensor in collected} == {id(nested), id(flat)}


def test_bfs_ids_handle_cycles(tensor_runtime, monkeypatch):
    first = {}
    second = {"first": first}
    first.update(second=second, weight=torch.ones(2))
    model = torch.nn.Module()
    model.impl = {"first": first, "second": second}
    monkeypatch.setattr(tensor_runtime.tensor_layout, "is_transferable_tensor", lambda _tensor: True)

    collected = tensor_runtime.tensor_layout.collect_transferable_tensors(model, True)
    model.impl = dict(reversed(list(model.impl.items())))
    reordered = tensor_runtime.tensor_layout.collect_transferable_tensors(model, True)

    assert len(collected) == 1
    assert collected[0][1] is first["weight"]
    assert collected[0][0] == reordered[0][0]


@pytest.mark.parametrize("reverse", [False, True])
def test_bfs_filters_alias_edges_before_deduplication(tensor_runtime, monkeypatch, reverse):
    model = torch.nn.Module()
    weight = torch.ones(2)
    entries = [("topk_indices_buffer", weight), ("container", {"weight": weight})]
    model.impl = dict(reversed(entries) if reverse else entries)
    monkeypatch.setattr(tensor_runtime.tensor_layout, "is_transferable_tensor", lambda _tensor: True)

    collected = tensor_runtime.tensor_layout.collect_transferable_tensors(model, True)
    del model.impl["topk_indices_buffer"]
    without_excluded_alias = tensor_runtime.tensor_layout.collect_transferable_tensors(model, True)

    assert len(collected) == 1
    assert collected[0][1] is weight
    assert collected[0][0] == without_excluded_alias[0][0]


def test_bfs_is_rebuilt_for_each_collection(tensor_runtime, monkeypatch):
    model = torch.nn.Module()
    model.impl = {}
    monkeypatch.setattr(tensor_runtime.tensor_layout, "is_transferable_tensor", lambda _tensor: True)
    assert tensor_runtime.tensor_layout.collect_transferable_tensors(model, True) == []
    model.impl["weight"] = torch.ones(2)

    collected = tensor_runtime.tensor_layout.collect_transferable_tensors(model, True)

    assert len(collected) == 1
    assert collected[0][1] is model.impl["weight"]


@pytest.mark.parametrize("processed_layout", [False, True])
def test_bfs_ids_ignore_parameter_buffer_module_and_attribute_order(tensor_runtime, monkeypatch, processed_layout):
    def make_model(reverse):
        model = torch.nn.Module()
        parameter = torch.nn.Parameter(torch.ones(2))
        buffer = torch.ones(3)
        child = torch.nn.Module()
        child.weight = torch.nn.Parameter(torch.ones(4))
        shared = {"weight": torch.ones(5)}
        pairs = lambda entries: reversed(entries) if reverse else entries
        for name in pairs(["a", "z"]):
            model.register_parameter(name, parameter)
        for name in pairs(["b", "y"]):
            model.register_buffer(name, buffer)
        for name in pairs(["left", "right"]):
            model.add_module(name, child)
        child.add_module("back", model)
        model.impl = SimpleNamespace(**dict(pairs([("a", shared), ("b", shared)])))
        return model

    layout = tensor_runtime.tensor_layout
    monkeypatch.setattr(layout, "is_transferable_tensor", lambda _tensor: True)
    manifests = [
        {
            name: tuple(tensor.shape)
            for name, tensor in layout.collect_transferable_tensors(make_model(reverse), processed_layout)
        }
        for reverse in (False, True)
    ]

    assert manifests[0] == manifests[1]
    assert set(manifests[0].values()) == {(2,), (3,), (4,), (5,)}


def test_bfs_finalizes_all_same_depth_candidates_before_expansion(tensor_runtime, monkeypatch):
    layout = tensor_runtime.tensor_layout
    root_id = hashlib.sha256(b"rfork-tensor-id").digest()
    impl_id = layout._tensor_edge_id(root_id, "attribute", "impl")
    candidates = {
        name: layout._tensor_edge_id(layout._tensor_edge_id(impl_id, "key", name), "key", "shared")
        for name in ("a", "b")
    }
    shared = {"weight": torch.ones(2)}
    model = torch.nn.Module()
    # Enqueue the worse candidate first to catch premature expansion/ID finalization.
    model.impl = {name: {"shared": shared} for name in sorted(candidates, key=candidates.get, reverse=True)}
    monkeypatch.setattr(layout, "is_transferable_tensor", lambda _tensor: True)

    collected = layout.collect_transferable_tensors(model, True)

    expected = layout._tensor_edge_id(min(candidates.values()), "key", "weight")
    assert collected[0][0] == expected.hex()


def test_bfs_uses_shortest_eligible_path(tensor_runtime, monkeypatch):
    layout = tensor_runtime.tensor_layout
    model = torch.nn.Module()
    tensor = torch.ones(2)
    model.impl = {"long": {"weight": tensor}, "short": tensor}
    monkeypatch.setattr(layout, "is_transferable_tensor", lambda _tensor: True)

    collected = layout.collect_transferable_tensors(model, True)

    root_id = hashlib.sha256(b"rfork-tensor-id").digest()
    impl_id = layout._tensor_edge_id(root_id, "attribute", "impl")
    expected = layout._tensor_edge_id(impl_id, "key", "short")
    assert len(collected) == 1
    assert collected[0][0] == expected.hex()


def test_bfs_visits_tensor_bearing_diamond_once(tensor_runtime, monkeypatch):
    class CountingDict(dict):
        scans = 0

        def items(self):
            self.scans += 1
            return super().items()

    depth = 30  # Enumerating every alias would produce over a billion tensor paths.
    value = CountingDict(weight=torch.ones(2))
    nodes = [value]
    for _ in range(depth):
        value = CountingDict(left=value, right=value)
        nodes.append(value)
    model = torch.nn.Module()
    model.impl = value
    layout = tensor_runtime.tensor_layout
    eligible = Mock(return_value=True)
    edge_id = Mock(wraps=layout._tensor_edge_id)
    monkeypatch.setattr(layout, "is_transferable_tensor", eligible)
    monkeypatch.setattr(layout, "_tensor_edge_id", edge_id)

    collected = layout.collect_transferable_tensors(model, True)

    assert len(collected) == 1
    assert all(node.scans == 1 for node in nodes)
    assert eligible.call_count == 1
    assert edge_id.call_count == 2 * depth + 2


def test_bfs_preserves_device_empty_meta_and_layout_checks(tensor_runtime, monkeypatch):
    model = torch.nn.Module()
    weight = torch.ones(2)
    cpu = torch.ones(3)
    model.impl = {"weight": weight, "cpu": cpu, "empty": torch.empty(0), "meta": torch.empty(2, device="meta")}
    layout = tensor_runtime.tensor_layout
    monkeypatch.setattr(layout, "is_tensor_on_transfer_device", lambda tensor: tensor is not cpu)

    collected = layout.collect_transferable_tensors(model, True)

    assert len(collected) == 1
    assert collected[0][1] is weight
    model.impl["gapped"] = torch.ones(4)[::2]
    with pytest.raises(ValueError, match="gapped or overlapping"):
        layout.collect_transferable_tensors(model, True)


def test_bfs_runtime_named_container_still_exposes_valid_tensor(tensor_runtime, monkeypatch):
    model = torch.nn.Module()
    model.impl = {"topk_indices_buffer": {"weight": torch.ones(2)}}
    monkeypatch.setattr(tensor_runtime.tensor_layout, "is_transferable_tensor", lambda _tensor: True)

    collected = tensor_runtime.tensor_layout.collect_transferable_tensors(model, True)

    assert len(collected) == 1
    assert collected[0][1] is model.impl["topk_indices_buffer"]["weight"]


def test_bfs_distinguishes_typed_dictionary_keys(tensor_runtime, monkeypatch):
    model = torch.nn.Module()
    model.impl = {1: torch.ones(2), "1": torch.ones(3), (1, "a"): torch.ones(4)}
    monkeypatch.setattr(tensor_runtime.tensor_layout, "is_transferable_tensor", lambda _tensor: True)

    collected = tensor_runtime.tensor_layout.collect_transferable_tensors(model, True)

    assert len(dict(collected)) == 3


def test_bfs_rejects_tensor_id_collisions(tensor_runtime, monkeypatch):
    model = torch.nn.Module()
    model.impl = {"a": torch.ones(2), "b": torch.ones(3)}
    layout = tensor_runtime.tensor_layout
    monkeypatch.setattr(layout, "is_transferable_tensor", lambda _tensor: True)
    monkeypatch.setattr(layout, "_tensor_edge_id", lambda *_args: bytes(32))

    with pytest.raises(ValueError, match="conflicting tensor IDs"):
        layout.collect_transferable_tensors(model, True)


@pytest.mark.parametrize("exclude_shared", [False, True])
def test_bfs_ids_match_seed_reads_and_shared_exclusions(tensor_runtime, monkeypatch, exclude_shared):
    def make_model(shared_value, own_value, reverse):
        model = torch.nn.Module()
        shared = torch.full((4,), shared_value, dtype=torch.float32)
        own = torch.full((3,), own_value, dtype=torch.float32)
        entries = [("a", shared), ("b", shared), ("own", own)]
        model.impl = dict(reversed(entries) if reverse else entries)
        return model

    layout = tensor_runtime.tensor_layout
    transfer = tensor_runtime.transfer_backend
    manifest = sys.modules["vllm_ascend.model_loader.rfork.manifest"]
    monkeypatch.setattr(layout, "is_transferable_tensor", lambda _tensor: True)
    monkeypatch.setattr(manifest, "read_npu_format", lambda _tensor: 0)
    source = make_model(7, 5, False)
    destination = make_model(11, 13, True)
    source_tensors = layout.collect_transferable_tensors(source, True)
    destination_tensors = layout.collect_transferable_tensors(destination, True)
    assert layout.build_structural_digest(source_tensors) == layout.build_structural_digest(destination_tensors)
    shared = destination.impl["a"]
    excluded = [(shared.data_ptr(), shared.numel() * shared.element_size())] if exclude_shared else []
    reads = []

    def read(_session, client_ptrs, seed_ptrs, lengths):
        for client_ptr, seed_ptr, length in zip(client_ptrs, seed_ptrs, lengths, strict=True):
            reads.append((client_ptr, seed_ptr, length))
            ctypes.memmove(client_ptr, seed_ptr, length)
        return SimpleNamespace(is_error=lambda: False)

    backend = tensor_runtime.RForkTransferBackend()
    backend.transfer_engine = SimpleNamespace(batch_transfer_sync_read=read)
    backend._registered_transferable_tensors, _ = transfer._split_tensors_by_excluded_blocks(
        destination_tensors, excluded
    )
    backend.excluded_weight_blocks = excluded
    seed_info = tensor_runtime.SeedTransferInfo(
        "seed-session",
        {
            name: (tensor.data_ptr(), tensor.numel(), tensor.element_size(), tuple(tensor.shape), str(tensor.dtype))
            for name, tensor in source_tensors
        },
        formats={name: 0 for name, _ in source_tensors},
    )

    assert backend.read_weights_from_seed(destination, seed_info, True)
    assert len(reads) == (1 if exclude_shared else 2)
    assert destination.impl["a"] is destination.impl["b"]
    torch.testing.assert_close(destination.impl["a"], torch.full((4,), 11.0 if exclude_shared else 7.0))
    torch.testing.assert_close(destination.impl["own"], torch.full((3,), 5.0))


def test_bfs_preserves_public_attribute_scan_scope(tensor_runtime, monkeypatch):
    class CallableImpl:
        def __call__(self):
            pass

    model = torch.nn.Module()
    impl = CallableImpl()
    impl.weight = torch.ones(2)
    impl._private = torch.ones(3)
    impl.function = lambda: None
    impl.function.weight = torch.ones(4)
    impl.module = torch.nn.Linear(2, 2)
    model.impl = impl
    model.other = SimpleNamespace(weight=torch.ones(5))
    model._private = torch.ones(6)
    monkeypatch.setattr(tensor_runtime.tensor_layout, "is_transferable_tensor", lambda _tensor: True)

    collected = tensor_runtime.tensor_layout.collect_transferable_tensors(model, True)

    assert len(collected) == 1
    assert collected[0][1] is impl.weight


@pytest.mark.parametrize("processed_layout", [False, True])
def test_bfs_only_follows_registered_module_edges(tensor_runtime, monkeypatch, processed_layout):
    model = torch.nn.Module()
    model.child = torch.nn.Linear(2, 2, bias=False)
    # These references deliberately bypass the module registry.
    object.__setattr__(model, "alias", model.child)
    object.__setattr__(model, "impl", torch.nn.Linear(3, 3, bias=False))
    model.module_list = [torch.nn.Linear(4, 4, bias=False)]
    monkeypatch.setattr(tensor_runtime.tensor_layout, "is_transferable_tensor", lambda _tensor: True)

    collected = tensor_runtime.tensor_layout.collect_transferable_tensors(model, processed_layout)

    assert len(collected) == 1
    assert collected[0][1] is model.child.weight


def test_bfs_inconsistent_tensor_sharing_rejects_transfer(tensor_runtime, monkeypatch):
    layout = tensor_runtime.tensor_layout
    monkeypatch.setattr(layout, "is_transferable_tensor", lambda _tensor: True)
    source = torch.nn.Module()
    tensor = torch.ones(2)
    source.impl = {"a": tensor, "b": tensor}
    destination = torch.nn.Module()
    destination.impl = {"a": torch.empty(2), "b": torch.empty(2)}
    backend = tensor_runtime.RForkTransferBackend()
    read = Mock()
    backend.transfer_engine = SimpleNamespace(batch_transfer_sync_read=read)
    backend._registered_transferable_tensors = layout.collect_transferable_tensors(destination, True)
    source_tensors = layout.collect_transferable_tensors(source, True)
    seed_info = tensor_runtime.SeedTransferInfo(
        "seed-session",
        {name: (tensor.data_ptr(), 2, 4, (2,), "float32") for name, tensor in source_tensors},
        formats={name: 0 for name, _ in source_tensors},
    )

    assert not backend.read_weights_from_seed(destination, seed_info, True)
    read.assert_not_called()


def test_seed_metadata_accepts_stable_ids_with_existing_protocol(tensor_runtime, monkeypatch):
    types = _load_module(monkeypatch, "vllm_ascend.model_loader.rfork.types", "types.py")
    client = _load_module(monkeypatch, "rfork_bfs_seed_client", "seed_client.py")
    model = torch.nn.Module()
    model.weight = torch.nn.Parameter(torch.ones(2))
    monkeypatch.setattr(tensor_runtime.tensor_layout, "is_transferable_tensor", lambda _tensor: True)
    collected = tensor_runtime.tensor_layout.collect_transferable_tensors(model, True)
    weights = {name: [tensor.data_ptr(), 2, 4, [2], "float32"] for name, tensor in collected}
    formats = {name: 0 for name in weights}
    monkeypatch.setattr(
        client.requests,
        "get",
        Mock(
            return_value=SimpleNamespace(
                status_code=200,
                json=lambda: {
                    "rfork_protocol_version": types.RFORK_PROTOCOL_VERSION,
                    "rfork_transfer_engine_info": ["seed-session", weights],
                    "rfork_transfer_engine_format_info": formats,
                },
            )
        ),
    )

    assert types.RFORK_PROTOCOL_VERSION == 1
    info = client.fetch_seed_transfer_info("http://seed", "key", 1.0)
    assert info is not None
    assert info.weights == weights
    assert info.formats == formats
