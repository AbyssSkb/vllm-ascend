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
def test_bfs_discards_plain_leaves_before_classification(tensor_runtime, monkeypatch, processed_layout):
    layout = tensor_runtime.tensor_layout
    monkeypatch.setattr(layout, "is_tensor_on_transfer_device", lambda _tensor: True)
    model = torch.nn.Module()
    model.weight = torch.nn.Parameter(torch.ones(2))
    model.register_buffer("scale", torch.tensor(1.0))
    model.impl = SimpleNamespace(weight=model.weight, packed_weight=torch.ones(3))
    expected = layout.collect_transferable_tensors(model, processed_layout)

    leaves = [None, True, False, 1, 1.5, 1j, "text", b"bytes"]
    model.public_leaves = leaves
    model.impl.metadata = SimpleNamespace(
        strings=[f"value_{index}" for index in range(10_000)],
        values=leaves,
        tuple_values=tuple(leaves),
        mapping=dict(enumerate(leaves)),
    )
    classify = Mock(wraps=layout._tensor_child_key)
    monkeypatch.setattr(layout, "_tensor_child_key", classify)

    collected = layout.collect_transferable_tensors(model, processed_layout)

    assert [(name, id(tensor)) for name, tensor in collected] == [(name, id(tensor)) for name, tensor in expected]
    assert len(collected) == 3
    assert classify.call_count < 30
    assert all(
        type(call.args[2]) not in {str, bytes, int, float, bool, complex, type(None)}
        for call in classify.call_args_list
    )


@pytest.mark.parametrize("processed_layout", [False, True])
@pytest.mark.parametrize("number_type, value", [(int, 1), (float, 1.5), (complex, 1j)])
def test_bfs_preserves_scan_scope_and_numeric_subclasses(
    tensor_runtime, monkeypatch, processed_layout, number_type, value
):
    class Number(number_type):
        pass

    class Implementation:
        class_weight = torch.ones(4)

        def __call__(self):
            pass

        def method(self):
            pass

    number = Number(value)
    number.weight = torch.ones(2)
    Implementation.method.weight = torch.ones(6)
    impl = Implementation()
    impl.weight = torch.tensor(1.0)
    impl.number = number
    impl._private = torch.ones(3)
    impl.function = lambda: None
    impl.function.weight = torch.ones(5)
    impl.class_object = Implementation
    impl.bound_method = impl.method
    impl.module = torch.nn.Linear(2, 2)
    for name, base, leaf_value in (("text", str, "text"), ("bytes", bytes, b"bytes")):
        leaf = type("Leaf", (base,), {})(leaf_value)
        leaf.weight = torch.ones(7)
        setattr(impl, name, leaf)
    model = torch.nn.Module()
    model.impl = impl
    model.other = SimpleNamespace(weight=torch.ones(8))
    model._private = torch.ones(9)
    monkeypatch.setattr(tensor_runtime.tensor_layout, "is_tensor_on_transfer_device", lambda _tensor: True)

    collected = tensor_runtime.tensor_layout.collect_transferable_tensors(model, processed_layout)

    assert {id(tensor) for _, tensor in collected} == {id(impl.weight), id(number.weight)}


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
        digest = backend.log_model_layout_summary(
            model,
            False,
            stage="receiver_after_post_load",
            peer_session_id="seed-session",
        )

    message = next(record.getMessage() for record in caplog.records if "RFork tensor layout summary" in record.message)
    assert "stage=receiver_after_post_load" in message
    assert "session=receiver-session peer_session=seed-session" in message
    assert "tensors=1" in message
    assert digest == tensor_layout.build_structural_digest(tensor_layout.collect_transferable_tensors(model, False))
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
@pytest.mark.parametrize("exclude_shared", [False, True])
@pytest.mark.parametrize("deferred", [False, True])
def test_session_reuses_inventory_and_final_digest(runtime, monkeypatch, processed_layout, exclude_shared, deferred):
    prefix = "vllm_ascend.model_loader.rfork"
    layout = sys.modules[f"{prefix}.tensor_layout"]
    transfer = _load_module(monkeypatch, f"{prefix}.transfer_backend", "transfer_backend.py")
    monkeypatch.setattr(runtime.session, "RForkTransferBackend", transfer.RForkTransferBackend)
    monkeypatch.setattr(layout, "is_tensor_on_transfer_device", lambda tensor: True)
    monkeypatch.setattr(layout, "read_npu_format", lambda tensor: 0)
    monkeypatch.setattr(transfer, "read_npu_format", lambda tensor: 0)
    monkeypatch.setattr(sys.modules[f"{prefix}.manifest"], "read_npu_format", lambda tensor: 0)
    model = torch.nn.Module()
    model.weight = torch.nn.Parameter(torch.arange(4, dtype=torch.float32))
    model.register_buffer("scale", torch.tensor(1.0))
    inventory = layout.collect_transferable_tensors(model, processed_layout)
    expected_digest = layout.build_structural_digest(inventory)
    excluded = [(model.scale.data_ptr(), model.scale.element_size())] if exclude_shared else None
    success = SimpleNamespace(is_error=lambda: False)
    blocks = [
        {"address": tensor.data_ptr(), "size": tensor.numel() * tensor.element_size(), "state": "active_allocated"}
        for _, tensor in inventory
    ]
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(memory=SimpleNamespace(memory_snapshot=lambda: [{"blocks": blocks}])),
        raising=False,
    )
    collect = Mock(wraps=layout.collect_transferable_tensors)
    monkeypatch.setattr(transfer, "collect_transferable_tensors", collect)
    monkeypatch.setattr(runtime.session, "collect_transferable_tensors", collect)
    session = runtime.session.RForkSession(runtime.config, runtime.identity)
    backend = session.transfer_backend
    backend.transfer_engine = SimpleNamespace(
        batch_register_memory_ex=lambda registrations: success,
        batch_unregister_memory=lambda addresses: success,
        batch_transfer_sync_read=lambda *args: success,
        finalize=lambda: success,
    )
    backend._memory_registration_cls = lambda *fields: fields
    backend.transfer_session_id = "receiver-session"
    backend._is_initialized = True
    monkeypatch.setattr(session, "_ensure_lease_release_retry_locked", lambda: None)
    monkeypatch.setattr(session, "_run_seed_heartbeat", lambda *args: None)
    monkeypatch.setattr(session.planner, "report_seed_once", lambda *args, **kwargs: True)
    monkeypatch.setattr(session.planner, "remove_seed", lambda: True)
    monkeypatch.setattr(runtime.session, "start_rfork_server", Mock(return_value=Mock(is_alive=True, port=1234)))
    verify = Mock(wraps=session.planner.verify_structural_digest)
    monkeypatch.setattr(session.planner, "verify_structural_digest", verify)

    with pytest.raises(RuntimeError, match="unavailable"):
        backend.snapshot_registered_tensor_inventory()
    try:
        assert session.register_destination(model, processed_layout, excluded)
        assert collect.call_count == 1
        assert session.planner.structural_digest == expected_digest
        assert len(backend._registered_transferable_tensors) == (1 if exclude_shared else 2)
        assert backend.snapshot_registered_tensor_inventory() == inventory
        backend.snapshot_registered_tensor_inventory().clear()
        assert len(backend.snapshot_registered_tensor_inventory()) == 2
        seed_info = runtime.types.SeedTransferInfo(
            "seed-session", dict(backend.weight_manifest), formats=dict(backend.weight_formats)
        )
        monkeypatch.setattr(runtime.session, "fetch_seed_transfer_info", lambda *args: seed_info)
        session.state = runtime.types.RForkLifecycleState.LEASED
        session.seed_lease = runtime.lease
        assert session.transfer_from_seed(model, processed_layout)
        assert collect.call_count == 1
        if not deferred:
            session.seed_lease = None
        model.eval()
        digest = session.log_transferred_model_layout(model, processed_layout)
        assert digest == expected_digest
        assert collect.call_count == 2
        result = session.start_seed_service(model, processed_layout, excluded, structural_digest=digest)
        # Checkpoint-layout promotion needs a new registration after post-load processing.
        expected_scans = 2 + (not processed_layout)
        assert collect.call_count == expected_scans
        if deferred:
            assert result is runtime.types.RForkSeedServiceStartResult.DEFERRED
            verify.assert_not_called()
            model.register_buffer("late_buffer", torch.ones(1))
            live_digest = layout.build_structural_digest(layout.collect_transferable_tensors(model, processed_layout))
            assert live_digest != expected_digest
            session.seed_lease = None
            session._promote_deferred_seed()
            assert collect.call_count == expected_scans + 1
            verify.assert_called_once_with(live_digest)
        else:
            assert result is runtime.types.RForkSeedServiceStartResult.STARTED
            verify.assert_called_once_with(expected_digest)
    finally:
        session.seed_lease = None
        assert session.shutdown()
    with pytest.raises(RuntimeError, match="unavailable"):
        backend.snapshot_registered_tensor_inventory()


@pytest.mark.parametrize("processed_layout", [False, True])
def test_bfs_ids_preserve_graph_structure_and_ignore_order(tensor_runtime, monkeypatch, processed_layout):
    def make_model(reverse):
        model = torch.nn.Module()
        parameter = torch.nn.Parameter(torch.ones(2))
        buffer = torch.ones(3)
        child = torch.nn.Module()
        child.weight = torch.nn.Parameter(torch.ones(4))
        shared = {"weight": torch.ones(5)}
        alias = torch.ones(14)
        pairs = lambda entries: reversed(entries) if reverse else entries
        for name in pairs(["a", "z"]):
            model.register_parameter(name, parameter)
        for name in pairs(["b", "y"]):
            model.register_buffer(name, buffer)
        for name in pairs(["left", "right"]):
            model.add_module(name, child)
        child.add_module("back", model)
        entries = [
            ("a", shared),
            ("b", shared),
            ("b.weight", torch.ones(6)),
            (1, torch.ones(7)),
            ("1", torch.ones(8)),
            ((1, "a"), torch.ones(9)),
            ("topk_indices_buffer", {"weight": torch.ones(10)}),
            ("state", SimpleNamespace(weight=torch.ones(12))),
            ("alias", dict(pairs([("topk_indices_buffer", alias), ("valid", alias)]))),
        ]
        model.impl = dict(pairs(entries))
        model.impl["cycle"] = model.impl
        model.metadata = model.impl
        object.__setattr__(model, "unregistered", torch.nn.Linear(15, 15, bias=False))
        return model

    layout = tensor_runtime.tensor_layout
    monkeypatch.setattr(layout, "is_transferable_tensor", lambda _tensor: True)
    manifests = []
    for reverse in (False, True):
        model = make_model(reverse)
        manifest = {
            name: tuple(tensor.shape) for name, tensor in layout.collect_transferable_tensors(model, processed_layout)
        }
        del model.impl["alias"]["topk_indices_buffer"]
        assert manifest == {
            name: tuple(tensor.shape) for name, tensor in layout.collect_transferable_tensors(model, processed_layout)
        }
        manifests.append(manifest)

    assert manifests[0] == manifests[1]
    assert len(manifests[0]) == 11
    assert set(manifests[0].values()) == {(size,) for size in (*range(2, 11), 12, 14)}


@pytest.mark.parametrize("short_path", [False, True])
def test_bfs_chooses_shortest_path_then_smallest_id(tensor_runtime, monkeypatch, short_path):
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
    if short_path:
        model.impl["short"] = shared["weight"]
    monkeypatch.setattr(layout, "is_transferable_tensor", lambda _tensor: True)

    collected = layout.collect_transferable_tensors(model, True)

    expected = layout._tensor_edge_id(min(candidates.values()), "key", "weight")
    if short_path:
        expected = layout._tensor_edge_id(impl_id, "key", "short")
    assert len(collected) == 1
    assert collected[0][0] == expected.hex()


@pytest.mark.parametrize("processed_layout", [False, True])
@pytest.mark.parametrize("contains_tensor", [False, True])
def test_bfs_expands_shared_diamond_once(tensor_runtime, monkeypatch, processed_layout, contains_tensor):
    class CountingDict(dict):
        scans = 0

        def items(self):
            self.scans += 1
            return super().items()

    depth = 30  # Enumerating every alias would produce over a billion tensor paths.
    weight = torch.ones(2)
    value = CountingDict(value=weight if contains_tensor else 1)
    nodes = [value]
    for _ in range(depth):
        value = CountingDict(left=value, right=value)
        nodes.append(value)
    model = torch.nn.Module()
    model.impl = {"left": value, "right": value, "weight": weight}
    layout = tensor_runtime.tensor_layout
    eligible = Mock(return_value=True)
    edge_id = Mock(wraps=layout._tensor_edge_id)
    monkeypatch.setattr(layout, "is_transferable_tensor", eligible)
    monkeypatch.setattr(layout, "_tensor_edge_id", edge_id)

    collected = layout.collect_transferable_tensors(model, processed_layout)

    assert len(collected) == 1
    assert all(node.scans == 1 for node in nodes)
    assert eligible.call_count == 1
    assert edge_id.call_count == 2 * depth + 4


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


@pytest.mark.parametrize("exclude_shared, matching_aliases", [(False, True), (True, True), (False, False)])
def test_bfs_ids_match_seed_reads_and_shared_exclusions(tensor_runtime, monkeypatch, exclude_shared, matching_aliases):
    def make_model(shared_value, own_value, reverse, matching_aliases=True):
        model = torch.nn.Module()
        shared = torch.full((4,), shared_value, dtype=torch.float32)
        own = torch.full((3,), own_value, dtype=torch.float32)
        entries = [("a", shared), ("b", shared if matching_aliases else shared.clone()), ("own", own)]
        model.impl = dict(reversed(entries) if reverse else entries)
        return model

    layout = tensor_runtime.tensor_layout
    transfer = tensor_runtime.transfer_backend
    manifest = sys.modules["vllm_ascend.model_loader.rfork.manifest"]
    monkeypatch.setattr(layout, "is_transferable_tensor", lambda _tensor: True)
    monkeypatch.setattr(manifest, "read_npu_format", lambda _tensor: 0)
    source = make_model(7, 5, False)
    destination = make_model(11, 13, True, matching_aliases)
    source_tensors = layout.collect_transferable_tensors(source, True)
    destination_tensors = layout.collect_transferable_tensors(destination, True)
    assert (
        layout.build_structural_digest(source_tensors) == layout.build_structural_digest(destination_tensors)
    ) == matching_aliases
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

    success = backend.read_weights_from_seed(destination, seed_info, True)
    if not matching_aliases:
        assert not success
        assert reads == []
        return
    assert success
    assert len(reads) == (1 if exclude_shared else 2)
    assert destination.impl["a"] is destination.impl["b"]
    torch.testing.assert_close(destination.impl["a"], torch.full((4,), 11.0 if exclude_shared else 7.0))
    torch.testing.assert_close(destination.impl["own"], torch.full((3,), 5.0))
