# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Collect live model tensors and adapt their layout for RFork transfer."""

import hashlib
import json
import logging
from collections.abc import Iterator
from types import FunctionType, MethodType
from typing import Any

import torch
from torch import nn
from vllm.logger import logger

from vllm_ascend.model_loader.rfork.manifest import (
    normalize_dtype_name,
    numel_from_shape,
    read_npu_format,
)

TENSOR_LAYOUT_SAMPLE_LIMIT = 3

# Exact types only: numeric subclasses can carry tensor attributes.
_TENSOR_ATTRIBUTE_LEAF_TYPES = frozenset({str, bytes, int, float, bool, complex, type(None)})

# Runtime scratch tensors are local execution state, not checkpoint-derived
# model state.  They must keep the capacity selected by the receiving instance
# instead of being copied from a seed that may use different scheduler limits.
_RUNTIME_ONLY_TENSOR_NAMES = frozenset({"topk_indices_buffer"})


def _is_runtime_only_tensor(name: str) -> bool:
    return name.rsplit(".", 1)[-1] in _RUNTIME_ONLY_TENSOR_NAMES


def _layout_digest(records: list[dict[str, Any]]) -> str:
    payload = json.dumps(records, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(payload.encode()).hexdigest()


def build_structural_digest(tensors: list[tuple[str, torch.Tensor]]) -> str:
    """Digest the transferable tensor set's names, shapes, dtypes, and NPU formats.

    This summarizes exactly what ``validate_weight_manifest`` compares between a
    seed and a receiver, derived from the built model rather than from
    configuration fields.  Pass the output of ``collect_transferable_tensors``
    so runtime-only scratch buffers stay excluded.  ``read_npu_format`` yields
    ``None`` off-device, which keeps the digest well defined during CPU-only
    inspection.
    """
    records = [
        {
            "name": name,
            "shape": tuple(int(dim) for dim in tensor.shape),
            "dtype": normalize_dtype_name(tensor.dtype),
            "npu_format": read_npu_format(tensor),
        }
        for name, tensor in sorted(tensors, key=lambda item: item[0])
    ]
    return _layout_digest(records)


def log_tensor_layout_summary(
    tensors: list[tuple[str, torch.Tensor]],
    *,
    stage: str,
    session_id: str | None,
    processed_layout: bool,
    peer_session_id: str | None = None,
    known_formats: dict[str, int] | None = None,
) -> None:
    """Log a bounded summary of logical and physical tensor layouts at INFO."""
    if not logger.isEnabledFor(logging.INFO):
        return

    try:
        import torch_npu
    except Exception:
        torch_npu = None

    semantic_records: list[dict[str, Any]] = []
    physical_records: list[dict[str, Any]] = []
    samples: list[dict[str, Any]] = []
    fallback_samples: list[dict[str, Any]] = []
    format_counts: dict[str, int] = {}
    error_counts: dict[str, int] = {}
    logical_bytes_total = 0
    unique_storage_bytes = 0
    unique_storages: set[tuple[str, int]] = set()
    storage_view_tensors = 0
    physical_nonlogical_tensors = 0

    def capture(read, field: str):
        try:
            return read()
        except Exception as exc:
            error_name = f"{field}:{type(exc).__name__}"
            error_counts[error_name] = error_counts.get(error_name, 0) + 1
            return "unavailable"

    for name, tensor in sorted(tensors, key=lambda item: item[0]):
        device = capture(lambda tensor=tensor: str(tensor.device), "device")
        dtype = capture(lambda tensor=tensor: str(tensor.dtype), "dtype")
        shape = capture(lambda tensor=tensor: tuple(int(value) for value in tensor.shape), "shape")
        stride = capture(lambda tensor=tensor: tuple(int(value) for value in tensor.stride()), "stride")
        numel = capture(lambda tensor=tensor: int(tensor.numel()), "numel")
        element_size = capture(lambda tensor=tensor: int(tensor.element_size()), "element_size")
        logical_bytes = (
            numel * element_size if isinstance(numel, int) and isinstance(element_size, int) else "unavailable"
        )
        storage_offset = capture(lambda tensor=tensor: int(tensor.storage_offset()), "storage_offset")
        storage_bytes = capture(lambda tensor=tensor: int(tensor.untyped_storage().nbytes()), "storage_bytes")
        storage_ptr = capture(lambda tensor=tensor: int(tensor.untyped_storage().data_ptr()), "storage_ptr")
        if known_formats is not None and name in known_formats:
            npu_format: Any = known_formats[name]
        elif getattr(getattr(tensor, "device", None), "type", None) == "npu" and torch_npu is not None:
            npu_format = capture(
                lambda tensor=tensor: int(torch_npu.get_npu_format(tensor)),
                "npu_format",
            )
        else:
            npu_format = "unavailable"
        if getattr(getattr(tensor, "device", None), "type", None) == "npu" and torch_npu is not None:
            npu_storage_numel: Any = capture(
                lambda tensor=tensor: int(torch_npu.get_storage_size(tensor)),
                "npu_storage_numel",
            )
        else:
            npu_storage_numel = "unavailable"

        if isinstance(logical_bytes, int):
            logical_bytes_total += logical_bytes
        if isinstance(storage_ptr, int) and isinstance(storage_bytes, int):
            storage_key = (str(device), storage_ptr)
            if storage_key not in unique_storages:
                unique_storages.add(storage_key)
                unique_storage_bytes += storage_bytes
        is_storage_view = (
            isinstance(storage_offset, int)
            and isinstance(storage_bytes, int)
            and isinstance(logical_bytes, int)
            and (storage_offset != 0 or storage_bytes != logical_bytes)
        )
        is_physical_nonlogical = (
            isinstance(npu_storage_numel, int) and isinstance(numel, int) and npu_storage_numel != numel
        )
        storage_view_tensors += int(is_storage_view)
        physical_nonlogical_tensors += int(is_physical_nonlogical)

        semantic = {
            "name": name,
            "dtype": dtype,
            "shape": shape,
            "stride": stride,
            "logical_bytes": logical_bytes,
            "npu_format": npu_format,
        }
        physical = {
            "name": name,
            "storage_offset": storage_offset,
            "storage_bytes": storage_bytes,
            "npu_storage_numel": npu_storage_numel,
        }
        semantic_records.append(semantic)
        physical_records.append(physical)
        sample = {**semantic, **physical, "device": device}
        if len(fallback_samples) < TENSOR_LAYOUT_SAMPLE_LIMIT:
            fallback_samples.append(sample)
        if (is_storage_view or is_physical_nonlogical) and len(samples) < TENSOR_LAYOUT_SAMPLE_LIMIT:
            samples.append(sample)

        format_key = str(npu_format)
        format_counts[format_key] = format_counts.get(format_key, 0) + 1

    if not samples:
        samples = fallback_samples
    logger.info(
        "RFork tensor layout summary: stage=%s session=%s peer_session=%s layout=%s tensors=%d "
        "logical_bytes=%d unique_storage_bytes=%d storage_view_tensors=%d physical_nonlogical_tensors=%d "
        "formats=%s semantic_digest=%s physical_digest=%s samples=%s errors=%s",
        stage,
        session_id,
        peer_session_id,
        "processed" if processed_layout else "checkpoint",
        len(semantic_records),
        logical_bytes_total,
        unique_storage_bytes,
        storage_view_tensors,
        physical_nonlogical_tensors,
        format_counts,
        _layout_digest(semantic_records),
        _layout_digest(physical_records),
        samples,
        error_counts,
    )


def reshape_tensor_to_seed_shape(
    name: str,
    tensor: torch.Tensor,
    seed_shape: tuple[int, ...] | None,
    reshape_events: list[tuple[str, tuple[int, ...], tuple[int, ...]]] | None = None,
) -> bool:
    if seed_shape is None or tuple(tensor.shape) == seed_shape:
        return True
    if tensor.numel() != numel_from_shape(seed_shape):
        logger.error("Weight shape mismatch for %s: local=%s, seed=%s", name, tuple(tensor.shape), seed_shape)
        return False
    local_shape = tuple(tensor.shape)
    try:
        tensor.data = tensor.data.view(seed_shape)
    except Exception as exc:
        logger.error("Failed to reshape RFork tensor %s from %s to %s: %s", name, local_shape, seed_shape, exc)
        return False
    if reshape_events is not None:
        reshape_events.append((name, local_shape, seed_shape))
    return True


def is_tensor_on_transfer_device(tensor: torch.Tensor) -> bool:
    return tensor.device.type == "npu"


def is_transferable_tensor(tensor: torch.Tensor) -> bool:
    return not tensor.is_meta and tensor.numel() > 0 and is_tensor_on_transfer_device(tensor)


def is_non_overlapping_dense_tensor(tensor: torch.Tensor) -> bool:
    """Return whether logical elements occupy one contiguous byte range."""
    if tensor.numel() <= 1:
        return True

    dense_stride = 1
    for stride, size in sorted(
        (int(stride), int(size)) for size, stride in zip(tensor.shape, tensor.stride(), strict=True) if size > 1
    ):
        if stride != dense_stride:
            return False
        dense_stride *= size
    return True


def validate_transferable_tensor_layout(name: str, tensor: torch.Tensor) -> None:
    """Reject tensor views that cannot be represented by RFork byte ranges."""
    if is_non_overlapping_dense_tensor(tensor):
        return
    raise ValueError(
        "RFork cannot transfer a tensor with gapped or overlapping storage: "
        f"{name!r}; shape={tuple(tensor.shape)}, stride={tuple(tensor.stride())}."
    )


def _tensor_edge_id(parent_id: bytes, kind: str, name: Any) -> bytes:
    # Typed, framed labels distinguish e.g. dict keys 1 / "1" and "a.b" / a -> b.
    name_type = type(name)
    label = json.dumps(
        (kind, name_type.__module__, name_type.__qualname__, name),
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    ).encode()
    return hashlib.sha256(parent_id + label).digest()


def _iter_tensor_children(
    value: Any, scan_objects: bool, processed_layout: bool
) -> Iterator[tuple[str, Any, Any, bool]]:
    if isinstance(value, nn.Module):
        # Read local registries: named_parameters/named_modules discard aliases before we can canonicalize them.
        for kind, members in (("module", value._modules), ("parameter", value._parameters), ("buffer", value._buffers)):
            for name, item in members.items():
                if type(item) not in _TENSOR_ATTRIBUTE_LEAF_TYPES:
                    yield kind, name, item, False
        if processed_layout:
            for name, item in vars(value).items():
                if not name.startswith("_") and type(item) not in _TENSOR_ATTRIBUTE_LEAF_TYPES:
                    yield "attribute", name, item, name == "impl"
        else:
            impl = getattr(value, "impl", None)
            if type(impl) not in _TENSOR_ATTRIBUTE_LEAF_TYPES:
                yield "attribute", "impl", impl, True
    elif isinstance(value, (list, tuple)):
        for index, item in enumerate(value):
            if type(item) not in _TENSOR_ATTRIBUTE_LEAF_TYPES:
                yield "index", index, item, scan_objects
    elif isinstance(value, dict):
        for name, item in value.items():
            if type(item) not in _TENSOR_ATTRIBUTE_LEAF_TYPES:
                yield "key", name, item, scan_objects
    else:
        for name, item in vars(value).items():
            if not name.startswith("_") and type(item) not in _TENSOR_ATTRIBUTE_LEAF_TYPES:
                yield "attribute", name, item, scan_objects


def _tensor_child_key(
    kind: str,
    name: Any,
    value: Any,
    scan_objects: bool,
    tensor_metadata: dict[int, tuple[torch.Tensor, tuple[Any, ...] | None]],
) -> tuple[Any, ...] | None:
    """Filter a candidate edge and return its local tensor-range or object key."""
    if isinstance(value, torch.Tensor):
        # Reject this alias edge only; another alias may still collect the tensor.
        if _is_runtime_only_tensor(f"{name}"):
            return None
        metadata = tensor_metadata.get(id(value))
        if metadata is not None:
            return metadata[1]
        signature = None
        if is_transferable_tensor(value):
            validate_transferable_tensor_layout(f"{name}", value)
            signature = (
                value.data_ptr(),
                value.numel(),
                tuple(value.shape),
                value.dtype,
                value.device,
                tuple(value.stride()),
            )
        tensor_metadata[id(value)] = (value, signature)
        return signature
    if isinstance(value, nn.Module):
        # Follow modules only through their registration edges.
        return (id(value), False) if kind == "module" else None
    if isinstance(value, (str, bytes)):
        return None
    if isinstance(value, (list, tuple, dict)):
        return (id(value), scan_objects)
    if not scan_objects or isinstance(value, (FunctionType, MethodType, type)):
        return None
    return (id(value), scan_objects) if hasattr(value, "__dict__") else None


def collect_transferable_tensors(model: nn.Module, processed_layout: bool) -> list[tuple[str, torch.Tensor]]:
    """Collect each tensor range once using order-independent, shortest-path IDs.

    Finish an entire BFS layer before expanding the next. Each node takes the
    smallest ID offered by its shortest-path predecessors; no alias paths are
    enumerated. Addresses are used only for local tensor-range deduplication.
    """
    root_key = (id(model), False)
    frontier = {root_key: hashlib.sha256(b"rfork-tensor-id").digest()}
    # Keep strong references throughout the scan so object IDs cannot be reused.
    objects: dict[tuple[Any, ...], tuple[Any, bool]] = {root_key: (model, False)}
    visited: set[tuple[Any, ...]] = set()
    tensor_metadata: dict[int, tuple[torch.Tensor, tuple[Any, ...] | None]] = {}
    collected: list[tuple[str, torch.Tensor]] = []
    collected_ids: set[str] = set()

    while frontier:
        # IDs in this layer are final. Ignore back edges, cycles, and longer paths.
        visited.update(frontier)
        next_frontier: dict[tuple[Any, ...], bytes] = {}
        for key, node_id in frontier.items():
            value, scan_objects = objects[key]
            if isinstance(value, torch.Tensor):
                tensor_id = node_id.hex()
                if tensor_id in collected_ids:
                    raise ValueError(f"RFork encountered conflicting tensor IDs: {tensor_id!r}")
                collected_ids.add(tensor_id)
                collected.append((tensor_id, value))
                continue

            for kind, name, item, child_scan_objects in _iter_tensor_children(value, scan_objects, processed_layout):
                child_key = _tensor_child_key(kind, name, item, child_scan_objects, tensor_metadata)
                if child_key is None or child_key in visited:
                    continue
                candidate_id = _tensor_edge_id(node_id, kind, name)
                previous_id = next_frontier.get(child_key)
                if previous_id is None:
                    objects[child_key] = (item, child_scan_objects)
                    next_frontier[child_key] = candidate_id
                elif candidate_id < previous_id:
                    next_frontier[child_key] = candidate_id
        frontier = next_frontier
    return collected


def find_non_npu_state_tensors(model: Any) -> list[str]:
    if not isinstance(model, nn.Module):
        return []
    return [
        name
        for iterator in (model.named_parameters(), model.named_buffers())
        for name, tensor in iterator
        if not tensor.is_meta and tensor.numel() > 0 and not is_tensor_on_transfer_device(tensor)
    ]
