# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Collect live model tensors and adapt their layout for RFork transfer."""

import enum
import json
from collections.abc import Iterator
from types import FunctionType, MethodType
from typing import Any

import regex as re
import torch
from torch import nn
from vllm.logger import logger

from vllm_ascend.model_loader.rfork.manifest import numel_from_shape

# Exact types only: numeric subclasses can carry tensor attributes.
_TENSOR_ATTRIBUTE_LEAF_TYPES = frozenset({str, bytes, int, float, bool, complex, type(None)})
_PLAIN_PATH_SEGMENT = re.compile(r"[A-Za-z0-9_]+")

# Runtime scratch tensors are local execution state, not checkpoint-derived
# model state.  They must keep the capacity selected by the receiving instance
# instead of being copied from a seed that may use different scheduler limits.
_RUNTIME_ONLY_TENSOR_NAMES = frozenset({"topk_indices_buffer"})


def _is_runtime_only_tensor(name: str) -> bool:
    return name.rsplit(".", 1)[-1] in _RUNTIME_ONLY_TENSOR_NAMES


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


def _format_tensor_key(key: Any) -> str | None:
    """Render a container key as a typed label that is identical across processes."""
    # Order matters: bool and Enum members subclass int/str.
    if key is None or isinstance(key, bool):
        return json.dumps(key)
    if isinstance(key, enum.Enum):
        key_type = type(key)
        enum_type = f"{key_type.__module__}.{key_type.__qualname__}"
        if isinstance(key, enum.Flag):
            return f"enum({enum_type}:{int(key.value)})" if isinstance(key.value, int) else None
        return f"enum({enum_type}.{key.name})" if key.name is not None else None
    if isinstance(key, str):
        return json.dumps(key, ensure_ascii=True)
    if isinstance(key, int):
        return str(int(key))
    if isinstance(key, float):
        return f"float({float(key).hex()})"
    if isinstance(key, torch.dtype):
        return f"dtype({key})"
    if type(key) is tuple:
        items: list[str] = []
        for item in key:
            label = _format_tensor_key(item)
            if label is None:
                return None
            items.append(label)
        return f"({items[0]},)" if len(items) == 1 else f"({','.join(items)})"
    # Default reprs embed addresses, so unknown key types cannot yield stable IDs.
    return None


def _tensor_child_path(parent_path: str, kind: str, name: Any) -> str:
    if kind in ("index", "key"):
        label = _format_tensor_key(name)
        if label is None:
            name_type = type(name)
            raise ValueError(
                "RFork cannot derive a stable tensor ID from container key type "
                f"{name_type.__module__}.{name_type.__qualname__}"
            )
        return f"{parent_path}[{label}]"
    name = str(name)
    if not _PLAIN_PATH_SEGMENT.fullmatch(name):
        return f"{parent_path}[{json.dumps(name, ensure_ascii=True)}]"
    return f"{parent_path}.{name}" if parent_path else name


def _iter_tensor_children(
    value: Any, scan_objects: bool, processed_layout: bool, registered_only: bool
) -> Iterator[tuple[str, Any, Any, bool]]:
    if isinstance(value, nn.Module):
        # Read local registries: named_parameters/named_modules discard aliases before we can canonicalize them.
        for kind, members in (("module", value._modules), ("parameter", value._parameters), ("buffer", value._buffers)):
            for name, item in members.items():
                if type(item) not in _TENSOR_ATTRIBUTE_LEAF_TYPES:
                    yield kind, name, item, False
        if registered_only:
            return
        if processed_layout:
            for name, item in vars(value).items():
                if not name.startswith("_") and type(item) not in _TENSOR_ATTRIBUTE_LEAF_TYPES:
                    yield "attribute", name, item, name == "impl"
        else:
            impl = getattr(value, "impl", None)
            if type(impl) not in _TENSOR_ATTRIBUTE_LEAF_TYPES:
                yield "attribute", "impl", impl, True
    elif registered_only:
        return
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
    parent_path: str,
) -> tuple[Any, ...] | None:
    if isinstance(value, torch.Tensor):
        # Reject this alias edge only; another alias may still collect the tensor.
        if _is_runtime_only_tensor(str(name)):
            return None
        metadata = tensor_metadata.get(id(value))
        if metadata is not None:
            return metadata[1]
        signature = None
        if is_transferable_tensor(value):
            validate_transferable_tensor_layout(_tensor_child_path(parent_path, kind, name), value)
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
        return (id(value), False) if kind == "module" else None
    if isinstance(value, (str, bytes)):
        return None
    if isinstance(value, (list, tuple, dict)):
        return (id(value), scan_objects)
    if not scan_objects or isinstance(value, (FunctionType, MethodType, type)):
        return None
    return (id(value), scan_objects) if hasattr(value, "__dict__") else None


def _collect_shortest_path_tensors(
    model: nn.Module,
    processed_layout: bool,
    registered_only: bool,
    claimed: set[tuple[Any, ...]],
    tensor_metadata: dict[int, tuple[torch.Tensor, tuple[Any, ...] | None]],
    collected: list[tuple[str, torch.Tensor]],
) -> None:
    root_key = (id(model), False)
    frontier = {root_key: ""}
    # Keep strong references throughout the scan so object IDs cannot be reused.
    objects: dict[tuple[Any, ...], tuple[Any, bool]] = {root_key: (model, False)}
    visited = set(claimed)

    while frontier:
        # IDs in this layer are final. Ignore back edges, cycles, and longer paths.
        visited.update(frontier)
        next_frontier: dict[tuple[Any, ...], str] = {}
        for key, path in frontier.items():
            value, scan_objects = objects[key]
            if isinstance(value, torch.Tensor):
                claimed.add(key)
                collected.append((path, value))
                continue

            children = _iter_tensor_children(value, scan_objects, processed_layout, registered_only)
            child_paths: set[str] = set()
            for kind, name, item, child_scan_objects in children:
                child_key = _tensor_child_key(kind, name, item, child_scan_objects, tensor_metadata, path)
                if child_key is None:
                    continue
                candidate_path = _tensor_child_path(path, kind, name)
                if candidate_path in child_paths:
                    raise ValueError(
                        f"RFork found distinct container entries with one tensor ID label: {candidate_path!r}"
                    )
                child_paths.add(candidate_path)
                if child_key in visited:
                    continue
                previous_path = next_frontier.get(child_key)
                if previous_path is None or candidate_path < previous_path:
                    # Same-signature aliases can be distinct objects; keep the one the ID names.
                    objects[child_key] = (item, child_scan_objects)
                    next_frontier[child_key] = candidate_path
        frontier = next_frontier


def collect_transferable_tensors(model: nn.Module, processed_layout: bool) -> list[tuple[str, torch.Tensor]]:
    """Collect each tensor range once under its canonical shortest-path ID.

    Registered parameters and buffers take priority over attribute aliases so
    in-place layout changes reach the registered tensor.
    """
    claimed: set[tuple[Any, ...]] = set()
    tensor_metadata: dict[int, tuple[torch.Tensor, tuple[Any, ...] | None]] = {}
    collected: list[tuple[str, torch.Tensor]] = []
    for registered_only in (True, False):
        _collect_shortest_path_tensors(model, processed_layout, registered_only, claimed, tensor_metadata, collected)
    tensor_ids = [tensor_id for tensor_id, _ in collected]
    if len(tensor_ids) != len(set(tensor_ids)):
        raise ValueError("RFork encountered conflicting tensor IDs in one model.")
    return collected


def collect_processed_layout_tensors(model: nn.Module) -> list[tuple[str, torch.Tensor]]:
    return collect_transferable_tensors(model, True)


def collect_checkpoint_layout_tensors(model: nn.Module) -> list[tuple[str, torch.Tensor]]:
    return collect_transferable_tensors(model, False)


def find_non_npu_state_tensors(model: Any) -> list[str]:
    if not isinstance(model, nn.Module):
        return []
    return [
        name
        for iterator in (model.named_parameters(), model.named_buffers())
        for name, tensor in iterator
        if not tensor.is_meta and tensor.numel() > 0 and not is_tensor_on_transfer_device(tensor)
    ]
