# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

"""Collect live model tensors and adapt their layout for RFork transfer."""

import inspect
from collections.abc import Iterator
from typing import Any

import torch
from torch import nn
from vllm.logger import logger

from vllm_ascend.model_loader.rfork.manifest import numel_from_shape


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


def _iter_tensors_in_value(
    prefix: str,
    value: Any,
    visited_object_ids: set[int],
    scan_objects: bool = False,
) -> Iterator[tuple[str, torch.Tensor]]:
    if isinstance(value, torch.Tensor):
        yield prefix, value
        return
    if isinstance(value, (nn.Module, str, bytes)):
        return
    # Function, method, and class objects are executable code containers.  A
    # custom callable instance, however, can own runtime tensors in its public
    # attributes (this is how several quantization implementations expose
    # packed weights).  Scan only the latter when object scanning is enabled.
    if inspect.isfunction(value) or inspect.ismethod(value) or inspect.isclass(value):
        return
    if isinstance(value, (list, tuple)):
        value_id = id(value)
        if value_id in visited_object_ids:
            return
        visited_object_ids.add(value_id)
        try:
            for index, item in enumerate(value):
                yield from _iter_tensors_in_value(f"{prefix}.{index}", item, visited_object_ids, scan_objects)
        finally:
            visited_object_ids.remove(value_id)
        return
    if isinstance(value, dict):
        value_id = id(value)
        if value_id in visited_object_ids:
            return
        visited_object_ids.add(value_id)
        try:
            for key, item in value.items():
                yield from _iter_tensors_in_value(f"{prefix}.{key}", item, visited_object_ids, scan_objects)
        finally:
            visited_object_ids.remove(value_id)
        return
    if callable(value) and (not scan_objects or not hasattr(value, "__dict__")):
        return
    if not scan_objects or not hasattr(value, "__dict__"):
        return
    value_id = id(value)
    if value_id in visited_object_ids:
        return
    visited_object_ids.add(value_id)
    try:
        for attr_name, attr_value in vars(value).items():
            if not attr_name.startswith("_"):
                yield from _iter_tensors_in_value(
                    f"{prefix}.{attr_name}",
                    attr_value,
                    visited_object_ids,
                    scan_objects,
                )
    finally:
        visited_object_ids.remove(value_id)


def _try_collect(
    name: str,
    tensor: torch.Tensor,
    seen_names: dict[str, int],
    collected: list[tuple[str, torch.Tensor]],
) -> None:
    if not is_transferable_tensor(tensor):
        return
    data_ptr = tensor.data_ptr()
    existing_index = seen_names.get(name)
    if existing_index is None:
        seen_names[name] = len(collected)
        collected.append((name, tensor))
        return

    # The same logical name can be encountered through named_parameters and a
    # public implementation attribute.  Deduplicate it only when it describes
    # exactly the same logical tensor.  A same-pointer view with a different
    # range or layout is a conflicting manifest entry and must fail loudly;
    # silently choosing the smaller or larger view can make a seed and receiver
    # disagree about the bytes represented by that name.
    existing_tensor = collected[existing_index][1]
    if existing_tensor is tensor or (
        existing_tensor.data_ptr() == data_ptr
        and existing_tensor.numel() == tensor.numel()
        and tuple(existing_tensor.shape) == tuple(tensor.shape)
        and existing_tensor.dtype == tensor.dtype
        and tuple(existing_tensor.stride()) == tuple(tensor.stride())
    ):
        return

    raise ValueError(
        "RFork encountered conflicting tensor entries for logical name "
        f"{name!r}; shape, dtype, stride, or storage differs."
    )


def collect_processed_layout_tensors(model: nn.Module) -> list[tuple[str, torch.Tensor]]:
    seen: dict[str, int] = {}
    collected: list[tuple[str, torch.Tensor]] = []
    for name, tensor in model.named_parameters():
        _try_collect(name, tensor, seen, collected)
    for name, tensor in model.named_buffers():
        _try_collect(name, tensor, seen, collected)
    for module_prefix, module in model.named_modules():
        for attr_name, attr_value in vars(module).items():
            if attr_name.startswith("_") or isinstance(attr_value, nn.Module):
                continue
            for tensor_name, tensor in _iter_tensors_in_value(
                attr_name,
                attr_value,
                set(),
                attr_name == "impl",
            ):
                full_name = f"{module_prefix}.{tensor_name}" if module_prefix else tensor_name
                _try_collect(full_name, tensor, seen, collected)
    return collected


def collect_checkpoint_layout_tensors(model: nn.Module) -> list[tuple[str, torch.Tensor]]:
    seen: dict[str, int] = {}
    collected: list[tuple[str, torch.Tensor]] = []
    for name, tensor in model.named_parameters():
        _try_collect(name, tensor, seen, collected)
    for name, tensor in model.named_buffers():
        _try_collect(name, tensor, seen, collected)
    for module_prefix, module in model.named_modules():
        impl = getattr(module, "impl", None)
        if impl is None or isinstance(impl, nn.Module):
            continue
        for tensor_name, tensor in _iter_tensors_in_value("impl", impl, set(), scan_objects=True):
            full_name = f"{module_prefix}.{tensor_name}" if module_prefix else tensor_name
            _try_collect(full_name, tensor, seen, collected)
    return collected


def collect_transferable_tensors(model: nn.Module, processed_layout: bool) -> list[tuple[str, torch.Tensor]]:
    if processed_layout:
        return collect_processed_layout_tensors(model)
    return collect_checkpoint_layout_tensors(model)


def find_non_npu_state_tensors(model: Any) -> list[str]:
    if not isinstance(model, nn.Module):
        return []
    return [
        name
        for iterator in (model.named_parameters(), model.named_buffers())
        for name, tensor in iterator
        if not tensor.is_meta and tensor.numel() > 0 and not is_tensor_on_transfer_device(tensor)
    ]
