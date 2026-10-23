# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Resolve one checkpoint's mounted weights without changing its logical identity."""

import math
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from copy import copy
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.config import ModelConfig, VllmConfig

DEFAULT_MODEL_MOUNT_TIMEOUT_SEC = 600.0


class AsyncModelMount:
    def __init__(self, timeout_sec: float = DEFAULT_MODEL_MOUNT_TIMEOUT_SEC) -> None:
        if (
            isinstance(timeout_sec, bool)
            or not isinstance(timeout_sec, (int, float))
            or not math.isfinite(timeout_sec)
            or timeout_sec <= 0
        ):
            raise ValueError("Async model mount timeout must be a finite positive number")
        self.deadline = time.monotonic() + timeout_sec
        self.done = threading.Event()
        self.path: str | None = None
        self.error: Exception | None = None
        self.completed_at: float | None = None
        # shortcut: the manager has no cancellation hook; add cooperative cancellation when its interface supports it.
        threading.Thread(target=self._resolve, daemon=True, name="AsyncModelMount").start()

    def _resolve(self) -> None:
        try:
            # The optional manager is needed only in an enabled worker process.
            try:
                from model_manager.apis import get_model_path_from_manager
            except ImportError as exc:
                raise ImportError("Async model mount requires model_manager.apis.get_model_path_from_manager") from exc
            path = get_model_path_from_manager(with_weights=True)
            if not isinstance(path, str) or not path.strip():
                raise ValueError("Model manager returned an empty or invalid model path")
            if not Path(path).is_dir():
                raise FileNotFoundError(f"Mounted model directory does not exist: {path}")
            self.path = path
        except Exception as exc:
            self.error = exc
        finally:
            self.completed_at = time.monotonic()
            if self.error is None and self.completed_at >= self.deadline:
                self.path = None
                self.error = TimeoutError("Model manager completed after the async model mount deadline")
            self.done.set()

    def remaining(self) -> float:
        return max(0.0, self.deadline - time.monotonic())

    def wait(self) -> str:
        if not self.done.wait(self.remaining()):
            raise TimeoutError("Timed out waiting for async model mount")
        if self.error is not None:
            raise self.error
        assert self.path is not None
        return self.path


def validate_async_mount_model(vllm_config: "VllmConfig", model_config: "ModelConfig | None" = None) -> None:
    speculative_config = getattr(vllm_config, "speculative_config", None)
    draft_model_config = getattr(speculative_config, "draft_model_config", None)
    if draft_model_config is None:
        return
    target_model_config = getattr(speculative_config, "target_model_config", None) or vllm_config.model_config
    if draft_model_config.model != target_model_config.model:
        raise ValueError("Async model mount supports one checkpoint; an independent draft checkpoint is unsupported")
    if model_config is not None and model_config.model != target_model_config.model:
        raise ValueError("Async model mount model must belong to the configured checkpoint")


def mounted_model_config(
    vllm_config: "VllmConfig", model_config: "ModelConfig", path: str
) -> tuple["VllmConfig", "ModelConfig"]:
    mounted_model = copy(model_config)
    mounted_model.model = path
    mounted_config = copy(vllm_config)
    mounted_config.model_config = (
        mounted_model if vllm_config.model_config is model_config else copy(vllm_config.model_config)
    )
    mounted_config.model_config.model = path
    return mounted_config, mounted_model


@contextmanager
def mounted_model_context(worker, path: str) -> Iterator[None]:
    """Give the ordinary loader and its same-checkpoint drafter local weight paths."""
    validate_async_mount_model(worker.vllm_config)
    original_config, original_model = worker.vllm_config, worker.model_config
    mounted_config, mounted_model = mounted_model_config(original_config, original_model, path)
    speculative_config = getattr(mounted_config, "speculative_config", None)
    if speculative_config is not None:
        speculative_config = copy(speculative_config)
        for name in ("target_model_config", "draft_model_config"):
            config = getattr(speculative_config, name, None)
            if config is not None:
                config = copy(config)
                config.model = path
                setattr(speculative_config, name, config)
        mounted_config.speculative_config = speculative_config

    originals = []

    def replace(owner, name, value):
        if hasattr(owner, name):
            originals.append((owner, name, getattr(owner, name)))
            setattr(owner, name, value)

    try:
        runner = worker.model_runner
        for owner in (worker, runner, getattr(runner, "drafter", None), getattr(runner, "speculator", None)):
            if owner is None:
                continue
            owner_config = getattr(owner, "vllm_config", original_config)
            owner_model = getattr(owner, "model_config", owner_config.model_config)
            if owner_config is original_config and owner_model is original_model:
                owner_config, owner_model = mounted_config, mounted_model
            else:
                owner_config, owner_model = mounted_model_config(owner_config, owner_model, path)
                if speculative_config is not None:
                    owner_config.speculative_config = speculative_config
            replace(owner, "vllm_config", owner_config)
            replace(owner, "model_config", owner_model)
            if speculative_config is not None:
                replace(owner, "speculative_config", speculative_config)
                replace(owner, "spec_config", speculative_config)
                replace(owner, "draft_model_config", getattr(speculative_config, "draft_model_config", None))
        yield
    finally:
        for owner, name, value in reversed(originals):
            setattr(owner, name, value)
