# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Resolve externally managed checkpoint files only when they are needed."""

from copy import copy
from pathlib import Path

from vllm_ascend import envs


def enabled() -> bool:
    return envs.VLLM_ASCEND_ASYNC_MODEL_MOUNT


def mounted_configs(vllm_config, model_config):
    spec = getattr(vllm_config, "speculative_config", None)
    target = getattr(spec, "target_model_config", None) or vllm_config.model_config
    if model_config.model != target.model:
        raise ValueError("The model manager cannot select an independent draft checkpoint")

    from model_manager.apis import get_model_path_from_manager

    path = get_model_path_from_manager(with_weights=True)
    if not isinstance(path, str) or not path.strip():
        raise ValueError("Model manager returned an invalid model path")
    if not Path(path).is_dir():
        raise FileNotFoundError(f"Managed model directory does not exist: {path}")

    mounted_model = copy(model_config)
    mounted_model.model = path
    mounted_config = copy(vllm_config)
    mounted_config.model_config = (
        mounted_model if vllm_config.model_config is model_config else copy(vllm_config.model_config)
    )
    mounted_config.model_config.model = path
    return mounted_config, mounted_model
