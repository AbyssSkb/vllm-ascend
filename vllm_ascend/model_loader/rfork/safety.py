# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""RFork loader guards for configurations with mutable model storage."""

from typing import Any


def mutable_weights_bypass_reason(vllm_config: Any, model_config: Any) -> str | None:
    """Return why RFork must not register weights that can later be replaced."""
    if any(
        bool(getattr(config, "enable_sleep_mode", False))
        for config in (model_config, getattr(vllm_config, "model_config", None))
    ):
        return "sleep mode"
    if getattr(vllm_config, "weight_transfer_config", None) is not None:
        return "online weight transfer (weight_transfer_config)"
    if any(
        architecture in getattr(model_config, "architectures", ())
        for architecture in ("Qwen3DSparkModel", "Qwen3OmniDSparkModel", "Glm5DSparkForCausalLM", "K3DSparkModel")
    ):
        from vllm_ascend.utils import get_rotation_path

        # The proposer rotates registered draft weights after seed publication.
        if get_rotation_path(vllm_config) is not None:
            return "DSpark draft weight rotation"
    return None
