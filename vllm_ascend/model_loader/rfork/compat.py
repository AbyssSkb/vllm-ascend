# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Compatibility with Ascend releases predating hardware profiles."""

from types import SimpleNamespace
from typing import Any


def get_current_hardware_profile() -> Any:
    # Resolve lazily so RFork can be inspected without initializing NPU utilities.
    try:
        from vllm_ascend.device.hardware_profile import get_current_hardware_profile as get_native_profile
    except ModuleNotFoundError as exc:
        if exc.name != "vllm_ascend.device.hardware_profile":
            raise

        from vllm_ascend.utils import is_310p

        policy_name = "FORCE_NZ" if is_310p() else "CONFIGURABLE"
        return SimpleNamespace(weight_layout_policy=SimpleNamespace(name=policy_name))

    return get_native_profile()
