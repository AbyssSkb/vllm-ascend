# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import builtins
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

RFORK_ROOT = Path(__file__).resolve().parents[4] / "vllm_ascend/model_loader/rfork"
HARDWARE_PROFILE_MODULE = "vllm_ascend.device.hardware_profile"
UTILS_MODULE = "vllm_ascend.utils"
_MISSING = object()


def _load_compat(monkeypatch):
    module_name = "vllm_ascend.model_loader.rfork._compat_isolated_test"
    spec = importlib.util.spec_from_file_location(module_name, RFORK_ROOT / "compat.py")
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


def _route_imports(monkeypatch, *, hardware_profile, is_310p):
    original_import = builtins.__import__
    calls = []

    def import_router(name, globals=None, locals=None, fromlist=(), level=0):
        if name == HARDWARE_PROFILE_MODULE:
            calls.append(name)
            if hardware_profile is _MISSING:
                raise ModuleNotFoundError(
                    f"No module named '{HARDWARE_PROFILE_MODULE}'",
                    name=HARDWARE_PROFILE_MODULE,
                )
            if isinstance(hardware_profile, BaseException):
                raise hardware_profile
            return hardware_profile
        if name == UTILS_MODULE:
            calls.append(name)
            return SimpleNamespace(is_310p=is_310p)
        return original_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", import_router)
    return calls


def test_native_hardware_profile_is_loaded_lazily_and_returned_unchanged(monkeypatch):
    expected = object()
    calls = []

    def native_get_current_hardware_profile():
        calls.append("native")
        return expected

    imports = _route_imports(
        monkeypatch,
        hardware_profile=SimpleNamespace(get_current_hardware_profile=native_get_current_hardware_profile),
        is_310p=lambda: pytest.fail("legacy fallback should not be imported"),
    )

    compat = _load_compat(monkeypatch)
    assert imports == []
    assert compat.get_current_hardware_profile() is expected
    assert calls == ["native"]
    assert imports == [HARDWARE_PROFILE_MODULE]


@pytest.mark.parametrize(
    ("is_310p", "expected_layout"),
    [(True, "FORCE_NZ"), (False, "CONFIGURABLE")],
)
def test_missing_hardware_profile_uses_legacy_layout(monkeypatch, is_310p, expected_layout):
    compat = _load_compat(monkeypatch)
    is_310p_calls = []

    def detect_310p():
        is_310p_calls.append(True)
        return is_310p

    imports = _route_imports(
        monkeypatch,
        hardware_profile=_MISSING,
        is_310p=detect_310p,
    )

    profile = compat.get_current_hardware_profile()

    assert profile.weight_layout_policy.name == expected_layout
    assert is_310p_calls == [True]
    assert imports == [HARDWARE_PROFILE_MODULE, UTILS_MODULE]


def test_nested_hardware_profile_dependency_error_is_not_swallowed(monkeypatch):
    compat = _load_compat(monkeypatch)
    nested_error = ModuleNotFoundError(
        "No module named 'native_profile_dependency'",
        name="native_profile_dependency",
    )
    imports = _route_imports(
        monkeypatch,
        hardware_profile=nested_error,
        is_310p=lambda: pytest.fail("nested dependency errors must not use fallback"),
    )

    with pytest.raises(ModuleNotFoundError, match="native_profile_dependency"):
        compat.get_current_hardware_profile()

    assert imports == [HARDWARE_PROFILE_MODULE]


def test_native_hardware_profile_call_error_is_not_replaced_by_fallback(monkeypatch):
    compat = _load_compat(monkeypatch)
    native_error = RuntimeError("native hardware profile failed")

    def fail_native_profile():
        raise native_error

    imports = _route_imports(
        monkeypatch,
        hardware_profile=SimpleNamespace(get_current_hardware_profile=fail_native_profile),
        is_310p=lambda: pytest.fail("native call errors must not use fallback"),
    )

    with pytest.raises(RuntimeError, match="native hardware profile failed"):
        compat.get_current_hardware_profile()

    assert imports == [HARDWARE_PROFILE_MODULE]
