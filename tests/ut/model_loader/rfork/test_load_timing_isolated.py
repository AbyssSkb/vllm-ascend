# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import json
import logging
import sys
from contextlib import contextmanager
from types import SimpleNamespace

import pytest
import torch


@pytest.mark.parametrize("path", ["bypass", "error"])
def test_loader_timing_records_exit_and_all_elapsed_time(rfork_helpers, monkeypatch, caplog, path):
    module = rfork_helpers.loader
    timing_module = sys.modules["vllm_ascend.model_loader.rfork.timing"]
    clock = SimpleNamespace(ns=0)
    monkeypatch.setattr(timing_module.time, "perf_counter_ns", lambda: clock.ns)

    @contextmanager
    def dtype_context(_dtype):
        clock.ns += 1_000_000
        try:
            yield
        finally:
            clock.ns += 2_000_000

    module_config = SimpleNamespace(dtype=torch.float32)
    config = SimpleNamespace(device_config=SimpleNamespace(device="cpu"), model_config=module_config)
    loader = module.RForkModelLoader(SimpleNamespace(device=None, model_loader_extra_config={}))
    monkeypatch.setattr(module, "set_default_torch_dtype", dtype_context)
    monkeypatch.setattr(module, "mutable_weights_bypass_reason", lambda *args: "test" if path == "bypass" else None)
    monkeypatch.setattr(module, "get_ascend_config", lambda: SimpleNamespace(eplb_config=SimpleNamespace()))
    monkeypatch.setattr(module, "_is_dynamic_eplb_enabled", lambda *args: False)
    monkeypatch.setattr(loader, "_requires_processed_layout_transfer", lambda *_: False)
    monkeypatch.setattr(module, "_load_with_default_loader", lambda *args: object())

    def fail(*args):
        raise module.RForkConfigurationError("missing")

    monkeypatch.setattr(loader, "_ensure_rfork_session", fail)
    with caplog.at_level(logging.INFO, logger=module.logger.name):
        if path == "error":
            with pytest.raises(module.RForkConfigurationError, match="missing"):
                loader.load_model(config, module_config)
        else:
            loader.load_model(config, module_config)
    lines = [r.getMessage() for r in caplog.records if r.getMessage().startswith("RFork load timing:")]
    assert len(lines) == 1
    assert f"source={path} outcome={'error' if path == 'error' else 'success'}" in lines[0]
    stages = json.loads(lines[0].split("stages_s=", 1)[1])
    assert sum(stages.values()) == pytest.approx(clock.ns / 1e9, abs=1e-6)
    assert "context_exit" in stages
