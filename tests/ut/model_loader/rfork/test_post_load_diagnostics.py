# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import logging
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import nullcontext
from types import ModuleType, SimpleNamespace

import pytest
import torch
from torch import nn

from .test_restored_lifecycle import rfork_helpers as _rfork_helpers

rfork_helpers = _rfork_helpers


@pytest.fixture
def diagnostics(rfork_helpers, monkeypatch):
    loader = rfork_helpers.loader
    clock = SimpleNamespace(now=0.0)
    events = []
    calls = []
    outputs = []
    state = SimpleNamespace(error=None)

    def native_cast(weight, *args, **kwargs):
        calls.append((weight, args, kwargs))
        destination_format = args[0] if args else kwargs.get("acl_format")
        events.append(f"cast:{destination_format}")
        clock.now += 3.0
        if state.error is not None:
            raise state.error
        return outputs.pop(0) if outputs else weight

    def synchronize(device):
        assert device == torch.device("cpu")
        events.append("sync")
        clock.now += 2.0 if events.count("sync") == 1 else 5.0

    def get_npu_format(tensor):
        events.append("metadata:format")
        return 29

    def get_storage_size(tensor):
        events.append("metadata:physical")
        return 128

    torch_npu = ModuleType("torch_npu")
    torch_npu.npu_format_cast = native_cast
    torch_npu.get_npu_format = get_npu_format
    torch_npu.get_storage_size = get_storage_size
    monkeypatch.setitem(sys.modules, "torch_npu", torch_npu)
    monkeypatch.setattr(torch, "npu", SimpleNamespace(synchronize=synchronize), raising=False)
    monkeypatch.setattr(loader.time, "perf_counter", lambda: clock.now)
    return SimpleNamespace(
        loader=loader,
        clock=clock,
        events=events,
        calls=calls,
        outputs=outputs,
        state=state,
        torch_npu=torch_npu,
        native_cast=native_cast,
        session=SimpleNamespace(identity=SimpleNamespace(is_draft_model=False, tp_rank=3)),
        device=torch.device("cpu"),
    )


def _record(caplog, marker):
    records = [record for record in caplog.records if marker in record.getMessage()]
    assert len(records) == 1
    return records[0]


def test_nz_diagnostics_times_real_calls_without_warmup(diagnostics, caplog):
    caplog.set_level(logging.DEBUG, logger=diagnostics.loader.logger.name)
    source = torch.arange(12, dtype=torch.int8).reshape(3, 4)[1:, ::2]
    outputs = [torch.empty(4, dtype=torch.int8) for _ in range(3)]
    diagnostics.outputs.extend(outputs)

    with diagnostics.loader._rfork_post_load_timing(nn.Module(), diagnostics.session, diagnostics.device):
        assert diagnostics.torch_npu.npu_format_cast(source, 2) is outputs[0]
        assert diagnostics.torch_npu.npu_format_cast(source, 29, customize_dtype=torch.int8) is outputs[1]
        assert diagnostics.torch_npu.npu_format_cast(source, acl_format=29) is outputs[2]

    assert diagnostics.torch_npu.npu_format_cast is diagnostics.native_cast
    assert len(diagnostics.calls) == 3
    assert all(call[0] is source for call in diagnostics.calls)
    assert diagnostics.calls[1][1:] == ((29,), {"customize_dtype": torch.int8})
    assert diagnostics.calls[2][1:] == ((), {"acl_format": 29})
    assert diagnostics.events[:4] == ["cast:2", "sync", "cast:29", "sync"]
    assert diagnostics.events.count("sync") == 2

    records = [record for record in caplog.records if "NZ conversion call:" in record.getMessage()]
    assert len(records) == 2
    first = records[0].getMessage()
    assert "synchronized=True pre_wait=2.0000s call_host=3.0000s post_wait=5.0000s" in first
    assert "synchronized=False pre_wait=0.0000s call_host=3.0000s post_wait=0.0000s" in records[1].getMessage()
    metadata = records[0].args[-2]
    assert metadata["shape"] == (2, 2)
    assert metadata["dtype"] == "torch.int8"
    assert metadata["stride"] == (4, 2)
    assert metadata["storage_offset"] == 4
    assert metadata["ptr_offset_bytes"] == 4
    assert metadata["logical_bytes"] == 4
    assert metadata["storage_bytes"] == 12
    assert metadata["npu_storage_numel"] == 128
    assert metadata["format"] == 29
    summary = _record(caplog, "NZ conversion summary:").getMessage()
    assert "tp_rank=3 calls=2 call_host=6.0000s pre_wait=2.0000s post_wait=5.0000s" in summary


def test_info_diagnostics_does_not_enumerate_wrap_or_synchronize(diagnostics, monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger=diagnostics.loader.logger.name)

    class Model(nn.Module):
        def named_modules(self, *args, **kwargs):
            raise AssertionError("INFO diagnostics must not enumerate the model")

    def unexpected_clock():
        raise AssertionError("INFO diagnostics must not read the clock")

    monkeypatch.setattr(diagnostics.loader.time, "perf_counter", unexpected_clock)
    tensor = torch.ones(1)
    with diagnostics.loader._rfork_post_load_timing(Model(), diagnostics.session, diagnostics.device):
        assert diagnostics.torch_npu.npu_format_cast is diagnostics.native_cast
        assert diagnostics.torch_npu.npu_format_cast(tensor, 29) is tensor
    assert diagnostics.events == ["cast:29"]
    assert not caplog.records


def test_shared_nested_hooks_preserve_results_paths_and_instance_attributes(diagnostics, caplog):
    caplog.set_level(logging.DEBUG, logger=diagnostics.loader.logger.name)

    class ActualScheme:
        def process_weights_after_loading(self, layer):
            layer.weight.add_(1)
            diagnostics.clock.now += 1.0
            return layer.weight.clone()

    class Adapter:
        def __init__(self):
            self.quant_method = ActualScheme()

        def process_weights_after_loading(self, layer):
            return self.quant_method.process_weights_after_loading(layer)

    class ActualAttentionImpl:
        pass

    class Layer(nn.Module):
        def __init__(self, adapter):
            super().__init__()
            self.quant_method = adapter
            self.impl = ActualAttentionImpl()
            self.register_buffer("weight", torch.zeros(2))

        def process_weights_after_loading(self, dtype):
            return self.quant_method.process_weights_after_loading(self)

    def model_and_instance_hook():
        model = nn.Module()
        adapter = Adapter()
        model.left = Layer(adapter)
        model.right = Layer(adapter)
        instance_hook = lambda dtype: adapter.process_weights_after_loading(model.right)
        model.right.process_weights_after_loading = instance_hook
        return model, adapter, instance_hook

    def process(model):
        results = []
        for layer in (model.left, model.right):
            results.append(layer.quant_method.process_weights_after_loading(layer))
            results.append(layer.process_weights_after_loading(torch.float32))
        return results

    baseline, _, _ = model_and_instance_hook()
    expected = process(baseline)
    model, adapter, instance_hook = model_and_instance_hook()
    with diagnostics.loader._rfork_post_load_timing(model, diagnostics.session, diagnostics.device):
        actual = process(model)

    assert all(torch.equal(left, right) for left, right in zip(actual, expected))
    assert torch.equal(model.left.weight, baseline.left.weight)
    assert torch.equal(model.right.weight, baseline.right.weight)
    assert "process_weights_after_loading" not in vars(adapter)
    assert "process_weights_after_loading" not in vars(model.left)
    assert model.right.process_weights_after_loading is instance_hook
    summary = _record(caplog, "post-load hooks:")
    assert "process_host=4.0000s hook_host=4.0000s other_host=0.0000s calls=4" in summary.getMessage()
    assert "ActualScheme" in summary.getMessage()
    assert "ActualAttentionImpl" in summary.getMessage()
    assert {entry[1] for entry in summary.args[-1]} == {"left", "right"}
    assert diagnostics.events == []


@pytest.mark.parametrize("failure", ["cast", "hook"])
def test_exceptions_propagate_and_restore_cast_and_hooks(diagnostics, caplog, failure):
    caplog.set_level(logging.DEBUG, logger=diagnostics.loader.logger.name)
    error = RuntimeError("original operation failed")

    class QuantMethod:
        def process_weights_after_loading(self, layer):
            if failure == "hook":
                raise error
            return diagnostics.torch_npu.npu_format_cast(layer.weight, 29)

    model = nn.Module()
    model.layer = nn.Module()
    model.layer.quant_method = QuantMethod()
    model.layer.weight = torch.ones(2)
    if failure == "cast":
        diagnostics.state.error = error
    with (
        pytest.raises(RuntimeError) as raised,
        diagnostics.loader._rfork_post_load_timing(model, diagnostics.session, diagnostics.device),
    ):
        model.layer.quant_method.process_weights_after_loading(model.layer)
    assert raised.value is error
    assert diagnostics.torch_npu.npu_format_cast is diagnostics.native_cast
    assert "process_weights_after_loading" not in vars(model.layer.quant_method)
    assert "failures': 1" in _record(caplog, "post-load hooks:").getMessage()


def test_other_thread_delegates_to_original_cast_and_hook(diagnostics, caplog):
    caplog.set_level(logging.DEBUG, logger=diagnostics.loader.logger.name)

    class QuantMethod:
        def process_weights_after_loading(self, layer):
            layer.weight.add_(1)
            return diagnostics.torch_npu.npu_format_cast(layer.weight, 29)

    model = nn.Module()
    model.layer = nn.Module()
    model.layer.weight = torch.zeros(2)
    model.layer.quant_method = QuantMethod()
    with diagnostics.loader._rfork_post_load_timing(model, diagnostics.session, diagnostics.device):
        with ThreadPoolExecutor(max_workers=1) as pool:
            result = pool.submit(model.layer.quant_method.process_weights_after_loading, model.layer).result()
        assert result is model.layer.weight
        assert diagnostics.events == ["cast:29"]
        assert model.layer.quant_method.process_weights_after_loading(model.layer) is model.layer.weight
    assert len(diagnostics.calls) == 2
    assert diagnostics.events.count("sync") == 2
    assert torch.equal(model.layer.weight, torch.full((2,), 2.0))
    assert "calls=1" in _record(caplog, "post-load hooks:").getMessage()
    assert "calls=1" in _record(caplog, "NZ conversion summary:").getMessage()
    assert diagnostics.torch_npu.npu_format_cast is diagnostics.native_cast


def test_post_conversion_sync_failure_preserves_call_record_and_restores_dispatch(diagnostics, monkeypatch, caplog):
    caplog.set_level(logging.DEBUG, logger=diagnostics.loader.logger.name)
    error = RuntimeError("post-conversion synchronization failed")
    sync_calls = []

    def synchronize(device):
        sync_calls.append(device)
        diagnostics.clock.now += 2.0 if len(sync_calls) == 1 else 5.0
        if len(sync_calls) == 2:
            raise error

    monkeypatch.setattr(torch.npu, "synchronize", synchronize)

    class QuantMethod:
        def process_weights_after_loading(self, layer):
            return diagnostics.torch_npu.npu_format_cast(layer.weight, 29)

    model = nn.Module()
    model.layer = nn.Module()
    model.layer.weight = torch.zeros(2)
    model.layer.quant_method = QuantMethod()
    instance_hook = lambda dtype: model.layer.quant_method.process_weights_after_loading(model.layer)
    model.layer.process_weights_after_loading = instance_hook
    with (
        pytest.raises(RuntimeError) as raised,
        diagnostics.loader._rfork_post_load_timing(model, diagnostics.session, diagnostics.device),
    ):
        model.layer.process_weights_after_loading(torch.float32)

    assert raised.value is error
    assert sync_calls == [diagnostics.device, diagnostics.device]
    assert len(diagnostics.calls) == 1
    assert diagnostics.torch_npu.npu_format_cast is diagnostics.native_cast
    assert "process_weights_after_loading" not in vars(model.layer.quant_method)
    assert model.layer.process_weights_after_loading is instance_hook
    summary = _record(caplog, "NZ conversion summary:").getMessage()
    assert "calls=1 call_host=3.0000s pre_wait=2.0000s post_wait=5.0000s" in summary
    assert "post_sync_error': 'RuntimeError'" in summary


def test_slowest_hook_samples_are_bounded(diagnostics, caplog):
    caplog.set_level(logging.DEBUG, logger=diagnostics.loader.logger.name)

    class Layer(nn.Module):
        def __init__(self, elapsed):
            super().__init__()
            self.elapsed = elapsed

        def process_weights_after_loading(self, dtype):
            diagnostics.clock.now += self.elapsed

    model = nn.Module()
    model.layers = nn.ModuleList(Layer(index) for index in range(12))
    with diagnostics.loader._rfork_post_load_timing(model, diagnostics.session, diagnostics.device):
        for layer in model.layers:
            layer.process_weights_after_loading(torch.float32)
    slowest = _record(caplog, "post-load hooks:").args[-1]
    assert len(slowest) == 10
    assert [record[1] for record in slowest] == [f"layers.{index}" for index in range(11, 1, -1)]


def test_unavailable_physical_metadata_does_not_change_conversion(diagnostics, monkeypatch, caplog):
    caplog.set_level(logging.DEBUG, logger=diagnostics.loader.logger.name)
    monkeypatch.setattr(diagnostics.torch_npu, "get_storage_size", lambda tensor: (16, 16))
    tensor = torch.ones(2)
    with diagnostics.loader._rfork_post_load_timing(nn.Module(), diagnostics.session, diagnostics.device):
        assert diagnostics.torch_npu.npu_format_cast(tensor, 29) is tensor
    metadata = _record(caplog, "NZ conversion call:").args[-2]
    assert metadata["npu_storage_numel"].startswith("unavailable")
    assert len(diagnostics.calls) == 1


@pytest.mark.parametrize("level,processed", [(logging.DEBUG, True), (logging.INFO, True), (logging.DEBUG, False)])
def test_load_model_connects_diagnostics_only_to_processed_debug_path(
    diagnostics, monkeypatch, caplog, level, processed
):
    caplog.set_level(level, logger=diagnostics.loader.logger.name)
    sync_arguments = []
    wrapped_calls = []

    class QuantMethod:
        def process_weights_after_loading(self, layer):
            return diagnostics.torch_npu.npu_format_cast(layer.weight, 29)

    model = nn.Module()
    model.layer = nn.Module()
    model.layer.weight = torch.ones(2)
    model.layer.quant_method = QuantMethod()

    def initialize_model(**kwargs):
        diagnostics.events.append("initialize")
        return model

    def process_weights_after_loading(actual_model, model_config, target_device):
        assert actual_model is model
        diagnostics.events.append("posthooks")
        wrapped_calls.append(diagnostics.torch_npu.npu_format_cast is not diagnostics.native_cast)
        actual_model.layer.quant_method.process_weights_after_loading(actual_model.layer)

    def synchronize(*args):
        diagnostics.events.append("sync")
        sync_arguments.append(args)
        diagnostics.clock.now += 1.0

    def stage(name):
        def run(*args, **kwargs):
            diagnostics.events.append(name)
            return True

        return run

    monkeypatch.setattr(diagnostics.loader, "initialize_model", initialize_model)
    monkeypatch.setattr(diagnostics.loader, "process_weights_after_loading", process_weights_after_loading)
    monkeypatch.setattr(torch.npu, "synchronize", synchronize)
    monkeypatch.setattr(diagnostics.loader, "_start_rfork_seed_service", stage("seed"))
    monkeypatch.setattr(
        diagnostics.loader, "_rfork_skip_unquantized_moe_post_load_processing", lambda model: nullcontext()
    )
    monkeypatch.setattr(
        diagnostics.loader,
        "get_ascend_config",
        lambda: SimpleNamespace(
            weight_nz_mode=0,
            eplb_config=SimpleNamespace(expert_map_path=None, dynamic_eplb=False, expert_map_record_path=None),
        ),
    )
    session = diagnostics.session
    session.register_destination = stage("register")
    session.acquire_seed = stage("acquire")
    session.transfer_from_seed = stage("transfer")
    session.log_transferred_model_layout = lambda *args: None
    loader = diagnostics.loader.RForkModelLoader.__new__(diagnostics.loader.RForkModelLoader)
    loader.load_config = SimpleNamespace(device=None)
    loader._ensure_rfork_session = lambda *args: session
    model_config = SimpleNamespace(dtype=torch.float32, quantization="test" if processed else None)
    vllm_config = SimpleNamespace(
        model_config=model_config,
        device_config=SimpleNamespace(device="cpu"),
        parallel_config=SimpleNamespace(enable_eplb=False),
    )

    assert loader.load_model(vllm_config, model_config) is model
    assert len(diagnostics.calls) == 1
    assert wrapped_calls == [processed and level == logging.DEBUG]
    assert diagnostics.torch_npu.npu_format_cast is diagnostics.native_cast
    assert "process_weights_after_loading" not in vars(model.layer.quant_method)
    events = [event for event in diagnostics.events if not event.startswith("metadata:")]
    if processed and level == logging.DEBUG:
        assert events == [
            "initialize",
            "sync",
            "posthooks",
            "sync",
            "cast:29",
            "sync",
            "sync",
            "register",
            "acquire",
            "transfer",
            "seed",
        ]
        assert sync_arguments == [(diagnostics.device,)] * 3 + [()]
    elif processed:
        assert events == ["initialize", "posthooks", "cast:29", "sync", "register", "acquire", "transfer", "seed"]
        assert sync_arguments == [()]
    else:
        assert events == ["initialize", "register", "acquire", "transfer", "posthooks", "cast:29", "seed"]
        assert sync_arguments == []
