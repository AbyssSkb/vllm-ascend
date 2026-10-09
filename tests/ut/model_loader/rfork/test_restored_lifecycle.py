# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import logging
import sys
import weakref
from contextlib import nullcontext
from functools import wraps
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from torch.nn import Module
from torch.utils.hooks import RemovableHandle

from .rfork_test_support import _load_module, _stub


@pytest.fixture
def rfork_helpers(monkeypatch):
    """Load the real RFork helpers with only their external imports stubbed."""

    def install(name, **attributes):
        module = _stub(monkeypatch, name, **attributes)
        parent_name, _, child_name = name.rpartition(".")
        parent = sys.modules.get(parent_name)
        if parent is not None:
            monkeypatch.setattr(parent, child_name, module, raising=False)
        return module

    def package(name):
        return install(name, __path__=[])

    package("vllm")
    package("vllm.config")
    install("vllm.config", ModelConfig=object, VllmConfig=object)
    install("vllm.config.load", LoadConfig=object)
    install("vllm.distributed", get_tensor_model_parallel_rank=lambda: 0)
    install("vllm.distributed.parallel_state", get_ep_group=lambda: None, get_pp_group=lambda: None)
    install("vllm.logger", logger=logging.getLogger("rfork-restored-lifecycle-test"))

    package("vllm.model_executor")
    package("vllm.model_executor.model_loader")
    install("vllm.model_executor.model_loader", register_model_loader=lambda name: lambda cls: cls)

    class BaseModelLoader:
        def __init__(self, load_config):
            self.load_config = load_config

    install("vllm.model_executor.model_loader.base_loader", BaseModelLoader=BaseModelLoader)
    install(
        "vllm.model_executor.model_loader.utils",
        initialize_model=lambda **kwargs: None,
        process_weights_after_loading=lambda *args, **kwargs: None,
    )
    package("vllm.utils")
    install("vllm.utils.torch_utils", set_default_torch_dtype=lambda dtype: nullcontext())
    package("vllm.model_executor.layers")
    rope_dict = {"baseline": object()}
    install("vllm.model_executor.layers.rotary_embedding", _ROPE_DICT=rope_dict)

    package("vllm_ascend")
    package("vllm_ascend.device")
    install("vllm_ascend.ascend_config", get_ascend_config=lambda: SimpleNamespace(enable_fused_mc2=0))
    install("vllm_ascend.utils", is_310p=lambda: False, maybe_trans_nz=lambda weight: weight)
    package("vllm_ascend.model_loader")
    package("vllm_ascend.model_loader.rfork")

    safety = _load_module(monkeypatch, "rfork_restored_lifecycle_safety", "safety.py")
    install("vllm_ascend.model_loader.rfork.safety", mutable_weights_bypass_reason=safety.mutable_weights_bypass_reason)
    install("vllm_ascend.model_loader.rfork.config", RForkConfig=object)
    install(
        "vllm_ascend.model_loader.rfork.identity",
        _resolve_sharded_dp_rank=lambda *args, **kwargs: None,
        build_compatibility_fingerprint=lambda *args, **kwargs: "fingerprint",
    )
    install("vllm_ascend.model_loader.rfork.session", RForkSession=object)
    install(
        "vllm_ascend.model_loader.rfork.types",
        RForkFallbackCleanupResult=object,
        RForkIdentity=object,
        RForkLifecycleState=object,
        RForkSeedServiceStartResult=object,
    )

    package("vllm_ascend.eplb")
    package("vllm_ascend.eplb.adaptor")

    class VllmEplbAdaptor:
        _registered_moe_layers = []

    adaptor_module = install("vllm_ascend.eplb.adaptor.vllm_adaptor", VllmEplbAdaptor=VllmEplbAdaptor)
    package("vllm_ascend.ops")
    package("vllm_ascend.ops.fused_moe")

    class AscendMoERunner(Module):
        moe_counter = -1

        def __init__(self, quant_method=None):
            super().__init__()
            self._quant_method = quant_method

    class AscendUnquantizedFusedMoEMethod:
        pass

    install(
        "vllm_ascend.ops.fused_moe.fused_moe",
        AscendMoERunner=AscendMoERunner,
        AscendUnquantizedFusedMoEMethod=AscendUnquantizedFusedMoEMethod,
    )

    rotary_module = install(
        "vllm_ascend.ops.rotary_embedding",
        _cos_sin_cache=None,
        _cos_cache=None,
        _sin_cache=None,
    )

    class DynamoHookRegistry(dict):
        pass

    dynamo_hooks = DynamoHookRegistry()
    install("torch._dynamo.convert_frame", _bytecode_hooks=dynamo_hooks)

    loader = _load_module(monkeypatch, "rfork_restored_lifecycle_loader", "rfork_loader.py")
    return SimpleNamespace(
        loader=loader,
        safety=safety,
        AscendMoERunner=AscendMoERunner,
        adaptor_module=adaptor_module,
        rotary_module=rotary_module,
        dynamo_hooks=dynamo_hooks,
        rope_dict=rope_dict,
    )


@pytest.mark.parametrize(
    ("model_sleep", "vllm_sleep", "weight_transfer", "expected"),
    [
        (True, False, False, "sleep mode"),
        (False, True, False, "sleep mode"),
        (False, False, True, "online weight transfer (weight_transfer_config)"),
        (False, False, False, None),
        (True, False, True, "sleep mode"),
    ],
)
def test_mutable_weights_bypass_reason_reports_sleep_and_weight_transfer(
    rfork_helpers, model_sleep, vllm_sleep, weight_transfer, expected
):
    vllm_config = SimpleNamespace(
        model_config=SimpleNamespace(enable_sleep_mode=vllm_sleep),
        weight_transfer_config=object() if weight_transfer else None,
    )
    model_config = SimpleNamespace(enable_sleep_mode=model_sleep)

    assert rfork_helpers.safety.mutable_weights_bypass_reason(vllm_config, model_config) == expected


def _add_dynamo_hook(hooks, owner):
    handle = RemovableHandle(hooks)
    hooks[handle.id] = owner.bytecode_hook
    owner._bytecode_hook_handle = handle
    return handle.id


def test_fallback_reset_restores_ascend_globals_and_only_removes_new_dynamo_hooks(rfork_helpers):
    class HookOwner(Module):
        def bytecode_hook(self, *args, **kwargs):
            pass

    adaptor = rfork_helpers.adaptor_module.VllmEplbAdaptor
    baseline_layer = object()
    baseline_registry = [baseline_layer]
    adaptor._registered_moe_layers = baseline_registry
    rfork_helpers.AscendMoERunner.moe_counter = 17

    baseline_caches = (object(), object(), object())
    (
        rfork_helpers.rotary_module._cos_sin_cache,
        rfork_helpers.rotary_module._cos_cache,
        rfork_helpers.rotary_module._sin_cache,
    ) = baseline_caches

    baseline_owner = HookOwner()
    baseline_hook_id = _add_dynamo_hook(rfork_helpers.dynamo_hooks, baseline_owner)
    vllm_config = SimpleNamespace(compilation_config=SimpleNamespace())
    snapshot = rfork_helpers.loader._snapshot_process_global_model_state(vllm_config)

    baseline_registry.append(object())
    adaptor._registered_moe_layers = [object()]
    rfork_helpers.AscendMoERunner.moe_counter = 999
    dirty_caches = (object(), object(), object())
    (
        rfork_helpers.rotary_module._cos_sin_cache,
        rfork_helpers.rotary_module._cos_cache,
        rfork_helpers.rotary_module._sin_cache,
    ) = dirty_caches

    discarded_owner = HookOwner()
    discarded_hook_id = _add_dynamo_hook(rfork_helpers.dynamo_hooks, discarded_owner)
    assert discarded_hook_id not in snapshot.dynamo_bytecode_hook_ids

    rfork_helpers.loader._reset_process_global_model_state(vllm_config, snapshot=snapshot)

    assert adaptor._registered_moe_layers is baseline_registry
    assert adaptor._registered_moe_layers == [baseline_layer]
    assert rfork_helpers.AscendMoERunner.moe_counter == 17
    assert (
        rfork_helpers.rotary_module._cos_sin_cache,
        rfork_helpers.rotary_module._cos_cache,
        rfork_helpers.rotary_module._sin_cache,
    ) == baseline_caches
    assert baseline_hook_id in rfork_helpers.dynamo_hooks
    assert rfork_helpers.dynamo_hooks[baseline_hook_id].__self__ is baseline_owner
    assert discarded_hook_id not in rfork_helpers.dynamo_hooks


@pytest.fixture
def post_load_clock(rfork_helpers, monkeypatch):
    clock = SimpleNamespace(now=0.0)
    monkeypatch.setattr(rfork_helpers.loader.time, "perf_counter", lambda: clock.now)
    monkeypatch.setattr(
        torch,
        "npu",
        SimpleNamespace(synchronize=lambda: pytest.fail("hook timing must not add NPU synchronization")),
        raising=False,
    )
    return clock


def _post_load_log_args(caplog):
    records = [record for record in caplog.records if "post-load hooks:" in record.message]
    assert len(records) == 1
    assert records[0].levelno == logging.DEBUG
    return records[0].args


def test_post_load_timing_preserves_tensor_results_and_accounts_for_host_time(rfork_helpers, post_load_clock, caplog):
    class Scheme:
        def process_weights_after_loading(self, layer):
            layer.weight.mul_(2)
            post_load_clock.now += 3
            return layer.weight

    class Adapter:
        def __init__(self):
            self.quant_method = Scheme()

        def process_weights_after_loading(self, layer):
            return self.quant_method.process_weights_after_loading(layer)

    class AttentionImpl:
        def __init__(self, weight):
            self.weight = weight

        def process_weights_after_loading(self, dtype):
            assert dtype == torch.float32
            self.weight.add_(1)
            post_load_clock.now += 2
            return self.weight

    model = Module()
    model.linear = Module()
    model.linear.register_buffer("weight", torch.arange(4, dtype=torch.float32))
    model.linear.quant_method = Adapter()
    model.attention = Module()
    model.attention.register_buffer("weight", torch.arange(4, dtype=torch.float32))
    model.attention.impl = AttentionImpl(model.attention.weight)
    original_attention_hook = model.attention.impl.process_weights_after_loading
    model.attention.process_weights_after_loading = original_attention_hook
    session = SimpleNamespace(identity=SimpleNamespace(is_draft_model=False, tp_rank=3))

    with (
        caplog.at_level(logging.DEBUG, logger=rfork_helpers.loader.logger.name),
        rfork_helpers.loader._rfork_post_load_timing(model, session),
    ):
        assert model.linear.quant_method.process_weights_after_loading(model.linear) is model.linear.weight
        post_load_clock.now += 7
        assert model.attention.process_weights_after_loading(torch.float32) is model.attention.weight

    assert torch.equal(model.linear.weight, torch.arange(4, dtype=torch.float32) * 2)
    assert torch.equal(model.attention.weight, torch.arange(4, dtype=torch.float32) + 1)
    assert "process_weights_after_loading" not in vars(model.linear.quant_method)
    assert model.attention.process_weights_after_loading is original_attention_hook
    kind, rank, process_host, hook_host, other_host, calls, groups, slowest = _post_load_log_args(caplog)
    assert (kind, rank, process_host, hook_host, other_host, calls) == ("main", 3, 12, 5, 7, 2)
    assert groups[("quant", Scheme.__qualname__)] == {"calls": 1, "host_s": 3, "failures": 0}
    assert groups[("attention", AttentionImpl.__qualname__)] == {"calls": 1, "host_s": 2, "failures": 0}
    assert [(entry["module"], entry["host_s"]) for entry in slowest] == [("linear", 3), ("attention", 2)]


def test_post_load_timing_assigns_shared_quant_hooks_to_layers_without_counting_nested_hooks(
    rfork_helpers, post_load_clock, caplog
):
    class Scheme:
        def process_weights_after_loading(self, layer):
            layer.weight.add_(1)
            post_load_clock.now += 3
            return layer.weight

    class SharedAdapter:
        def __init__(self):
            self.quant_method = Scheme()

        def process_weights_after_loading(self, layer):
            post_load_clock.now += 1
            return layer.process_weights_after_loading()

    class Layer(Module):
        def __init__(self, adapter):
            super().__init__()
            self.quant_method = adapter
            self.register_buffer("weight", torch.zeros(2))

        def process_weights_after_loading(self):
            post_load_clock.now += 2
            return self.quant_method.quant_method.process_weights_after_loading(self)

    adapter = SharedAdapter()
    model = Module()
    model.first = Layer(adapter)
    model.second = Layer(adapter)
    session = SimpleNamespace(identity=SimpleNamespace(is_draft_model=False, tp_rank=0))

    with (
        caplog.at_level(logging.DEBUG, logger=rfork_helpers.loader.logger.name),
        rfork_helpers.loader._rfork_post_load_timing(model, session),
    ):
        assert adapter.process_weights_after_loading(model.first) is model.first.weight
        post_load_clock.now += 4
        assert adapter.process_weights_after_loading(layer=model.second) is model.second.weight

    assert torch.equal(model.first.weight, torch.ones(2))
    assert torch.equal(model.second.weight, torch.ones(2))
    for owner in (adapter, model.first, model.second):
        assert "process_weights_after_loading" not in vars(owner)
    _, _, process_host, hook_host, other_host, calls, groups, slowest = _post_load_log_args(caplog)
    assert (process_host, hook_host, other_host, calls) == (16, 12, 4, 2)
    assert groups == {("quant", Scheme.__qualname__): {"calls": 2, "host_s": 12, "failures": 0}}
    assert [(entry["module"], entry["host_s"]) for entry in slowest] == [("first", 6), ("second", 6)]


@pytest.mark.parametrize("level", [logging.INFO, logging.ERROR])
def test_post_load_timing_disabled_does_not_scan_read_clock_or_replace_hooks(rfork_helpers, monkeypatch, caplog, level):
    class UnscannableModel(Module):
        def named_modules(self, *args, **kwargs):
            pytest.fail("disabled diagnostics must not traverse the model")

    model = UnscannableModel()
    original_hook = lambda layer: layer
    model.quant_method = SimpleNamespace(process_weights_after_loading=original_hook)
    monkeypatch.setattr(
        rfork_helpers.loader.time, "perf_counter", lambda: pytest.fail("disabled diagnostics must not read the clock")
    )
    session = SimpleNamespace(identity=SimpleNamespace(is_draft_model=False, tp_rank=0))

    with (
        caplog.at_level(level, logger=rfork_helpers.loader.logger.name),
        rfork_helpers.loader._rfork_post_load_timing(model, session),
    ):
        assert model.quant_method.process_weights_after_loading is original_hook
        assert model.quant_method.process_weights_after_loading(model) is model

    assert model.quant_method.process_weights_after_loading is original_hook
    assert not any("post-load hooks:" in record.message for record in caplog.records)


def test_post_load_timing_restores_hooks_and_records_failure_when_processing_raises(
    rfork_helpers, post_load_clock, caplog
):
    error = ValueError("post-load failed")

    class QuantMethod:
        def process_weights_after_loading(self, layer):
            post_load_clock.now += 2
            raise error

    model = Module()
    model.quant_method = QuantMethod()
    original_module_hook = lambda dtype: dtype
    model.process_weights_after_loading = original_module_hook
    session = SimpleNamespace(identity=SimpleNamespace(is_draft_model=False, tp_rank=0))

    with (
        caplog.at_level(logging.DEBUG, logger=rfork_helpers.loader.logger.name),
        pytest.raises(ValueError) as raised,
        rfork_helpers.loader._rfork_post_load_timing(model, session),
    ):
        model.quant_method.process_weights_after_loading(model)

    assert raised.value is error
    assert "process_weights_after_loading" not in vars(model.quant_method)
    assert model.process_weights_after_loading is original_module_hook
    _, _, process_host, hook_host, other_host, calls, groups, slowest = _post_load_log_args(caplog)
    assert (process_host, hook_host, other_host, calls) == (2, 2, 0, 1)
    assert groups == {("quant", QuantMethod.__qualname__): {"calls": 1, "host_s": 2, "failures": 1}}
    assert slowest[0]["ok"] is False


def test_post_load_timing_limits_slowest_samples_while_aggregating_every_call(rfork_helpers, post_load_clock, caplog):
    class Layer(Module):
        def __init__(self, duration):
            super().__init__()
            self.duration = duration

        def process_weights_after_loading(self, dtype):
            post_load_clock.now += self.duration

    model = Module()
    model.layers = torch.nn.ModuleList(Layer(duration) for duration in range(1, 13))
    session = SimpleNamespace(identity=SimpleNamespace(is_draft_model=False, tp_rank=0))

    with (
        caplog.at_level(logging.DEBUG, logger=rfork_helpers.loader.logger.name),
        rfork_helpers.loader._rfork_post_load_timing(model, session),
    ):
        for layer in model.layers:
            layer.process_weights_after_loading(torch.float32)

    _, _, process_host, hook_host, other_host, calls, groups, slowest = _post_load_log_args(caplog)
    assert (process_host, hook_host, other_host, calls) == (78, 78, 0, 12)
    assert groups == {("attention", Layer.__qualname__): {"calls": 12, "host_s": 78, "failures": 0}}
    assert len(slowest) == 10
    assert [entry["host_s"] for entry in slowest] == list(range(12, 2, -1))
    assert [entry["module"] for entry in slowest] == [f"layers.{index}" for index in range(11, 1, -1)]
    assert all("process_weights_after_loading" not in vars(layer) for layer in model.layers)


@pytest.mark.parametrize("should_fail", [False, True])
def test_post_load_timing_preserves_moe_validation_bypass_and_restores_both_contexts(
    rfork_helpers, post_load_clock, caplog, should_fail
):
    validation_calls = []

    class QuantMethod:
        def process_weights_after_loading(self, layer):
            layer.weight.add_(1)
            post_load_clock.now += 3
            if should_fail:
                raise RuntimeError("moe processing failed")
            return layer.weight

    quant_method = QuantMethod()
    original_hook = quant_method.process_weights_after_loading

    @wraps(original_hook)
    def validated_hook(*args, **kwargs):
        result = original_hook(*args, **kwargs)
        validation_calls.append("validate")
        return result

    quant_method.process_weights_after_loading = validated_hook
    model = Module()
    model.runner = rfork_helpers.AscendMoERunner(quant_method)
    model.experts = Module()
    model.experts.quant_method = quant_method
    model.experts.register_buffer("weight", torch.zeros(2))
    session = SimpleNamespace(identity=SimpleNamespace(is_draft_model=False, tp_rank=0))
    expectation = pytest.raises(RuntimeError, match="moe processing failed") if should_fail else nullcontext()

    with (
        caplog.at_level(logging.DEBUG, logger=rfork_helpers.loader.logger.name),
        expectation,
        rfork_helpers.loader._rfork_pre_transfer_weight_processing(model),
    ):
        assert quant_method.process_weights_after_loading is original_hook
        with rfork_helpers.loader._rfork_post_load_timing(model, session):
            assert quant_method.process_weights_after_loading(model.experts) is model.experts.weight
        assert quant_method.process_weights_after_loading is original_hook

    assert quant_method.process_weights_after_loading is validated_hook
    assert validation_calls == []
    assert torch.equal(model.experts.weight, torch.ones(2))
    _, _, process_host, hook_host, other_host, calls, groups, slowest = _post_load_log_args(caplog)
    assert (process_host, hook_host, other_host, calls) == (3, 3, 0, 1)
    assert groups == {("quant", QuantMethod.__qualname__): {"calls": 1, "host_s": 3, "failures": int(should_fail)}}
    assert slowest[0]["module"] == "experts"
    assert slowest[0]["ok"] is not should_fail


@pytest.mark.parametrize("processed_layout", [False, True])
def test_load_limits_nz_probe_to_processed_layout_and_preserves_transfer_order(
    rfork_helpers, post_load_clock, monkeypatch, caplog, processed_layout
):
    """Run the temporary experiment before hooks without changing raw-layout loading or weights."""
    loader_module = rfork_helpers.loader
    events = []

    class QuantMethod:
        def process_weights_after_loading(self, layer):
            layer.weight.mul_(2)
            post_load_clock.now += 3

    model = Module()
    model.register_buffer("weight", torch.ones(2))
    model.quant_method = QuantMethod()
    model_config = SimpleNamespace(dtype=torch.float32, quantization="w8a8" if processed_layout else None)
    vllm_config = SimpleNamespace(device_config=SimpleNamespace(device="cpu"))

    def initialize_model(**kwargs):
        events.append("initialize")
        post_load_clock.now += 5
        return model

    def process_weights_after_loading(loaded_model, config, target_device):
        assert loaded_model is model and config is model_config
        events.append("process")
        loaded_model.quant_method.process_weights_after_loading(loaded_model)

    def synchronize(device=None):
        is_init_drain = events[-1] == "initialize"
        assert device == (torch.device("cpu") if is_init_drain else None)
        events.append("init_drain" if is_init_drain else "synchronize")
        post_load_clock.now += 6 if is_init_drain else 4

    def register_destination(loaded_model, transfer_processed_layout, exclude_blocks):
        assert loaded_model is model and transfer_processed_layout == processed_layout and exclude_blocks == []
        if processed_layout:
            assert events[-1] == "synchronize"
            assert post_load_clock.now == 19
        else:
            assert events[-1] == "initialize" and post_load_clock.now == 5
        events.append("register")
        return True

    def transfer_from_seed(loaded_model, transfer_processed_layout):
        assert transfer_processed_layout == processed_layout
        assert torch.equal(loaded_model.weight, torch.full((2,), 2.0 if processed_layout else 1.0))
        events.append("transfer")
        loaded_model.weight.fill_(9)
        return True

    session = SimpleNamespace(
        identity=SimpleNamespace(is_draft_model=False, tp_rank=0),
        register_destination=register_destination,
        acquire_seed=lambda: events.append("acquire") or True,
        transfer_from_seed=transfer_from_seed,
        log_transferred_model_layout=lambda *args: None,
    )
    loader = object.__new__(loader_module.RForkModelLoader)
    loader.load_config = SimpleNamespace(device=None)
    loader.rfork_config = SimpleNamespace(post_load_init_sync=False)

    def probe_nz_conversion(target_device, probe_session):
        assert processed_layout
        assert target_device == torch.device("cpu") and probe_session is session
        assert events[-1] == "init_drain"
        assert torch.equal(model.weight, torch.ones(2))
        events.append("probe")
        post_load_clock.now += 1

    monkeypatch.setattr(loader_module, "_rfork_probe_nz_conversion", probe_nz_conversion)
    monkeypatch.setattr(loader, "_ensure_rfork_session", lambda *args: session)
    monkeypatch.setattr(loader, "_get_target_registered_blocks", lambda *args: [])
    monkeypatch.setattr(loader_module, "initialize_model", initialize_model)
    monkeypatch.setattr(loader_module, "process_weights_after_loading", process_weights_after_loading)
    monkeypatch.setattr(loader_module, "_snapshot_process_global_model_state", lambda *args: None)
    monkeypatch.setattr(
        loader_module, "_start_rfork_seed_service", lambda *args, **kwargs: events.append("seed") or True
    )
    monkeypatch.setattr(
        loader_module,
        "get_ascend_config",
        lambda: SimpleNamespace(
            eplb_config=SimpleNamespace(dynamic_eplb=False, expert_map_record_path=None, expert_map_path=None)
        ),
    )
    monkeypatch.setattr(torch.npu, "synchronize", synchronize)

    with caplog.at_level(logging.DEBUG, logger=loader_module.logger.name):
        assert loader.load_model(vllm_config, model_config) is model

    if processed_layout:
        expected_events = ["initialize", "init_drain", "probe"]
        assert events == expected_events + ["process", "synchronize", "register", "acquire", "transfer", "seed"]
    else:
        assert events == ["initialize", "register", "acquire", "transfer", "process", "seed"]
    assert torch.equal(model.weight, torch.full((2,), 9.0 if processed_layout else 18.0))
    assert not model.training
    stages = [record for record in caplog.records if "layout processing stages:" in record.message]
    assert len(stages) == int(processed_layout)
    if processed_layout:
        assert stages[0].args == ("main", 3, 4, 7)
    drain_logs = [record for record in caplog.records if "post-load initialization drain:" in record.message]
    assert len(drain_logs) == int(processed_layout)
    if drain_logs:
        assert drain_logs[0].args == ("main", 0, 6)
    assert "process_weights_after_loading" not in vars(model.quant_method)
    if processed_layout:
        _, _, process_host, hook_host, other_host, calls, _, _ = _post_load_log_args(caplog)
        assert (process_host, hook_host, other_host, calls) == (3, 3, 0, 1)
    else:
        assert not any("post-load hooks:" in record.message for record in caplog.records)


@pytest.mark.parametrize("is_draft", [False, True])
@pytest.mark.parametrize("converted", [False, True])
def test_nz_probe_uses_independent_inputs_and_separates_call_from_device_wait(
    rfork_helpers, post_load_clock, monkeypatch, caplog, is_draft, converted
):
    loader_module = rfork_helpers.loader
    device = torch.device("cpu")
    events = []
    inputs = []
    outputs = []
    empty = torch.empty
    call_times = (5, 2)
    wait_times = iter((3, 7, 11))

    def allocate(shape, *, dtype, device):
        assert device == torch.device("cpu")
        weight = empty(shape, dtype=dtype, device="cpu")
        inputs.append(weight)
        events.append("allocate")
        post_load_clock.now += 2
        return weight

    def synchronize(sync_device):
        assert sync_device == device
        events.append("synchronize")
        post_load_clock.now += next(wait_times)

    def cast(weight):
        iteration = len(outputs)
        assert weight is inputs[iteration]
        assert events[-1] == "synchronize"
        if iteration:
            assert inputs[0] is not weight and inputs[0].data_ptr() != weight.data_ptr()
            assert outputs[0]() is not None
        events.append("cast")
        post_load_clock.now += call_times[iteration]
        output = weight.clone() if converted else weight
        outputs.append(weakref.ref(output))
        return output

    monkeypatch.setattr(torch, "empty", allocate)
    monkeypatch.setattr(torch.npu, "synchronize", synchronize)
    monkeypatch.setattr(loader_module, "maybe_trans_nz", cast)
    monkeypatch.setattr(loader_module, "get_ascend_config", lambda: SimpleNamespace(weight_nz_mode=int(converted)))
    session = SimpleNamespace(identity=SimpleNamespace(is_draft_model=is_draft, tp_rank=3))

    with caplog.at_level(logging.DEBUG, logger=loader_module.logger.name):
        loader_module._rfork_probe_nz_conversion(device, session)

    assert events == ["allocate", "allocate", "synchronize", "cast", "synchronize", "cast", "synchronize"]
    kind = "draft" if is_draft else "main"
    preparation = [record for record in caplog.records if "NZ conversion probe inputs:" in record.message]
    assert len(preparation) == 1
    assert preparation[0].args == (kind, 3, device, 4, 3)
    probes = [record for record in caplog.records if "NZ conversion probe:" in record.message]
    assert [record.levelno for record in probes] == [logging.DEBUG, logging.DEBUG]
    assert [record.args for record in probes] == [
        (kind, 3, 1, (4096, 1024), "torch.int8", 4194304, int(converted), converted, 5, 7, 12),
        (kind, 3, 2, (4096, 1024), "torch.int8", 4194304, int(converted), converted, 2, 11, 13),
    ]
    if converted:
        assert all(output() is None for output in outputs)


def test_nz_probe_propagates_conversion_error_without_running_second_sample(
    rfork_helpers, post_load_clock, monkeypatch
):
    loader_module = rfork_helpers.loader
    device = torch.device("cpu")
    error = RuntimeError("NZ conversion failed")
    cast = Mock(side_effect=error)
    synchronize = Mock()
    monkeypatch.setattr(loader_module, "maybe_trans_nz", cast)
    monkeypatch.setattr(torch.npu, "synchronize", synchronize)
    session = SimpleNamespace(identity=SimpleNamespace(is_draft_model=False, tp_rank=0))

    with pytest.raises(RuntimeError) as raised:
        loader_module._rfork_probe_nz_conversion(device, session)

    assert raised.value is error
    assert cast.call_count == 1
    synchronize.assert_called_once_with(device)


@pytest.mark.parametrize(
    ("level", "had_marker", "should_fail"),
    [
        (logging.DEBUG, False, False),
        (logging.DEBUG, True, False),
        (logging.DEBUG, False, True),
        (logging.DEBUG, True, True),
        (logging.INFO, False, False),
    ],
)
def test_first_w8a8_wq_a_stages_are_sampled_once_and_marker_is_restored(
    rfork_helpers, post_load_clock, caplog, level, had_marker, should_fail
):
    """Sample the first wq_a and restore its marker even when nested processing fails."""
    observed_markers = []
    original_marker = {"preserved": True}
    error = ValueError("first W8A8 conversion failed")

    def process_weights_after_loading(self, layer):
        marker = getattr(layer, "_rfork_post_load_stages", None)
        observed_markers.append(marker)
        if marker is not None:
            assert marker is not original_marker
            marker.update(transpose_copy_host=1.0, nz_cast_host=2.0, weight_shape=(2,), weight_dtype="torch.float32")
        layer.weight.add_(1)
        post_load_clock.now += 3
        if should_fail and layer is model.layers[0].wq_a:
            raise error
        return layer.weight

    scheme_type = type(
        "AscendW8A8DynamicLinearMethod", (), {"process_weights_after_loading": process_weights_after_loading}
    )

    class Adapter:
        def __init__(self):
            self.quant_method = scheme_type()

        def process_weights_after_loading(self, layer):
            return layer.process_weights_after_loading()

    class Layer(Module):
        def __init__(self, adapter):
            super().__init__()
            self.quant_method = adapter
            self.register_buffer("weight", torch.zeros(2))

        def process_weights_after_loading(self):
            return self.quant_method.quant_method.process_weights_after_loading(self)

    adapter = Adapter()
    model = Module()
    model.q_proj = Layer(adapter)
    model.layers = torch.nn.ModuleList([Module(), Module()])
    model.layers[0].wq_a = Layer(adapter)
    model.layers[1].wq_a = Layer(adapter)
    first = model.layers[0].wq_a
    second = model.layers[1].wq_a
    if had_marker:
        first._rfork_post_load_stages = original_marker
    session = SimpleNamespace(identity=SimpleNamespace(is_draft_model=False, tp_rank=2))
    expectation = pytest.raises(ValueError) if should_fail else nullcontext()

    with (
        caplog.at_level(level, logger=rfork_helpers.loader.logger.name),
        expectation as raised,
        rfork_helpers.loader._rfork_post_load_timing(model, session),
    ):
        assert adapter.process_weights_after_loading(model.q_proj) is model.q_proj.weight
        assert adapter.process_weights_after_loading(first) is first.weight
        assert adapter.process_weights_after_loading(second) is second.weight

    if should_fail:
        assert raised.value is error
    if had_marker:
        assert first._rfork_post_load_stages is original_marker
        assert original_marker == {"preserved": True}
    else:
        assert "_rfork_post_load_stages" not in vars(first)
    assert "_rfork_post_load_stages" not in vars(second)
    assert "process_weights_after_loading" not in vars(adapter)
    assert all("process_weights_after_loading" not in vars(layer) for layer in (model.q_proj, first, second))
    assert observed_markers[0] is None
    sample_logs = [record for record in caplog.records if "first W8A8 linear stages:" in record.message]
    if level == logging.INFO:
        assert observed_markers == [None, None, None]
        assert sample_logs == []
        assert not any("post-load hooks:" in record.message for record in caplog.records)
        return

    assert isinstance(observed_markers[1], dict)
    if not should_fail:
        assert observed_markers[2] is None
    assert len(sample_logs) == 1
    kind, rank, details = sample_logs[0].args
    assert (kind, rank) == ("main", 2)
    assert details == {
        "module": "layers.0.wq_a",
        "host_s": 3,
        "ok": not should_fail,
        "transpose_copy_host": 1.0,
        "nz_cast_host": 2.0,
        "weight_shape": (2,),
        "weight_dtype": "torch.float32",
    }
    _, _, process_host, hook_host, other_host, calls, groups, _ = _post_load_log_args(caplog)
    expected_calls = 2 if should_fail else 3
    assert (process_host, hook_host, other_host, calls) == (3 * expected_calls, 3 * expected_calls, 0, expected_calls)
    assert groups == {
        ("quant", "AscendW8A8DynamicLinearMethod"): {
            "calls": expected_calls,
            "host_s": 3 * expected_calls,
            "failures": int(should_fail),
        }
    }
