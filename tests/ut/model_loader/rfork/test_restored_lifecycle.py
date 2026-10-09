# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import logging
import math
import sys
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager, nullcontext
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
    install("vllm_ascend.utils", ACL_FORMAT_FRACTAL_NZ=29, ACL_FORMAT_ND=2, is_310p=lambda: False)
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
def test_load_limits_nz_allocation_to_processed_layout_and_preserves_transfer_order(
    rfork_helpers, post_load_clock, monkeypatch, caplog, processed_layout
):
    """Allocate NZ only while processed-layout hooks run, preserving transfer and raw-layout behavior."""
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
            assert post_load_clock.now == 18
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

    @contextmanager
    def allocate_nz_weights(allocation_session):
        assert processed_layout
        assert allocation_session is session
        assert events[-1] == "init_drain"
        assert torch.equal(model.weight, torch.ones(2))
        events.append("allocation_enter")
        try:
            yield
        finally:
            events.append("allocation_exit")

    monkeypatch.setattr(loader_module, "_rfork_allocate_nz_weights", allocate_nz_weights)
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
        assert events == [
            "initialize",
            "init_drain",
            "allocation_enter",
            "process",
            "allocation_exit",
            "synchronize",
            "register",
            "acquire",
            "transfer",
            "seed",
        ]
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


def _nz_weight(
    shape=(4096, 1024),
    dtype=torch.int8,
    device_type="npu",
    contiguous=True,
    npu_format=2,
    storage_offset=0,
    storage_bytes=None,
    base_shape=None,
    base_stride=None,
    base_dtype=None,
):
    logical_bytes = math.prod(shape) * (1 if dtype == torch.int8 else 4)
    physical_bytes = logical_bytes if storage_bytes is None else storage_bytes
    stride = tuple(math.prod(shape[index + 1 :]) for index in range(len(shape)))
    descriptor_shape = shape if base_shape is None else base_shape
    descriptor_stride = tuple(math.prod(descriptor_shape[index + 1 :]) for index in range(len(descriptor_shape)))
    storage = SimpleNamespace(
        nbytes=lambda: physical_bytes,
        base_shape=descriptor_shape,
        base_stride=descriptor_stride if base_stride is None else base_stride,
        base_dtype=dtype if base_dtype is None else base_dtype,
    )
    return SimpleNamespace(
        shape=shape,
        dtype=dtype,
        device=SimpleNamespace(type=device_type, index=0),
        npu_format=npu_format,
        is_contiguous=lambda: contiguous,
        stride=lambda: stride,
        numel=lambda: math.prod(shape),
        element_size=lambda: 1 if dtype == torch.int8 else 4,
        storage_offset=lambda: storage_offset,
        untyped_storage=lambda: storage,
    )


@pytest.fixture
def nz_backend(rfork_helpers, post_load_clock, monkeypatch):
    def allocate(*, size, dtype, device, acl_format):
        post_load_clock.now += 2
        return _nz_weight(shape=size, dtype=dtype, device_type=device.type, npu_format=acl_format)

    return _stub(
        monkeypatch,
        "torch_npu",
        npu_format_cast=Mock(return_value=object()),
        empty_with_format=Mock(side_effect=allocate),
        get_npu_format=lambda weight: weight.npu_format,
        _C=SimpleNamespace(
            _tensor_construct_from_storage=Mock(
                side_effect=lambda storage: SimpleNamespace(
                    shape=storage.base_shape,
                    stride=lambda: storage.base_stride,
                    dtype=storage.base_dtype,
                )
            )
        ),
    )


@pytest.mark.parametrize("shape", [(4096, 1024), (8, 4096, 1024)])
@pytest.mark.parametrize("is_draft", [False, True])
@pytest.mark.parametrize("use_keywords", [False, True])
def test_nz_allocation_preserves_tensor_metadata_and_restores_cast(
    rfork_helpers, nz_backend, caplog, shape, is_draft, use_keywords
):
    weight = _nz_weight(shape=shape)
    original_cast = nz_backend.npu_format_cast
    session = SimpleNamespace(identity=SimpleNamespace(is_draft_model=is_draft, tp_rank=3))

    with (
        caplog.at_level(logging.DEBUG, logger=rfork_helpers.loader.logger.name),
        rfork_helpers.loader._rfork_allocate_nz_weights(session),
    ):
        result = (
            nz_backend.npu_format_cast(self=weight, acl_format=29)
            if use_keywords
            else nz_backend.npu_format_cast(weight, 29)
        )
        assert result is not weight
        assert result.shape == weight.shape and result.dtype == weight.dtype
        assert result.device.type == weight.device.type and result.npu_format == 29

    assert nz_backend.npu_format_cast is original_cast
    original_cast.assert_not_called()
    nz_backend.empty_with_format.assert_called_once_with(
        size=weight.shape, dtype=weight.dtype, device=weight.device, acl_format=29
    )
    records = [record for record in caplog.records if "NZ destination allocations:" in record.message]
    assert len(records) == 1 and records[0].levelno == logging.DEBUG
    assert records[0].args == (
        "draft" if is_draft else "main",
        3,
        1,
        math.prod(shape),
        2,
        {"shape": shape, "dtype": "torch.int8", "format": 29, "storage_bytes": math.prod(shape)},
    )


@pytest.mark.parametrize("weight_options", [{}, {"storage_offset": 1, "storage_bytes": 8388608}])
def test_nz_allocation_preserves_already_nz_tensor_alias(rfork_helpers, nz_backend, weight_options):
    weight = _nz_weight(npu_format=29, **weight_options)
    original_cast = nz_backend.npu_format_cast
    session = SimpleNamespace(identity=SimpleNamespace(is_draft_model=False, tp_rank=0))

    with rfork_helpers.loader._rfork_allocate_nz_weights(session):
        assert nz_backend.npu_format_cast(weight, 29) is weight

    assert nz_backend.npu_format_cast is original_cast
    original_cast.assert_not_called()
    nz_backend.empty_with_format.assert_not_called()


@pytest.mark.parametrize(
    ("weight_options", "acl_format", "args", "kwargs"),
    [
        ({}, 2, (), {}),
        ({"dtype": torch.float32}, 29, (), {}),
        ({"device_type": "cpu"}, 29, (), {}),
        ({"contiguous": False}, 29, (), {}),
        ({"shape": (4096,)}, 29, (), {}),
        ({"npu_format": 3}, 29, (), {}),
        ({"storage_offset": 1}, 29, (), {}),
        ({"storage_bytes": 8388608}, 29, (), {}),
        ({"base_shape": (1024, 4096)}, 29, (), {}),
        ({"base_stride": (1, 4096)}, 29, (), {}),
        ({"base_dtype": torch.float32}, 29, (), {}),
        ({}, 29, (0,), {}),
        ({}, 29, (), {"customize_dtype": 0}),
    ],
)
def test_nz_allocation_preserves_unsupported_conversion_calls(
    rfork_helpers, nz_backend, weight_options, acl_format, args, kwargs
):
    weight = _nz_weight(**weight_options)
    original_cast = nz_backend.npu_format_cast
    session = SimpleNamespace(identity=SimpleNamespace(is_draft_model=False, tp_rank=0))

    with rfork_helpers.loader._rfork_allocate_nz_weights(session):
        assert nz_backend.npu_format_cast(weight, acl_format, *args, **kwargs) is original_cast.return_value

    assert nz_backend.npu_format_cast is original_cast
    original_cast.assert_called_once_with(weight, acl_format, *args, **kwargs)
    nz_backend.empty_with_format.assert_not_called()


def test_nz_allocation_accepts_native_self_keyword_when_passing_cpu_tensor_through(rfork_helpers, nz_backend):
    weight = _nz_weight(device_type="cpu")
    original_cast = nz_backend.npu_format_cast
    session = SimpleNamespace(identity=SimpleNamespace(is_draft_model=False, tp_rank=0))

    with rfork_helpers.loader._rfork_allocate_nz_weights(session):
        assert nz_backend.npu_format_cast(self=weight, acl_format=29) is original_cast.return_value

    original_cast.assert_called_once_with(weight, 29)
    assert nz_backend.npu_format_cast is original_cast
    nz_backend.empty_with_format.assert_not_called()


@pytest.mark.parametrize("error", [None, RuntimeError("descriptor unavailable"), TypeError("unsupported storage")])
def test_nz_allocation_preserves_original_cast_when_base_descriptor_is_unavailable(
    rfork_helpers, nz_backend, monkeypatch, error
):
    weight = _nz_weight()
    original_cast = nz_backend.npu_format_cast
    if error is None:
        monkeypatch.delattr(nz_backend._C, "_tensor_construct_from_storage")
    else:
        nz_backend._C._tensor_construct_from_storage.side_effect = error
    session = SimpleNamespace(identity=SimpleNamespace(is_draft_model=False, tp_rank=0))

    with rfork_helpers.loader._rfork_allocate_nz_weights(session):
        assert nz_backend.npu_format_cast(weight, 29) is original_cast.return_value

    assert nz_backend.npu_format_cast is original_cast
    original_cast.assert_called_once_with(weight, 29)
    nz_backend.empty_with_format.assert_not_called()


@pytest.mark.parametrize("use_keywords", [False, True])
def test_nz_allocation_does_not_intercept_other_threads(rfork_helpers, nz_backend, use_keywords):
    weight = _nz_weight()
    original_cast = nz_backend.npu_format_cast
    session = SimpleNamespace(identity=SimpleNamespace(is_draft_model=False, tp_rank=0))

    with rfork_helpers.loader._rfork_allocate_nz_weights(session), ThreadPoolExecutor(max_workers=1) as executor:
        conversion = (
            executor.submit(nz_backend.npu_format_cast, self=weight, acl_format=29)
            if use_keywords
            else executor.submit(nz_backend.npu_format_cast, weight, 29)
        )
        result = conversion.result(timeout=5)

    assert result is original_cast.return_value
    assert nz_backend.npu_format_cast is original_cast
    original_cast.assert_called_once_with(weight, 29)
    nz_backend.empty_with_format.assert_not_called()


@pytest.mark.parametrize("failure", ["body", "allocator", "output_format"])
def test_nz_allocation_restores_cast_after_failure(rfork_helpers, nz_backend, failure):
    weight = _nz_weight()
    original_cast = nz_backend.npu_format_cast
    error = RuntimeError("allocation or processing failed")
    if failure == "allocator":
        nz_backend.empty_with_format.side_effect = error
    elif failure == "output_format":
        nz_backend.empty_with_format.side_effect = None
        nz_backend.empty_with_format.return_value = _nz_weight(npu_format=2)
    session = SimpleNamespace(identity=SimpleNamespace(is_draft_model=False, tp_rank=0))

    with pytest.raises(RuntimeError) as raised, rfork_helpers.loader._rfork_allocate_nz_weights(session):
        nz_backend.npu_format_cast(weight, 29)
        if failure == "body":
            raise error

    if failure == "output_format":
        assert "did not preserve format 29" in str(raised.value)
    else:
        assert raised.value is error
    assert nz_backend.npu_format_cast is original_cast
    original_cast.assert_not_called()
    assert nz_backend.empty_with_format.call_count == 1


def test_nz_allocation_is_restored_before_seed_miss_reloads_a_new_model(rfork_helpers, nz_backend, monkeypatch):
    loader_module = rfork_helpers.loader
    original_cast = nz_backend.npu_format_cast
    empty_model = Module()
    local_model = Module()
    local_model.register_buffer("weight", torch.full((2,), 7.0))
    reset = Mock()
    model_config = SimpleNamespace(dtype=torch.float32, quantization="ascend")
    vllm_config = SimpleNamespace(device_config=SimpleNamespace(device="cpu"))
    session = SimpleNamespace(
        identity=SimpleNamespace(is_draft_model=False, tp_rank=0),
        register_destination=Mock(return_value=True),
        acquire_seed=lambda: False,
        prepare_for_fallback=lambda: SimpleNamespace(can_schedule_seed=True),
    )
    loader = object.__new__(loader_module.RForkModelLoader)
    loader.load_config = SimpleNamespace(device=None)
    loader.rfork_config = SimpleNamespace(post_load_init_sync=False)

    def process_model(model, config, target_device):
        assert model is empty_model
        assert nz_backend.npu_format_cast is not original_cast
        model.destination = nz_backend.npu_format_cast(_nz_weight(), 29)
        assert model.destination.npu_format == 29

    def load_local(*args):
        assert nz_backend.npu_format_cast is original_cast
        reset.assert_called_once_with(vllm_config, empty_model, None)
        assert nz_backend.npu_format_cast(_nz_weight(), 29) is original_cast.return_value
        return local_model

    monkeypatch.setattr(loader, "_ensure_rfork_session", lambda *args: session)
    monkeypatch.setattr(loader, "_get_target_registered_blocks", lambda *args: [])
    monkeypatch.setattr(loader_module, "initialize_model", lambda **kwargs: empty_model)
    monkeypatch.setattr(loader_module, "process_weights_after_loading", process_model)
    monkeypatch.setattr(loader_module, "_snapshot_process_global_model_state", lambda *args: None)
    monkeypatch.setattr(loader_module, "_reset_process_global_model_state", reset)
    monkeypatch.setattr(loader_module, "_load_with_default_loader", load_local)
    monkeypatch.setattr(loader_module, "_start_rfork_seed_service", Mock(return_value=True))
    monkeypatch.setattr(loader_module.gc, "collect", Mock())
    monkeypatch.setattr(
        loader_module,
        "get_ascend_config",
        lambda: SimpleNamespace(
            eplb_config=SimpleNamespace(dynamic_eplb=False, expert_map_record_path=None, expert_map_path=None)
        ),
    )
    monkeypatch.setattr(torch.npu, "synchronize", lambda device=None: None)
    monkeypatch.setattr(torch.npu, "empty_cache", Mock(), raising=False)

    assert loader.load_model(vllm_config, model_config) is local_model
    assert torch.equal(local_model.weight, torch.full((2,), 7.0))
    assert nz_backend.npu_format_cast is original_cast
    assert nz_backend.empty_with_format.call_count == 1 and original_cast.call_count == 1


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
