# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import logging
import sys
import weakref
from contextlib import nullcontext
from dataclasses import dataclass
from types import SimpleNamespace

import pytest
import torch
from safetensors.torch import save_file
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
    install("vllm_ascend.device.hardware_profile", get_current_hardware_profile=lambda: SimpleNamespace())
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
        def __init__(self, quant_method=None):
            super().__init__()
            self._quant_method = quant_method

    install("vllm_ascend.ops.fused_moe.fused_moe", AscendMoERunner=AscendMoERunner)

    class AscendRoutedExperts:
        moe_counter = -1

    routed_experts_module = install("vllm_ascend.ops.fused_moe.routed_experts", AscendRoutedExperts=AscendRoutedExperts)
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
        routed_experts_module=routed_experts_module,
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
    rfork_helpers.routed_experts_module.AscendRoutedExperts.moe_counter = 17

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
    rfork_helpers.routed_experts_module.AscendRoutedExperts.moe_counter = 999
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
    assert rfork_helpers.routed_experts_module.AscendRoutedExperts.moe_counter == 17
    assert (
        rfork_helpers.rotary_module._cos_sin_cache,
        rfork_helpers.rotary_module._cos_cache,
        rfork_helpers.rotary_module._sin_cache,
    ) == baseline_caches
    assert baseline_hook_id in rfork_helpers.dynamo_hooks
    assert rfork_helpers.dynamo_hooks[baseline_hook_id].__self__ is baseline_owner
    assert discarded_hook_id not in rfork_helpers.dynamo_hooks


@pytest.fixture
def checkpoint_loader(monkeypatch, rfork_helpers):
    @dataclass
    class Source:
        model_or_path: str
        revision: str | None
        subfolder: str | None = None
        prefix: str = ""
        fall_back_to_pt: bool = True
        allow_patterns_overrides: list[str] | None = None

    calls = []
    selections = {}
    configs = []

    class DefaultModelLoader:
        def __init__(self, load_config):
            configs.append(load_config)

        def _prepare_weights(self, *args):
            calls.append(args)
            return "unused", *selections[args[0]]

    DefaultModelLoader.Source = Source
    _stub(monkeypatch, "vllm.model_executor.model_loader.default_loader", DefaultModelLoader=DefaultModelLoader)
    return SimpleNamespace(Source=Source, calls=calls, selections=selections, configs=configs)


def test_checkpoint_names_follow_selected_sources_and_prefixes(rfork_helpers, checkpoint_loader, tmp_path):
    primary = tmp_path / "selected.safetensors"
    secondary = tmp_path / "secondary.safetensors"
    save_file({"model.layers.3.shared_head.head.weight": torch.ones(2)}, primary)
    save_file({"head.weight": torch.zeros(2)}, secondary)
    save_file({"unselected.weight": torch.ones(2)}, tmp_path / "stale.safetensors")
    checkpoint_loader.selections.update(primary=([str(primary)], True), secondary=([str(secondary)], True))
    model = Module()
    model.fall_back_to_pt_during_load = False
    model.allow_patterns_overrides = ["selected*.safetensors"]
    model.secondary_weights = [
        checkpoint_loader.Source("secondary", "secondary-revision", "subfolder", "aux.", False, ["secondary*"])
    ]
    load_config = SimpleNamespace(load_format="rfork", model_loader_extra_config={"rfork_scheduler_url": "planner"})

    names = rfork_helpers.loader._get_checkpoint_weight_names(
        model, SimpleNamespace(model="primary", revision="primary-revision"), load_config
    )

    assert names == {"model.layers.3.shared_head.head.weight", "aux.head.weight"}
    assert checkpoint_loader.calls == [
        ("primary", None, "primary-revision", False, ["selected*.safetensors"]),
        ("secondary", "subfolder", "secondary-revision", False, ["secondary*"]),
    ]
    assert checkpoint_loader.configs[0].load_format == "auto"
    assert checkpoint_loader.configs[0].model_loader_extra_config == {}
    assert load_config.load_format == "rfork"
    assert load_config.model_loader_extra_config == {"rfork_scheduler_url": "planner"}


@pytest.mark.parametrize("metadata", ["pt", "corrupt", "empty", "missing"])
def test_checkpoint_names_reject_unavailable_metadata(rfork_helpers, checkpoint_loader, tmp_path, metadata):
    path = tmp_path / "model.safetensors"
    if metadata == "corrupt":
        path.write_bytes(b"invalid safetensors")
    elif metadata == "empty":
        save_file({}, path)
    checkpoint_loader.selections["model"] = ([str(path)], metadata != "pt")
    config = SimpleNamespace(load_format="rfork", model_loader_extra_config={})

    with pytest.raises((OSError, ValueError)):
        rfork_helpers.loader._get_checkpoint_weight_names(
            Module(), SimpleNamespace(model="model", revision=None), config
        )


@pytest.mark.parametrize("processed_layout", [False, True])
@pytest.mark.parametrize(
    "outcome", ["transfer", "shared", "local", "transfer_error", "metadata_error", "hook_error", "no_hook"]
)
def test_model_state_preparation_precedes_registration_and_guards_fallback(
    rfork_helpers, monkeypatch, processed_layout, outcome
):
    module = rfork_helpers.loader
    # Pytest's log capture would retain fallback exceptions and their model-owning tracebacks.
    monkeypatch.setattr(module.logger, "propagate", False)
    events = []
    names = {"model.layers.3.shared_head.head.weight"}
    fallback_model = Module()
    model_ref = lambda: None

    class Model(Module):
        def prepare_for_rfork(self, weight_names):
            assert weight_names == names
            events.append("prepare")
            if outcome == "hook_error":
                raise ValueError("head state unavailable")
            self.has_own_lm_head = True

    def initialize(**kwargs):
        nonlocal model_ref
        model = Module() if outcome == "no_hook" else Model()
        model_ref = weakref.ref(model)
        events.append("initialize")
        return model

    def default_load(*args):
        # Fallback must release the discarded model before allocating its replacement.
        assert model_ref() is None
        return record("default", fallback_model)

    def metadata(*args):
        assert outcome != "no_hook"
        events.append("metadata")
        if outcome == "metadata_error":
            raise ValueError("checkpoint metadata unavailable")
        return names

    def record(event, result=None):
        events.append(event)
        if event in ("shared", "process", "register") and outcome != "no_hook":
            assert model_ref().has_own_lm_head is True
        return result

    session = SimpleNamespace(
        can_reuse_shared_weights=lambda *args: record("shared", outcome == "shared"),
        register_destination=lambda *args: record("register", True),
        acquire_seed=lambda: record("acquire", outcome != "local"),
        transfer_from_seed=lambda *args: record("transfer", outcome != "transfer_error"),
        log_transferred_model_layout=lambda *args: record("digest", "prepared-digest"),
        prepare_for_fallback=lambda: record("cleanup", SimpleNamespace(can_schedule_seed=True)),
    )
    loader = module.RForkModelLoader.__new__(module.RForkModelLoader)
    loader.load_config = SimpleNamespace(device=None)
    monkeypatch.setattr(loader, "_ensure_rfork_session", lambda *args: session)
    monkeypatch.setattr(loader, "_get_target_registered_blocks", lambda *args: [(1, 2)])
    monkeypatch.setattr(loader, "_requires_processed_layout_transfer", lambda *args: processed_layout)
    monkeypatch.setattr(module, "_get_checkpoint_weight_names", metadata)
    monkeypatch.setattr(module, "initialize_model", initialize)
    monkeypatch.setattr(module, "process_weights_after_loading", lambda *args: record("process"))
    monkeypatch.setattr(module, "_rfork_skip_unquantized_moe_post_load_processing", lambda *args: nullcontext())
    monkeypatch.setattr(module, "_start_rfork_seed_service", lambda *args, **kwargs: record("seed", True))
    monkeypatch.setattr(module, "_load_with_default_loader", default_load)
    monkeypatch.setattr(
        torch, "npu", SimpleNamespace(synchronize=lambda: None, empty_cache=lambda: None), raising=False
    )
    monkeypatch.setattr(
        module,
        "get_ascend_config",
        lambda: SimpleNamespace(eplb_config=SimpleNamespace(dynamic_eplb=False, expert_map_record_path=None)),
    )
    model_config = SimpleNamespace(dtype=torch.float32)
    vllm_config = SimpleNamespace(device_config=SimpleNamespace(device="cpu"), model_config=model_config)

    result = loader.load_model(vllm_config, model_config)

    prefix = ["initialize"] + ([] if outcome == "no_hook" else ["metadata"])
    if outcome not in ("no_hook", "metadata_error"):
        prefix.append("prepare")
    if outcome in ("metadata_error", "hook_error"):
        assert result is fallback_model
        assert events == prefix + ["cleanup", "default"]
    elif outcome == "shared":
        assert result is model_ref()
        assert events == prefix + ["shared"]
    else:
        assert result is (fallback_model if outcome in ("local", "transfer_error") else model_ref())
        expected = prefix + ["shared"] + (["process"] if processed_layout else []) + ["register", "acquire"]
        if outcome in ("local", "transfer_error"):
            expected += ([] if outcome == "local" else ["transfer"]) + ["cleanup", "default", "seed"]
        else:
            expected += ["transfer"] + ([] if processed_layout else ["process"]) + ["digest", "seed"]
        assert events == expected
