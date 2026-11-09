# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import ast
import json
import typing
import weakref
from collections.abc import Callable, Iterable
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest
import regex as re
import torch
from safetensors.torch import save_file
from torch.nn import Module
from torch.utils.hooks import RemovableHandle

from .rfork_test_support import RFORK_ROOT, _load_module, _stub


@pytest.fixture(params=["DeepSeekMTPModel", "DeepseekV32MTPModel"])
def mtp_model(request, rfork_helpers, tensor_runtime, monkeypatch):
    class WeightsMapper:
        def __init__(self, *, orig_to_new_prefix):
            self.orig_to_new_prefix = orig_to_new_prefix

        def _map_name(self, name):
            for prefix, replacement in self.orig_to_new_prefix.items():
                if name.startswith(prefix):
                    return replacement + name[len(prefix) :] if replacement is not None else None
            return name

        def apply_list(self, names):
            return [mapped for name in names if (mapped := self._map_name(name)) is not None]

        def apply(self, weights):
            return ((mapped, tensor) for name, tensor in weights if (mapped := self._map_name(name)) is not None)

    class DeepSeekMTP(torch.nn.Module):
        def load_weights(self, weights):
            return {name for name, _ in weights}

    _stub(monkeypatch, "vllm.compilation.decorators", support_torch_compile=lambda cls: cls)
    _stub(monkeypatch, "vllm.config", VllmConfig=object)
    _stub(monkeypatch, "vllm.model_executor.models.deepseek_mtp", DeepSeekMTP=DeepSeekMTP)
    _stub(monkeypatch, "vllm.model_executor.models.deepseek_v2", GlmMoeDsaForCausalLM=torch.nn.Module)
    _stub(monkeypatch, "vllm.model_executor.models.utils", AutoWeightsLoader=object, WeightsMapper=WeightsMapper)
    _stub(monkeypatch, "vllm.sequence", IntermediateTensors=object)
    _stub(monkeypatch, "vllm_ascend.utils", is_rot_weight_used=lambda _config: False)
    module = _load_module(monkeypatch, "_rfork_test_mtp_model", "../../models/deepseek_mtp.py")
    model = module.AscendDeepSeekMTP.__new__(module.AscendDeepSeekMTP)
    torch.nn.Module.__init__(model)
    model.config = SimpleNamespace(num_hidden_layers=3)
    model.quant_config = None
    model.model = torch.nn.Module()
    model.model.mtp_start_layer_idx = 3
    layer = torch.nn.Module()
    layer.shared_head = torch.nn.Module()
    layer.shared_head.head = torch.nn.Linear(2, 2, bias=False)
    model.model.layers = torch.nn.ModuleDict({"3": layer})
    monkeypatch.setattr(tensor_runtime.tensor_layout, "is_tensor_on_transfer_device", lambda _tensor: True)
    return SimpleNamespace(model=model, mapper=WeightsMapper, architecture=request.param)


@pytest.mark.parametrize(
    "head_name, has_own_head",
    [
        ("model.layers.3.shared_head.head.weight", True),
        ("lm_head.weight", False),
        ("layers.3.shared_head.head.weight", False),
        ("model.language_model.layers.3.shared_head.head.weight", False),
        (None, False),
    ],
)
@pytest.mark.parametrize("processed_layout", [False, True])
def test_mtp_checkpoint_preparation_preserves_registered_structure(
    rfork_helpers, tensor_runtime, mtp_model, monkeypatch, head_name, has_own_head, processed_layout
):
    model = mtp_model.model
    layout = tensor_runtime.tensor_layout
    head = model.model.layers["3"].shared_head.head
    original = head.weight.detach().clone()
    assert not hasattr(model, "lm_head")
    names = {head_name} if head_name is not None else set()
    unprepared_digest = layout.build_structural_digest(layout.collect_transferable_tensors(model, processed_layout))

    monkeypatch.setattr(rfork_helpers.loader, "_get_checkpoint_weight_names", lambda *args: names)
    rfork_helpers.loader._prepare_rfork_model_state(model, SimpleNamespace(architectures=[mtp_model.architecture]))
    before = layout.collect_transferable_tensors(model, processed_layout)
    digest = layout.build_structural_digest(before)
    loaded = model.load_weights((name, original) for name in names)
    after = layout.collect_transferable_tensors(model, processed_layout)

    assert loaded == names
    assert model.has_own_lm_head is has_own_head
    assert hasattr(model, "lm_head") is has_own_head
    if has_own_head:
        assert model.lm_head is head
    expected = "lm_head.weight" if has_own_head else "model.layers.3.shared_head.head.weight"
    assert [name for name, _ in before] == [expected]
    assert [name for name, _ in after] == [expected]
    assert (digest != unprepared_digest) is has_own_head
    assert layout.build_structural_digest(after) == digest
    torch.testing.assert_close(head.weight, original)


def test_mtp_checkpoint_preparation_ignores_nonlocal_head(rfork_helpers, mtp_model, monkeypatch):
    model = mtp_model.model
    model.model.layers = torch.nn.ModuleDict()
    monkeypatch.setattr(
        rfork_helpers.loader,
        "_get_checkpoint_weight_names",
        lambda *args: {"model.layers.3.shared_head.head.weight"},
    )

    rfork_helpers.loader._prepare_rfork_model_state(model, SimpleNamespace(architectures=[mtp_model.architecture]))

    assert model.has_own_lm_head is False
    assert not hasattr(model, "lm_head")


def test_mtp_preparation_applies_same_checkpoint_mappers_as_loading(rfork_helpers, mtp_model, monkeypatch):
    model = mtp_model.model
    mapper = mtp_model.mapper(
        orig_to_new_prefix={"checkpoint_head.": "model.layers.3.shared_head.head.", "checkpoint_rot.": "rot."}
    )
    model.quant_config = SimpleNamespace(get_cache_scale_mapper=lambda: mapper)
    raw_names = {"checkpoint_head.weight", "checkpoint_rot.weight"}

    monkeypatch.setattr(rfork_helpers.loader, "_get_checkpoint_weight_names", lambda *args: raw_names)
    rfork_helpers.loader._prepare_rfork_model_state(model, SimpleNamespace(architectures=[mtp_model.architecture]))
    assert model.has_own_lm_head is True
    assert model.lm_head is model.model.layers["3"].shared_head.head
    loaded = model.load_weights((name, torch.ones(2)) for name in raw_names)

    assert loaded == {"model.layers.3.shared_head.head.weight", "model.layers.3.rot.weight"}
    assert model.has_own_lm_head is True
    model.load_weights([])
    assert model.has_own_lm_head is False


@pytest.fixture(params=["deepseek_v4", "deepseek_v41"])
def dspark_model(request, rfork_helpers, monkeypatch):
    """Use the real model loading methods with small CPU parameters."""
    classes = {}

    def process_eagle_weight(model, name):
        if "embed_tokens" in name:
            model.has_own_embed_tokens = True
        if "lm_head" in name:
            model.has_own_lm_head = True

    _stub(monkeypatch, "vllm.model_executor.models", __path__=[])
    _stub(monkeypatch, "vllm.model_executor.models.utils", process_eagle_weight=process_eagle_weight)
    for family, class_name in (
        ("deepseek_v4", "DSparkDeepseekV4ForCausalLM"),
        ("deepseek_v41", "DSparkDeepseekV41ForCausalLM"),
    ):
        path = RFORK_ROOT.parents[1] / "models" / family / "dspark.py"
        tree = ast.parse(path.read_text(encoding="utf-8"))
        source_class = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == class_name)
        methods = [
            node
            for node in source_class.body
            if isinstance(node, ast.FunctionDef)
            and node.name in {"load_weights", "_remap_dspark_name", "_remap_checkpoint_name"}
        ]

        namespace = {
            "Iterable": Iterable,
            "typing": typing,
            "torch": torch,
            "re": re,
            "process_eagle_weight": process_eagle_weight,
            "get_tensor_model_parallel_rank": lambda: 0,
            "get_tensor_model_parallel_world_size": lambda: 1,
            "default_weight_loader": lambda param, weight: param.data.copy_(weight),
            "logger": SimpleNamespace(info_once=lambda *args: None),
        }
        exec(compile(ast.Module(body=methods, type_ignores=[]), str(path), "exec"), namespace)

        class DraftModel(torch.nn.Module):
            has_own_embed_tokens = False
            has_own_lm_head = False

            def __init__(self, rotation_path, layers):
                super().__init__()
                self.rotation_path = rotation_path
                self.config = SimpleNamespace(num_attention_heads=1, num_hidden_layers=3)
                self.model = torch.nn.Module()
                self.model.num_dspark_layers = layers
                self.model.get_expert_mapping = lambda: []
                self.model.embed_tokens = torch.nn.Embedding(2, 2)
                self.lm_head = torch.nn.Linear(2, 2, bias=False)

        for method in methods:
            setattr(DraftModel, method.name, namespace[method.name])
        classes[family] = DraftModel
    return request.param, classes[request.param]


@pytest.mark.parametrize(
    ("names", "rotation", "layers", "expected"),
    [
        ({"mtp.0.embed.weight", "mtp.2.head.weight"}, "rotation", 3, (True, True)),
        ({"mtp.0.embed.weight"}, "rotation", 3, (True, False)),
        ({"mtp.2.head.weight"}, "rotation", 3, (False, True)),
        ({"unrelated.weight"}, "rotation", 3, (False, False)),
        ({"embed.weight", "head.weight"}, None, 3, (True, True)),
        ({"embed.weight", "head.weight"}, "rotation", 3, (False, False)),
        ({"mtp.0.embed.weight", "mtp.3.head.weight"}, "rotation", 4, (True, True)),
    ],
)
def test_dspark_checkpoint_preparation_matches_local_loading(
    rfork_helpers, dspark_model, monkeypatch, names, rotation, layers, expected
):
    family, model_class = dspark_model
    local = model_class(rotation, layers)
    receiver = model_class(rotation, layers)
    before = {name: param.detach().clone() for name, param in receiver.named_parameters()}
    local.load_weights((name, torch.ones(2, 2)) for name in names)
    monkeypatch.setattr(rfork_helpers.loader, "_get_checkpoint_weight_names", lambda *_args: names)

    architecture = "DSparkDraftModel" if family == "deepseek_v4" else "DeepseekV41DSparkModel"
    rfork_helpers.loader._prepare_rfork_model_state(receiver, SimpleNamespace(architectures=[architecture]))

    assert (local.has_own_embed_tokens, local.has_own_lm_head) == expected
    assert (receiver.has_own_embed_tokens, receiver.has_own_lm_head) == expected
    for name, param in receiver.named_parameters():
        torch.testing.assert_close(param, before[name])


def test_dspark_preparation_uses_quantized_index_without_reading_shards(
    rfork_helpers, dspark_model, checkpoint_files, monkeypatch, tmp_path
):
    family, model_class = dspark_model
    receiver = model_class("rotation", 3)
    weights = {name: "weights.safetensors" for name in ("mtp.0.embed.weight", "mtp.2.head.weight")}
    (tmp_path / "quant_model_weights.safetensors.index.json").write_text(
        json.dumps({"weight_map": weights}), encoding="utf-8"
    )

    def fail_open(*args, **kwargs):
        raise AssertionError("checkpoint preparation must not read tensor shards when an index is available")

    monkeypatch.setattr("safetensors.safe_open", fail_open)
    architecture = "DSparkDraftModel" if family == "deepseek_v4" else "DeepseekV41DSparkModel"
    config = SimpleNamespace(model=str(tmp_path), revision=None, architectures=[architecture])
    rfork_helpers.loader._prepare_rfork_model_state(receiver, config)

    assert receiver.has_own_embed_tokens
    assert receiver.has_own_lm_head


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
def checkpoint_files(monkeypatch, rfork_helpers):
    def get_model_file(model, filename, revision=None):
        path = Path(model) / filename
        return path if path.is_file() else None

    _stub(monkeypatch, "vllm_ascend.quantization", __path__=[])
    _stub(monkeypatch, "vllm.transformers_utils.repo_utils", list_filtered_repo_files=lambda *args, **kwargs: [])
    return _stub(monkeypatch, "vllm_ascend.quantization.utils", get_model_file=get_model_file)


@pytest.mark.parametrize(
    "index_file",
    [
        "model.safetensors.index.json",
        "quant_model_weights.safetensors.index.json",
        "consolidated.safetensors.index.json",
        "custom.safetensors.index.json",
    ],
)
def test_checkpoint_names_use_index_without_opening_shards(
    rfork_helpers, checkpoint_files, monkeypatch, tmp_path, index_file
):
    weight_map = {
        "model.layers.0.weight": "shard-a.safetensors",
        "model.layers.1.weight": "shard-b.safetensors",
    }
    (tmp_path / index_file).write_text(json.dumps({"weight_map": weight_map}), encoding="utf-8")

    def fail_open(*args, **kwargs):
        raise AssertionError("indexed checkpoint must not open every shard")

    monkeypatch.setattr("safetensors.safe_open", fail_open)
    names = rfork_helpers.loader._get_checkpoint_weight_names(SimpleNamespace(model=str(tmp_path), revision=None))
    assert names == set(weight_map)


def test_checkpoint_names_read_safetensors_headers_without_index(rfork_helpers, checkpoint_files, tmp_path):
    save_file({"model.layers.0.weight": torch.ones(2)}, tmp_path / "shard-a.safetensors")
    save_file({"model.layers.1.weight": torch.ones(2)}, tmp_path / "shard-b.safetensors")

    names = rfork_helpers.loader._get_checkpoint_weight_names(SimpleNamespace(model=str(tmp_path), revision=None))

    assert names == {"model.layers.0.weight", "model.layers.1.weight"}


def test_checkpoint_names_find_remote_index_with_revision(rfork_helpers, checkpoint_files, monkeypatch, tmp_path):
    index = tmp_path / "custom.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {"draft.weight": "draft.safetensors"}}), encoding="utf-8")
    calls = []

    def get_model_file(model, filename, revision=None):
        calls.append((model, filename, revision))
        return index if filename == index.name else None

    def list_files(model, patterns, revision=None):
        assert (model, patterns, revision) == ("org/model", ["*.safetensors.index.json"], "v1")
        return ["nested/ignored.safetensors.index.json", index.name]

    _stub(monkeypatch, "vllm.transformers_utils.repo_utils", list_filtered_repo_files=list_files)
    checkpoint_files.get_model_file = get_model_file
    names = rfork_helpers.loader._get_checkpoint_weight_names(SimpleNamespace(model="org/model", revision="v1"))

    assert names == {"draft.weight"}
    assert all(model == "org/model" and revision == "v1" for model, _, revision in calls)


def test_checkpoint_names_use_first_index_in_sorted_order(rfork_helpers, checkpoint_files, tmp_path):
    for name, key in (("z.safetensors.index.json", "later.weight"), ("a.safetensors.index.json", "first.weight")):
        (tmp_path / name).write_text(json.dumps({"weight_map": {key: "weights.safetensors"}}), encoding="utf-8")

    names = rfork_helpers.loader._get_checkpoint_weight_names(SimpleNamespace(model=str(tmp_path), revision=None))

    assert names == {"first.weight"}


def test_mtp_preparation_uses_quantized_index(rfork_helpers, mtp_model, checkpoint_files, tmp_path):
    head_name = "model.layers.3.shared_head.head.weight"
    (tmp_path / "quant_model_weights.safetensors.index.json").write_text(
        json.dumps({"weight_map": {head_name: "quant_model_weights-00001.safetensors"}}), encoding="utf-8"
    )

    model = mtp_model.model
    config = SimpleNamespace(model=str(tmp_path), revision=None, architectures=[mtp_model.architecture])
    rfork_helpers.loader._prepare_rfork_model_state(model, config)

    assert model.has_own_lm_head is True
    assert model.lm_head is model.model.layers["3"].shared_head.head


@pytest.mark.parametrize(
    ("metadata", "expected_error"),
    [
        ("corrupt", "could not read checkpoint metadata"),
        ("empty", "received empty checkpoint metadata"),
        ("missing", "received empty checkpoint metadata"),
    ],
)
def test_checkpoint_names_reject_unavailable_metadata(
    rfork_helpers, checkpoint_files, tmp_path, metadata, expected_error
):
    path = tmp_path / "model.safetensors"
    if metadata == "corrupt":
        path.write_bytes(b"invalid safetensors")
    elif metadata == "empty":
        save_file({}, path)

    with pytest.raises((OSError, ValueError), match=expected_error):
        rfork_helpers.loader._get_checkpoint_weight_names(SimpleNamespace(model=str(tmp_path), revision=None))


@pytest.mark.parametrize("processed_layout", [False, True])
@pytest.mark.parametrize(
    "outcome", ["transfer", "shared", "local", "transfer_error", "metadata_error", "state_error", "other_model"]
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
    model_ref: Callable[[], Module | None] = lambda: None

    class Model(Module):
        quant_config = None

        def __init__(self):
            super().__init__()
            self.head = torch.nn.Parameter(torch.ones(1))

        def named_parameters(self):
            return ((name, self.head) for name in names)

        def _maybe_set_own_lm_head(self, weight_names):
            assert weight_names == names
            events.append("prepare")
            if outcome == "state_error":
                raise ValueError("head state unavailable")
            self.has_own_lm_head = True

    class OtherModel(Module):
        def _maybe_set_own_lm_head(self, weight_names):
            raise AssertionError("RFork must not prepare unrelated model types")

    def initialize(**kwargs):
        nonlocal model_ref
        model = OtherModel() if outcome == "other_model" else Model()
        model_ref = weakref.ref(model)
        events.append("initialize")
        return model

    def default_load(*args):
        # Fallback must release the discarded model before allocating its replacement.
        assert model_ref() is None
        return record("default", fallback_model)

    def metadata(*args):
        assert outcome != "other_model"
        events.append("metadata")
        if outcome == "metadata_error":
            raise ValueError("checkpoint metadata unavailable")
        return names

    def record(event, result=None):
        events.append(event)
        if event in ("shared", "process", "register") and outcome != "other_model":
            live_model = model_ref()
            assert isinstance(live_model, Model)
            assert live_model.has_own_lm_head is True
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
    architecture = "UnrelatedModel" if outcome == "other_model" else "DeepSeekMTPModel"
    model_config = SimpleNamespace(dtype=torch.float32, architectures=[architecture])
    vllm_config = SimpleNamespace(device_config=SimpleNamespace(device="cpu"), model_config=model_config)

    result = loader.load_model(vllm_config, model_config)

    prefix = ["initialize"] + ([] if outcome == "other_model" else ["metadata"])
    if outcome not in ("other_model", "metadata_error"):
        prefix.append("prepare")
    if outcome in ("metadata_error", "state_error"):
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
