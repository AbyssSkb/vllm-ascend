# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from .rfork_test_support import _stub


@pytest.fixture
def process_eagle(monkeypatch, rfork_helpers):
    def process_eagle_weight(model, name):
        if "embed_tokens" in name:
            model.has_own_embed_tokens = True
        if "lm_head" in name:
            model.has_own_lm_head = True

    _stub(monkeypatch, "vllm.model_executor.models", __path__=[])
    _stub(monkeypatch, "vllm.model_executor.models.utils", process_eagle_weight=process_eagle_weight)


def make_dspark(architecture, *, vocab_size=16, draft_vocab_size=16, target_vocab_size=16):
    model = nn.Module()
    model.model = nn.Module()
    model.model.confidence_head = nn.Linear(2, 2)
    model.config = SimpleNamespace(vocab_size=vocab_size, draft_vocab_size=draft_vocab_size)
    model.target_vocab_size = target_vocab_size
    model.draft_id_to_target_id = (
        nn.Parameter(torch.zeros(draft_vocab_size, dtype=torch.long), requires_grad=False)
        if draft_vocab_size < target_vocab_size
        else None
    )
    model.has_own_embed_tokens = model.has_own_lm_head = architecture == "Glm5DSparkForCausalLM"
    model.enable_confidence_head = True
    config = SimpleNamespace(architectures=[architecture], hf_config=SimpleNamespace(architectures=[architecture]))
    return model, config


@pytest.mark.parametrize("architecture", ["Qwen3DSparkModel", "Qwen3OmniDSparkModel"])
@pytest.mark.parametrize(
    ("names", "own_embedding", "own_head", "confidence"),
    [
        ({"embed_tokens.weight", "lm_head.weight", "confidence_head.weight"}, True, True, True),
        ({"layer.weight"}, False, False, False),
        ({"embed_tokens.weight", "t2d_lm_head.weight"}, True, False, False),
    ],
)
def test_qwen_dspark_restores_checkpoint_state(
    rfork_helpers, process_eagle, monkeypatch, architecture, names, own_embedding, own_head, confidence
):
    model, config = make_dspark(architecture)
    checkpoint_head = model.model.confidence_head
    monkeypatch.setattr(rfork_helpers.loader, "_get_checkpoint_weight_names", lambda *_: names)

    rfork_helpers.loader._prepare_rfork_model_state(model, config)

    assert model.has_own_embed_tokens is own_embedding
    assert model.has_own_lm_head is own_head
    assert model.model.confidence_head is (checkpoint_head if confidence else None)


@pytest.mark.parametrize(
    ("kwargs", "names", "message"),
    [
        ({"vocab_size": 20}, {"lm_head.weight"}, "input vocab_size"),
        ({"draft_vocab_size": 8}, {"d2t"}, "lm_head"),
        ({"draft_vocab_size": 8}, {"lm_head.weight"}, "d2t"),
    ],
)
def test_qwen_dspark_validates_vocab_checkpoint(rfork_helpers, process_eagle, monkeypatch, kwargs, names, message):
    model, config = make_dspark("Qwen3DSparkModel", **kwargs)
    monkeypatch.setattr(rfork_helpers.loader, "_get_checkpoint_weight_names", lambda *_: names)

    with pytest.raises(ValueError, match=message):
        rfork_helpers.loader._prepare_rfork_model_state(model, config)


@pytest.mark.parametrize(
    ("names", "own_embedding", "own_head", "confidence"),
    [
        ({"embed_tokens.weight", "lm_head.weight", "confidence_head.weight"}, True, True, True),
        ({"layer.weight"}, False, False, False),
        ({"embed_tokens.weight", "lm_head.bias"}, True, False, False),
    ],
)
def test_glm_dspark_restores_checkpoint_state(rfork_helpers, monkeypatch, names, own_embedding, own_head, confidence):
    model, config = make_dspark("Glm5DSparkForCausalLM")
    checkpoint_head = model.model.confidence_head
    monkeypatch.setattr(rfork_helpers.loader, "_get_checkpoint_weight_names", lambda *_: names)

    rfork_helpers.loader._prepare_rfork_model_state(model, config)

    assert model.has_own_embed_tokens is own_embedding
    assert model.has_own_lm_head is own_head
    assert model.enable_confidence_head is confidence
    assert model.model.confidence_head is checkpoint_head


@pytest.mark.parametrize(
    ("kwargs", "names", "message"),
    [
        ({"vocab_size": 20}, {"lm_head.weight"}, "embedding"),
        ({"draft_vocab_size": 8}, {"d2t"}, "d2t and draft lm_head"),
        ({"draft_vocab_size": 8}, {"lm_head.weight"}, "d2t and draft lm_head"),
    ],
)
def test_glm_dspark_validates_vocab_checkpoint(rfork_helpers, monkeypatch, kwargs, names, message):
    model, config = make_dspark("Glm5DSparkForCausalLM", **kwargs)
    monkeypatch.setattr(rfork_helpers.loader, "_get_checkpoint_weight_names", lambda *_: names)

    with pytest.raises(ValueError, match=message):
        rfork_helpers.loader._prepare_rfork_model_state(model, config)


@pytest.mark.parametrize(
    ("architecture", "mapping"),
    [
        ("Qwen3DSparkModel", "d2t"),
        ("Glm5DSparkForCausalLM", "d2t"),
        ("Glm5DSparkForCausalLM", "draft_id_to_target_id"),
    ],
)
def test_dspark_reduced_vocab_keeps_own_head(rfork_helpers, process_eagle, monkeypatch, architecture, mapping):
    model, config = make_dspark(architecture, draft_vocab_size=8)
    monkeypatch.setattr(rfork_helpers.loader, "_get_checkpoint_weight_names", lambda *_: {"lm_head.weight", mapping})

    rfork_helpers.loader._prepare_rfork_model_state(model, config)

    assert model.has_own_lm_head
    assert model.draft_id_to_target_id is not None


def test_kimi_dspark_does_not_read_checkpoint_metadata(rfork_helpers, monkeypatch):
    model, config = make_dspark("K3DSparkModel")
    model.has_own_embed_tokens = True  # PP > 1 owns its embedding by construction.
    monkeypatch.setattr(
        rfork_helpers.loader,
        "_get_checkpoint_weight_names",
        lambda *_: pytest.fail("K3 owns or shares weights based on construction, not checkpoint names"),
    )

    rfork_helpers.loader._prepare_rfork_model_state(model, config)

    assert model.has_own_embed_tokens


@pytest.mark.parametrize(
    "architecture",
    ["Qwen3DSparkModel", "Qwen3OmniDSparkModel", "Glm5DSparkForCausalLM", "K3DSparkModel"],
)
@pytest.mark.parametrize("rotation", [None, "rotation.safetensors"])
def test_dspark_rotation_bypasses_transfer(rfork_helpers, monkeypatch, architecture, rotation):
    _stub(monkeypatch, "vllm_ascend.utils", get_rotation_path=lambda _config: rotation)
    model_config = SimpleNamespace(architectures=[architecture])
    vllm_config = SimpleNamespace(model_config=model_config, weight_transfer_config=None)

    reason = rfork_helpers.safety.mutable_weights_bypass_reason(vllm_config, model_config)

    assert reason == ("DSpark draft weight rotation" if rotation else None)


def test_deepseek_dspark_does_not_bypass_for_rotation(rfork_helpers):
    model_config = SimpleNamespace(architectures=["DeepseekV41DSparkModel"])
    vllm_config = SimpleNamespace(model_config=model_config, weight_transfer_config=None)

    assert rfork_helpers.safety.mutable_weights_bypass_reason(vllm_config, model_config) is None


def test_rotated_dspark_uses_default_loader_before_creating_session(rfork_helpers, monkeypatch):
    _stub(monkeypatch, "vllm_ascend.utils", get_rotation_path=lambda _config: "rotation.safetensors")
    runtime = rfork_helpers.loader
    loader = runtime.RForkModelLoader.__new__(runtime.RForkModelLoader)
    loader.load_config = SimpleNamespace(device=None)
    model_config = SimpleNamespace(architectures=["Qwen3DSparkModel"], dtype=torch.float32, quantization=None)
    vllm_config = SimpleNamespace(
        model_config=model_config, device_config=SimpleNamespace(device="cpu"), weight_transfer_config=None
    )
    expected_model = nn.Module()
    monkeypatch.setattr(runtime, "_load_with_default_loader", lambda *_args: expected_model)
    monkeypatch.setattr(loader, "_ensure_rfork_session", lambda *_args: pytest.fail("RFork session created"))

    assert loader.load_model(vllm_config, model_config) is expected_model
