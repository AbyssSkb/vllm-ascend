# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import ast
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch


@pytest.fixture
def alignment(tmp_path):
    source = Path(__file__).resolve().parents[4] / "vllm_ascend/models/qwen3_dspark.py"
    tree = ast.parse(source.read_text())
    function = next(
        node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "align_draft_weights"
    )
    function.decorator_list = []
    bootstrap = tmp_path / "bootstrap"
    mounted = tmp_path / "mounted"
    bootstrap.mkdir()
    mounted.mkdir()
    events = []
    target_config = SimpleNamespace(model=str(bootstrap), hf_text_config=SimpleNamespace(vocab_size=2, hidden_size=2))
    config = SimpleNamespace(
        model_config=target_config, load_config=SimpleNamespace(load_format="rfork"), rotated=False
    )
    mounted_target = SimpleNamespace(model=str(mounted), hf_text_config=target_config.hf_text_config)
    mounted_config = SimpleNamespace(model_config=mounted_target, load_config=config.load_config, rotated=False)
    manager = SimpleNamespace(enabled=Mock(return_value=True), mounted_configs=Mock())

    def resolve(config_arg, target_arg):
        events.append("manager")
        assert target_arg is target_config
        mounted_config.rotated = config_arg.rotated
        return mounted_config, mounted_target

    manager.mounted_configs.side_effect = resolve
    pp = SimpleNamespace(world_size=1)

    def layer(*args, **kwargs):
        return SimpleNamespace(quant_method=SimpleNamespace(process_weights_after_loading=Mock()))

    def read_rotation(path):
        events.append(("rotation", str(path)))
        return torch.eye(2)

    def read_weight(layer, path, names, rotation, name):
        events.append(("weight", path, name))

    namespace = {
        "torch": torch,
        "managed_weights": manager,
        "get_pp_group": lambda: pp,
        "get_rotation_path": lambda cfg: Path(cfg.model_config.model) / "rotation.safetensors" if cfg.rotated else None,
        "get_rotation_matrix": read_rotation,
        "process_weight": lambda weight, rotation: weight,
        "VocabParallelEmbedding": layer,
        "ParallelLMHead": layer,
        "TARGET_EMBED_WEIGHT_NAMES": ("embed_tokens.weight",),
        "TARGET_LM_HEAD_WEIGHT_NAMES": ("lm_head.weight",),
        "load_quarot_target_layer": read_weight,
    }
    exec(compile(ast.Module(body=[function], type_ignores=[]), str(source), "exec"), namespace)
    model = SimpleNamespace(model=SimpleNamespace(), has_own_embed_tokens=False, has_own_lm_head=False)
    projection = SimpleNamespace(weight=torch.eye(2))
    return SimpleNamespace(
        run=lambda: namespace["align_draft_weights"](model, projection, config),
        model=model,
        manager=manager,
        config=config,
        pp=pp,
        events=events,
        bootstrap=bootstrap,
        mounted=mounted,
    )


@pytest.mark.parametrize("own_embedding,pp_size", [(False, 1), (True, 2)])
def test_no_rotation_without_local_embedding_read_does_not_wait(alignment, own_embedding, pp_size):
    alignment.model.has_own_embed_tokens = own_embedding
    alignment.pp.world_size = pp_size
    alignment.run()
    alignment.manager.mounted_configs.assert_not_called()
    assert alignment.events == []


def test_pipeline_embedding_waits_just_before_loading_target_shard(alignment):
    alignment.pp.world_size = 2
    alignment.run()
    assert alignment.events == ["manager", ("weight", str(alignment.mounted), "draft embed_tokens.weight")]
    assert alignment.model.has_own_embed_tokens
    assert not alignment.model.has_own_lm_head
    alignment.manager.mounted_configs.assert_called_once()


def test_rotation_in_bootstrap_needs_no_manager_when_draft_owns_weights(alignment):
    alignment.config.rotated = True
    alignment.model.has_own_embed_tokens = alignment.model.has_own_lm_head = True
    (alignment.bootstrap / "rotation.safetensors").touch()
    alignment.run()
    alignment.manager.mounted_configs.assert_not_called()
    assert alignment.events == [("rotation", str(alignment.bootstrap / "rotation.safetensors"))]


def test_rotation_in_bootstrap_delays_manager_until_target_weights(alignment):
    alignment.config.rotated = True
    (alignment.bootstrap / "rotation.safetensors").touch()
    alignment.run()
    assert alignment.events == [
        ("rotation", str(alignment.bootstrap / "rotation.safetensors")),
        "manager",
        ("weight", str(alignment.mounted), "draft embed_tokens.weight"),
        ("weight", str(alignment.mounted), "draft lm_head.weight"),
    ]
    alignment.manager.mounted_configs.assert_called_once()


def test_missing_rotation_waits_once_and_reuses_target_directory(alignment):
    alignment.config.rotated = True
    alignment.run()
    assert alignment.events == [
        "manager",
        ("rotation", str(alignment.mounted / "rotation.safetensors")),
        ("weight", str(alignment.mounted), "draft embed_tokens.weight"),
        ("weight", str(alignment.mounted), "draft lm_head.weight"),
    ]
    alignment.manager.mounted_configs.assert_called_once()


def test_manager_error_stops_before_checkpoint_read(alignment):
    alignment.pp.world_size = 2
    alignment.manager.mounted_configs.side_effect = RuntimeError("manager failed")
    with pytest.raises(RuntimeError, match="manager failed"):
        alignment.run()
    assert alignment.events == []


@pytest.mark.parametrize("mode", ["disabled", "ordinary"])
def test_unmanaged_checkpoint_read_retains_original_path(alignment, mode):
    alignment.pp.world_size = 2
    if mode == "disabled":
        alignment.manager.enabled.return_value = False
    else:
        alignment.config.load_config.load_format = "auto"
    alignment.run()
    alignment.manager.mounted_configs.assert_not_called()
    assert alignment.events == [("weight", str(alignment.bootstrap), "draft embed_tokens.weight")]
