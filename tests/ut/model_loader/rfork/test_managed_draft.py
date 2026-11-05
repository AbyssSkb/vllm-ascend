# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import ast
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from .rfork_test_support import _stub


@pytest.fixture
def draft_load(monkeypatch):
    path = Path(__file__).resolve().parents[4] / "vllm_ascend/spec_decode/llm_base_proposer.py"
    tree = ast.parse(path.read_text())
    cls = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AscendSpecDecodeBaseProposer"
    )
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_get_model")
    manager = SimpleNamespace(enabled=Mock(return_value=True), mounted_configs=Mock())
    get_model = Mock(return_value=object())
    namespace = {
        "nn": SimpleNamespace(Module=object),
        "managed_weights": manager,
        "get_model": get_model,
        "logger": Mock(),
    }
    _stub(monkeypatch, "vllm.compilation.backends", set_model_tag=lambda name: nullcontext())
    exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
    draft = SimpleNamespace(model="bootstrap")
    draft_config = SimpleNamespace(load_format="auto")
    config = SimpleNamespace(
        load_config=SimpleNamespace(load_format="rfork"),
        parallel_config=SimpleNamespace(world_size=1, data_parallel_size=1),
    )
    draft_vllm_config = object()
    proposer = SimpleNamespace(
        vllm_config=config,
        speculative_config=SimpleNamespace(draft_model_config=draft, draft_load_config=draft_config),
        _create_draft_vllm_config=lambda: draft_vllm_config,
        _maybe_remove_d2t=Mock(),
        method="mtp",
    )
    return SimpleNamespace(
        load=namespace["_get_model"],
        proposer=proposer,
        manager=manager,
        get_model=get_model,
        draft=draft,
        draft_vllm_config=draft_vllm_config,
    )


def test_managed_ordinary_draft_loads_mounted_config(draft_load):
    mounted_config, mounted_draft = object(), object()
    draft_load.manager.mounted_configs.return_value = mounted_config, mounted_draft
    result = draft_load.load(draft_load.proposer)
    draft_load.manager.mounted_configs.assert_called_once_with(draft_load.draft_vllm_config, draft_load.draft)
    draft_load.get_model.assert_called_once_with(
        vllm_config=mounted_config,
        model_config=mounted_draft,
        load_config=draft_load.proposer.speculative_config.draft_load_config,
    )
    assert result is draft_load.get_model.return_value


@pytest.mark.parametrize("world_size,dp_size", [(2, 1), (1, 2)])
def test_managed_ordinary_draft_allows_multiple_processes(draft_load, world_size, dp_size):
    parallel = draft_load.proposer.vllm_config.parallel_config
    parallel.world_size, parallel.data_parallel_size = world_size, dp_size
    mounted_config, mounted_draft = object(), object()
    draft_load.manager.mounted_configs.return_value = mounted_config, mounted_draft
    draft_load.load(draft_load.proposer)
    draft_load.manager.mounted_configs.assert_called_once_with(draft_load.draft_vllm_config, draft_load.draft)
    assert draft_load.get_model.call_args.kwargs["vllm_config"] is mounted_config
    assert draft_load.get_model.call_args.kwargs["model_config"] is mounted_draft


def test_managed_ordinary_draft_propagates_manager_failure(draft_load):
    draft_load.manager.mounted_configs.side_effect = RuntimeError("manager failed")
    with pytest.raises(RuntimeError, match="manager failed"):
        draft_load.load(draft_load.proposer)
    draft_load.get_model.assert_not_called()


@pytest.mark.parametrize("case", ["disabled", "ordinary_target", "rfork_draft", "implicit_draft"])
def test_other_draft_loading_does_not_call_manager(draft_load, case):
    proposer = draft_load.proposer
    if case == "disabled":
        draft_load.manager.enabled.return_value = False
    elif case == "ordinary_target":
        proposer.vllm_config.load_config.load_format = "auto"
    elif case == "rfork_draft":
        proposer.speculative_config.draft_load_config.load_format = "rfork"
    else:
        proposer.speculative_config.draft_load_config = None
    draft_load.load(proposer)
    draft_load.manager.mounted_configs.assert_not_called()
    assert draft_load.get_model.call_args.kwargs["model_config"] is draft_load.draft
