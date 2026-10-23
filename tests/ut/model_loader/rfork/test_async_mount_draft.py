# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import __future__

import ast
from contextlib import contextmanager, nullcontext
from pathlib import Path
from types import SimpleNamespace

import pytest

from .rfork_test_support import _load_module, _stub


@pytest.fixture
def draft_runtime(monkeypatch):
    """Exercise the real draft-loading method with external runtime imports replaced."""
    mount_module = _load_module(monkeypatch, "_async_mount_draft_test", "../async_mount.py")
    _stub(monkeypatch, "vllm", __path__=[])
    _stub(monkeypatch, "vllm.compilation", __path__=[])
    _stub(monkeypatch, "vllm.compilation.backends", set_model_tag=lambda tag: nullcontext())
    _stub(monkeypatch, "vllm_ascend", __path__=[])
    _stub(monkeypatch, "vllm_ascend.model_loader", __path__=[])
    _stub(monkeypatch, "vllm_ascend.model_loader.async_mount", **vars(mount_module))
    events = []
    phases = []
    captured = []
    mount_error = [None]
    load_error = [None]
    cleanup_error = [None]

    @contextmanager
    def phase(name):
        phases.append(name)
        try:
            yield
        except Exception:
            events.append(("failed", name))
            raise

    def coordinate(config, *, is_draft):
        assert is_draft is True
        assert config.model_config.runner_type == "draft"
        return SimpleNamespace(phase=phase)

    _stub(monkeypatch, "vllm_ascend.model_loader.rfork.source_race", RForkLoadCoordinator=coordinate)

    def wait():
        events.append("mount")
        if mount_error[0] is not None:
            raise mount_error[0]
        return "mounted"

    target = SimpleNamespace(model="bootstrap", hf_config=object(), tokenizer="bootstrap", runner_type="generate")
    target_derived = SimpleNamespace(model="bootstrap", hf_config=target.hf_config, runner_type="draft")
    draft = SimpleNamespace(model="bootstrap", hf_config=object(), tokenizer="bootstrap", runner_type="draft")
    main_load = SimpleNamespace(load_format="rfork", async_model_mount=SimpleNamespace(wait=wait))
    draft_load = SimpleNamespace(load_format="auto")
    spec = SimpleNamespace(target_model_config=target, draft_model_config=draft, draft_load_config=draft_load)
    config = SimpleNamespace(model_config=target, speculative_config=spec, load_config=main_load)
    draft_config = SimpleNamespace(model_config=target_derived, speculative_config=spec, load_config=main_load)
    loaded_model = object()

    def load(**kwargs):
        events.append("load")
        captured.append(kwargs)
        if load_error[0] is not None:
            raise load_error[0]
        return loaded_model

    def cleanup(model):
        assert model is loaded_model
        events.append("cleanup")
        if cleanup_error[0] is not None:
            raise cleanup_error[0]

    proposer = SimpleNamespace(
        vllm_config=config,
        speculative_config=spec,
        method="mtp",
        _create_draft_vllm_config=lambda: draft_config,
        _maybe_remove_d2t=cleanup,
    )
    envs = SimpleNamespace(VLLM_ASCEND_ASYNC_MODEL_MOUNT=True)
    source = Path(__file__).resolve().parents[4] / "vllm_ascend/spec_decode/llm_base_proposer.py"
    tree = ast.parse(source.read_text())
    cls = next(
        node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "AscendSpecDecodeBaseProposer"
    )
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "_get_model")
    namespace = {
        "nullcontext": nullcontext,
        "envs": envs,
        "logger": SimpleNamespace(info=lambda *args: None),
        "get_model": load,
    }
    exec(
        compile(ast.Module(body=[method], type_ignores=[]), str(source), "exec", __future__.annotations.compiler_flag),
        namespace,
    )
    return SimpleNamespace(
        get_model=namespace["_get_model"],
        proposer=proposer,
        config=config,
        draft_config=draft_config,
        target=target,
        target_derived=target_derived,
        draft=draft,
        envs=envs,
        captured=captured,
        events=events,
        phases=phases,
        mount_error=mount_error,
        load_error=load_error,
        cleanup_error=cleanup_error,
        loaded_model=loaded_model,
    )


def test_ordinary_draft_reuses_main_rfork_mount_and_preserves_target_config(draft_runtime):
    rt = draft_runtime
    assert rt.get_model(rt.proposer) is rt.loaded_model
    assert rt.events == ["mount", "load", "cleanup"]
    assert rt.phases == ["draft mounted directory", "mounted draft loading"]
    loaded = rt.captured[0]
    assert loaded["vllm_config"].model_config.hf_config is rt.target.hf_config
    assert loaded["vllm_config"].model_config.runner_type == "draft"
    assert loaded["model_config"].hf_config is rt.draft.hf_config
    assert loaded["vllm_config"].model_config is not loaded["model_config"]
    assert loaded["vllm_config"].model_config.model == loaded["model_config"].model == "mounted"
    assert loaded["load_config"] is rt.proposer.speculative_config.draft_load_config
    assert rt.target.model == rt.target_derived.model == rt.draft.model == "bootstrap"
    assert rt.proposer.vllm_config is rt.config
    assert rt.proposer.speculative_config.draft_model_config is rt.draft


@pytest.mark.parametrize("unchanged_path", ["disabled", "ordinary-main", "rfork-draft", "inherited-rfork"])
def test_other_draft_paths_are_unchanged(draft_runtime, unchanged_path):
    rt = draft_runtime
    if unchanged_path == "disabled":
        rt.envs.VLLM_ASCEND_ASYNC_MODEL_MOUNT = False
    elif unchanged_path == "ordinary-main":
        rt.config.load_config.load_format = "auto"
    elif unchanged_path == "rfork-draft":
        rt.proposer.speculative_config.draft_load_config.load_format = "rfork"
    else:
        rt.proposer.speculative_config.draft_load_config = None
    assert rt.get_model(rt.proposer) is rt.loaded_model
    assert rt.events == ["load", "cleanup"]
    assert rt.phases == []
    assert rt.captured[0]["vllm_config"] is rt.draft_config
    assert rt.captured[0]["model_config"] is rt.draft
    assert rt.captured[0]["load_config"] is rt.proposer.speculative_config.draft_load_config


@pytest.mark.parametrize("failure_stage", ["mount", "load", "cleanup"])
def test_ordinary_draft_mount_and_load_failures_are_coordinated(draft_runtime, failure_stage):
    rt = draft_runtime
    getattr(rt, f"{failure_stage}_error")[0] = TimeoutError("original deadline")
    with pytest.raises(TimeoutError, match="original deadline"):
        rt.get_model(rt.proposer)
    phase_name = "draft mounted directory" if failure_stage == "mount" else "mounted draft loading"
    assert ("failed", phase_name) in rt.events
    assert len(rt.captured) == (0 if failure_stage == "mount" else 1)
    assert rt.target.model == rt.target_derived.model == rt.draft.model == "bootstrap"
