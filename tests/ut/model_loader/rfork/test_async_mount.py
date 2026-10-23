# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import ast
import sys
from contextlib import contextmanager, nullcontext
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import pytest

from .rfork_test_support import _load_module, _stub


@pytest.fixture
def mount_module(monkeypatch):
    return _load_module(monkeypatch, "_async_mount_test", "../async_mount.py")


def install_manager(monkeypatch, resolve):
    _stub(monkeypatch, "model_manager", __path__=[])
    _stub(monkeypatch, "model_manager.apis", get_model_path_from_manager=resolve)


def test_mount_resolves_weights_in_background(mount_module, monkeypatch, tmp_path):
    called = Event()
    release = Event()

    def resolve(*, with_weights):
        assert with_weights is True
        called.set()
        assert release.wait(5)
        return str(tmp_path)

    install_manager(monkeypatch, resolve)
    mount = mount_module.AsyncModelMount()
    try:
        assert called.wait(5)
        assert not mount.done.is_set()
    finally:
        release.set()
    assert mount.wait() == str(tmp_path)
    assert mount.done.is_set()
    assert mount.error is None
    assert mount.completed_at is not None


@pytest.mark.parametrize("value", [None, "", " ", 42])
def test_mount_rejects_invalid_paths(mount_module, monkeypatch, value):
    install_manager(monkeypatch, lambda **kwargs: value)
    with pytest.raises(ValueError, match="invalid model path"):
        mount_module.AsyncModelMount().wait()


def test_mount_rejects_missing_directory(mount_module, monkeypatch, tmp_path):
    install_manager(monkeypatch, lambda **kwargs: str(tmp_path / "absent"))
    with pytest.raises(FileNotFoundError, match="Mounted model directory"):
        mount_module.AsyncModelMount().wait()


def test_mount_propagates_manager_error(mount_module, monkeypatch):
    error = RuntimeError("mount failed")

    def resolve(**kwargs):
        raise error

    install_manager(monkeypatch, resolve)
    mount = mount_module.AsyncModelMount()
    with pytest.raises(RuntimeError, match="mount failed") as raised:
        mount.wait()
    assert raised.value is error
    assert mount.done.is_set()


def test_mount_missing_manager_has_clear_error(mount_module, monkeypatch):
    _stub(monkeypatch, "model_manager", __path__=[])
    monkeypatch.delitem(sys.modules, "model_manager.apis", raising=False)
    with pytest.raises(ImportError, match="requires model_manager.apis"):
        mount_module.AsyncModelMount().wait()


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan"), True, "1", None])
def test_mount_timeout_validation(mount_module, timeout):
    with pytest.raises(ValueError, match="finite positive"):
        mount_module.AsyncModelMount(timeout)


def test_mount_wait_uses_original_deadline(mount_module, monkeypatch, tmp_path):
    clock = [10.0]
    called = Event()
    release = Event()
    monkeypatch.setattr(mount_module.time, "monotonic", lambda: clock[0])

    def resolve(**kwargs):
        called.set()
        assert release.wait(5)
        return str(tmp_path)

    install_manager(monkeypatch, resolve)
    mount = mount_module.AsyncModelMount(5)
    try:
        assert called.wait(5)
        clock[0] = 13.0
        assert mount.remaining() == 2.0
        clock[0] = 15.0
        with pytest.raises(TimeoutError, match="Timed out"):
            mount.wait()
    finally:
        release.set()
    assert mount.done.wait(5)
    assert mount.path is None
    assert isinstance(mount.error, TimeoutError)
    with pytest.raises(TimeoutError, match="after.*deadline"):
        mount.wait()


def model_configs(*, draft_path="bootstrap"):
    model = SimpleNamespace(
        model="bootstrap", tokenizer="bootstrap", served_model_name="logical", enable_sleep_mode=False
    )
    draft = SimpleNamespace(model=draft_path, tokenizer="bootstrap", runner_type="draft")
    spec = SimpleNamespace(target_model_config=model, draft_model_config=draft)
    config = SimpleNamespace(
        model_config=model,
        speculative_config=spec,
        load_config=SimpleNamespace(load_format="auto"),
        compilation_config=SimpleNamespace(static_forward_context={}),
        weight_transfer_config=None,
    )
    return config, model, spec, draft


def test_mounted_config_preserves_identity_and_shared_runtime(mount_module):
    config, model, _, _ = model_configs()
    mounted_config, mounted_model = mount_module.mounted_model_config(config, model, "mounted")
    assert mounted_config is not config
    assert mounted_model is not model
    assert mounted_config.model_config is mounted_model
    assert mounted_model.model == "mounted"
    assert mounted_model.tokenizer == "bootstrap"
    assert mounted_model.served_model_name == "logical"
    assert mounted_config.compilation_config is config.compilation_config
    assert mounted_config.load_config is config.load_config
    assert model.model == "bootstrap"


def test_mounted_draft_config_preserves_target_derived_model(mount_module):
    config, target, _, draft = model_configs()
    target.hf_config = object()
    draft.hf_config = object()
    target.runner_type = "draft"
    mounted_config, mounted_draft = mount_module.mounted_model_config(config, draft, "mounted")
    assert mounted_config.model_config is not mounted_draft
    assert mounted_config.model_config.hf_config is target.hf_config
    assert mounted_config.model_config.runner_type == "draft"
    assert mounted_draft.hf_config is draft.hf_config
    assert mounted_config.model_config.model == mounted_draft.model == "mounted"
    assert config.model_config is target
    assert target.model == draft.model == "bootstrap"


def test_mounted_context_preserves_drafter_target_derived_config(mount_module):
    config, model, spec, draft = model_configs()
    target_derived_model = SimpleNamespace(model="bootstrap", hf_config=object(), runner_type="draft")
    draft_config = SimpleNamespace(model_config=target_derived_model, speculative_config=spec)
    drafter = SimpleNamespace(vllm_config=draft_config, model_config=draft, speculative_config=spec)
    runner = SimpleNamespace(vllm_config=config, model_config=model, drafter=drafter)
    worker = SimpleNamespace(vllm_config=config, model_config=model, model_runner=runner)
    with mount_module.mounted_model_context(worker, "mounted"):
        assert drafter.vllm_config is not worker.vllm_config
        assert drafter.vllm_config.model_config.hf_config is target_derived_model.hf_config
        assert drafter.vllm_config.model_config.runner_type == "draft"
        assert drafter.vllm_config.model_config is not drafter.model_config
        assert drafter.vllm_config.model_config.model == drafter.model_config.model == "mounted"
    assert drafter.vllm_config is draft_config
    assert drafter.model_config is draft
    assert target_derived_model.model == draft.model == "bootstrap"


@pytest.mark.parametrize("fail_load", [False, True])
@pytest.mark.parametrize("draft_attr", ["drafter", "speculator"])
def test_mounted_context_copies_same_checkpoint_draft_and_restores(mount_module, fail_load, draft_attr):
    config, model, spec, draft = model_configs()
    drafter = SimpleNamespace(vllm_config=config, speculative_config=spec, draft_model_config=draft)
    runner = SimpleNamespace(vllm_config=config, model_config=model, speculative_config=spec)
    setattr(runner, draft_attr, drafter)
    worker = SimpleNamespace(vllm_config=config, model_config=model, model_runner=runner)
    try:
        with mount_module.mounted_model_context(worker, "mounted"):
            assert worker.vllm_config is runner.vllm_config is drafter.vllm_config
            assert worker.model_config is runner.model_config
            assert runner.model_config.model == "mounted"
            assert drafter.draft_model_config.model == "mounted"
            assert drafter.speculative_config.target_model_config.model == "mounted"
            assert drafter.draft_model_config is drafter.speculative_config.draft_model_config
            assert model.model == draft.model == "bootstrap"
            if fail_load:
                raise RuntimeError("load failed")
    except RuntimeError:
        assert fail_load
    assert worker.vllm_config is runner.vllm_config is drafter.vllm_config is config
    assert worker.model_config is runner.model_config is model
    assert runner.speculative_config is drafter.speculative_config is spec
    assert drafter.draft_model_config is draft


def test_independent_draft_is_rejected(mount_module):
    config, _, _, _ = model_configs(draft_path="independent")
    with pytest.raises(ValueError, match="independent draft"):
        mount_module.validate_async_mount_model(config)


def worker_load_method(namespace):
    source = Path(__file__).resolve().parents[4] / "vllm_ascend/worker/worker.py"
    tree = ast.parse(source.read_text())
    cls = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "NPUWorker")
    method = next(node for node in cls.body if isinstance(node, ast.FunctionDef) and node.name == "load_model")
    module = ast.Module(body=[method], type_ignores=[])
    exec(compile(module, str(source), "exec"), namespace)
    return namespace["load_model"]


@pytest.mark.parametrize("enabled, load_format", [(False, "auto"), (True, "rfork"), (True, "auto")])
def test_worker_uses_mount_only_for_enabled_ordinary_loader(mount_module, monkeypatch, enabled, load_format):
    config, model, _, _ = model_configs()
    config.load_config.load_format = load_format
    runner = SimpleNamespace(vllm_config=config, model_config=model)
    worker = SimpleNamespace(vllm_config=config, model_config=model, model_runner=runner)
    calls = []
    mounted = enabled and load_format != "rfork"
    _stub(monkeypatch, "vllm_ascend", __path__=[])
    _stub(monkeypatch, "vllm_ascend.model_loader", __path__=[])
    phases = []
    _stub(
        monkeypatch,
        "vllm_ascend.model_loader.rfork.source_race",
        RForkLoadCoordinator=lambda config, **kwargs: SimpleNamespace(
            phase=lambda name: phases.append(name) or nullcontext()
        ),
    )
    monkeypatch.setitem(sys.modules, "vllm_ascend.model_loader.async_mount", mount_module)
    monkeypatch.setattr(mount_module, "AsyncModelMount", lambda: SimpleNamespace(wait=lambda: "mounted"))

    @contextmanager
    def current_config(active):
        assert active is worker.vllm_config
        assert active.model_config.model == ("mounted" if mounted else "bootstrap")
        calls.append("context")
        yield

    def load_model():
        assert runner.model_config.model == ("mounted" if mounted else "bootstrap")
        calls.append("load")

    runner.load_model = load_model
    load = worker_load_method(
        {
            "nullcontext": nullcontext,
            "envs_ascend": SimpleNamespace(VLLM_ASCEND_ASYNC_MODEL_MOUNT=enabled),
            "set_current_vllm_config": current_config,
        }
    )
    load(worker)
    assert calls == ["context", "load"]
    assert phases == (["async mounted directory", "mounted model loading"] if mounted else [])
    assert worker.vllm_config is runner.vllm_config is config
    assert worker.model_config is runner.model_config is model


@pytest.mark.parametrize("failure_stage", ["wait", "load"])
def test_worker_coordinates_mount_and_load_errors(mount_module, monkeypatch, failure_stage):
    config, model, _, _ = model_configs()
    runner = SimpleNamespace(vllm_config=config, model_config=model)
    worker = SimpleNamespace(vllm_config=config, model_config=model, model_runner=runner)
    _stub(monkeypatch, "vllm_ascend", __path__=[])
    _stub(monkeypatch, "vllm_ascend.model_loader", __path__=[])
    monkeypatch.setitem(sys.modules, "vllm_ascend.model_loader.async_mount", mount_module)
    failed_phases = []

    @contextmanager
    def phase(name):
        try:
            yield
        except RuntimeError:
            failed_phases.append(name)
            raise

    _stub(
        monkeypatch,
        "vllm_ascend.model_loader.rfork.source_race",
        RForkLoadCoordinator=lambda config, **kwargs: SimpleNamespace(phase=phase),
    )

    def fail():
        raise RuntimeError("expected failure")

    monkeypatch.setattr(
        mount_module,
        "AsyncModelMount",
        lambda: SimpleNamespace(wait=fail if failure_stage == "wait" else lambda: "mounted"),
    )
    runner.load_model = fail
    load = worker_load_method(
        {
            "nullcontext": nullcontext,
            "envs_ascend": SimpleNamespace(VLLM_ASCEND_ASYNC_MODEL_MOUNT=True),
            "set_current_vllm_config": lambda config: nullcontext(),
        }
    )
    with pytest.raises(RuntimeError, match="expected failure"):
        load(worker)
    expected_phase = "async mounted directory" if failure_stage == "wait" else "mounted model loading"
    assert failed_phases == [expected_phase]
    assert worker.vllm_config is runner.vllm_config is config
    assert worker.model_config is runner.model_config is model


@pytest.mark.parametrize("value, expected", [(None, False), ("0", False), ("1", True)])
def test_async_mount_env_flag(monkeypatch, value, expected):
    envs = _load_module(monkeypatch, "_async_mount_env_test", "../../envs.py")
    if value is None:
        monkeypatch.delenv("VLLM_ASCEND_ASYNC_MODEL_MOUNT", raising=False)
    else:
        monkeypatch.setenv("VLLM_ASCEND_ASYNC_MODEL_MOUNT", value)
    assert envs.VLLM_ASCEND_ASYNC_MODEL_MOUNT is expected


def test_async_mount_env_rejects_invalid_flag(monkeypatch):
    envs = _load_module(monkeypatch, "_async_mount_env_test", "../../envs.py")
    monkeypatch.setenv("VLLM_ASCEND_ASYNC_MODEL_MOUNT", "true")
    with pytest.raises(ValueError, match="either '0' or '1'"):
        _ = envs.VLLM_ASCEND_ASYNC_MODEL_MOUNT
