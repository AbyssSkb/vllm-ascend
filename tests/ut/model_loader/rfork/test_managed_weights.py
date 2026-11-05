# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import json
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from .rfork_test_support import _load_module, _stub


@pytest.fixture
def managed_weights(monkeypatch):
    package = _stub(monkeypatch, "vllm_ascend", __path__=[])
    envs = _stub(monkeypatch, "vllm_ascend.envs", VLLM_ASCEND_ASYNC_MODEL_MOUNT=False)
    package.envs = envs
    return _load_module(monkeypatch, "managed_weights_test", "managed_weights.py")


def configs(*, draft_path=None):
    model = SimpleNamespace(model="bootstrap", tokenizer="bootstrap", served_model_name="logical")
    draft = SimpleNamespace(model=draft_path) if draft_path is not None else None
    spec = SimpleNamespace(target_model_config=model, draft_model_config=draft) if draft else None
    return SimpleNamespace(model_config=model, speculative_config=spec), model


def manager(monkeypatch, result):
    _stub(monkeypatch, "model_manager", __path__=[])
    resolve = Mock(return_value=result)
    _stub(monkeypatch, "model_manager.apis", get_model_path_from_manager=resolve)
    return resolve


def test_enabled_follows_configuration(managed_weights, monkeypatch):
    assert not managed_weights.enabled()
    monkeypatch.setattr(managed_weights.envs, "VLLM_ASCEND_ASYNC_MODEL_MOUNT", True)
    assert managed_weights.enabled()


def test_mounted_configs_uses_manager_lazily_and_keeps_original_identity(managed_weights, monkeypatch, tmp_path):
    resolve = manager(monkeypatch, str(tmp_path))
    config, model = configs()
    resolve.assert_not_called()
    mounted_config, mounted_model = managed_weights.mounted_configs(config, model)
    resolve.assert_called_once_with(with_weights=True)
    assert mounted_config is not config
    assert mounted_model is not model
    assert mounted_config.model_config is mounted_model
    assert mounted_model.model == str(tmp_path)
    assert mounted_model.tokenizer == "bootstrap"
    assert mounted_model.served_model_name == "logical"
    assert config.model_config is model
    assert model.model == "bootstrap"


@pytest.mark.parametrize("result", [None, "", " ", 42])
def test_mounted_configs_rejects_invalid_manager_path(managed_weights, monkeypatch, result):
    manager(monkeypatch, result)
    config, model = configs()
    with pytest.raises((ValueError, TypeError)):
        managed_weights.mounted_configs(config, model)


def test_mounted_configs_rejects_missing_directory(managed_weights, monkeypatch, tmp_path):
    manager(monkeypatch, str(tmp_path / "missing"))
    config, model = configs()
    with pytest.raises((FileNotFoundError, ValueError)):
        managed_weights.mounted_configs(config, model)


def test_mounted_configs_propagates_manager_error(managed_weights, monkeypatch):
    resolve = manager(monkeypatch, None)
    resolve.side_effect = RuntimeError("manager failed")
    config, model = configs()
    with pytest.raises(RuntimeError, match="manager failed"):
        managed_weights.mounted_configs(config, model)


def test_mounted_configs_rejects_independent_draft(managed_weights, monkeypatch, tmp_path):
    resolve = manager(monkeypatch, str(tmp_path))
    config, model = configs(draft_path="other-checkpoint")
    with pytest.raises(ValueError, match="draft"):
        managed_weights.mounted_configs(config, config.speculative_config.draft_model_config)
    resolve.assert_not_called()


def test_mounted_configs_allows_target_with_independent_draft(managed_weights, monkeypatch, tmp_path):
    manager(monkeypatch, str(tmp_path))
    config, model = configs(draft_path="other-checkpoint")
    mounted_config, mounted_model = managed_weights.mounted_configs(config, model)
    assert mounted_model.model == str(tmp_path)
    assert mounted_config.speculative_config.draft_model_config.model == "other-checkpoint"


def test_mounted_configs_keeps_draft_configuration_distinct(managed_weights, monkeypatch, tmp_path):
    manager(monkeypatch, str(tmp_path))
    config, model = configs(draft_path="bootstrap")
    draft = config.speculative_config.draft_model_config
    mounted_config, mounted_draft = managed_weights.mounted_configs(config, draft)
    assert mounted_draft is not draft
    assert mounted_config.model_config is not mounted_draft
    assert mounted_config.model_config is not model
    assert mounted_config.model_config.model == mounted_draft.model == str(tmp_path)
    assert model.model == draft.model == "bootstrap"


@pytest.fixture
def loading(rfork_helpers, monkeypatch, tmp_path):
    module = rfork_helpers.loader
    events = []
    config, model_config = configs()
    config.device_config = SimpleNamespace(device="cpu")
    model_config.dtype = torch.float32
    model = torch.nn.Module()
    fallback_model = torch.nn.Module()
    session = SimpleNamespace(
        loading=nullcontext,
        identity=SimpleNamespace(tp_rank=1, global_rank=0, is_draft_model=False),
        register_receiver_weights=lambda *args: True,
        acquire_seed_lease=Mock(return_value=True),
        read_weights_from_seed=lambda: True,
        validate_transferred_model_layout=lambda *args: None,
        prepare_for_fallback=lambda: events.append("cleanup") or True,
    )
    loader = module.RForkModelLoader.__new__(module.RForkModelLoader)
    loader.load_config = SimpleNamespace(device=None)
    loader._get_session = lambda *args: session
    monkeypatch.setattr(module.managed_weights.envs, "VLLM_ASCEND_ASYNC_MODEL_MOUNT", True)
    resolver = manager(monkeypatch, str(tmp_path))
    resolver.side_effect = lambda **kwargs: events.append("manager") or str(tmp_path)
    monkeypatch.setattr(module, "_rfork_bypass_reason", lambda *args: None)
    monkeypatch.setattr(module, "_snapshot_process_global_model_state", lambda *args: None)
    monkeypatch.setattr(module, "_reset_process_global_model_state", lambda *args: None)
    monkeypatch.setattr(module, "initialize_model", lambda **kwargs: events.append("construct") or model)
    monkeypatch.setattr(module, "_prepare_rfork_model_state", lambda *args: None)
    monkeypatch.setattr(module, "load_local_weights", lambda *args: None)
    monkeypatch.setattr(module, "release_local_weights", lambda *args: None)
    monkeypatch.setattr(module, "record_loaded_local_weights", lambda *args: None)
    monkeypatch.setattr(module, "process_weights_after_loading", lambda *args: None)
    monkeypatch.setattr(module, "_rfork_pre_transfer_weight_processing", lambda *args: nullcontext())
    monkeypatch.setattr(module, "_refresh_rfork_flatquant_state", lambda *args: None)
    monkeypatch.setattr(module, "_validate_rfork_shared_experts", lambda *args: None)
    monkeypatch.setattr(module, "_schedule_seed_publication", lambda *args, **kwargs: None)

    def default_load(mounted_config, mounted_model, *args):
        assert mounted_config.model_config is mounted_model
        assert mounted_model.model == str(tmp_path)
        events.append("default")
        return fallback_model

    monkeypatch.setattr(module, "_load_with_default_loader", default_load)
    monkeypatch.setattr(
        torch, "npu", SimpleNamespace(synchronize=lambda: None, empty_cache=lambda: None), raising=False
    )
    return SimpleNamespace(
        module=module,
        loader=loader,
        config=config,
        model_config=model_config,
        model=model,
        fallback_model=fallback_model,
        session=session,
        resolver=resolver,
        events=events,
        helpers=rfork_helpers,
    )


def test_transfer_does_not_wait_for_manager(loading):
    assert loading.loader.load_model(loading.config, loading.model_config) is loading.model
    loading.resolver.assert_not_called()


def test_seed_miss_waits_for_manager_after_cleanup(loading):
    loading.session.acquire_seed_lease.return_value = False
    assert loading.loader.load_model(loading.config, loading.model_config) is loading.fallback_model
    assert loading.events.index("cleanup") < loading.events.index("manager") < loading.events.index("default")


def test_seed_miss_stops_if_manager_fails(loading):
    loading.session.acquire_seed_lease.return_value = False
    loading.resolver.side_effect = RuntimeError("manager unavailable")
    with pytest.raises(RuntimeError, match="manager unavailable"):
        loading.loader.load_model(loading.config, loading.model_config)
    assert "cleanup" in loading.events
    assert "default" not in loading.events


def test_engram_waits_for_manager_before_construction(loading):
    loading.config.engram_config = SimpleNamespace(cpu_offload=False)
    loading.model_config.hf_config = SimpleNamespace(engram_layer_ids=[1])
    assert loading.loader.load_model(loading.config, loading.model_config) is loading.model
    assert loading.events.index("manager") < loading.events.index("construct")


@pytest.mark.parametrize("has_index", [True, False])
def test_mtp_uses_bootstrap_index_or_waits_for_manager(loading, monkeypatch, tmp_path, has_index):
    loading.model_config.model = str(tmp_path)
    if has_index:
        (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"lm_head.weight": "a"}}))
    mtp = loading.helpers.mtp_module.AscendDeepSeekMTP()
    monkeypatch.setattr(loading.module, "initialize_model", lambda **kwargs: mtp)
    assert loading.loader.load_model(loading.config, loading.model_config) is mtp
    assert loading.resolver.call_count == (0 if has_index else 1)
