# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import sys
from types import SimpleNamespace

import pytest

pytest_plugins = [f"{__package__}.test_managed_weights"]


@pytest.mark.parametrize("hf_config_attr", ["hf_config", "hf_text_config"])
def test_v4_mtp_with_engram_config_skips_world_group_and_manager(loading, monkeypatch, hf_config_attr):
    loading.config.engram_config = SimpleNamespace(cpu_offload=False)
    loading.config.parallel_config = SimpleNamespace(pipeline_parallel_size=2)
    loading.model_config.runner_type = "draft"
    setattr(
        loading.model_config,
        hf_config_attr,
        SimpleNamespace(architectures=["DeepSeekV4MTPModel"], engram_layer_ids=[1]),
    )
    monkeypatch.setattr(
        sys.modules["vllm.distributed.parallel_state"],
        "get_world_group",
        lambda: pytest.fail("Draft loading must not enter a world group containing non-participating PP ranks"),
    )

    assert loading.loader.load_model(loading.config, loading.model_config) is loading.model
    loading.resolver.assert_not_called()
    assert "construct" in loading.events


def test_other_draft_with_engram_config_rejects_multi_stage_pipeline(loading, monkeypatch):
    loading.config.engram_config = SimpleNamespace(cpu_offload=False)
    loading.config.parallel_config = SimpleNamespace(pipeline_parallel_size=2)
    loading.model_config.runner_type = "draft"
    loading.model_config.hf_config = SimpleNamespace(architectures=["OtherMTPModel"], engram_layer_ids=[1])
    monkeypatch.setattr(
        sys.modules["vllm.distributed.parallel_state"],
        "get_world_group",
        lambda: pytest.fail("Unsupported draft loading must not enter the world group"),
    )

    with pytest.raises(NotImplementedError):
        loading.loader.load_model(loading.config, loading.model_config)
    loading.resolver.assert_not_called()
    assert "construct" not in loading.events
