# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import importlib.util
import logging
import sys
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import pytest
import requests
import torch

RFORK_ROOT = Path(__file__).resolve().parents[4] / "vllm_ascend/model_loader/rfork"


def _load_module(monkeypatch, module_name: str, file_name: str):
    spec = importlib.util.spec_from_file_location(module_name, RFORK_ROOT / file_name)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, module_name, module)
    spec.loader.exec_module(module)
    return module


def _stub(monkeypatch, module_name: str, **attributes):
    module = ModuleType(module_name)
    module.__dict__.update(attributes)
    monkeypatch.setitem(sys.modules, module_name, module)
    return module


@pytest.fixture
def identity_module(monkeypatch):
    _stub(monkeypatch, "vllm.config", ModelConfig=object, VllmConfig=object)
    _stub(monkeypatch, "vllm_ascend.ascend_config", get_ascend_config=lambda: SimpleNamespace(weight_nz_mode=None))
    _stub(
        monkeypatch,
        "vllm_ascend.device.hardware_profile",
        get_current_hardware_profile=lambda: SimpleNamespace(weight_layout_policy=None),
    )
    return _load_module(monkeypatch, "rfork_test_identity", "identity.py")


@pytest.fixture
def manifest_module(monkeypatch):
    _stub(monkeypatch, "vllm.logger", logger=logging.getLogger("rfork-manifest-validation-test"))
    _stub(monkeypatch, "vllm_ascend.model_loader.rfork.types", SeedTransferInfo=object)
    return _load_module(monkeypatch, "rfork_test_manifest", "manifest.py")


@pytest.fixture
def seed_module(monkeypatch):
    _stub(monkeypatch, "vllm.logger", logger=logging.getLogger("rfork-seed-validation-test"))

    @dataclass(frozen=True)
    class SeedTransferInfo:
        session_id: str
        weights: dict
        shapes: dict | None = None

    _stub(monkeypatch, "vllm_ascend.model_loader.rfork.types", SeedTransferInfo=SeedTransferInfo)
    return _load_module(monkeypatch, "rfork_test_seed_client", "seed_client.py")


@pytest.fixture
def planner_module(monkeypatch):
    _stub(monkeypatch, "vllm.logger", logger=logging.getLogger("rfork-planner-validation-test"))
    _stub(monkeypatch, "vllm.utils.network_utils", get_ip=lambda: "127.0.0.1")
    _stub(monkeypatch, "vllm_ascend.model_loader.rfork.config", RForkConfig=object)
    _stub(monkeypatch, "vllm_ascend.model_loader.rfork.identity", build_seed_key=lambda **kwargs: "seed-key")

    class LeaseReleaseResult(Enum):
        RELEASED = auto()
        RETRYABLE = auto()
        REJECTED = auto()

    @dataclass(frozen=True)
    class SeedLease:
        seed_ip: str
        seed_port: int
        user_id: str
        seed_rank: int
        seed_key: str

    _stub(
        monkeypatch,
        "vllm_ascend.model_loader.rfork.types",
        LeaseReleaseResult=LeaseReleaseResult,
        SeedLease=SeedLease,
        RForkIdentity=object,
        SeedAdvertisement=object,
    )
    return _load_module(monkeypatch, "rfork_test_planner_client", "planner_client.py")


def _model_config(*, dtype=torch.float32, quantization=None, revision="rev-a"):
    return SimpleNamespace(
        dtype=dtype,
        quantization=quantization,
        revision=revision,
        hf_config=SimpleNamespace(
            model_type="test-model",
            architectures=["TestModel"],
            quantization_config=None,
        ),
    )


def _vllm_config(tp_size=1):
    return SimpleNamespace(
        parallel_config=SimpleNamespace(
            tensor_parallel_size=tp_size,
            pipeline_parallel_size=1,
            expert_parallel_size=None,
            data_parallel_size=None,
            enable_expert_parallel=False,
            is_moe_model=False,
        )
    )


def _fingerprint(identity_module, *, dtype=torch.float32, quantization=None, revision="rev-a", tp_size=1):
    return identity_module.build_compatibility_fingerprint(
        _vllm_config(tp_size),
        _model_config(dtype=dtype, quantization=quantization, revision=revision),
        model_url="model",
        model_deploy_strategy_name="strategy",
    )


def test_canonicalization_is_deterministic_for_dicts_sets_and_dtype(identity_module):
    first = {"set": {3, 1, 2}, "mapping": {"b": 2, "a": 1}, "dtype": torch.float16}
    second = {"dtype": torch.float16, "mapping": {"a": 1, "b": 2}, "set": {2, 3, 1}}

    assert identity_module._canonicalize_fingerprint_value(first) == identity_module._canonicalize_fingerprint_value(
        second
    )


@pytest.mark.parametrize(
    "change",
    [
        {"dtype": torch.float16},
        {"quantization": "compressed"},
        {"revision": "rev-b"},
        {"tp_size": 2},
    ],
)
def test_fingerprint_changes_when_seed_compatibility_changes(identity_module, change):
    base = _fingerprint(identity_module)
    assert _fingerprint(identity_module, **change) != base


def test_fingerprint_quantization_config_changes(identity_module):
    base_config = _model_config()
    changed_config = _model_config()
    base_config.hf_config.quantization_config = {"bits": 8, "group_size": 128}
    changed_config.hf_config.quantization_config = {"bits": 4, "group_size": 128}

    build = identity_module.build_compatibility_fingerprint
    kwargs = {"model_url": "model", "model_deploy_strategy_name": "strategy"}
    assert build(_vllm_config(), base_config, **kwargs) != build(_vllm_config(), changed_config, **kwargs)


def test_recursive_fingerprint_values_fail_with_actionable_error(identity_module):
    recursive_dict = {}
    recursive_dict["self"] = recursive_dict
    recursive_list = []
    recursive_list.append(recursive_list)
    recursive_object = SimpleNamespace()
    recursive_object.child = recursive_object

    for value in (recursive_dict, recursive_list, recursive_object):
        with pytest.raises(ValueError, match="recursive reference"):
            identity_module._canonicalize_fingerprint_value(value)


def test_shared_config_objects_are_not_treated_as_cycles(identity_module):
    shared = {"bits": 8}
    assert identity_module._canonicalize_fingerprint_value({"first": shared, "second": shared}) == {
        "first": {"bits": 8},
        "second": {"bits": 8},
    }


@pytest.mark.parametrize("raw", [[], "model", 1])
def test_extra_config_requires_json_object(monkeypatch, raw):
    config = _load_module(monkeypatch, "rfork_test_config", "config.py")
    with pytest.raises(RuntimeError, match="JSON object"):
        config.RForkConfig.from_extra_config(raw)


def test_build_seed_key_rejects_missing_identity_values(identity_module):
    build = identity_module.build_seed_key
    with pytest.raises(RuntimeError, match="model_url"):
        build(0, "", "strategy", "fingerprint")
    with pytest.raises(RuntimeError, match="model_deploy_strategy_name"):
        build(0, "model", "", "fingerprint")
    with pytest.raises(RuntimeError, match="compatibility fingerprint"):
        build(0, "model", "strategy", "")
    with pytest.raises(TypeError):
        build(0, "model", "strategy")


def test_manifest_rejects_conflicting_mapping_and_tuple_dtype(manifest_module):
    assert manifest_module.parse_weight_info((1, 1, 4, {"shape": [1], "dtype": "float32"}, "float16")) is None
    parsed = manifest_module.parse_weight_info((1, 1, 4, {"shape": [1], "dtype": "torch.float32"}, torch.float32))
    assert parsed == (1, 1, 4, (1,), "float32")


def test_seed_url_appends_independent_port_and_rejects_conflicts(seed_module):
    assert seed_module.build_seed_url("http://seed-host", 1234) == "http://seed-host:1234"
    assert seed_module.build_seed_url("https://seed-host:1234", 1234) == "https://seed-host:1234"
    assert seed_module.build_seed_url("[::1]", 1234) == "http://[::1]:1234"
    assert seed_module.build_seed_url("[::1]:1234", 1234) == "http://[::1]:1234"
    with pytest.raises(ValueError, match="conflicts"):
        seed_module.build_seed_url("http://seed-host:4321", 1234)
    with pytest.raises(ValueError, match="conflicts"):
        seed_module.build_seed_url("[::1]:4321", 1234)


def test_shape_request_failure_can_use_complete_weight_metadata(seed_module, monkeypatch, caplog):
    main_response = SimpleNamespace(
        status_code=200,
        json=lambda: {"rfork_transfer_engine_info": ["session", {"weight": (1, 1, 4, [1], "float32")}]},
    )
    get = Mock(side_effect=[main_response, requests.RequestException("shape endpoint unavailable")])
    monkeypatch.setattr(seed_module.requests, "get", get)

    with caplog.at_level(logging.WARNING):
        result = seed_module.fetch_seed_transfer_info("http://seed:1234", "key", 1.0)

    assert result is not None
    assert result.shapes is None
    assert "complete weight metadata" in caplog.text


def test_shape_request_failure_remains_conservative_without_complete_metadata(seed_module, monkeypatch):
    main_response = SimpleNamespace(
        status_code=200,
        json=lambda: {"rfork_transfer_engine_info": ["session", {"weight": (1, 1, 4)}]},
    )
    get = Mock(side_effect=[main_response, requests.RequestException("shape endpoint unavailable")])
    monkeypatch.setattr(seed_module.requests, "get", get)

    assert seed_module.fetch_seed_transfer_info("http://seed:1234", "key", 1.0) is None


@pytest.mark.parametrize(
    ("seed_port", "seed_rank", "message"),
    [("not-a-port", "0", "non-integer"), ("0", "-1", "invalid")],
)
def test_planner_rejects_invalid_lease_headers_with_error(
    planner_module, monkeypatch, caplog, seed_port, seed_rank, message
):
    config = SimpleNamespace(
        request_timeout_sec=1.0,
        planner_url="http://planner",
        model_url="model",
        model_deploy_strategy_name="strategy",
        lease_release_max_attempts=1,
        lease_release_retry_interval_sec=1.0,
        heartbeat_interval_sec=1.0,
    )
    identity = SimpleNamespace(
        tp_rank=0, pp_rank=None, ep_rank=None, is_draft_model=False, compatibility_fingerprint="fp"
    )
    client = planner_module.RForkPlannerClient(config, identity)
    response = SimpleNamespace(
        status_code=200,
        headers={"SEED_IP": "127.0.0.1", "SEED_PORT": seed_port, "USER_ID": "lease", "SEED_RANK": seed_rank},
    )
    monkeypatch.setattr(planner_module.requests, "get", Mock(return_value=response))

    with caplog.at_level(logging.WARNING):
        assert client.acquire_seed() is None
    assert message in caplog.text
