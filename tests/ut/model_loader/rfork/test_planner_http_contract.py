# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import importlib.util
import logging
import sys
import threading
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from starlette.requests import Request

from examples.rfork.rfork_planner import Scheduler, Store, build_router


@pytest.fixture
def runtime(monkeypatch):
    root = Path(__file__).resolve().parents[4]
    prefix = "vllm_ascend.model_loader.rfork"

    def stub(name, **attrs):
        module = ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
        return module

    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, root / path)
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, name, module)
        spec.loader.exec_module(module)
        return module

    package = stub("vllm_ascend", __path__=[str(root / "vllm_ascend")])
    package.envs = load("vllm_ascend.envs", "vllm_ascend/envs.py")
    stub("vllm.logger", logger=logging.getLogger("rfork-http-contract-test"))
    stub("vllm.utils.network_utils", get_ip=lambda: "127.0.0.1")
    stub(f"{prefix}.identity", build_seed_key=lambda **kwargs: "model-key")
    modules = {
        name: load(f"{prefix}.{name}", f"vllm_ascend/model_loader/rfork/{name}.py")
        for name in ("types", "config", "planner_client")
    }
    return SimpleNamespace(**modules)


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("is_draft", [False, True])
def test_model_identity_headers_preserve_planner_lifecycle(runtime, monkeypatch, enabled, is_draft):
    config = runtime.config.RForkConfig("model", "strategy", "http://planner", send_model_identity_headers=enabled)
    identity = runtime.types.RForkIdentity(2, 2, is_draft_model=is_draft, compatibility_fingerprint="fingerprint")
    client = runtime.planner_client.RForkPlannerClient(config, identity)
    store = Store(heartbeat_ttl_sec=60, default_resource_points=1, scheduler=Scheduler())
    routes = {route.path: route.endpoint for route in build_router(store).routes}
    calls = []
    heartbeat_stop = threading.Event()

    def request(url, **kwargs):
        path = url.removeprefix("http://planner")
        calls.append((path, kwargs))
        headers = [(key.lower().encode("ascii"), value.encode("ascii")) for key, value in kwargs["headers"].items()]
        response = routes[path](Request({"type": "http", "headers": headers}))
        assert response.status_code == 200
        if path == "/add_seed" and len(calls) == 3:
            heartbeat_stop.set()
        return SimpleNamespace(status_code=response.status_code, headers=response.headers, text=response.body.decode())

    monkeypatch.setattr(runtime.planner_client.requests, "get", request)
    monkeypatch.setattr(runtime.planner_client.requests, "post", request)

    assert client.report_seed_once(1234)
    lease = client.acquire_seed()
    assert lease is not None
    assert store.debug_snapshot()["lease_count"] == 1
    client.run_seed_heartbeat(1234, stop_event=heartbeat_stop)
    assert store.debug_snapshot()["lease_count"] == 1
    assert client.release_seed_once(lease) is runtime.types.LeaseReleaseResult.RELEASED
    assert store.debug_snapshot()["lease_count"] == 0
    assert client.remove_seed()
    assert store.debug_snapshot()["seed_count"] == 0

    identity_headers = (
        {"MODEL_URL": "model", "DEPLOY_STRATEGY": "strategy", "IS_DRAFT": str(is_draft).lower()} if enabled else {}
    )
    advertisement = {"SEED_KEY": "model-key", "SEED_IP": "127.0.0.1", "SEED_PORT": "1234", "SEED_RANK": "2"}
    expected = [
        ("/add_seed", {**advertisement, "SEED_REFCNT": "0"}),
        ("/get_seed", {"SEED_KEY": "model-key"}),
        ("/add_seed", {**advertisement, "SEED_REFCNT": "0"}),
        (
            "/put_seed",
            {"SEED_IP": "127.0.0.1", "SEED_PORT": "1234", "USER_ID": lease.user_id, "SEED_RANK": "2"},
        ),
        ("/remove_seed", advertisement),
    ]
    assert len(calls) == len(expected)
    for (path, kwargs), (expected_path, headers) in zip(calls, expected):
        assert path == expected_path
        assert kwargs["headers"] == {**headers, **identity_headers}
        assert kwargs["timeout"] == config.request_timeout_sec
    assert calls[3][1]["allow_redirects"] is False
