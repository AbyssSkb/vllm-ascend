# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import importlib.util
import logging
import queue
import socket
import sys
import threading
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


@pytest.fixture
def seed_server(monkeypatch):
    """Load seed_server with only the logger dependency supplied by the test."""
    vllm_module = ModuleType("vllm")
    vllm_module.__path__ = []
    monkeypatch.setitem(sys.modules, "vllm", vllm_module)
    logger_module = ModuleType("vllm.logger")
    logger_module.logger = logging.getLogger("rfork-seed-server-cleanup-test")
    monkeypatch.setitem(sys.modules, "vllm.logger", logger_module)

    requests_module = ModuleType("requests")
    requests_module.get = None
    monkeypatch.setitem(sys.modules, "requests", requests_module)

    class FastAPI:
        def get(self, *args, **kwargs):
            return lambda function: function

    fastapi_module = ModuleType("fastapi")
    fastapi_module.FastAPI = FastAPI
    monkeypatch.setitem(sys.modules, "fastapi", fastapi_module)

    responses_module = ModuleType("fastapi.responses")
    responses_module.Response = lambda status_code: SimpleNamespace(status_code=status_code)
    monkeypatch.setitem(sys.modules, "fastapi.responses", responses_module)

    uvicorn_module = ModuleType("uvicorn")
    uvicorn_module.Config = None
    uvicorn_module.Server = None
    monkeypatch.setitem(sys.modules, "uvicorn", uvicorn_module)

    ascend_module = ModuleType("vllm_ascend")
    ascend_module.__path__ = []
    monkeypatch.setitem(sys.modules, "vllm_ascend", ascend_module)
    model_loader_module = ModuleType("vllm_ascend.model_loader")
    model_loader_module.__path__ = []
    monkeypatch.setitem(sys.modules, "vllm_ascend.model_loader", model_loader_module)
    rfork_module = ModuleType("vllm_ascend.model_loader.rfork")
    rfork_module.__path__ = []
    monkeypatch.setitem(sys.modules, "vllm_ascend.model_loader.rfork", rfork_module)
    types_module = ModuleType("vllm_ascend.model_loader.rfork.types")
    types_module.SeedTransferInfo = object
    monkeypatch.setitem(sys.modules, "vllm_ascend.model_loader.rfork.types", types_module)

    path = Path(__file__).resolve().parents[4] / "vllm_ascend/model_loader/rfork/seed_server.py"
    name = "vllm_ascend.model_loader.rfork._seed_server_cleanup_isolated"
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, name, module)
    spec.loader.exec_module(module)
    return module


class _ControlledSocket:
    family = socket.AF_INET

    def __init__(self):
        self.closed = threading.Event()

    def getsockname(self):
        return ("127.0.0.1", 32123)

    def close(self):
        self.closed.set()


class _ControlledServer:
    instances = []
    run_entered = threading.Event()
    allow_run_exit = threading.Event()

    def __init__(self, config):
        self.should_exit = False
        self.config = config
        self.__class__.instances.append(self)

    def run(self, sockets):
        self.__class__.run_entered.set()
        self.__class__.allow_run_exit.wait(timeout=5.0)


class _BlockingQueue:
    def __init__(self, real_queue, maxsize=0):
        self._queue = real_queue(maxsize=maxsize)
        self.put_entered = threading.Event()
        self.allow_put = threading.Event()

    def put(self, item):
        self.put_entered.set()
        if not self.allow_put.wait(timeout=5.0):
            raise RuntimeError("test queue put was not released")
        self._queue.put(item)

    def get(self, timeout=None):
        return self._queue.get(timeout=timeout)


def _configure_server_fakes(seed_server, monkeypatch, *, blocking_queue=False):
    _ControlledServer.instances.clear()
    _ControlledServer.run_entered.clear()
    _ControlledServer.allow_run_exit.clear()
    sockets = []

    def create_socket(bind_host):
        sock = _ControlledSocket()
        sockets.append(sock)
        return sock

    monkeypatch.setattr(seed_server, "_create_bound_socket", create_socket)
    monkeypatch.setattr(seed_server.uvicorn, "Config", lambda *args, **kwargs: SimpleNamespace())
    monkeypatch.setattr(seed_server.uvicorn, "Server", _ControlledServer)

    queue_holder = {}
    if blocking_queue:
        real_queue = queue.Queue

        def queue_factory(maxsize=0):
            test_queue = _BlockingQueue(real_queue, maxsize=maxsize)
            queue_holder["queue"] = test_queue
            return test_queue

        monkeypatch.setattr(seed_server.queue, "Queue", queue_factory)

    return sockets, queue_holder


def _seed_info():
    return SimpleNamespace(session_id="session", weights={}, shapes={})


def test_queue_timeout_can_stop_startup_resource_published_before_put(seed_server, monkeypatch):
    sockets, queue_holder = _configure_server_fakes(seed_server, monkeypatch, blocking_queue=True)
    original_stop_startup_thread = seed_server._stop_startup_thread

    def stop_startup_thread(thread, startup_state, stop_event, timeout=0.01):
        return original_stop_startup_thread(thread, startup_state, stop_event, timeout=timeout)

    monkeypatch.setattr(seed_server, "_stop_startup_thread", stop_startup_thread)
    monkeypatch.setattr(seed_server, "SERVER_STARTUP_QUEUE_TIMEOUT_SEC", 0.05)

    with pytest.raises(seed_server.RForkSeedServerStartupError) as raised:
        seed_server.start_rfork_server("key", _seed_info(), health_timeout_sec=0.05)

    test_queue = queue_holder["queue"]
    assert test_queue.put_entered.is_set()
    assert sockets[0].closed.is_set()
    assert raised.value.handle is not None
    assert raised.value.handle.is_alive

    test_queue.allow_put.set()
    _ControlledServer.allow_run_exit.set()
    assert raised.value.handle.stop(timeout=1.0)


def test_health_failure_retains_handle_when_server_thread_outlives_stop_timeout(seed_server, monkeypatch):
    sockets, _ = _configure_server_fakes(seed_server, monkeypatch)
    original_handle_stop = seed_server.RForkSeedServerHandle.stop

    def stop_handle(handle, timeout=5.0):
        return original_handle_stop(handle, timeout=0.01 if timeout == 5.0 else timeout)

    monkeypatch.setattr(seed_server.RForkSeedServerHandle, "stop", stop_handle)
    monkeypatch.setattr(seed_server, "HEALTH_POLL_INTERVAL_SEC", 0.001)
    monkeypatch.setattr(seed_server.requests, "get", lambda *args, **kwargs: SimpleNamespace(status_code=503))

    with pytest.raises(seed_server.RForkSeedServerStartupError) as raised:
        seed_server.start_rfork_server("key", _seed_info(), health_timeout_sec=0.05)

    handle = raised.value.handle
    assert handle is not None
    assert handle.is_alive
    assert sockets[0].closed.is_set()
    assert _ControlledServer.instances[0].should_exit

    _ControlledServer.allow_run_exit.set()
    assert handle.stop(timeout=1.0)
