# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.

import errno
import math
import socket
import threading
import time
from contextlib import suppress
from http import HTTPStatus

import requests
import uvicorn
from fastapi import FastAPI
from fastapi.responses import Response
from vllm.logger import logger

from vllm_ascend.model_loader.rfork.types import RFORK_PROTOCOL_VERSION, SeedTransferInfo

SERVER_STOP_TIMEOUT_SEC = 5.0
HEALTH_POLL_INTERVAL_SEC = 0.05


class RForkSeedServerHandle:
    """Own the listener and server thread until both are stopped."""

    def __init__(self, server: uvicorn.Server, sock: socket.socket) -> None:
        self.server = server
        self.sock = sock
        self.port = sock.getsockname()[1]
        self.thread = threading.Thread(target=self._run, daemon=True, name="RForkSeedServer")

    def _run(self) -> None:
        try:
            self.server.run(sockets=[self.sock])
        except Exception:
            logger.exception("[RFork Seed] server thread failed")
        finally:
            with suppress(OSError):
                self.sock.close()

    @property
    def is_alive(self) -> bool:
        return self.thread.is_alive()

    def stop(self, timeout: float = SERVER_STOP_TIMEOUT_SEC) -> bool:
        self.server.should_exit = True
        if self.thread.ident is not None and self.thread is not threading.current_thread():
            self.thread.join(timeout=max(float(timeout), 0.0))
        stopped = not self.thread.is_alive()
        if stopped:
            with suppress(OSError):
                self.sock.close()
        else:
            logger.warning("[RFork Seed] server thread did not stop within %.2fs", timeout)
        return stopped


class RForkSeedServerStartupError(RuntimeError):
    """Report startup failure and retain a server handle for cleanup."""

    def __init__(self, message: str, *, handle: RForkSeedServerHandle | None = None) -> None:
        super().__init__(message)
        self.handle = handle


def _create_bound_socket(bind_host: str, port: int = 0) -> socket.socket:
    """Create an IPv4 or IPv6 listener from the bind host's resolver result."""
    host = bind_host or "0.0.0.0"
    infos = socket.getaddrinfo(host, port, socket.AF_UNSPEC, socket.SOCK_STREAM, 0, socket.AI_PASSIVE)
    last_error: OSError | None = None
    for family, socktype, protocol, _, sockaddr in infos:
        sock: socket.socket | None = None
        try:
            sock = socket.socket(family, socktype, protocol)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(sockaddr)
            sock.listen(socket.SOMAXCONN)
            return sock
        except OSError as exc:
            last_error = exc
            if sock is not None:
                sock.close()
    if last_error is None:
        raise OSError(f"unable to resolve RFork seed bind host {host!r}")
    raise OSError(f"unable to bind RFork seed host {host!r} port {port}: {last_error}") from last_error


def _health_url(bind_host: str, sock: socket.socket, port: int) -> str:
    if sock.family == socket.AF_INET6:
        host = str(sock.getsockname()[0]) or "::1"
        if host in ("::", "0:0:0:0:0:0:0:0"):
            host = "::1"
        return f"http://[{host}]:{port}/health_check_with_key"
    host = str(sock.getsockname()[0]) or bind_host or "127.0.0.1"
    if host == "0.0.0.0":
        host = "127.0.0.1"
    return f"http://{host}:{port}/health_check_with_key"


def start_rfork_server(
    seed_key,
    rfork_transfer_engine_info: SeedTransferInfo,
    health_timeout_sec: float = 30.0,
    *,
    bind_host: str = "0.0.0.0",
    port: int = 0,
    fallback_to_dynamic_port: bool = False,
) -> RForkSeedServerHandle:
    if isinstance(health_timeout_sec, bool) or not isinstance(health_timeout_sec, (int, float)):
        raise ValueError("health_timeout_sec must be a finite positive number")
    if not math.isfinite(float(health_timeout_sec)) or float(health_timeout_sec) <= 0:
        raise ValueError("health_timeout_sec must be a finite positive number")

    try:
        try:
            sock = _create_bound_socket(bind_host, port)
        except OSError as exc:
            bind_error = exc.__cause__ if isinstance(exc.__cause__, OSError) else exc
            if port <= 0 or not fallback_to_dynamic_port or bind_error.errno != errno.EADDRINUSE:
                raise
            logger.warning(
                "[RFork Seed] configured port %d on %s is in use; using an OS-assigned port", port, bind_host
            )
            sock = _create_bound_socket(bind_host, 0)
    except Exception as exc:
        raise RForkSeedServerStartupError("RFork seed server failed to start") from exc

    try:
        app = FastAPI()

        @app.get("/get_rfork_transfer_engine_info")
        def get_rfork_transfer_engine_info(seed_key: str):
            if seed_key != expected_seed_key:
                return Response(status_code=HTTPStatus.NOT_FOUND)
            return {
                "rfork_protocol_version": RFORK_PROTOCOL_VERSION,
                "rfork_transfer_engine_info": (info.session_id, info.weights),
                "rfork_transfer_engine_format_info": dict(info.formats) if info.formats else None,
            }

        @app.get("/health_check_with_key")
        def health_check_with_key(seed_key: str):
            return Response(status_code=HTTPStatus.OK if seed_key == expected_seed_key else HTTPStatus.BAD_REQUEST)

        info = rfork_transfer_engine_info
        expected_seed_key = seed_key
        server = uvicorn.Server(uvicorn.Config(app, host=None, port=None, log_level="warning"))
        handle = RForkSeedServerHandle(server, sock)
        url = _health_url(bind_host, sock, handle.port)
        handle.thread.start()
    except Exception as exc:
        sock.close()
        raise RForkSeedServerStartupError("RFork seed server failed to start") from exc

    deadline = time.monotonic() + float(health_timeout_sec)
    last_error: object | None = None
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0 or not handle.is_alive:
            break
        try:
            response = requests.get(url, params={"seed_key": seed_key}, timeout=min(10.0, remaining))
            if response.status_code == HTTPStatus.OK:
                return handle
            last_error = f"unexpected status code {response.status_code} from health check"
        except Exception as exc:
            last_error = str(exc)
        remaining = deadline - time.monotonic()
        if remaining > 0:
            time.sleep(min(HEALTH_POLL_INTERVAL_SEC, remaining))

    if not handle.stop():
        logger.warning("[RFork Seed] health check cleanup is incomplete; retaining the server handle for retry")
    raise RForkSeedServerStartupError(f"RFork seed server health check failed: {last_error}", handle=handle)
