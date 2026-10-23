# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import threading
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from .test_legacy_planner import _LegacyPlannerAdapter, _make_legacy_session, _wait_for_release


def test_source_wait_config_defaults(runtime):
    config = type(runtime.config).from_extra_config({"rfork_model_identity_headers": False})

    assert config.source_wait_timeout_sec == 600.0
    assert config.seed_retry_interval_sec == 1.0


def test_source_wait_config_uses_explicit_positive_values(runtime):
    config = type(runtime.config).from_extra_config(
        {
            "rfork_model_identity_headers": False,
            "rfork_source_wait_timeout_sec": 42,
            "rfork_seed_retry_interval_sec": 0.25,
        }
    )

    assert config.source_wait_timeout_sec == 42
    assert config.seed_retry_interval_sec == 0.25


@pytest.mark.parametrize("name", ["rfork_source_wait_timeout_sec", "rfork_seed_retry_interval_sec"])
@pytest.mark.parametrize("value", [True, "1", 0, -1, float("nan"), float("inf")])
def test_source_wait_config_rejects_invalid_values(runtime, name, value):
    with pytest.raises(ValueError, match=f"{name} must be a finite positive JSON number"):
        type(runtime.config).from_extra_config({"rfork_model_identity_headers": False, name: value})


@pytest.mark.parametrize(("timeout_sec", "expected"), [(None, 0.1), (0.025, 0.025), (1.0, 0.1)])
def test_planner_seed_timeout_is_bounded_without_changing_other_requests(runtime, monkeypatch, timeout_sec, expected):
    get = Mock(return_value=SimpleNamespace(status_code=404))
    monkeypatch.setattr(runtime.client.requests, "get", get)
    client = runtime.client.RForkPlannerClient(runtime.config, runtime.identity)
    client.bind_structural_digest("digest")

    assert client.acquire_seed(timeout_sec=timeout_sec) is None

    get.assert_called_once_with(
        "http://planner/get_seed",
        headers={"SEED_KEY": client.seed_key},
        timeout=expected,
        allow_redirects=False,
    )
    assert client.request_timeout_sec == runtime.config.request_timeout_sec == 0.1


@pytest.mark.parametrize("timeout_sec", [True, False, 0, -1, float("nan"), float("inf"), -float("inf"), "1"])
def test_planner_rejects_invalid_seed_timeout_before_request(runtime, monkeypatch, timeout_sec):
    get = Mock()
    monkeypatch.setattr(runtime.client.requests, "get", get)
    client = runtime.client.RForkPlannerClient(runtime.config, runtime.identity)

    with pytest.raises(ValueError, match="finite positive number"):
        client.acquire_seed(timeout_sec=timeout_sec)

    get.assert_not_called()


@pytest.mark.parametrize("timeout_sec", [None, 0.025])
def test_session_passes_seed_timeout_only_when_requested(runtime, timeout_sec):
    session = runtime.session.RForkSession(runtime.config, runtime.identity)
    session.state = runtime.types.RForkLifecycleState.REGISTERED
    session.planner = Mock()
    session.planner.acquire_seed.return_value = None
    session.planner.take_rejected_lease.return_value = None

    assert not session.acquire_seed(timeout_sec=timeout_sec)

    if timeout_sec is None:
        session.planner.acquire_seed.assert_called_once_with()
    else:
        session.planner.acquire_seed.assert_called_once_with(timeout_sec=timeout_sec)
    assert session.state is runtime.types.RForkLifecycleState.REGISTERED


@pytest.mark.parametrize("timeout_sec", [True, False, 0, -1, float("nan"), float("inf"), -float("inf"), "1"])
def test_session_rejects_invalid_seed_timeout_before_lifecycle_changes(runtime, timeout_sec):
    session = runtime.session.RForkSession(runtime.config, runtime.identity)
    session.planner = Mock()
    session.seed_lease = runtime.lease

    with pytest.raises(ValueError, match="finite positive number"):
        session.acquire_seed(timeout_sec=timeout_sec)

    assert session.planner.mock_calls == []
    assert session.seed_lease is runtime.lease
    assert session.state is runtime.types.RForkLifecycleState.INITIALIZED


@pytest.mark.parametrize("initial_state", ["LEASED", "CLEANUP_REQUIRED", "TRANSFERRED"])
@pytest.mark.parametrize("raises", [False, True])
def test_unconfirmed_renewal_invalidates_only_a_waiting_lease(runtime, initial_state, raises):
    session = runtime.session.RForkSession(runtime.config, runtime.identity)
    session.state = getattr(runtime.types.RForkLifecycleState, initial_state)
    session.seed_lease = runtime.lease
    session.planner = Mock()
    session.planner.renew_seed_once.return_value = False
    if raises:
        session.planner.renew_seed_once.side_effect = RuntimeError("renewal failed")
    session.lease_renew_thread = threading.current_thread()
    stop_event = Mock()
    stop_event.wait.side_effect = [False, True]

    session._renew_seed_lease(runtime.lease, stop_event)

    expected_state = "CLEANUP_REQUIRED" if initial_state == "LEASED" else initial_state
    assert session.state is getattr(runtime.types.RForkLifecycleState, expected_state)
    assert session.seed_lease is runtime.lease
    assert session.lease_renew_thread is None
    assert stop_event.wait.call_count == 1
    session.planner.renew_seed_once.assert_called_once_with(runtime.lease)
    session.planner.release_seed_once.assert_not_called()
    session.transfer_backend.unregister_memory_region.assert_not_called()


def test_failed_renewal_does_not_invalidate_a_replacement_lease(runtime):
    session = runtime.session.RForkSession(runtime.config, runtime.identity)
    session.state = runtime.types.RForkLifecycleState.LEASED
    session.seed_lease = replace(runtime.lease, user_id="replacement-lease")
    session.planner = Mock()
    session.planner.renew_seed_once.return_value = False
    stop_event = Mock()
    stop_event.wait.side_effect = [False, True]

    session._renew_seed_lease(runtime.lease, stop_event)

    assert session.state is runtime.types.RForkLifecycleState.LEASED
    assert session.seed_lease.user_id == "replacement-lease"


def test_bounded_acquisition_finds_a_late_legacy_seed_without_renewal(runtime, monkeypatch):
    planner = _LegacyPlannerAdapter()
    session, _ = _make_legacy_session(runtime, monkeypatch, planner)
    planner.advertisements.clear()

    assert not session.acquire_seed(timeout_sec=0.025)
    assert session.seed_lease is None
    assert session.state is runtime.types.RForkLifecycleState.REGISTERED

    planner.publish(session.planner.seed_key)
    assert session.acquire_seed(timeout_sec=0.025)
    assert session.seed_lease.lease_ttl_sec is None
    assert session.lease_renew_thread is None
    planner.advance(601)
    assert session.state is runtime.types.RForkLifecycleState.LEASED
    assert all(options["timeout"] == 0.025 for options in planner.request_options)
    assert not any(path == "/renew_seed_lease" for _, path in planner.calls)

    session.prepare_for_fallback()
    _wait_for_release(session)
    assert planner.active_count == 0


def test_bounded_rejected_allocation_is_released_by_session_without_reallocation(runtime, monkeypatch):
    entered, resume = threading.Event(), threading.Event()
    planner = _LegacyPlannerAdapter(
        lease_ttl_header="invalid",
        put_entered=entered,
        put_gate=resume,
        block_put_on_call=1,
    )
    session, backend = _make_legacy_session(runtime, monkeypatch, planner)

    assert session.planner.acquire_seed(timeout_sec=0.025) is None
    assert session.planner.acquire_seed(timeout_sec=0.025) is None
    assert planner.calls == [("GET", "/get_seed")]
    assert planner.active_count == 1

    assert not session.acquire_seed(timeout_sec=0.025)
    release_worker = session.lease_release_thread
    try:
        assert release_worker is not None
        assert entered.wait(2)
        assert session.seed_lease is not None
        assert session.planner.take_rejected_lease() is None
        assert not session.acquire_seed(timeout_sec=0.025)
        assert not session.transfer_from_seed(object(), True)
        backend.read_weights_from_seed.assert_not_called()
        assert planner.calls == [("GET", "/get_seed"), ("POST", "/put_seed")]
        assert planner.request_options[0]["timeout"] == 0.025
        assert planner.request_options[1]["timeout"] == runtime.config.request_timeout_sec
    finally:
        resume.set()
        _wait_for_release(session)

    assert session.seed_lease is None
    assert planner.active_count == 0
    session.prepare_for_fallback()
