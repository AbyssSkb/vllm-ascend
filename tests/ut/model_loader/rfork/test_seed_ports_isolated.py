# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

from tests.ut.model_loader.rfork.test_late_lease_isolated import loader_runtime as loader_runtime
from tests.ut.model_loader.rfork.test_lease_release import runtime as runtime


@pytest.fixture
def port_runtime(request):
    return request.getfixturevalue("runtime")


@pytest.mark.parametrize("value", [True, False, -1, 65536, 1.5, 3.0, float("nan"), float("inf"), None, "18000", [], {}])
def test_seed_port_base_requires_json_integer(port_runtime, value):
    with pytest.raises(ValueError, match="rfork_seed_port_base"):
        type(port_runtime.config).from_extra_config({"rfork_seed_port_base": value})
    with pytest.raises(ValueError, match="rfork_seed_port_base"):
        replace(port_runtime.config, seed_port_base=value)


@pytest.mark.parametrize("port", [0, 18000, 65535])
def test_seed_port_base_accepts_range(port_runtime, port):
    config = type(port_runtime.config).from_extra_config({"rfork_seed_port_base": port})
    assert config.seed_port_base == port


def test_seed_port_base_defaults_to_dynamic(port_runtime, monkeypatch):
    monkeypatch.setenv("RFORK_SEED_PORT_BASE", "18000")
    assert type(port_runtime.config).from_extra_config({}).seed_port_base == 0


def _start_seed_service(port_runtime, monkeypatch, base, rank, is_draft):
    config = replace(port_runtime.config, seed_port_base=base)
    identity = replace(port_runtime.identity, global_rank=rank, is_draft_model=is_draft)
    session = port_runtime.session.RForkSession(config, identity)
    session.planner = Mock(seed_key="model-key")
    session.state = port_runtime.types.RForkLifecycleState.READY
    monkeypatch.setattr(session, "_seed_transfer_info", Mock())
    start_server = Mock(return_value=SimpleNamespace(port=32123))
    monkeypatch.setattr(port_runtime.session, "start_rfork_server", start_server)
    monkeypatch.setattr(port_runtime.session.threading, "Thread", Mock())
    assert session.start_seed_service(object(), True)
    # Advertise the actual bound port, including after a dynamic assignment.
    session.planner.report_seed_once.assert_called_once_with(32123, seed_ip=config.seed_advertise_host)
    return start_server.call_args.kwargs["port"]


def test_seed_ports_reserve_unique_main_and_draft_slots_per_rank(port_runtime, monkeypatch):
    ports = [
        _start_seed_service(port_runtime, monkeypatch, 18000, rank, draft)
        for rank in range(3)
        for draft in (False, True)
    ]
    assert ports == [18000, 18001, 18002, 18003, 18004, 18005]


@pytest.mark.parametrize("is_moe", [False, True], ids=["dense", "moe"])
def test_loader_seed_ports_include_dp_index_once(request, port_runtime, monkeypatch, is_moe):
    r = request.getfixturevalue("loader_runtime")
    ports = []
    monkeypatch.setattr(torch.distributed, "get_rank", lambda: r.world_group.rank_in_group)
    monkeypatch.setattr(port_runtime.session.threading, "Thread", Mock())
    start_server = Mock(side_effect=lambda *args, **kwargs: SimpleNamespace(port=kwargs["port"]))
    monkeypatch.setattr(port_runtime.session, "start_rfork_server", start_server)
    for dp_rank in range(2):
        for tp_rank in range(2):
            # Dense engines reset world rank per replica; MoE's world spans DP.
            r.world_group.rank_in_group = dp_rank * 2 + tp_rank if is_moe else tp_rank
            monkeypatch.setattr(r.module, "get_tensor_model_parallel_rank", Mock(return_value=tp_rank))
            parallel_config = SimpleNamespace(
                data_parallel_index=dp_rank,
                data_parallel_rank=dp_rank if is_moe else 0,
                data_parallel_size=2 if is_moe else 1,
                tensor_parallel_size=2,
                pipeline_parallel_size=1,
                prefill_context_parallel_size=1,
                enable_expert_parallel=False,
                is_moe_model=is_moe,
            )
            loader = r.module.RForkModelLoader(
                SimpleNamespace(
                    model_loader_extra_config={
                        "rfork_seed_port_base": 18000,
                        "rfork_scheduler_url": "http://planner",
                        "model_url": "model",
                        "model_deploy_strategy_name": "strategy",
                    }
                )
            )
            for draft in (False, True):
                model_config = SimpleNamespace(**vars(r.config), runner_type="draft" if draft else "generate")
                vc = SimpleNamespace(model_config=model_config, parallel_config=parallel_config)
                session = loader._ensure_rfork_session(vc, model_config)
                session.planner = Mock(seed_key="model-key")
                session.state = port_runtime.types.RForkLifecycleState.READY
                monkeypatch.setattr(session, "_seed_transfer_info", Mock())
                assert session.start_seed_service(object(), True)
                ports.append(start_server.call_args.kwargs["port"])
    assert ports == list(range(18000, 18008))


@pytest.mark.parametrize("draft", [False, True])
def test_seed_port_zero_stays_dynamic_at_every_rank(port_runtime, monkeypatch, draft):
    assert _start_seed_service(port_runtime, monkeypatch, 0, 17, draft) == 0


@pytest.mark.parametrize(
    ("base", "rank", "draft", "expected"),
    [
        (65535, 0, False, 65535),
        (65534, 0, True, 65535),
        (65533, 1, False, 65535),
        (65535, 0, True, 0),
        (65534, 1, False, 0),
    ],
)
def test_seed_port_overflow_falls_back_to_dynamic(port_runtime, monkeypatch, base, rank, draft, expected):
    assert _start_seed_service(port_runtime, monkeypatch, base, rank, draft) == expected
