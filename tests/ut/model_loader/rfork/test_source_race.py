# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import logging
import threading
from types import SimpleNamespace

import pytest
import torch

from .rfork_test_support import _load_module, _stub


@pytest.fixture
def race_module(monkeypatch):
    _stub(monkeypatch, "vllm", __path__=[])
    _stub(monkeypatch, "vllm.logger", logger=logging.getLogger("rfork-source-race-test"))
    for package in ("vllm_ascend", "vllm_ascend.model_loader", "vllm_ascend.model_loader.rfork"):
        _stub(monkeypatch, package, __path__=[])
    _load_module(monkeypatch, "vllm_ascend.model_loader.rfork.types", "types.py")
    return _load_module(monkeypatch, "rfork_source_race_test", "source_race.py")


class Clock:
    def __init__(self):
        self.now = 0.0
        self.on_wait = lambda: None
        self.waits = []

    def monotonic(self):
        return self.now

    def sleep(self, delay):
        self.waits.append(delay)
        self.now += delay
        self.on_wait()


class Mount:
    def __init__(self, clock, deadline=5.0):
        self.clock = clock
        self.deadline = deadline
        self.path = None
        self.error = None
        self.completed_at = None
        self.done = SimpleNamespace(is_set=lambda: self.completed_at is not None, wait=clock.sleep)

    def remaining(self):
        return max(0.0, self.deadline - self.clock.now)

    def complete(self, error=None):
        self.path = "/mounted" if error is None else None
        self.error = error
        self.completed_at = self.clock.now


class Session:
    def __init__(self, module, actions):
        self.states = module.RForkLifecycleState
        self.state = self.states.REGISTERED
        self.seed_lease = None
        self.identity = SimpleNamespace(tp_rank=0)
        self.actions = iter(actions)
        self.timeouts = []

    def acquire_seed(self, timeout_sec):
        self.timeouts.append(timeout_sec)
        action = next(self.actions, False)
        acquired = action() if callable(action) else action
        if acquired:
            self.seed_lease = object()
            self.state = self.states.LEASED
        return acquired


def coordinator(module):
    result = module.RForkLoadCoordinator.__new__(module.RForkLoadCoordinator)
    result.groups = []
    return result


def run_race(module, monkeypatch, session, mount, **kwargs):
    monkeypatch.setattr(module, "time", mount.clock)
    return module.race_sources(session, mount, coordinator(module), 1.0, **kwargs)


def test_mount_wins_without_seed(race_module, monkeypatch, caplog):
    clock = Clock()
    mount = Mount(clock)
    clock.on_wait = mount.complete
    session = Session(race_module, [False])
    with caplog.at_level(logging.INFO):
        assert run_race(race_module, monkeypatch, session, mount) == "mount"
    assert session.timeouts == [5.0]
    assert "Async model mount won RFork fallback race" in caplog.text


def test_ready_mount_does_not_require_seed_lookup(race_module, monkeypatch):
    clock = Clock()
    mount = Mount(clock)
    mount.complete()
    session = Session(race_module, [True])

    assert run_race(race_module, monkeypatch, session, mount) == "mount"
    assert session.timeouts == []


def test_late_seed_wins_without_mount(race_module, monkeypatch, caplog):
    clock = Clock()
    session = Session(race_module, [False, True])
    with caplog.at_level(logging.INFO):
        assert run_race(race_module, monkeypatch, session, Mount(clock)) == "seed"
    assert session.timeouts == [5.0, 4.0]
    assert "RFork seed retry won fallback race" in caplog.text


def test_seed_wins_when_mount_completes_during_request(race_module, monkeypatch):
    clock = Clock()
    mount = Mount(clock)

    def acquire():
        clock.now = 0.1
        mount.complete()
        clock.now = 0.2
        return True

    session = Session(race_module, [acquire])
    assert run_race(race_module, monkeypatch, session, mount) == "seed"
    assert session.seed_lease is not None


def test_simultaneous_ready_sources_prefer_seed(race_module, monkeypatch):
    clock = Clock()
    mount = Mount(clock)

    def acquire():
        mount.complete()
        return True

    session = Session(race_module, [acquire])
    assert run_race(race_module, monkeypatch, session, mount) == "seed"


@pytest.mark.parametrize("lookup", ["seed", "miss", "late_seed"])
def test_mount_required_for_preparation_allows_one_bounded_seed_lookup(race_module, monkeypatch, lookup):
    clock = Clock()
    mount = Mount(clock)
    mount.complete()
    clock.now = 0.5

    def acquire():
        if lookup == "late_seed":
            clock.now = mount.deadline + 1.0
        return lookup != "miss"

    session = Session(race_module, [acquire])
    assert run_race(race_module, monkeypatch, session, mount, mount_required_for_preparation=True) == (
        "seed" if lookup == "seed" else "mount"
    )
    assert session.timeouts == [4.5]
    assert clock.waits == []
    if lookup == "late_seed":
        assert session.seed_lease is not None  # Existing fallback cleanup releases the late lease.


def test_mount_required_for_preparation_does_not_restart_expired_budget(race_module, monkeypatch):
    clock = Clock()
    mount = Mount(clock, deadline=1.0)
    mount.complete()
    clock.now = 2.0
    session = Session(race_module, [True])
    assert run_race(race_module, monkeypatch, session, mount, mount_required_for_preparation=True) == "mount"
    assert session.timeouts == []


def test_mount_error_keeps_retrying_seed_without_busy_loop(race_module, monkeypatch):
    clock = Clock()
    mount = Mount(clock)
    mount.complete(OSError("mount failed"))
    session = Session(race_module, [False, True])
    assert run_race(race_module, monkeypatch, session, mount) == "seed"
    assert clock.waits == [1.0]


def test_unavailable_sources_share_one_deadline(race_module, monkeypatch):
    clock = Clock()
    mount = Mount(clock, deadline=2.5)
    mount.complete(OSError("mount failed"))
    session = Session(race_module, [False])
    with pytest.raises(TimeoutError, match="Timed out racing"):
        run_race(race_module, monkeypatch, session, mount)
    assert clock.now == 2.5
    assert session.timeouts == [2.5, 1.5, 0.5]


def test_seed_returned_after_deadline_is_not_selected(race_module, monkeypatch):
    clock = Clock()
    mount = Mount(clock, deadline=1.0)

    def acquire():
        clock.now = 2.0
        return True

    session = Session(race_module, [acquire])
    with pytest.raises(TimeoutError):
        run_race(race_module, monkeypatch, session, mount)
    assert session.seed_lease is not None


@pytest.mark.parametrize("state", ["CLEANUP_REQUIRED", "FINALIZED"])
def test_invalid_lifecycle_is_not_a_retryable_seed_miss(race_module, monkeypatch, state):
    clock = Clock()
    session = Session(race_module, [])
    session.state = getattr(session.states, state)
    with pytest.raises(RuntimeError, match="cannot be retried"):
        run_race(race_module, monkeypatch, session, Mount(clock))
    assert session.timeouts == []


def test_acquisition_exception_is_reported_before_retry(race_module, monkeypatch):
    clock = Clock()

    def acquire():
        raise ValueError("invalid lifecycle")

    session = Session(race_module, [acquire])
    with pytest.raises(RuntimeError, match="cannot be retried") as failure:
        run_race(race_module, monkeypatch, session, Mount(clock))
    assert isinstance(failure.value.__cause__, ValueError)
    assert clock.waits == []


@pytest.mark.parametrize("shutdown_in_progress", [False, True])
def test_session_stopped_during_acquisition_is_not_retried(race_module, monkeypatch, shutdown_in_progress):
    clock = Clock()
    session = Session(race_module, [])

    def stop_during_acquisition():
        if shutdown_in_progress:
            session._shutdown_in_progress = True
        else:
            session.lease_release_stop_event = threading.Event()
            session.lease_release_stop_event.set()
        return False

    session.actions = iter([stop_during_acquisition])
    with pytest.raises(RuntimeError, match="cannot be retried"):
        run_race(race_module, monkeypatch, session, Mount(clock))
    assert session.timeouts == [5.0]
    assert clock.waits == []


def test_group_selection_excludes_pp_for_draft(race_module, monkeypatch):
    groups = {name: SimpleNamespace(world_size=2, cpu_group=name) for name in ("tp", "pcp", "pp")}
    _stub(
        monkeypatch,
        "vllm.distributed.parallel_state",
        get_tp_group=lambda: groups["tp"],
        get_pcp_group=lambda: groups["pcp"],
        get_pp_group=lambda: groups["pp"],
        get_ep_group=lambda: None,
    )
    config = SimpleNamespace(parallel_config=SimpleNamespace(prefill_context_parallel_size=2, pipeline_parallel_size=2))
    assert race_module.RForkLoadCoordinator(config, is_draft=False).groups == ["tp", "pcp", "pp"]
    assert race_module.RForkLoadCoordinator(config, is_draft=True).groups == ["tp", "pcp"]


def test_group_selection_uses_ep_and_rejects_unmatched_draft_participants(race_module, monkeypatch):
    ep = SimpleNamespace(world_size=4, cpu_group="ep")
    tp = SimpleNamespace(world_size=2, cpu_group="tp")
    _stub(
        monkeypatch,
        "vllm.distributed.parallel_state",
        get_tp_group=lambda: tp,
        get_pcp_group=lambda: None,
        get_pp_group=lambda: None,
        get_ep_group=lambda: ep,
    )
    config = SimpleNamespace(parallel_config=SimpleNamespace(enable_expert_parallel=True, data_parallel_size=2))
    assert race_module.RForkLoadCoordinator(config, is_draft=True).groups == ["ep"]
    tp.world_size = 1
    with pytest.raises(ValueError, match="participants to match"):
        race_module.RForkLoadCoordinator(config, is_draft=True)


@pytest.mark.parametrize("dp_size,pcp_size", [(2, 1), (1, 2), (2, 2)])
@pytest.mark.parametrize("config_field", ["hf_text_config", "hf_config"])
def test_engram_group_coordinates_dp_and_pcp_without_ep(race_module, monkeypatch, dp_size, pcp_size, config_field):
    groups = {
        "tp": SimpleNamespace(world_size=2, cpu_group="tp"),
        "pcp": SimpleNamespace(world_size=pcp_size, cpu_group="pcp"),
        "pp": SimpleNamespace(world_size=2, cpu_group="pp"),
        "edp": SimpleNamespace(world_size=dp_size * pcp_size, cpu_group="edp"),
    }
    _stub(
        monkeypatch,
        "vllm.distributed.parallel_state",
        get_tp_group=lambda: groups["tp"],
        get_pcp_group=lambda: groups["pcp"],
        get_pp_group=lambda: groups["pp"],
        get_ep_group=lambda: pytest.fail("EP is disabled"),
    )
    getter_calls = []

    def existing_engram_group():
        getter_calls.append(True)
        return groups["edp"]

    _stub(monkeypatch, "vllm.distributed", get_engram_dp_group=existing_engram_group)
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(
            data_parallel_size=dp_size,
            prefill_context_parallel_size=pcp_size,
            pipeline_parallel_size=2,
            enable_expert_parallel=False,
        ),
        engram_config=object(),
        model_config=SimpleNamespace(**{config_field: SimpleNamespace(engram_layer_ids=[1])}),
    )
    participant_groups = ["tp", *(["pcp"] if pcp_size > 1 else []), "edp"]
    assert race_module.RForkLoadCoordinator(config, is_draft=False).groups == [*participant_groups, "pp"]
    assert race_module.RForkLoadCoordinator(config, is_draft=True).groups == participant_groups
    assert getter_calls == [True, True]


@pytest.mark.parametrize("engram_config,layer_ids", [(None, [1]), (object(), []), (object(), None)])
def test_inactive_engram_does_not_import_its_group_getter(race_module, monkeypatch, engram_config, layer_ids):
    _stub(
        monkeypatch,
        "vllm.distributed.parallel_state",
        get_tp_group=lambda: SimpleNamespace(world_size=2, cpu_group="tp"),
        get_pcp_group=lambda: None,
        get_pp_group=lambda: None,
        get_ep_group=lambda: None,
    )
    # The optional Engram getter is absent in runtimes without Engram support.
    _stub(monkeypatch, "vllm.distributed")
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(),
        engram_config=engram_config,
        model_config=SimpleNamespace(hf_text_config=SimpleNamespace(engram_layer_ids=layer_ids)),
    )
    assert race_module.RForkLoadCoordinator(config, is_draft=False).groups == ["tp"]


def test_engram_singleton_group_needs_no_additional_collective(race_module, monkeypatch):
    _stub(
        monkeypatch,
        "vllm.distributed.parallel_state",
        get_tp_group=lambda: SimpleNamespace(world_size=1, cpu_group="tp"),
        get_pcp_group=lambda: None,
        get_pp_group=lambda: None,
        get_ep_group=lambda: None,
    )
    _stub(monkeypatch, "vllm.distributed", get_engram_dp_group=lambda: None)
    config = SimpleNamespace(
        parallel_config=SimpleNamespace(),
        engram_config=object(),
        model_config=SimpleNamespace(hf_config=SimpleNamespace(engram_layer_ids=[1])),
    )
    assert race_module.RForkLoadCoordinator(config, is_draft=False).groups == []


class ThreadGroup:
    """Two CPU rank substitutes that execute the same blocking collective sequence."""

    def __init__(self):
        self.barrier = threading.Barrier(2)
        self.tensors = {}
        self.result = None

    def reduce(self, tensor, op, group):
        self.tensors[threading.current_thread().name] = tensor.clone()
        leader = self.barrier.wait(timeout=5) == 0
        if leader:
            stacked = torch.stack(list(self.tensors.values()))
            self.result = stacked.amin(0) if op is torch.distributed.ReduceOp.MIN else stacked.amax(0)
        self.barrier.wait(timeout=5)
        tensor.copy_(self.result)
        self.barrier.wait(timeout=5)


def run_ranks(callback, rank_count=2):
    results = [None] * rank_count

    def run(rank):
        try:
            results[rank] = callback(rank)
        except Exception as exc:
            results[rank] = exc

    threads = [
        threading.Thread(target=run, args=(rank,), name=f"rank-{rank}", daemon=True) for rank in range(rank_count)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join(timeout=10)
        assert not thread.is_alive(), "rank collective deadlocked"
    return results


@pytest.mark.parametrize("complete_mount", [False, True])
def test_ranks_retain_early_lease_until_all_seeds_are_ready(race_module, monkeypatch, complete_mount):
    group = ThreadGroup()
    monkeypatch.setattr(race_module.torch.distributed, "all_reduce", group.reduce)
    clocks = [Clock(), Clock()]
    mounts = [Mount(clock) for clock in clocks]
    monkeypatch.setattr(
        race_module,
        "time",
        SimpleNamespace(
            monotonic=lambda: clocks[int(threading.current_thread().name[-1])].monotonic(),
            sleep=lambda delay: clocks[int(threading.current_thread().name[-1])].sleep(delay),
        ),
    )

    def acquire(rank):
        if complete_mount:
            mounts[rank].complete()
            clocks[rank].now += 0.1
        return True

    sessions = [Session(race_module, [lambda: acquire(0)]), Session(race_module, [False, lambda: acquire(1)])]

    def run(rank):
        coord = coordinator(race_module)
        coord.groups = [group]
        return race_module.race_sources(sessions[rank], mounts[rank], coord, 1.0)

    assert run_ranks(run) == ["seed", "seed"]
    assert len(sessions[0].timeouts) == 1
    assert len(sessions[1].timeouts) == 2


@pytest.mark.parametrize("outcome", ["mount", "timeout", "invalid_lifecycle"])
def test_partial_rank_readiness_and_failure_are_coordinated(race_module, monkeypatch, outcome):
    group = ThreadGroup()
    monkeypatch.setattr(race_module.torch.distributed, "all_reduce", group.reduce)
    clocks = [Clock(), Clock()]
    mounts = [Mount(clocks[0], deadline=1.0), Mount(clocks[1])]
    sessions = [Session(race_module, [False]), Session(race_module, [False])]
    if outcome == "mount":
        mounts[0].complete()
        clocks[1].on_wait = mounts[1].complete
    elif outcome == "invalid_lifecycle":
        sessions[0].state = sessions[0].states.CLEANUP_REQUIRED
    monkeypatch.setattr(
        race_module,
        "time",
        SimpleNamespace(
            monotonic=lambda: clocks[int(threading.current_thread().name[-1])].monotonic(),
            sleep=lambda delay: clocks[int(threading.current_thread().name[-1])].sleep(delay),
        ),
    )

    def run(rank):
        coord = coordinator(race_module)
        coord.groups = [group]
        return race_module.race_sources(sessions[rank], mounts[rank], coord, 1.0)

    results = run_ranks(run)
    if outcome == "mount":
        assert results == ["mount", "mount"]
        assert clocks[0].waits == [1.0]
    else:
        error_type = TimeoutError if outcome == "timeout" else RuntimeError
        assert all(isinstance(result, error_type) for result in results)


@pytest.mark.parametrize("phase_error", [False, True])
def test_tp_pp_grid_propagates_readiness_and_errors_to_every_rank(race_module, monkeypatch, phase_error):
    tp_groups = [ThreadGroup(), ThreadGroup()]
    pp_groups = [ThreadGroup(), ThreadGroup()]
    monkeypatch.setattr(
        race_module.torch.distributed,
        "all_reduce",
        lambda tensor, op, group: group.reduce(tensor, op, group),
    )
    clocks = [Clock() for _ in range(4)]
    sessions = [Session(race_module, [True]) for _ in range(3)] + [Session(race_module, [False, True])]
    monkeypatch.setattr(
        race_module,
        "time",
        SimpleNamespace(
            monotonic=lambda: clocks[int(threading.current_thread().name[-1])].monotonic(),
            sleep=lambda delay: clocks[int(threading.current_thread().name[-1])].sleep(delay),
        ),
    )

    def run(rank):
        coord = coordinator(race_module)
        coord.groups = [tp_groups[rank // 2], pp_groups[rank % 2]]
        if phase_error:
            with coord.phase("transfer"):
                if rank == 3:
                    raise ValueError("one shard transfer failed")
        else:
            return race_module.race_sources(sessions[rank], Mount(clocks[rank]), coord, 1.0)

    results = run_ranks(run, rank_count=4)
    if phase_error:
        assert all(isinstance(result, RuntimeError) for result in results)
        assert isinstance(results[3].__cause__, ValueError)
    else:
        assert results == ["seed"] * 4
        assert [len(session.timeouts) for session in sessions] == [1, 1, 1, 2]


def test_phase_failure_reaches_every_rank(race_module, monkeypatch):
    group = ThreadGroup()
    monkeypatch.setattr(race_module.torch.distributed, "all_reduce", group.reduce)

    def run(rank):
        coord = coordinator(race_module)
        coord.groups = [group]
        with coord.phase("transfer"):
            if rank == 0:
                raise ValueError("transfer failed")

    results = run_ranks(run)
    assert all(isinstance(result, RuntimeError) for result in results)
    assert isinstance(results[0].__cause__, ValueError)
    assert results[1].__cause__ is None


def test_identity_disagreement_reaches_every_rank(race_module, monkeypatch):
    group = ThreadGroup()
    monkeypatch.setattr(race_module.torch.distributed, "all_reduce", group.reduce)

    def run(rank):
        coord = coordinator(race_module)
        coord.groups = [group]
        coord.check_identity(f"checkpoint-{rank}")

    assert all(isinstance(result, ValueError) for result in run_ranks(run))
