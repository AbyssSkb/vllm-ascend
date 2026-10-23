# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import pytest
import torch
from torch.nn import Module

from .rfork_test_support import _load_module, _stub


@pytest.fixture
def mounted_loader(monkeypatch, tmp_path, rfork_helpers):
    """Exercise the real loader/race with one CPU rank and a deterministic mount clock."""
    helpers = rfork_helpers
    module = helpers.loader
    monkeypatch.setattr(module.envs, "VLLM_ASCEND_ASYNC_MODEL_MOUNT", True)
    monkeypatch.setattr(module.envs, "RFORK_MODEL_IDENTITY_HEADERS", "0", raising=False)
    types = _load_module(monkeypatch, "vllm_ascend.model_loader.rfork.types", "types.py")
    config = _load_module(monkeypatch, "vllm_ascend.model_loader.rfork.config", "config.py")
    mount_module = _load_module(monkeypatch, "vllm_ascend.model_loader.async_mount", "../async_mount.py")
    race = _load_module(monkeypatch, "vllm_ascend.model_loader.rfork.source_race", "source_race.py")
    monkeypatch.setattr(module, "RForkConfig", config.RForkConfig)
    monkeypatch.setattr(module, "RForkLifecycleState", types.RForkLifecycleState)
    group = SimpleNamespace(world_size=1, cpu_group=None)
    for name in ("get_tp_group", "get_pcp_group"):
        monkeypatch.setattr("vllm.distributed.parallel_state." + name, lambda: group, raising=False)
    monkeypatch.setattr("vllm.distributed.get_engram_dp_group", lambda: group, raising=False)

    full_path = tmp_path / "mounted"
    full_path.mkdir()
    bootstrap = tmp_path / "bootstrap"
    bootstrap.mkdir()
    context = SimpleNamespace(
        module=module,
        race=race,
        types=types,
        helpers=helpers,
        full_path=str(full_path),
        bootstrap=str(bootstrap),
        mount_at=2.0,
        mount_error=None,
        acquire_outcomes=[],
        acquire_hook=None,
        transfer_hook=None,
        transfer_ok=True,
        cleanup_ok=True,
        bypass=None,
        initialization_error=None,
        local_weights_error=None,
        release_error=None,
        readiness_error=None,
        reuse_shared=False,
        exclude_blocks=[],
        events=[],
        defaults=[],
        initializations=[],
        model_factory=Module,
        mount=None,
        resolve=None,
    )

    class Clock:
        now = 0.0

        def advance(self, delay, *, until=None):
            end = self.now + delay
            if context.mount_at is not None and context.resolve is not None and self.now <= context.mount_at <= end:
                self.now = context.mount_at
                resolve, context.resolve = context.resolve, None
                resolve()
                if until is not None and until.is_set():
                    return
            self.now = end

    clock = Clock()
    context.clock = clock
    clock_module = SimpleNamespace(monotonic=lambda: clock.now, perf_counter=lambda: clock.now, sleep=clock.advance)
    for owner in (module, mount_module, race):
        monkeypatch.setattr(owner, "time", clock_module)

    class Event:
        ready = False

        def is_set(self):
            return self.ready

        def set(self):
            self.ready = True

        def wait(self, timeout):
            if not self.ready:
                clock.advance(timeout, until=self)
            return self.ready

    class Thread:
        def __init__(self, *, target, **kwargs):
            context.mount = target.__self__
            context.resolve = target

        def start(self):
            clock.advance(0)

    monkeypatch.setattr(mount_module, "threading", SimpleNamespace(Event=Event, Thread=Thread))
    _stub(monkeypatch, "model_manager", __path__=[])

    def resolve_path(*, with_weights):
        assert with_weights is True
        context.events.append("mount")
        if context.mount_error is not None:
            raise context.mount_error
        return context.full_path

    _stub(monkeypatch, "model_manager.apis", get_model_path_from_manager=resolve_path)

    class Session:
        identity = types.RForkIdentity(0, 0, compatibility_fingerprint="logical-checkpoint")
        seed_lease = None
        state = types.RForkLifecycleState.INITIALIZED

        def can_reuse_shared_weights(self, *args):
            context.events.append("shared")
            return context.reuse_shared

        def register_destination(self, *args):
            context.events.append("register")
            self.state = types.RForkLifecycleState.REGISTERED
            return True

        def acquire_seed(self, *, timeout_sec):
            context.events.append("acquire")
            assert 0 < timeout_sec <= context.mount.deadline - clock.now
            if context.acquire_hook is not None:
                context.acquire_hook()
            acquired = context.acquire_outcomes.pop(0) if context.acquire_outcomes else False
            if acquired:
                self.seed_lease = object()
                self.state = types.RForkLifecycleState.LEASED
            return acquired

        def transfer_from_seed(self, *args):
            context.events.append("transfer")
            if context.transfer_hook is not None:
                context.transfer_hook()
            return context.transfer_ok

        def log_transferred_model_layout(self, *args):
            return "transferred-digest"

        def prepare_for_fallback(self):
            context.events.append("cleanup")
            self.seed_lease = None
            self.state = types.RForkLifecycleState.INITIALIZED
            return types.RForkFallbackCleanupResult(True, True, context.cleanup_ok)

    session = Session()
    context.session = session
    load_config = SimpleNamespace(
        device=None,
        load_format="rfork",
        model_loader_extra_config={
            "model_url": "logical-model",
            "model_deploy_strategy_name": "deployment",
            "rfork_seed_retry_interval_sec": 1.0,
            "rfork_source_wait_timeout_sec": 10.0,
        },
    )
    loader = module.RForkModelLoader(load_config)
    context.loader = loader
    monkeypatch.setattr(loader, "_ensure_rfork_session", lambda *args: session)
    monkeypatch.setattr(loader, "_get_target_registered_blocks", lambda *args: context.exclude_blocks)
    monkeypatch.setattr(loader, "_requires_processed_layout_transfer", lambda *args: False)
    monkeypatch.setattr(module, "mutable_weights_bypass_reason", lambda *args: context.bypass)
    monkeypatch.setattr(module, "_snapshot_process_global_model_state", lambda *args: None)
    monkeypatch.setattr(module, "_reset_process_global_model_state", lambda *args: context.events.append("reset"))
    monkeypatch.setattr(module, "_rfork_skip_unquantized_moe_post_load_processing", lambda *args: nullcontext())
    monkeypatch.setattr(module, "process_weights_after_loading", lambda *args: context.events.append("post_load"))
    monkeypatch.setattr(module, "_start_rfork_seed_service", lambda *args, **kwargs: context.events.append("publish"))
    monkeypatch.setattr(
        module,
        "get_ascend_config",
        lambda: SimpleNamespace(eplb_config=SimpleNamespace(dynamic_eplb=False, expert_map_record_path=None)),
    )
    monkeypatch.setattr(torch, "npu", SimpleNamespace(empty_cache=lambda: None), raising=False)

    def initialize(**kwargs):
        context.events.append("initialize")
        initialization_config = kwargs["vllm_config"]
        initialization_model = kwargs["model_config"]
        needs_mount = getattr(context.vllm_config, "engram_config", None) is not None and any(
            bool(getattr(getattr(context.model_config, name, None), "engram_layer_ids", None))
            for name in ("hf_text_config", "hf_config")
        )
        expected_path = context.full_path if needs_mount else context.bootstrap
        assert initialization_config.model_config.model == initialization_model.model == expected_path
        if needs_mount:
            assert initialization_config.model_config is initialization_model
            assert context.mount.done.is_set()
        context.initializations.append((initialization_config, initialization_model))
        if context.initialization_error is not None:
            raise context.initialization_error
        return context.model_factory()

    def default_load(vllm_config, model_config, fallback_load_config, prefix):
        context.events.append("default")
        context.defaults.append((vllm_config, model_config, fallback_load_config, prefix))
        return Module()

    monkeypatch.setattr(module, "initialize_model", initialize)
    monkeypatch.setattr(module, "_load_with_default_loader", default_load)

    def local_weights(model):
        context.events.append("local_weights")
        if context.local_weights_error is not None:
            raise context.local_weights_error

    def release_weights(model):
        context.events.append("release_local")
        if context.release_error is not None:
            raise context.release_error

    def record_weights(model):
        context.events.append("local_ready")
        if context.readiness_error is not None:
            raise context.readiness_error

    monkeypatch.setattr(module, "load_local_weights", local_weights)
    monkeypatch.setattr(module, "release_local_weights", release_weights)
    monkeypatch.setattr(module, "record_loaded_local_weights", record_weights)
    context.model_config = SimpleNamespace(
        model=context.bootstrap,
        tokenizer="bootstrap-tokenizer",
        revision="checkpoint-revision",
        served_model_name="logical-model",
        dtype=torch.float32,
    )
    context.vllm_config = SimpleNamespace(
        model_config=context.model_config,
        load_config=load_config,
        device_config=SimpleNamespace(device="cpu"),
        parallel_config=SimpleNamespace(enable_expert_parallel=False, pipeline_parallel_size=1),
    )
    context.load = lambda: loader.load_model(context.vllm_config, context.model_config)
    return context


def _fail_remote_phase(context, monkeypatch, name, *, unsafe_initialization=False):
    coordinator = context.race.RForkLoadCoordinator
    original_phase, original_flags = coordinator.phase, coordinator.flags
    active_phase = None
    failure_reported = False

    @contextmanager
    def phase(self, phase_name):
        nonlocal active_phase
        active_phase = phase_name
        try:
            with original_phase(self, phase_name):
                yield
        finally:
            active_phase = None

    def flags(self, values, *, all_ranks=True):
        nonlocal failure_reported
        if active_phase == name:
            failure_reported = True
            return [False]
        if failure_reported and active_phase is None and not all_ranks and unsafe_initialization:
            return [True]
        return original_flags(self, values, all_ranks=all_ranks)

    monkeypatch.setattr(coordinator, "phase", phase)
    monkeypatch.setattr(coordinator, "flags", flags)


def test_mount_winner_uses_local_config_and_preserves_logical_identity(mounted_loader):
    context = mounted_loader

    assert isinstance(context.load(), Module)

    mounted_config, mounted_model, load_config, _ = context.defaults[0]
    assert mounted_config is not context.vllm_config
    assert mounted_model is not context.model_config
    assert mounted_config.model_config is mounted_model
    assert mounted_model.model == context.full_path
    assert context.vllm_config.model_config is context.model_config
    assert context.model_config.model == context.bootstrap
    assert mounted_model.tokenizer == context.model_config.tokenizer == "bootstrap-tokenizer"
    assert mounted_model.revision == context.model_config.revision == "checkpoint-revision"
    assert mounted_model.served_model_name == "logical-model"
    assert context.loader.rfork_config.model_url == "logical-model"
    assert context.session.identity.compatibility_fingerprint == "logical-checkpoint"
    assert load_config.load_format == "rfork"
    assert context.events.index("cleanup") < context.events.index("default")
    assert "transfer" not in context.events


def test_late_seed_wins_without_waiting_for_mount(mounted_loader):
    context = mounted_loader
    context.mount_at = None
    context.acquire_outcomes = [False, True]

    assert isinstance(context.load(), Module)

    assert context.events.count("acquire") == 2
    assert context.events.count("transfer") == 1
    assert context.defaults == []
    assert not context.mount.done.is_set()
    assert context.clock.now == 1.0


@pytest.mark.parametrize("on_time", [False, True])
def test_mount_completion_during_seed_request_prefers_timely_seed(mounted_loader, on_time):
    context = mounted_loader
    context.mount_at = 0.5
    context.acquire_outcomes = [True]
    context.acquire_hook = lambda: context.clock.advance(1.0 if on_time else 11.0)

    context.load()

    if on_time:
        assert context.events.count("transfer") == 1
        assert "cleanup" not in context.events
        assert context.defaults == []
    else:
        assert "transfer" not in context.events
        assert context.events.count("cleanup") == 1
        assert context.session.seed_lease is None
        assert len(context.defaults) == 1
        assert context.defaults[0][1].model == context.full_path


@pytest.mark.parametrize("mount_at, should_load", [(8.0, True), (11.0, False)])
def test_transfer_failure_reuses_original_mount_deadline(mounted_loader, mount_at, should_load):
    context = mounted_loader
    context.mount_at = mount_at
    context.acquire_outcomes = [True]
    context.transfer_ok = False
    context.transfer_hook = lambda: context.clock.advance(7.0)

    if should_load:
        context.load()
        assert len(context.defaults) == 1
        assert context.clock.now == 8.0
    else:
        with pytest.raises(RuntimeError, match="mounted fallback directory") as failure:
            context.load()
        assert isinstance(failure.value.__cause__, TimeoutError)
        assert context.defaults == []
        assert context.clock.now == 10.0
    assert context.mount.deadline == 10.0
    assert context.events.count("transfer") == 1


def test_mount_exception_does_not_prevent_seed_transfer(mounted_loader):
    context = mounted_loader
    context.mount_at = 0.0
    context.mount_error = OSError("mount failed")
    context.acquire_outcomes = [False, True]

    context.load()

    assert context.events.count("acquire") == 2
    assert "transfer" in context.events
    assert context.defaults == []
    assert context.clock.now == 1.0


@pytest.mark.parametrize("manager_error", [None, OSError("mount failed")])
def test_unavailable_sources_never_load_bootstrap_weights(mounted_loader, manager_error):
    context = mounted_loader
    context.mount_at = None if manager_error is None else 0.0
    context.mount_error = manager_error

    with pytest.raises(RuntimeError, match="mounted fallback directory"):
        context.load()

    assert context.defaults == []
    assert "transfer" not in context.events
    assert context.clock.now == 10.0


def test_failed_memory_cleanup_prevents_second_model_allocation(mounted_loader):
    context = mounted_loader
    context.mount_at = 0.0
    context.cleanup_ok = False

    with pytest.raises(RuntimeError, match="fallback memory safety") as failure:
        context.load()

    assert "Refusing to allocate a second model" in str(failure.value.__cause__)
    assert context.events.count("cleanup") == context.module.FALLBACK_CLEANUP_MAX_ATTEMPTS
    assert "release_local" not in context.events
    assert context.defaults == []


def test_bypass_waits_for_mounted_weights(mounted_loader):
    context = mounted_loader
    context.bypass = "sleep mode"

    context.load()

    assert context.events == ["mount", "default"]
    assert context.defaults[0][1].model == context.full_path
    assert context.model_config.model == context.bootstrap


def test_shutdown_during_source_race_prevents_fallback_allocation(mounted_loader):
    context = mounted_loader
    context.acquire_hook = lambda: setattr(context.session, "_shutdown_in_progress", True)

    with pytest.raises(RuntimeError, match="fallback memory safety") as failure:
        context.load()

    assert "shutting down" in str(failure.value.__cause__)
    assert context.defaults == []


def test_rank_identity_mismatch_aborts_before_model_initialization(mounted_loader, monkeypatch):
    context = mounted_loader

    def mismatch(self, identity):
        raise ValueError("ranks disagree on the logical model identity")

    monkeypatch.setattr(context.race.RForkLoadCoordinator, "check_identity", mismatch)

    with pytest.raises(ValueError, match="logical model identity"):
        context.load()

    assert "initialize" not in context.events
    assert context.defaults == []


@pytest.mark.parametrize("seed_available", [False, True])
def test_mtp_checkpoint_headers_use_mounted_directory_before_seed_lookup(mounted_loader, monkeypatch, seed_available):
    context = mounted_loader
    observed_paths = []
    context.acquire_outcomes = [seed_available]

    class Model(context.helpers.mtp_module.AscendDeepSeekMTP):
        quant_config = None

        def _maybe_set_own_lm_head(self, names):
            assert names == {"model.layers.3.shared_head.head.weight"}
            context.events.append("head_prepared")

    def checkpoint_names(model, model_config, load_config):
        observed_paths.append(model_config.model)
        return {"model.layers.3.shared_head.head.weight"}

    context.model_factory = Model
    monkeypatch.setattr(context.module, "_get_checkpoint_weight_names", checkpoint_names)

    context.load()

    assert observed_paths == [context.full_path]
    assert context.events.index("mount") < context.events.index("head_prepared") < context.events.index("register")
    assert context.events.index("register") < context.events.index("acquire")
    assert context.events.count("acquire") == 1
    assert ("transfer" in context.events) is seed_available
    assert len(context.defaults) == (0 if seed_available else 1)
    assert context.model_config.model == context.bootstrap


@pytest.mark.parametrize("model_kind", ["engram", "mtp"])
def test_mounted_preparation_does_not_restart_expired_seed_budget(mounted_loader, monkeypatch, model_kind):
    context = mounted_loader
    context.acquire_outcomes = [True]
    if model_kind == "engram":
        context.model_config.hf_text_config = SimpleNamespace(engram_layer_ids=[1])
        context.vllm_config.engram_config = SimpleNamespace(cpu_offload=False)
    else:

        class Model(context.helpers.mtp_module.AscendDeepSeekMTP):
            quant_config = None

            def _maybe_set_own_lm_head(self, names):
                pass

        context.model_factory = Model
        monkeypatch.setattr(context.module, "_get_checkpoint_weight_names", lambda *args: {"head.weight"})
    monkeypatch.setattr(context.module, "load_local_weights", lambda model: context.clock.advance(8.0))

    context.load()

    assert context.mount.deadline == context.clock.now == 10.0
    assert "acquire" not in context.events
    assert "transfer" not in context.events
    assert context.defaults[0][1].model == context.full_path


def test_other_mtp_class_can_transfer_before_mount(mounted_loader, monkeypatch):
    context = mounted_loader

    class DeepSeekV4MTP(Module):
        def _maybe_set_own_lm_head(self, names):
            pytest.fail("The distinct V4 class must not use AscendDeepSeekMTP header preparation")

    context.model_factory = DeepSeekV4MTP
    context.mount_at = None
    context.acquire_outcomes = [True]
    monkeypatch.setattr(
        context.module,
        "_get_checkpoint_weight_names",
        lambda *args: pytest.fail("V4 transfer must not require mounted checkpoint headers"),
    )

    context.load()

    assert "transfer" in context.events
    assert not context.mount.done.is_set()
    assert context.defaults == []


def test_main_and_same_checkpoint_draft_reuse_mount_task_and_deadline(mounted_loader):
    context = mounted_loader
    draft_config = SimpleNamespace(**vars(context.model_config), runner_type="draft")
    context.vllm_config.speculative_config = SimpleNamespace(
        target_model_config=context.model_config,
        draft_model_config=draft_config,
    )
    context.acquire_outcomes = [True]
    context.load()
    mount = context.vllm_config.load_config.async_model_mount
    assert not mount.done.is_set()

    context.clock.advance(2.0)
    context.session.seed_lease = None
    context.loader.load_model(context.vllm_config, draft_config)

    assert context.vllm_config.load_config.async_model_mount is mount
    assert context.mount is mount
    assert mount.deadline == 10.0
    assert context.events.count("mount") == 1
    assert context.defaults[0][1].model == context.full_path
    assert draft_config.model == context.bootstrap


@pytest.mark.parametrize("cpu_offload", [False, True])
def test_engram_construction_waits_for_mounted_checkpoint(mounted_loader, cpu_offload):
    context = mounted_loader
    context.model_config.hf_text_config = SimpleNamespace(engram_layer_ids=[1, 3])
    context.vllm_config.engram_config = SimpleNamespace(cpu_offload=cpu_offload)
    context.acquire_outcomes = [True]

    context.load()

    initialization_config, initialization_model = context.initializations[0]
    assert context.events.index("mount") < context.events.index("initialize")
    assert initialization_config is not context.vllm_config
    assert initialization_model is not context.model_config
    assert initialization_config.model_config is initialization_model
    assert initialization_model.model == context.full_path
    assert initialization_model.tokenizer == "bootstrap-tokenizer"
    assert context.vllm_config.model_config is context.model_config
    assert context.model_config.model == context.bootstrap
    assert context.model_config.tokenizer == "bootstrap-tokenizer"
    assert context.loader.rfork_config.model_url == "logical-model"
    assert context.session.identity.compatibility_fingerprint == "logical-checkpoint"
    assert "transfer" in context.events
    assert context.defaults == []


def test_shared_engram_config_without_layers_does_not_wait_for_mount(mounted_loader):
    context = mounted_loader
    context.model_config.hf_text_config = SimpleNamespace(engram_layer_ids=[])
    context.vllm_config.engram_config = SimpleNamespace(cpu_offload=True)
    context.mount_at = None
    context.acquire_outcomes = [True]

    context.load()

    assert context.initializations[0] == (context.vllm_config, context.model_config)
    assert not context.mount.done.is_set()
    assert "transfer" in context.events
    assert context.defaults == []


@pytest.mark.parametrize("reuse_shared", [False, True])
def test_local_weights_precede_shared_reuse_and_registration(mounted_loader, reuse_shared):
    context = mounted_loader
    context.model_config.hf_text_config = SimpleNamespace(engram_layer_ids=[1])
    context.vllm_config.engram_config = SimpleNamespace(cpu_offload=True)
    context.exclude_blocks = [(1, 2)]
    context.reuse_shared = reuse_shared
    context.acquire_outcomes = [True]

    context.load()

    assert context.events.index("local_weights") < context.events.index("shared")
    if reuse_shared:
        assert "register" not in context.events
        assert context.defaults == []
    else:
        assert context.events.index("local_weights") < context.events.index("register")


@pytest.mark.parametrize("release_fails", [False, True])
def test_local_weights_release_after_mr_cleanup_before_fallback(mounted_loader, release_fails):
    context = mounted_loader
    context.mount_at = 0.0
    if release_fails:
        context.release_error = RuntimeError("host buffer release failed")
        with pytest.raises(RuntimeError, match="discarded model memory"):
            context.load()
        assert context.defaults == []
    else:
        context.load()
        assert context.events.index("release_local") < context.events.index("default")
        assert context.events.index("default") < context.events.index("local_ready") < context.events.index("publish")
    assert context.events.index("cleanup") < context.events.index("release_local")


def test_failed_local_weights_are_released_before_mounted_fallback(mounted_loader):
    context = mounted_loader
    context.mount_at = 0.0
    context.local_weights_error = OSError("local checkpoint read failed")
    context.exclude_blocks = [(1, 2)]

    context.load()

    assert "register" not in context.events
    assert "shared" not in context.events
    assert context.events.index("local_weights") < context.events.index("cleanup")
    assert context.events.index("cleanup") < context.events.index("release_local") < context.events.index("default")
    assert len(context.defaults) == 1


@pytest.mark.parametrize("failed_rank", ["local", "remote"])
def test_partial_engram_initialization_aborts_all_ranks(mounted_loader, monkeypatch, failed_rank):
    context = mounted_loader
    context.model_config.hf_text_config = SimpleNamespace(engram_layer_ids=[1])
    context.vllm_config.engram_config = SimpleNamespace(cpu_offload=True)
    if failed_rank == "local":
        context.initialization_error = RuntimeError("partial host allocation")
    else:
        _fail_remote_phase(context, monkeypatch, "model initialization", unsafe_initialization=True)

    with pytest.raises(RuntimeError, match="cannot safely retry failed model construction"):
        context.load()

    assert len(context.initializations) == 1
    assert "cleanup" not in context.events
    assert "release_local" not in context.events
    assert context.defaults == []


def test_engram_mount_timeout_precedes_initialization_without_partial_allocation_error(mounted_loader):
    context = mounted_loader
    context.model_config.hf_text_config = SimpleNamespace(engram_layer_ids=[1])
    context.vllm_config.engram_config = SimpleNamespace(cpu_offload=True)
    context.mount_at = None

    with pytest.raises(RuntimeError, match="mounted fallback directory") as failure:
        context.load()

    assert isinstance(failure.value.__cause__, TimeoutError)
    assert "partial host allocations" not in str(failure.value)
    assert context.initializations == []
    assert "initialize" not in context.events
    assert "release_local" not in context.events
    assert context.defaults == []


@pytest.mark.parametrize("failed_rank", ["local", "remote"])
def test_fallback_local_weights_readiness_failure_propagates_to_all_ranks(mounted_loader, monkeypatch, failed_rank):
    context = mounted_loader
    context.mount_at = 0.0
    if failed_rank == "local":
        context.readiness_error = RuntimeError("incomplete local weights")
    else:
        _fail_remote_phase(context, monkeypatch, "fallback local weights readiness")

    with pytest.raises(RuntimeError, match="fallback local weights readiness"):
        context.load()

    assert len(context.defaults) == 1
    assert "local_ready" in context.events
    assert "publish" not in context.events


def test_remote_transfer_failure_makes_successful_rank_fall_back(mounted_loader, monkeypatch):
    context = mounted_loader
    _fail_remote_phase(context, monkeypatch, "seed transfer")
    context.acquire_outcomes = [True]

    context.load()

    assert context.events.count("transfer") == 1
    assert "post_load" not in context.events
    assert len(context.defaults) == 1
    assert context.events.index("cleanup") < context.events.index("default")
