import logging

from src.utils.tracking import (
    MLflowTracker,
    _flatten_params,
    build_run_metadata,
    get_git_revision,
)


def test_get_git_revision_returns_sha_in_repo():
    rev = get_git_revision()
    assert set(rev.keys()) == {"git_sha", "git_dirty"}
    assert len(rev["git_sha"]) in (0, 40)


def test_get_git_revision_no_repo(tmp_path):
    rev = get_git_revision(cwd=tmp_path)
    assert rev == {"git_sha": "", "git_dirty": ""}


def test_flatten_params_nested_dict():
    out: dict = {}
    _flatten_params("", {"a": 1, "b": {"c": 2, "d": [3, 4]}}, out)
    assert out == {"a": 1, "b.c": 2, "b.d": "3,4"}


def test_tracker_disabled_by_default():
    tracker = MLflowTracker.from_config({}, logger=logging.getLogger("test"))
    assert tracker.enabled is False
    tracker.start(tags={"x": "y"})
    tracker.log_params({"a": 1})
    tracker.log_metrics({"loss": 0.5}, step=1)
    tracker.end()


def test_tracker_disabled_when_flag_false():
    cfg = {"tracking": {"enabled": False, "experiment_name": "ignored"}}
    tracker = MLflowTracker.from_config(cfg, logger=logging.getLogger("test"))
    assert tracker.enabled is False


def test_tracker_context_manager_no_op_when_disabled():
    with MLflowTracker.from_config({}, logger=logging.getLogger("test")) as tracker:
        tracker.log_metrics({"loss": 1.0})
    assert tracker._active_run is None


def test_build_run_metadata_has_expected_keys():
    meta = build_run_metadata({"experiment_name": "unit_test", "seed": 42})
    expected = {
        "timestamp_utc",
        "experiment_name",
        "seed",
        "hostname",
        "python_version",
        "platform",
        "torch_version",
        "git_sha",
        "git_dirty",
    }
    assert set(meta.keys()) == expected
    assert meta["experiment_name"] == "unit_test"
    assert meta["seed"] == 42
    assert meta["python_version"].count(".") >= 1


def test_build_run_metadata_outside_repo(tmp_path):
    meta = build_run_metadata({"experiment_name": "x", "seed": 0}, cwd=tmp_path)
    assert meta["git_sha"] == ""
    assert meta["git_dirty"] == ""
