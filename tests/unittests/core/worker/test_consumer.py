#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Collection of tests for :mod:`orion.core.worker.consumer`."""
import os
import signal
import subprocess
import tempfile
import time

import pytest

import orion.core.io.experiment_builder as experiment_builder
import orion.core.io.resolve_config as resolve_config
import orion.core.utils.backward as backward
import orion.core.worker.consumer as consumer
from orion.core.utils.exceptions import BranchingEvent, MissingResultFile
from orion.core.utils.format_trials import tuple_to_trial

Consumer = consumer.Consumer


@pytest.fixture
def config(exp_config):
    """Return a configuration."""
    config = exp_config[0][0]
    config["metadata"]["user_args"] = ["--x~uniform(-50, 50)"]
    config["metadata"]["VCS"] = resolve_config.infer_versioning_metadata(
        config["metadata"]["user_script"]
    )
    config["name"] = "exp"
    config["working_dir"] = "/tmp/orion"
    backward.populate_space(config)
    config["space"] = config["metadata"]["priors"]
    return config


@pytest.mark.usefixtures("storage")
def test_trials_interrupted_sigterm(config, monkeypatch):
    """Check if a trial is set as interrupted when a signal is raised."""

    def mock_popen(self, *args, **kwargs):
        os.kill(os.getpid(), signal.SIGTERM)

    exp = experiment_builder.build(**config)

    monkeypatch.setattr(subprocess.Popen, "wait", mock_popen)

    trial = tuple_to_trial((1.0,), exp.space)

    con = Consumer(exp)

    with pytest.raises(KeyboardInterrupt):
        con(trial)


@pytest.mark.usefixtures("storage")
def test_trial_working_dir_is_changed(config):
    """Check that trial has its working_dir attribute changed."""
    exp = experiment_builder.build(**config)

    trial = tuple_to_trial((1.0,), exp.space)

    exp.register_trial(trial, status="reserved")

    con = Consumer(exp)
    con(trial)

    assert trial.working_dir is not None
    assert trial.working_dir == con.working_dir + "/exp_" + trial.id


@pytest.mark.usefixtures("storage")
def test_code_changed(config, monkeypatch):
    """Check that trial has its working_dir attribute changed."""
    exp = experiment_builder.build(**config)

    trial = tuple_to_trial((1.0,), exp.space)

    exp.register_trial(trial, status="reserved")

    con = Consumer(exp)

    def code_changed(user_script):
        return dict(
            type="git",
            is_dirty=True,
            HEAD_sha="changed",
            active_branch="new_branch",
            diff_sha="new_diff",
        )

    monkeypatch.setattr(consumer, "infer_versioning_metadata", code_changed)

    with pytest.raises(BranchingEvent) as exc:
        con(trial)

    assert exc.match("Code changed between execution of 2 trials")


@pytest.mark.usefixtures("storage")
def test_retrieve_result_nofile(config):
    """Test retrieve result"""
    results_file = tempfile.NamedTemporaryFile(
        mode="w", prefix="results_", suffix=".log", dir=".", delete=True
    )

    exp = experiment_builder.build(**config)

    con = Consumer(exp)

    with pytest.raises(MissingResultFile) as exec:
        con.retrieve_results(results_file)

    results_file.close()

    assert exec.match(r"Cannot parse result file")
