#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Perform a functional test for client helper functions."""
import os

import pytest

import orion.core.cli
from orion.core.worker.consumer import Consumer


def test_interrupt(storage, monkeypatch, capsys):
    """Test interruption from within user script."""
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))

    user_args = ["-x~uniform(-50, 50, precision=5)"]

    error_code = orion.core.cli.main(
        [
            "hunt",
            "--config",
            "./orion_config.yaml",
            "--exp-max-trials",
            "1",
            "--worker-trials",
            "1",
            "python",
            "black_box.py",
            "interrupt_trial",
        ]
        + user_args
    )

    assert error_code == 130

    captured = capsys.readouterr()
    assert captured.out == "Orion is interrupted.\n"
    assert captured.err == ""

    exp = list(storage.fetch_experiments({"name": "voila_voici"}))
    exp = exp[0]
    exp_id = exp["_id"]
    trials = list(storage.fetch_trials(uid=exp_id))
    assert len(trials) == 1
    assert trials[0].status == "interrupted"


def test_interrupt_diff_code(storage, monkeypatch, capsys):
    """Test interruption from within user script with custom int code"""
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))

    user_args = ["-x~uniform(-50, 50, precision=5)"]

    # Set local to 200
    orion.core.config.worker.interrupt_signal_code = 200

    # But child won't be passed ORION_INTERRUPT_CODE and therefore will send default code 130
    def empty_env(self, trial, results_file=None):
        return os.environ

    with monkeypatch.context() as m:
        m.setattr(Consumer, "get_execution_environment", empty_env)

        # Interrupt won't be interpreted properly and trials will be marked as broken
        error_code = orion.core.cli.main(
            [
                "hunt",
                "--config",
                "./orion_config.yaml",
                "--exp-max-trials",
                "2",
                "--worker-trials",
                "2",
                "python",
                "black_box.py",
                "interrupt_trial",
            ]
            + user_args
        )

        assert error_code == 0

        exp = list(storage.fetch_experiments({"name": "voila_voici"}))
        exp = exp[0]
        exp_id = exp["_id"]
        trials = list(storage.fetch_trials(uid=exp_id))
        assert len(trials) == 2
        assert trials[0].status == "broken"

    # This time we use true `get_execution_environment which pass properly int code to child.
    error_code = orion.core.cli.main(
        [
            "hunt",
            "--config",
            "./orion_config.yaml",
            "--exp-max-trial",
            "1",
            "--worker-trials",
            "1",
            "python",
            "black_box.py",
            "interrupt_trial",
        ]
        + user_args
    )

    assert error_code == 130

    captured = capsys.readouterr()
    assert "Orion is interrupted.\n" in captured.out
    assert captured.err == ""

    exp = list(storage.fetch_experiments({"name": "voila_voici"}))
    exp = exp[0]
    exp_id = exp["_id"]
    trials = list(storage.fetch_trials(uid=exp_id))
    assert len(trials) == 3
    assert trials[-1].status == "interrupted"


# TODO:

# test no call to any report

# Add all this in DOC.


@pytest.mark.parametrize("fct", ["report_bad_trial", "report_objective"])
def test_report_no_name(storage, monkeypatch, fct):
    """Test report helper functions with default names"""
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))

    user_args = ["-x~uniform(-50, 50, precision=5)"]

    orion.core.cli.main(
        [
            "hunt",
            "--config",
            "./orion_config.yaml",
            "--exp-max-trials",
            "2",
            "--worker-trials",
            "2",
            "python",
            "black_box.py",
            fct,
            "--objective",
            "1.0",
        ]
        + user_args
    )

    exp = list(storage.fetch_experiments({"name": "voila_voici"}))
    exp = exp[0]
    exp_id = exp["_id"]
    trials = list(storage.fetch_trials(uid=exp_id))
    assert len(trials) == 2
    assert trials[0].status == "completed"
    assert trials[0].results[0].name == "objective"
    assert trials[0].results[0].type == "objective"
    assert trials[0].results[0].value == 1.0


@pytest.mark.parametrize("fct", ["report_bad_trial", "report_objective"])
def test_report_with_name(storage, monkeypatch, fct):
    """Test report helper functions with custom names"""
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))

    user_args = ["-x~uniform(-50, 50, precision=5)"]

    orion.core.cli.main(
        [
            "hunt",
            "--config",
            "./orion_config.yaml",
            "--exp-max-trials",
            "2",
            "--worker-trials",
            "2",
            "python",
            "black_box.py",
            fct,
            "--objective",
            "1.0",
            "--name",
            "metric",
        ]
        + user_args
    )

    exp = list(storage.fetch_experiments({"name": "voila_voici"}))
    exp = exp[0]
    exp_id = exp["_id"]
    trials = list(storage.fetch_trials(uid=exp_id))
    assert len(trials) == 2
    assert trials[0].status == "completed"
    assert trials[0].results[0].name == "metric"
    assert trials[0].results[0].type == "objective"
    assert trials[0].results[0].value == 1.0


@pytest.mark.parametrize("fct", ["report_bad_trial", "report_objective"])
def test_report_with_bad_objective(storage, monkeypatch, fct):
    """Test report helper functions with bad objective types"""
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))

    user_args = ["-x~uniform(-50, 50, precision=5)"]

    with pytest.raises(ValueError) as exc:
        orion.core.cli.main(
            [
                "hunt",
                "--config",
                "./orion_config.yaml",
                "--exp-max-trials",
                "2",
                "--worker-trials",
                "2",
                "python",
                "black_box.py",
                fct,
                "--objective",
                "oh oh",
            ]
            + user_args
        )

    assert "must contain a type `objective` with type float/int" in str(exc.value)


def test_report_with_bad_trial_no_objective(storage, monkeypatch):
    """Test bad trial report helper function with default objective."""
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))

    user_args = ["-x~uniform(-50, 50, precision=5)"]

    orion.core.cli.main(
        [
            "hunt",
            "--config",
            "./orion_config.yaml",
            "--exp-max-trials",
            "2",
            "--worker-trials",
            "2",
            "python",
            "black_box.py",
            "report_bad_trial",
        ]
        + user_args
    )

    exp = list(storage.fetch_experiments({"name": "voila_voici"}))
    exp = exp[0]
    exp_id = exp["_id"]
    trials = list(storage.fetch_trials(uid=exp_id))
    assert len(trials) == 2
    assert trials[0].status == "completed"
    assert trials[0].results[0].name == "objective"
    assert trials[0].results[0].type == "objective"
    assert trials[0].results[0].value == 1e10


def test_report_with_bad_trial_with_data(storage, monkeypatch):
    """Test bad trial report helper function with additional data."""
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))

    user_args = ["-x~uniform(-50, 50, precision=5)"]

    orion.core.cli.main(
        [
            "hunt",
            "--config",
            "./orion_config.yaml",
            "--exp-max-trials",
            "2",
            "--worker-trials",
            "2",
            "python",
            "black_box.py",
            "report_bad_trial",
            "--data",
            "another",
        ]
        + user_args
    )

    exp = list(storage.fetch_experiments({"name": "voila_voici"}))
    exp = exp[0]
    exp_id = exp["_id"]
    trials = list(storage.fetch_trials(uid=exp_id))
    assert len(trials) == 2
    assert trials[0].status == "completed"
    assert trials[0].results[0].name == "objective"
    assert trials[0].results[0].type == "objective"
    assert trials[0].results[0].value == 1e10

    assert trials[0].results[1].name == "another"
    assert trials[0].results[1].type == "constraint"
    assert trials[0].results[1].value == 1.0


def test_no_report(storage, monkeypatch, capsys):
    """Test script call without any results reported."""
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))

    user_args = ["-x~uniform(-50, 50, precision=5)"]

    errorcode = orion.core.cli.main(
        [
            "hunt",
            "--config",
            "./orion_config.yaml",
            "--exp-max-trials",
            "2",
            "--worker-trials",
            "2",
            "python",
            "black_box.py",
            "no_report",
        ]
        + user_args
    )

    assert errorcode == 1

    captured = capsys.readouterr()
    assert captured.out == ""
    assert "Cannot parse result file" in captured.err
