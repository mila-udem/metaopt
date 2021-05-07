#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Perform a functional test for demo purposes."""
import os
import shutil
import subprocess
import tempfile
from collections import defaultdict

import numpy
import pytest
import yaml

import orion.core.cli
import orion.core.io.experiment_builder as experiment_builder
from orion.core.cli.hunt import workon
from orion.core.io.database.ephemeraldb import EphemeralDB
from orion.storage.base import get_storage
from orion.storage.legacy import Legacy
from orion.testing import OrionState


def test_demo_with_default_algo_cli_config_only(storage, monkeypatch):
    """Check that random algorithm is used, when no algo is chosen explicitly."""
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))

    orion.core.cli.main(
        [
            "hunt",
            "-n",
            "default_algo",
            "--max-trials",
            "10",
            "./black_box.py",
            "-x~uniform(-50, 50)",
        ]
    )

    exp = list(storage.fetch_experiments({"name": "default_algo"}))
    assert len(exp) == 1
    exp = exp[0]
    assert "_id" in exp
    assert exp["name"] == "default_algo"
    assert exp["pool_size"] == 1
    assert exp["max_trials"] == 10
    assert exp["max_broken"] == 3
    assert exp["algorithms"] == {"random": {"seed": None}}
    assert "user" in exp["metadata"]
    assert "datetime" in exp["metadata"]
    assert "orion_version" in exp["metadata"]
    assert "user_script" in exp["metadata"]
    assert exp["metadata"]["user_args"] == ["./black_box.py", "-x~uniform(-50, 50)"]

    trials = list(storage.fetch_trials(uid=exp["_id"]))
    assert len(trials) <= 10
    assert trials[-1].status == "completed"


def test_demo(storage, monkeypatch):
    """Test a simple usage scenario."""
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))

    user_args = [
        "./black_box.py",
        "-x~uniform(-50, 50, precision=10)",
        "--test-env",
        "--experiment-id",
        "{exp.id}",
        "--experiment-name",
        "{exp.name}",
        "--experiment-version",
        "{exp.version}",
        "--trial-id",
        "{trial.id}",
        "--working-dir",
        "{trial.working_dir}",
    ]

    orion.core.cli.main(["hunt", "--config", "./orion_config.yaml"] + user_args)

    exp = list(storage.fetch_experiments({"name": "voila_voici"}))
    assert len(exp) == 1
    exp = exp[0]
    assert "_id" in exp
    exp_id = exp["_id"]
    assert exp["name"] == "voila_voici"
    assert exp["pool_size"] == 1
    assert exp["max_trials"] == 20
    assert exp["max_broken"] == 5
    assert exp["algorithms"] == {
        "gradient_descent": {"learning_rate": 0.1, "dx_tolerance": 1e-5}
    }
    assert "user" in exp["metadata"]
    assert "datetime" in exp["metadata"]
    assert "orion_version" in exp["metadata"]
    assert "user_script" in exp["metadata"]
    assert exp["metadata"]["user_args"] == user_args
    trials = list(storage.fetch_trials(uid=exp_id))
    assert len(trials) <= 15
    assert trials[-1].status == "completed"
    trials = list(sorted(trials, key=lambda trial: trial.submit_time))
    for result in trials[-1].results:
        assert result.type != "constraint"
        if result.type == "objective":
            assert abs(result.value - 23.4) < 1e-6
            assert result.name == "example_objective"
        elif result.type == "gradient":
            res = numpy.asarray(result.value)
            assert 0.1 * numpy.sqrt(res.dot(res)) < 1e-4
            assert result.name == "example_gradient"
    params = trials[-1].params
    assert len(params) == 1
    px = params["/x"]
    assert isinstance(px, float)
    assert (px - 34.56789) < 1e-4


def test_demo_with_script_config(storage, monkeypatch):
    """Test a simple usage scenario."""
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))
    orion.core.cli.main(
        [
            "hunt",
            "--config",
            "./orion_config.yaml",
            "./black_box_w_config.py",
            "--config",
            "script_config.yaml",
        ]
    )

    exp = list(storage.fetch_experiments({"name": "voila_voici"}))
    assert len(exp) == 1
    exp = exp[0]
    assert "_id" in exp
    exp_id = exp["_id"]
    assert exp["name"] == "voila_voici"
    assert exp["pool_size"] == 1
    assert exp["max_trials"] == 20
    assert exp["max_broken"] == 5
    assert exp["algorithms"] == {
        "gradient_descent": {"learning_rate": 0.1, "dx_tolerance": 1e-5}
    }
    assert "user" in exp["metadata"]
    assert "datetime" in exp["metadata"]
    assert "orion_version" in exp["metadata"]
    assert "user_script" in exp["metadata"]
    assert exp["metadata"]["user_args"] == [
        "./black_box_w_config.py",
        "--config",
        "script_config.yaml",
    ]

    trials = list(storage.fetch_trials(uid=exp_id))
    assert len(trials) <= 15
    assert trials[-1].status == "completed"
    trials = list(sorted(trials, key=lambda trial: trial.submit_time))
    for result in trials[-1].results:
        assert result.type != "constraint"
        if result.type == "objective":
            assert abs(result.value - 23.4) < 1e-6
            assert result.name == "example_objective"
        elif result.type == "gradient":
            res = numpy.asarray(result.value)
            assert 0.1 * numpy.sqrt(res.dot(res)) < 1e-4
            assert result.name == "example_gradient"
    params = trials[-1].params
    assert len(params) == 1
    px = params["/x"]
    assert isinstance(px, float)
    assert (px - 34.56789) < 1e-4


def test_demo_with_python_and_script(storage, monkeypatch):
    """Test a simple usage scenario."""
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))
    orion.core.cli.main(
        [
            "hunt",
            "--config",
            "./orion_config.yaml",
            "python",
            "black_box_w_config.py",
            "--config",
            "script_config.yaml",
        ]
    )

    exp = list(storage.fetch_experiments({"name": "voila_voici"}))
    assert len(exp) == 1
    exp = exp[0]
    assert "_id" in exp
    exp_id = exp["_id"]
    assert exp["name"] == "voila_voici"
    assert exp["pool_size"] == 1
    assert exp["max_trials"] == 20
    assert exp["max_broken"] == 5
    assert exp["algorithms"] == {
        "gradient_descent": {"learning_rate": 0.1, "dx_tolerance": 1e-5}
    }
    assert "user" in exp["metadata"]
    assert "datetime" in exp["metadata"]
    assert "orion_version" in exp["metadata"]
    assert "user_script" in exp["metadata"]
    assert exp["metadata"]["user_args"] == [
        "python",
        "black_box_w_config.py",
        "--config",
        "script_config.yaml",
    ]

    trials = list(storage.fetch_trials(uid=exp_id))
    assert len(trials) <= 15
    assert trials[-1].status == "completed"
    trials = list(sorted(trials, key=lambda trial: trial.submit_time))
    for result in trials[-1].results:
        assert result.type != "constraint"
        if result.type == "objective":
            assert abs(result.value - 23.4) < 1e-6
            assert result.name == "example_objective"
        elif result.type == "gradient":
            res = numpy.asarray(result.value)
            assert 0.1 * numpy.sqrt(res.dot(res)) < 1e-4
            assert result.name == "example_gradient"
    params = trials[-1].params
    assert len(params) == 1
    px = params["/x"]
    assert isinstance(px, float)
    assert (px - 34.56789) < 1e-4


def test_demo_inexecutable_script(storage, monkeypatch, capsys):
    """Test error message when user script is not executable."""
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))
    script = tempfile.NamedTemporaryFile()
    orion.core.cli.main(
        [
            "hunt",
            "--config",
            "./orion_config.yaml",
            script.name,
            "--config",
            "script_config.yaml",
        ]
    )

    captured = capsys.readouterr().err
    assert "User script is not executable" in captured


def test_demo_two_workers(storage, monkeypatch):
    """Test a simple usage scenario."""
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))
    processes = []
    for _ in range(2):
        process = subprocess.Popen(
            [
                "orion",
                "hunt",
                "-n",
                "two_workers_demo",
                "--config",
                "./orion_config_random.yaml",
                "--max-trials",
                "20",
                "./black_box.py",
                "-x~norm(34, 3)",
            ]
        )
        processes.append(process)

    for process in processes:
        rcode = process.wait()
        assert rcode == 0

    exp = list(storage.fetch_experiments({"name": "two_workers_demo"}))
    assert len(exp) == 1
    exp = exp[0]
    assert "_id" in exp
    exp_id = exp["_id"]
    assert exp["name"] == "two_workers_demo"
    assert exp["pool_size"] == 2
    assert exp["max_trials"] == 20
    assert exp["max_broken"] == 5
    assert exp["algorithms"] == {"random": {"seed": None}}
    assert "user" in exp["metadata"]
    assert "datetime" in exp["metadata"]
    assert "orion_version" in exp["metadata"]
    assert "user_script" in exp["metadata"]
    assert exp["metadata"]["user_args"] == ["./black_box.py", "-x~norm(34, 3)"]

    trials = list(storage.fetch_trials(uid=exp_id))
    status = defaultdict(int)
    for trial in trials:
        status[trial.status] += 1
    assert 20 <= status["completed"] <= 21
    assert status["new"] < 5
    params = trials[-1].params
    assert len(params) == 1
    px = params["/x"]
    assert isinstance(px, float)


def test_workon():
    """Test scenario having a configured experiment already setup."""
    name = "voici_voila"
    config = {"name": name}
    config["algorithms"] = {"gradient_descent": {"learning_rate": 0.1}}
    config["pool_size"] = 1
    config["max_trials"] = 20
    config["exp_max_broken"] = 5
    config["user_args"] = [
        os.path.abspath(os.path.join(os.path.dirname(__file__), "black_box.py")),
        "-x~uniform(-50, 50, precision=None)",
    ]

    with OrionState():
        experiment = experiment_builder.build_from_args(config)

        workon(experiment, 20, 20, 20, 20, 20)

        storage = get_storage()

        exp = list(storage.fetch_experiments({"name": name}))
        assert len(exp) == 1
        exp = exp[0]
        assert "_id" in exp
        assert exp["name"] == name
        assert exp["pool_size"] == 1
        assert exp["max_trials"] == 20
        assert exp["max_broken"] == 5
        assert exp["algorithms"] == {
            "gradient_descent": {"learning_rate": 0.1, "dx_tolerance": 1e-5}
        }
        assert "user" in exp["metadata"]
        assert "datetime" in exp["metadata"]
        assert "user_script" in exp["metadata"]
        assert exp["metadata"]["user_args"] == config["user_args"]

        trials = list(storage.fetch_trials(experiment))
        assert len(trials) <= 15
        trials = list(sorted(trials, key=lambda trial: trial.submit_time))
        assert trials[-1].status == "completed"
        for result in trials[-1].results:
            assert result.type != "constraint"
            if result.type == "objective":
                assert abs(result.value - 23.4) < 1e-6
                assert result.name == "example_objective"
            elif result.type == "gradient":
                res = numpy.asarray(result.value)
                assert 0.1 * numpy.sqrt(res.dot(res)) < 1e-4
                assert result.name == "example_gradient"
        params = trials[-1].params
        assert len(params) == 1
        px = params["/x"]
        assert isinstance(px, float)
        assert (px - 34.56789) < 1e-4


def test_stress_unique_folder_creation(storage, monkeypatch, tmpdir, capfd):
    """Test integration with a possible framework that needs to create
    unique directories per trial.
    """
    # XXX: return and complete test when there is a way to control random
    # seed of Oríon
    how_many = 2
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))
    orion.core.cli.main(
        [
            "hunt",
            "--max-trials={}".format(how_many),
            "--pool-size=1",
            "--name=lalala",
            "--config",
            "./stress_gradient.yaml",
            "./dir_per_trial.py",
            "--dir={}".format(str(tmpdir)),
            "--other-name",
            "{exp.name}",
            "--name",
            "{trial.hash_name}",
            "-x~gaussian(30, 10)",
        ]
    )

    exp = list(storage.fetch_experiments({"name": "lalala"}))
    assert len(exp) == 1
    exp = exp[0]
    assert "_id" in exp
    exp_id = exp["_id"]

    # For contingent broken trials, which in this test means that a existing
    # directory was attempted to be created, it means that it's not md5 or
    # bad hash creation to blame, but the finite precision of the floating
    # point representation. Specifically, it seems that gradient descent
    # is able to reach such levels of precision jumping around the minimum
    # (notice that an appropriate learning rate was selected in this stress
    # test to create underdamped behaviour), that it begins to suggest same
    # things from the past. This is intended to be shown with the assertions
    # in the for-loop below.
    trials_c = list(storage.fetch_trials(uid=exp_id, where={"status": "completed"}))
    list_of_cx = [trial.params["/x"] for trial in trials_c]
    trials_b = list(storage.fetch_trials(uid=exp_id, where={"status": "broken"}))
    list_of_bx = [trial.params["/x"] for trial in trials_b]
    for bx in list_of_bx:
        assert bx in list_of_cx

    # ``exp.name`` has been delivered correctly (next 2 assertions)
    assert len(os.listdir(str(tmpdir))) == 1
    # Also, because of the way the demo gradient descent works `how_many` trials
    # can be completed
    assert len(os.listdir(str(tmpdir.join("lalala")))) == how_many
    assert len(trials_c) == how_many
    capfd.readouterr()  # Suppress fd level 1 & 2


def test_working_dir_argument_cmdline(storage, monkeypatch, tmp_path):
    """Check that a permanent directory is used instead of tmpdir"""
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))
    path = str(tmp_path) + "/test"
    assert not os.path.exists(path)
    orion.core.cli.main(
        [
            "hunt",
            "-n",
            "allo",
            "--working-dir",
            path,
            "--max-trials",
            "2",
            "--config",
            "./database_config.yaml",
            "./black_box.py",
            "-x~uniform(-50,50)",
        ]
    )

    exp = list(storage.fetch_experiments({"name": "allo"}))[0]
    assert exp["working_dir"] == path
    assert os.path.exists(path)
    assert os.listdir(path)

    shutil.rmtree(path)


def test_tmpdir_is_deleted(storage, monkeypatch, tmp_path):
    """Check that temporary directory is deletid tmpdir"""
    tmp_path = os.path.join(tempfile.gettempdir(), "orion")
    if os.path.exists(tmp_path):
        shutil.rmtree(tmp_path)

    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))
    orion.core.cli.main(
        [
            "hunt",
            "-n",
            "allo",
            "--max-trials",
            "2",
            "--config",
            "./database_config.yaml",
            "./black_box.py",
            "-x~uniform(-50,50)",
        ]
    )

    assert not os.listdir(tmp_path)


def test_working_dir_argument_config(storage, monkeypatch):
    """Check that workning dir argument is handled properly"""
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))
    dir_path = os.path.join("orion", "test")
    if os.path.exists(dir_path):
        shutil.rmtree(dir_path)

    orion.core.cli.main(
        [
            "hunt",
            "-n",
            "allo",
            "--max-trials",
            "2",
            "--config",
            "./working_dir_config.yaml",
            "./black_box.py",
            "-x~uniform(-50,50)",
        ]
    )

    exp = list(storage.fetch_experiments({"name": "allo"}))[0]
    assert exp["working_dir"] == dir_path
    assert os.path.exists(dir_path)
    assert os.listdir(dir_path)

    shutil.rmtree(dir_path)


def test_run_with_name_only(storage, monkeypatch):
    """Test hunt can be executed with experiment name only"""
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))
    orion.core.cli.main(
        [
            "hunt",
            "--init-only",
            "--config",
            "./orion_config_random.yaml",
            "./black_box.py",
            "-x~uniform(-50, 50)",
        ]
    )

    orion.core.cli.main(
        ["hunt", "--max-trials", "20", "--config", "./orion_config_random.yaml"]
    )

    exp = list(storage.fetch_experiments({"name": "demo_random_search"}))
    assert len(exp) == 1
    exp = exp[0]
    assert "_id" in exp
    exp_id = exp["_id"]
    trials = list(storage.fetch_trials(uid=exp_id))
    assert len(trials) == 20


def test_run_with_name_only_with_trailing_whitespace(storage, monkeypatch):
    """Test hunt can be executed with experiment name and trailing whitespace"""
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))
    orion.core.cli.main(
        [
            "hunt",
            "--init-only",
            "--config",
            "./orion_config_random.yaml",
            "./black_box.py",
            "-x~uniform(-50, 50)",
        ]
    )

    orion.core.cli.main(
        ["hunt", "--max-trials", "20", "--config", "./orion_config_random.yaml", ""]
    )

    exp = list(storage.fetch_experiments({"name": "demo_random_search"}))
    assert len(exp) == 1
    exp = exp[0]
    assert "_id" in exp
    exp_id = exp["_id"]
    trials = list(storage.fetch_trials(uid=exp_id))
    assert len(trials) == 20


@pytest.mark.parametrize("strategy", ["MaxParallelStrategy", "MeanParallelStrategy"])
def test_run_with_parallel_strategy(storage, monkeypatch, strategy):
    """Test hunt can be executed with max parallel strategies"""
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))

    with open("strategy_config.yaml") as f:
        config = yaml.safe_load(f.read())

    config_file = "{}_strategy_config.yaml".format(strategy)

    with open(config_file, "w") as f:
        config["producer"]["strategy"] = strategy
        f.write(yaml.dump(config))

    orion.core.cli.main(
        [
            "hunt",
            "--max-trials",
            "20",
            "--pool-size",
            "1",
            "--config",
            config_file,
            "./black_box.py",
            "-x~uniform(-50, 50)",
        ]
    )

    os.remove(config_file)

    exp = list(storage.fetch_experiments({"name": "strategy_demo"}))
    assert len(exp) == 1
    exp = exp[0]
    assert exp["producer"]["strategy"] == {strategy: {"default_result": float("inf")}}
    assert "_id" in exp
    exp_id = exp["_id"]
    trials = list(storage.fetch_trials(uid=exp_id))
    assert len(trials) == 20


def test_worker_trials(storage, monkeypatch):
    """Test number of trials executed is limited based on worker-trials"""
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))

    assert len(list(storage.fetch_experiments({"name": "demo_random_search"}))) == 0

    orion.core.cli.main(
        [
            "hunt",
            "--config",
            "./orion_config_random.yaml",
            "--pool-size",
            "1",
            "--worker-trials",
            "0",
            "./black_box.py",
            "-x~uniform(-50, 50)",
        ]
    )

    exp = list(storage.fetch_experiments({"name": "demo_random_search"}))
    assert len(exp) == 1
    exp = exp[0]
    assert "_id" in exp
    exp_id = exp["_id"]

    assert len(list(storage.fetch_trials(uid=exp_id))) == 0

    # Test only executes 2 trials
    orion.core.cli.main(
        ["hunt", "--name", "demo_random_search", "--worker-trials", "2"]
    )

    assert len(list(storage.fetch_trials(uid=exp_id))) == 2

    # Test only executes 3 more trials
    orion.core.cli.main(
        ["hunt", "--name", "demo_random_search", "--worker-trials", "3"]
    )

    assert len(list(storage.fetch_trials(uid=exp_id))) == 5

    # Test that max-trials has precedence over worker-trials
    orion.core.cli.main(
        [
            "hunt",
            "--name",
            "demo_random_search",
            "--worker-trials",
            "5",
            "--max-trials",
            "6",
        ]
    )

    assert len(list(storage.fetch_trials(uid=exp_id))) == 6


@pytest.mark.usefixtures("storage")
def test_resilience(monkeypatch):
    """Test if Oríon stops after enough broken trials."""
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))

    MAX_BROKEN = 3
    orion.core.config.worker.max_broken = 3

    orion.core.cli.main(
        [
            "hunt",
            "--config",
            "./orion_config_random.yaml",
            "./broken_box.py",
            "-x~uniform(-50, 50)",
        ]
    )

    exp = experiment_builder.build(name="demo_random_search")
    assert len(exp.fetch_trials_by_status("broken")) == MAX_BROKEN


@pytest.mark.usefixtures("storage")
def test_demo_with_shutdown_quickly(monkeypatch):
    """Check simple pipeline with random search is reasonably fast."""
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))

    monkeypatch.setattr(orion.core.config.worker, "heartbeat", 120)

    process = subprocess.Popen(
        [
            "orion",
            "hunt",
            "--config",
            "./orion_config_random.yaml",
            "--max-trials",
            "10",
            "./black_box.py",
            "-x~uniform(-50, 50)",
        ]
    )

    assert process.wait(timeout=40) == 0


def test_demo_with_nondefault_config_keyword(storage, monkeypatch):
    """Check that the user script configuration file is correctly used with a new keyword."""
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))
    orion.core.config.worker.user_script_config = "configuration"
    orion.core.cli.main(
        [
            "hunt",
            "--config",
            "./orion_config_other.yaml",
            "./black_box_w_config_other.py",
            "--configuration",
            "script_config.yaml",
        ]
    )

    exp = list(storage.fetch_experiments({"name": "voila_voici"}))
    assert len(exp) == 1
    exp = exp[0]
    assert "_id" in exp
    exp_id = exp["_id"]
    assert exp["name"] == "voila_voici"
    assert exp["pool_size"] == 1
    assert exp["max_trials"] == 20
    assert exp["algorithms"] == {
        "gradient_descent": {"learning_rate": 0.1, "dx_tolerance": 1e-5}
    }
    assert "user" in exp["metadata"]
    assert "datetime" in exp["metadata"]
    assert "orion_version" in exp["metadata"]
    assert "user_script" in exp["metadata"]
    assert exp["metadata"]["user_args"] == [
        "./black_box_w_config_other.py",
        "--configuration",
        "script_config.yaml",
    ]

    trials = list(storage.fetch_trials(uid=exp_id))
    assert len(trials) <= 15
    assert trials[-1].status == "completed"
    trials = list(sorted(trials, key=lambda trial: trial.submit_time))
    for result in trials[-1].results:
        assert result.type != "constraint"
        if result.type == "objective":
            assert abs(result.value - 23.4) < 1e-6
            assert result.name == "example_objective"
        elif result.type == "gradient":
            res = numpy.asarray(result.value)
            assert 0.1 * numpy.sqrt(res.dot(res)) < 1e-4
            assert result.name == "example_gradient"
    params = trials[-1].params
    assert len(params) == 1
    px = params["/x"]
    assert isinstance(px, float)
    assert (px - 34.56789) < 1e-4

    orion.core.config.worker.user_script_config = "config"


def test_demo_precision(storage, monkeypatch):
    """Test a simple usage scenario."""
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))

    user_args = ["-x~uniform(-50, 50, precision=5)"]

    orion.core.cli.main(
        [
            "hunt",
            "--config",
            "./orion_config.yaml",
            "--max-trials",
            "2",
            "./black_box.py",
        ]
        + user_args
    )

    exp = list(storage.fetch_experiments({"name": "voila_voici"}))
    exp = exp[0]
    exp_id = exp["_id"]
    trials = list(storage.fetch_trials(uid=exp_id))
    trials = list(sorted(trials, key=lambda trial: trial.submit_time))
    params = trials[-1].params
    value = params["/x"]

    assert value == float(numpy.format_float_scientific(value, precision=4))


@pytest.mark.usefixtures("setup_pickleddb_database")
def test_debug_mode(monkeypatch):
    """Test debug mode."""
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))

    user_args = ["-x~uniform(-50, 50, precision=5)"]

    orion.core.cli.main(
        [
            "--debug",
            "hunt",
            "--config",
            "./orion_config.yaml",
            "--max-trials",
            "2",
            "./black_box.py",
        ]
        + user_args
    )

    storage = get_storage()

    assert isinstance(storage, Legacy)
    assert isinstance(storage._db, EphemeralDB)


def test_no_args(capsys):
    """Test that help is printed when no args are given."""
    with pytest.raises(SystemExit):
        orion.core.cli.main([])

    captured = capsys.readouterr().out

    assert "usage:" in captured
    assert "Traceback" not in captured
