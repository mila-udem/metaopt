#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Perform a functional test for demo purposes."""
from collections import defaultdict
import os
import subprocess

import numpy
import pytest
import yaml

import orion.core.cli
from orion.core.io.database import Database
from orion.core.worker import workon
from orion.core.worker.experiment import Experiment


@pytest.mark.usefixtures("clean_db")
@pytest.mark.usefixtures("null_db_instances")
def test_demo_with_default_algo_cli_config_only(database, monkeypatch):
    """Check that random algorithm is used, when no algo is chosen explicitly."""
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))
    monkeypatch.setenv('ORION_DB_NAME', 'orion_test')
    monkeypatch.setenv('ORION_DB_ADDRESS', 'mongodb://user:pass@localhost')

    orion.core.cli.main(["hunt", "-n", "default_algo",
                         "--max-trials", "30",
                         "./black_box.py", "-x~uniform(-50, 50)"])

    exp = list(database.experiments.find({'name': 'default_algo'}))
    assert len(exp) == 1
    exp = exp[0]
    assert '_id' in exp
    assert exp['name'] == 'default_algo'
    assert exp['pool_size'] == 1
    assert exp['max_trials'] == 30
    assert exp['algorithms'] == {'random': {'seed': None}}
    assert 'user' in exp['metadata']
    assert 'datetime' in exp['metadata']
    assert 'orion_version' in exp['metadata']
    assert 'user_script' in exp['metadata']
    assert os.path.isabs(exp['metadata']['user_script'])
    assert exp['metadata']['user_args'] == ['-x~uniform(-50, 50)']


@pytest.mark.usefixtures("clean_db")
@pytest.mark.usefixtures("null_db_instances")
def test_demo(database, monkeypatch):
    """Test a simple usage scenario."""
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))
    orion.core.cli.main(["hunt", "--config", "./orion_config.yaml",
                         "./black_box.py", "-x~uniform(-50, 50)"])

    exp = list(database.experiments.find({'name': 'voila_voici'}))
    assert len(exp) == 1
    exp = exp[0]
    assert '_id' in exp
    exp_id = exp['_id']
    assert exp['name'] == 'voila_voici'
    assert exp['pool_size'] == 1
    assert exp['max_trials'] == 100
    assert exp['algorithms'] == {'gradient_descent': {'learning_rate': 0.1,
                                                      'dx_tolerance': 1e-7,
                                                      'seed': None}}
    assert 'user' in exp['metadata']
    assert 'datetime' in exp['metadata']
    assert 'orion_version' in exp['metadata']
    assert 'user_script' in exp['metadata']
    assert os.path.isabs(exp['metadata']['user_script'])
    assert exp['metadata']['user_args'] == ['-x~uniform(-50, 50)']

    trials = list(database.trials.find({'experiment': exp_id}))
    assert len(trials) <= 15
    assert trials[-1]['status'] == 'completed'
    for result in trials[-1]['results']:
        assert result['type'] != 'constraint'
        if result['type'] == 'objective':
            assert abs(result['value'] - 23.4) < 1e-6
            assert result['name'] == 'example_objective'
        elif result['type'] == 'gradient':
            res = numpy.asarray(result['value'])
            assert 0.1 * numpy.sqrt(res.dot(res)) < 1e-7
            assert result['name'] == 'example_gradient'
    params = trials[-1]['params']
    assert len(params) == 1
    assert params[0]['name'] == '/x'
    assert params[0]['type'] == 'real'
    assert (params[0]['value'] - 34.56789) < 1e-5


@pytest.mark.usefixtures("clean_db")
def test_demo_two_workers(database, monkeypatch):
    """Test a simple usage scenario."""
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))
    processes = []
    for _ in range(2):
        process = subprocess.Popen(["orion", "hunt", "-n", "two_workers_demo",
                                    "--config", "./orion_config_random.yaml",
                                    "--max-trials", "100",
                                    "./black_box.py", "-x~norm(34, 3)"])
        processes.append(process)

    for process in processes:
        rcode = process.wait()
        assert rcode == 0

    exp = list(database.experiments.find({'name': 'two_workers_demo'}))
    assert len(exp) == 1
    exp = exp[0]
    assert '_id' in exp
    exp_id = exp['_id']
    assert exp['name'] == 'two_workers_demo'
    assert exp['pool_size'] == 2
    assert exp['max_trials'] == 100
    assert exp['algorithms'] == {'random': {'seed': None}}
    assert 'user' in exp['metadata']
    assert 'datetime' in exp['metadata']
    assert 'orion_version' in exp['metadata']
    assert 'user_script' in exp['metadata']
    assert os.path.isabs(exp['metadata']['user_script'])
    assert exp['metadata']['user_args'] == ['-x~norm(34, 3)']

    trials = list(database.trials.find({'experiment': exp_id}))
    status = defaultdict(int)
    for trial in trials:
        status[trial['status']] += 1
    assert 100 <= status['completed'] <= 101
    assert status['new'] < 5
    params = trials[-1]['params']
    assert len(params) == 1
    assert params[0]['name'] == '/x'
    assert params[0]['type'] == 'real'


@pytest.mark.usefixtures("clean_db")
def test_workon(database):
    """Test scenario having a configured experiment already setup."""
    try:
        Database(of_type='MongoDB', name='orion_test',
                 username='user', password='pass')
    except (TypeError, ValueError):
        pass
    experiment = Experiment('voila_voici')
    config = experiment.configuration
    config['algorithms'] = {
        'gradient_descent': {
            'learning_rate': 0.1
            }
        }
    config['pool_size'] = 1
    config['max_trials'] = 100
    config['metadata']['user_script'] = os.path.abspath(os.path.join(
        os.path.dirname(__file__), "black_box.py"))
    config['metadata']['user_args'] = ["-x~uniform(-50, 50)"]
    experiment.configure(config)

    workon(experiment)

    exp = list(database.experiments.find({'name': 'voila_voici'}))
    assert len(exp) == 1
    exp = exp[0]
    assert '_id' in exp
    exp_id = exp['_id']
    assert exp['name'] == 'voila_voici'
    assert exp['pool_size'] == 1
    assert exp['max_trials'] == 100
    assert exp['algorithms'] == {'gradient_descent': {'learning_rate': 0.1,
                                                      'dx_tolerance': 1e-7,
                                                      'seed': None}}
    assert 'user' in exp['metadata']
    assert 'datetime' in exp['metadata']
    assert 'user_script' in exp['metadata']
    assert os.path.isabs(exp['metadata']['user_script'])
    assert exp['metadata']['user_args'] == ['-x~uniform(-50, 50)']

    trials = list(database.trials.find({'experiment': exp_id}))
    assert len(trials) <= 15
    assert trials[-1]['status'] == 'completed'
    for result in trials[-1]['results']:
        assert result['type'] != 'constraint'
        if result['type'] == 'objective':
            assert abs(result['value'] - 23.4) < 1e-6
            assert result['name'] == 'example_objective'
        elif result['type'] == 'gradient':
            res = numpy.asarray(result['value'])
            assert 0.1 * numpy.sqrt(res.dot(res)) < 1e-7
            assert result['name'] == 'example_gradient'
    params = trials[-1]['params']
    assert len(params) == 1
    assert params[0]['name'] == '/x'
    assert params[0]['type'] == 'real'
    assert (params[0]['value'] - 34.56789) < 1e-5


@pytest.mark.usefixtures("clean_db")
@pytest.mark.usefixtures("null_db_instances")
def test_stress_unique_folder_creation(database, monkeypatch, tmpdir, capfd):
    """Test integration with a possible framework that needs to create
    unique directories per trial.
    """
    # XXX: return and complete test when there is a way to control random
    # seed of Oríon
    how_many = 50
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))
    orion.core.cli.main(["hunt", "--max-trials={}".format(how_many),
                         "--pool-size=1",
                         "--name=lalala",
                         "--config", "./stress_gradient.yaml",
                         "./dir_per_trial.py",
                         "--dir={}".format(str(tmpdir)),
                         "--other-name~exp.name",
                         "--name~trial.hash_name",
                         "-x~gaussian(30, 10)"])

    exp = list(database.experiments.find({'name': 'lalala'}))
    assert len(exp) == 1
    exp = exp[0]
    assert '_id' in exp
    exp_id = exp['_id']

    # For contingent broken trials, which in this test means that a existing
    # directory was attempted to be created, it means that it's not md5 or
    # bad hash creation to blame, but the finite precision of the floating
    # point representation. Specifically, it seems that gradient descent
    # is able to reach such levels of precision jumping around the minimum
    # (notice that an appropriate learning rate was selected in this stress
    # test to create underdamped behaviour), that it begins to suggest same
    # things from the past. This is intended to be shown with the assertions
    # in the for-loop below.
    trials_c = list(database.trials.find({'experiment': exp_id, 'status': 'completed'}))
    list_of_cx = [trial['params'][0]['value'] for trial in trials_c]
    trials_b = list(database.trials.find({'experiment': exp_id, 'status': 'broken'}))
    list_of_bx = [trial['params'][0]['value'] for trial in trials_b]
    for bx in list_of_bx:
        assert bx in list_of_cx

    # ``exp.name`` has been delivered correctly (next 2 assertions)
    assert len(os.listdir(str(tmpdir))) == 1
    # Also, because of the way the demo gradient descent works `how_many` trials
    # can be completed
    assert len(os.listdir(str(tmpdir.join('lalala')))) == how_many
    assert len(trials_c) == how_many
    capfd.readouterr()  # Suppress fd level 1 & 2


@pytest.mark.usefixtures("clean_db")
@pytest.mark.usefixtures("null_db_instances")
def test_run_with_name_only(database, monkeypatch):
    """Test hunt can be executed with experiment name only"""
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))
    orion.core.cli.main(["init_only", "--config", "./orion_config_random.yaml",
                         "./black_box.py", "-x~uniform(-50, 50)"])

    orion.core.cli.main(["hunt", "--max-trials", "20", "--config", "./orion_config_random.yaml"])

    exp = list(database.experiments.find({'name': 'demo_random_search'}))
    assert len(exp) == 1
    exp = exp[0]
    print(exp['max_trials'])
    assert '_id' in exp
    exp_id = exp['_id']
    trials = list(database.trials.find({'experiment': exp_id}))
    assert len(trials) == 20


@pytest.mark.usefixtures("clean_db")
@pytest.mark.usefixtures("null_db_instances")
def test_run_with_name_only_with_trailing_whitespace(database, monkeypatch):
    """Test hunt can be executed with experiment name and trailing whitespace"""
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))
    orion.core.cli.main(["init_only", "--config", "./orion_config_random.yaml",
                         "./black_box.py", "-x~uniform(-50, 50)"])

    orion.core.cli.main(["hunt", "--max-trials", "20",
                         "--config", "./orion_config_random.yaml", ""])

    exp = list(database.experiments.find({'name': 'demo_random_search'}))
    assert len(exp) == 1
    exp = exp[0]
    print(exp['max_trials'])
    assert '_id' in exp
    exp_id = exp['_id']
    trials = list(database.trials.find({'experiment': exp_id}))
    assert len(trials) == 20


@pytest.mark.usefixtures("clean_db")
@pytest.mark.usefixtures("null_db_instances")
@pytest.mark.parametrize("strategy", ['MaxParallelStrategy', 'MeanParallelStrategy'])
def test_run_with_parallel_strategy(database, monkeypatch, strategy):
    """Test hunt can be executed with max parallel strategies"""
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))

    with open('strategy_config.yaml') as f:
        config = yaml.load(f.read())

    config_file = '{}_strategy_config.yaml'.format(strategy)

    with open(config_file, 'w') as f:
        config['producer']['strategy'] = strategy
        f.write(yaml.dump(config))

    with open(config_file, 'r') as f:
        print(yaml.load(f.read()))

    orion.core.cli.main(["hunt", "--max-trials", "20", "--pool-size", "1",
                         "--config", config_file,
                         "./black_box.py", "-x~uniform(-50, 50)"])

    os.remove(config_file)

    exp = list(database.experiments.find({'name': 'strategy_demo'}))
    assert len(exp) == 1
    exp = exp[0]
    assert exp['producer']['strategy'] == strategy
    print(exp['max_trials'])
    assert '_id' in exp
    exp_id = exp['_id']
    trials = list(database.trials.find({'experiment': exp_id}))
    assert len(trials) == 20


@pytest.mark.usefixtures("clean_db")
@pytest.mark.usefixtures("null_db_instances")
def test_worker_trials(database, monkeypatch):
    """Test number of trials executed is limited based on worker-trials"""
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))

    assert len(list(database.experiments.find({'name': 'demo_random_search'}))) == 0

    orion.core.cli.main(["hunt", "--config", "./orion_config_random.yaml", "--pool-size", "1",
                         "--worker-trials", "0",
                         "./black_box.py", "-x~uniform(-50, 50)"])

    exp = list(database.experiments.find({'name': 'demo_random_search'}))
    assert len(exp) == 1
    exp = exp[0]
    assert '_id' in exp
    exp_id = exp['_id']

    assert len(list(database.trials.find({'experiment': exp_id}))) == 0

    # Test only executes 2 trials
    orion.core.cli.main(["hunt", "--name", "demo_random_search", "--worker-trials", "2"])

    assert len(list(database.trials.find({'experiment': exp_id}))) == 2

    # Test only executes 3 more trials
    orion.core.cli.main(["hunt", "--name", "demo_random_search", "--worker-trials", "3"])

    assert len(list(database.trials.find({'experiment': exp_id}))) == 5

    # Test that max-trials has precedence over worker-trials
    orion.core.cli.main(["hunt", "--name", "demo_random_search", "--worker-trials", "5",
                         "--max-trials", "6"])

    assert len(list(database.trials.find({'experiment': exp_id}))) == 6
