#!/usr/bin/env python
"""Common fixtures and utils for tests."""

import copy
import datetime
import getpass
import os

import pytest

from orion.algo.space import (Categorical, Integer, Real, Space)
from orion.core.evc import conflicts
from orion.core.io.convert import (JSONConverter, YAMLConverter)
from orion.core.io.database import Database
from orion.core.io.space_builder import DimensionBuilder
from orion.core.worker.experiment import Experiment


TEST_DIR = os.path.dirname(os.path.abspath(__file__))
YAML_SAMPLE = os.path.join(TEST_DIR, 'sample_config.yml')
JSON_SAMPLE = os.path.join(TEST_DIR, 'sample_config.json')
UNKNOWN_SAMPLE = os.path.join(TEST_DIR, 'sample_config.txt')


@pytest.fixture()
def create_db_instance(null_db_instances, clean_db):
    """Create and save a singleton database instance."""
    database = Database('MongoDB', name='orion_test',
                        username='user', password='pass')
    return database


@pytest.fixture(scope='session')
def yaml_sample_path():
    """Return path with a yaml sample file."""
    return os.path.abspath(YAML_SAMPLE)


@pytest.fixture(scope='session')
def json_sample_path():
    """Return path with a json sample file."""
    return JSON_SAMPLE


@pytest.fixture(scope='session')
def unknown_type_sample_path():
    """Return path with a sample file of unknown configuration filetype."""
    return UNKNOWN_SAMPLE


@pytest.fixture(scope='session')
def some_sample_path():
    """Return path with a sample file of unknown configuration filetype."""
    return os.path.join(TEST_DIR, 'some_sample_config.txt')


@pytest.fixture(scope='session')
def yaml_converter():
    """Return a yaml converter."""
    return YAMLConverter()


@pytest.fixture(scope='session')
def json_converter():
    """Return a json converter."""
    return JSONConverter()


@pytest.fixture(scope='module')
def space():
    """Construct a simple space with every possible kind of Dimension."""
    space = Space()
    categories = {'asdfa': 0.1, 2: 0.2, 3: 0.3, 4: 0.4}
    dim = Categorical('yolo', categories, shape=2)
    space.register(dim)
    dim = Integer('yolo2', 'uniform', -3, 6)
    space.register(dim)
    dim = Real('yolo3', 'alpha', 0.9)
    space.register(dim)
    return space


@pytest.fixture(scope='module')
def fixed_suggestion():
    """Return the same tuple/sample from a possible space."""
    return (('asdfa', 2), 0, 3.5)


@pytest.fixture()
def with_user_tsirif(monkeypatch):
    """Make ``getpass.getuser()`` return ``'tsirif'``."""
    monkeypatch.setattr(getpass, 'getuser', lambda: 'tsirif')


@pytest.fixture()
def with_user_bouthilx(monkeypatch):
    """Make ``getpass.getuser()`` return ``'bouthilx'``."""
    monkeypatch.setattr(getpass, 'getuser', lambda: 'bouthilx')


@pytest.fixture()
def with_user_dendi(monkeypatch):
    """Make ``getpass.getuser()`` return ``'dendi'``."""
    monkeypatch.setattr(getpass, 'getuser', lambda: 'dendi')


@pytest.fixture()
def random_dt(monkeypatch):
    """Make ``datetime.datetime.utcnow()`` return an arbitrary date."""
    random_dt = datetime.datetime(1903, 4, 25, 0, 0, 0)

    class MockDatetime(datetime.datetime):
        @classmethod
        def utcnow(cls):
            return random_dt

    monkeypatch.setattr(datetime, 'datetime', MockDatetime)
    return random_dt


@pytest.fixture()
def hacked_exp(with_user_dendi, random_dt, clean_db, create_db_instance):
    """Return an `Experiment` instance with hacked _id to find trials in
    fake database.
    """
    try:
        Database(of_type=('orion.core.io.database.mongodb', 'MongoDB'), name='orion_test',
                 username='user', password='pass')
    except (TypeError, ValueError):
        pass
    exp = Experiment('supernaedo2')
    exp._id = 'supernaedo2'  # white box hack
    return exp


@pytest.fixture()
def trial_id_substitution(with_user_tsirif, random_dt, clean_db, create_db_instance):
    """Replace trial ids by the actual ids of the experiments."""
    db = create_db_instance
    experiments = db.read('experiments', {'metadata.user': 'tsirif'})
    experiment_dict = dict((experiment['name'], experiment) for experiment in experiments)
    trials = db.read('trials')

    for trial in trials:
        query = {'experiment': trial['experiment']}
        update = {'experiment': experiment_dict[trial['experiment']]['_id']}
        db.write('trials', update, query)


@pytest.fixture()
def refers_id_substitution(with_user_tsirif, random_dt, clean_db, create_db_instance):
    """Replace trial ids by the actual ids of the experiments."""
    db = create_db_instance
    query = {'metadata.user': 'tsirif'}
    selection = {'name': 1, 'refers': 1}
    experiments = db.read('experiments', query, selection)
    experiment_dict = dict((experiment['name'], experiment) for experiment in experiments)

    for experiment in experiments:
        query = {'_id': experiment['_id']}
        print(experiment['refers'])
        root_id = experiment_dict[experiment['refers']['root_id']]['_id']
        if experiment['refers']['parent_id'] is not None:
            parent_id = experiment_dict[experiment['refers']['parent_id']]['_id']
        else:
            parent_id = None
        update = {'refers.root_id': root_id, 'refers.parent_id': parent_id}
        db.write('experiments', update, query)


###
# Fixtures for EVC tests using conflicts, present in both ./evc and ./io.
# Note: Refactoring the EVC out of orion's core should take care of getting those
#       fixtures out of general conftest.py
###


@pytest.fixture
def new_config():
    """Generate a new experiment configuration"""
    return dict(
        name='test',
        algorithms='fancy',
        metadata={'VCS': 'to be changed',
                  'user_script': 'abs_path/black_box.py',
                  'user_args':
                  ['--new~normal(0,2)', '--changed~normal(0,2)'],
                  'user': 'some_user_name'})


@pytest.fixture
def old_config(create_db_instance):
    """Generate an old experiment configuration"""
    config = dict(
        name='test',
        algorithms='random',
        metadata={'VCS': {"type": "git",
                          "is_dirty": False,
                          "HEAD_sha": "test",
                          "active_branch": None,
                          "diff_sha": "diff",
                          },
                  'user_script': 'abs_path/black_box.py',
                  'user_args':
                  ['--missing~uniform(-10,10)', '--changed~uniform(-10,10)'],
                  'user': 'some_user_name'})

    create_db_instance.write('experiments', config)
    return config


@pytest.fixture
def new_dimension_conflict(old_config, new_config):
    """Generate a new dimension conflict for new experiment configuration"""
    name = 'new'
    prior = 'normal(0, 2)'
    dimension = DimensionBuilder().build(name, prior)
    return conflicts.NewDimensionConflict(old_config, new_config, dimension, prior)


@pytest.fixture
def new_dimension_with_default_conflict(old_config, new_config):
    """Generate a new dimension conflict with default value for new experiment configuration"""
    name = 'new'
    prior = 'normal(0, 2, default_value=0.001)'
    dimension = DimensionBuilder().build(name, prior)
    return conflicts.NewDimensionConflict(old_config, new_config, dimension, prior)


@pytest.fixture
def new_dimension_same_prior_conflict(old_config, new_config):
    """Generate a new dimension conflict with different prior for renaming tests"""
    name = 'new'
    prior = 'uniform(-10, 10)'
    dimension = DimensionBuilder().build(name, prior)
    return conflicts.NewDimensionConflict(old_config, new_config, dimension, prior)


@pytest.fixture
def changed_dimension_conflict(old_config, new_config):
    """Generate a changed dimension conflict"""
    name = 'changed'
    old_prior = 'uniform(-10, 10)'
    new_prior = 'normal(0, 2)'
    dimension = DimensionBuilder().build(name, old_prior)
    return conflicts.ChangedDimensionConflict(old_config, new_config,
                                              dimension, old_prior, new_prior)


@pytest.fixture
def missing_dimension_conflict(old_config, new_config):
    """Generate a missing dimension conflict"""
    name = 'missing'
    prior = 'uniform(-10, 10)'
    dimension = DimensionBuilder().build(name, prior)
    return conflicts.MissingDimensionConflict(old_config, new_config, dimension, prior)


@pytest.fixture
def missing_dimension_with_default_conflict(old_config, new_config):
    """Generate a missing dimension conflict with a default value"""
    name = 'missing'
    prior = 'uniform(-10, 10, default_value=0.0)'
    dimension = DimensionBuilder().build(name, prior)
    return conflicts.MissingDimensionConflict(old_config, new_config, dimension, prior)


@pytest.fixture
def algorithm_conflict(old_config, new_config):
    """Generate an algorithm configuration conflict"""
    return conflicts.AlgorithmConflict(old_config, new_config)


@pytest.fixture
def code_conflict(old_config, new_config):
    """Generate a code conflict"""
    return conflicts.CodeConflict(old_config, new_config)


@pytest.fixture
def cli_conflict(old_config, new_config):
    """Generate a commandline conflict"""
    new_config = copy.deepcopy(new_config)
    new_config['metadata']['user_args'].append("--some-new=args")
    return conflicts.CommandLineConflict(old_config, new_config)


@pytest.fixture
def config_conflict(old_config, new_config):
    """Generate a script config conflict"""
    return conflicts.ScriptConfigConflict(old_config, new_config)


@pytest.fixture
def experiment_name_conflict(old_config, new_config):
    """Generate an experiment name conflict"""
    return conflicts.ExperimentNameConflict(old_config, new_config)
