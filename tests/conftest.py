#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Common fixtures and utils for unittests and functional tests."""
import os

import numpy
from pymongo import MongoClient
import pytest
import yaml

from orion.algo.dumbalgo import DumbAlgo
from orion.core.io import resolve_config
from orion.core.io.database import Database
from orion.core.io.database.mongodb import MongoDB
from orion.core.worker.trial import Trial


@pytest.fixture(scope='session')
def dumbalgo():
    """Return stab algorithm class."""
    return DumbAlgo


@pytest.fixture()
def categorical_values():
    """Return a list of all the categorical points possible for `supernaedo2` and `supernaedo3`"""
    return [('rnn', 'rnn'), ('rnn', 'lstm_with_attention'), ('rnn', 'gru'),
            ('gru', 'rnn'), ('gru', 'lstm_with_attention'), ('gru', 'gru'),
            ('lstm', 'rnn'), ('lstm', 'lstm_with_attention'), ('lstm', 'gru')]


@pytest.fixture()
def exp_config():
    """Load an example database."""
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
              'unittests', 'core', 'experiment.yaml')) as f:
        exp_config = list(yaml.safe_load_all(f))

    for i, t_dict in enumerate(exp_config[1]):
        exp_config[1][i] = Trial(**t_dict).to_dict()
    for i, _ in enumerate(exp_config[0]):
        exp_config[0][i]["metadata"]["user_script"] = os.path.join(
            os.path.dirname(__file__), exp_config[0][i]["metadata"]["user_script"])

    return exp_config


@pytest.fixture(scope='session')
def database():
    """Return Mongo database object to test with example entries."""
    client = MongoClient(username='user', password='pass', authSource='orion_test')
    database = client.orion_test
    yield database
    client.close()


@pytest.fixture()
def clean_db(database, exp_config):
    """Clean insert example experiment entries to collections."""
    database.experiments.drop()
    database.experiments.insert_many(exp_config[0])
    database.lying_trials.drop()
    database.trials.drop()
    database.trials.insert_many(exp_config[1])
    database.workers.drop()
    database.workers.insert_many(exp_config[2])
    database.resources.drop()
    database.resources.insert_many(exp_config[3])


@pytest.fixture()
def null_db_instances():
    """Nullify singleton instance so that we can assure independent instantiation tests."""
    Database.instance = None
    MongoDB.instance = None


@pytest.fixture(scope='function')
def seed():
    """Return a fixed ``numpy.random.RandomState`` and global seed."""
    seed = 5
    rng = numpy.random.RandomState(seed)
    numpy.random.seed(seed)
    return rng


@pytest.fixture
def version_XYZ(monkeypatch):
    """Force orion version XYZ on output of resolve_config.fetch_metadata"""
    non_patched_fetch_metadata = resolve_config.fetch_metadata

    def fetch_metadata(cmdargs):
        metadata = non_patched_fetch_metadata(cmdargs)
        metadata['orion_version'] = 'XYZ'
        return metadata
    monkeypatch.setattr(resolve_config, "fetch_metadata", fetch_metadata)


@pytest.fixture()
def create_db_instance(null_db_instances, clean_db):
    """Create and save a singleton database instance."""
    try:
        db = Database('MongoDB', name='orion_test',
                      username='user', password='pass')
    except ValueError:
        db = Database()

    return db


@pytest.fixture()
def script_path():
    """Return a script path for mock"""
    return os.path.join(os.path.dirname(__file__), "functional/demo/black_box.py")


@pytest.fixture()
def mock_infer_versioning_metadata(monkeypatch):
    """Mock infer_versioning_metadata and create a VCS"""
    def fixed_dictionary(user_script):
        """Create VCS"""
        vcs = {}
        vcs['type'] = 'git'
        vcs['is_dirty'] = False
        vcs['HEAD_sha'] = "test"
        vcs['active_branch'] = None
        vcs['diff_sha'] = "diff"
        return vcs
    monkeypatch.setattr(resolve_config, "infer_versioning_metadata", fixed_dictionary)
