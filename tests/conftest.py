#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Common fixtures and utils for unittests and functional tests."""
import getpass
import os
import tempfile

import numpy
import pytest
import yaml
from pymongo import MongoClient

import orion.core
import orion.core.utils.backward as backward
from orion.algo.base import BaseAlgorithm, OptimizationAlgorithm
from orion.core.io import resolve_config
from orion.core.io.database import Database
from orion.core.io.database.mongodb import MongoDB
from orion.core.io.database.pickleddb import PickledDB
from orion.core.utils.singleton import update_singletons
from orion.core.worker.trial import Trial
from orion.storage.base import Storage, get_storage, setup_storage
from orion.storage.legacy import Legacy
from orion.testing import OrionState

# So that assert messages show up in tests defined outside testing suite.
pytest.register_assert_rewrite("orion.testing")


def pytest_addoption(parser):
    parser.addoption(
        "--mongodb",
        action="store_true",
        default=False,
        help="Include MongoDB tests and exclude non-MongoDB databases tests. "
        "Default behaviour includes non-MongoDB tests and excludes MongoDB "
        "databases tests.",
    )


def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "db_types_only(db_types): mark test to run only with listed database types",
    )
    config.addinivalue_line(
        "markers",
        "drop_collections(collections): mark test to drop collections prior running",
    )
    config.addinivalue_line(
        "markers",
        "insert_collections(collections): mark test to insert collections prior running",
    )


@pytest.fixture(scope="session", autouse=True)
def shield_from_user_config(request):
    """Do not read user's yaml global config."""
    _pop_out_yaml_from_config(orion.core.config)


def _pop_out_yaml_from_config(config):
    """Remove any configuration fetch from yaml file"""
    for key in config._config.keys():
        config._config[key].pop("yaml", None)

    for key in config._subconfigs.keys():
        _pop_out_yaml_from_config(config._subconfigs[key])


class DumbAlgo(BaseAlgorithm):
    """Stab class for `BaseAlgorithm`."""

    def __init__(
        self,
        space,
        value=5,
        scoring=0,
        judgement=None,
        suspend=False,
        done=False,
        seed=None,
        **nested_algo
    ):
        """Configure returns, allow for variable variables."""
        self._times_called_suspend = 0
        self._times_called_is_done = 0
        self._num = 0
        self._index = 0
        self._points = []
        self._suggested = None
        self._results = []
        self._score_point = None
        self._judge_point = None
        self._measurements = None
        self.pool_size = 1
        self.possible_values = [value]
        super(DumbAlgo, self).__init__(
            space,
            value=value,
            scoring=scoring,
            judgement=judgement,
            suspend=suspend,
            done=done,
            seed=seed,
            **nested_algo
        )

    def seed(self, seed):
        """Set the index to seed.

        Setting the seed as an index so that unit-tests can force the algorithm to suggest the same
        values as if seeded.
        """
        self._index = seed if seed is not None else 0

    @property
    def state_dict(self):
        """Return a state dict that can be used to reset the state of the algorithm."""
        _state_dict = super(DumbAlgo, self).state_dict
        _state_dict.update(
            {
                "index": self._index,
                "suggested": self._suggested,
                "num": self._num,
                "done": self.done,
            }
        )
        return _state_dict

    def set_state(self, state_dict):
        """Reset the state of the algorithm based on the given state_dict

        :param state_dict: Dictionary representing state of an algorithm
        """
        super(DumbAlgo, self).set_state(state_dict)
        self._index = state_dict["index"]
        self._suggested = state_dict["suggested"]
        self._num = state_dict["num"]
        self.done = state_dict["done"]

    def suggest(self, num):
        """Suggest based on `value`."""
        num = min(num, self.pool_size)
        self._num += num

        rval = []
        while len(rval) < num:
            value = self.possible_values[
                min(self._index, len(self.possible_values) - 1)
            ]
            self._index += 1
            rval.append(value)

        self._suggested = rval

        return rval

    def observe(self, points, results):
        """Log inputs."""
        super(DumbAlgo, self).observe(points, results)
        self._points += points
        self._results += results

    def score(self, point):
        """Log and return stab."""
        self._score_point = point
        return self.scoring

    def judge(self, point, measurements):
        """Log and return stab."""
        self._judge_point = point
        self._measurements = measurements
        return self.judgement

    @property
    def should_suspend(self):
        """Cound how many times it has been called and return `suspend`."""
        self._times_called_suspend += 1
        return self.suspend

    @property
    def is_done(self):
        """Cound how many times it has been called and return `done`."""
        self._times_called_is_done += 1
        return self.done


# Hack it into being discoverable
OptimizationAlgorithm.types.append(DumbAlgo)
OptimizationAlgorithm.typenames.append(DumbAlgo.__name__.lower())


@pytest.fixture()
def empty_config():
    """Return config purged from global definition"""
    orion.core.DEF_CONFIG_FILES_PATHS = []
    config = orion.core.build_config()
    orion.core.config = config
    resolve_config.config = config
    return config


@pytest.fixture()
def test_config(empty_config):
    """Return orion's config overwritten with local config file"""
    config_file = os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "orion_config.yaml"
    )
    empty_config.load_yaml(config_file)

    return empty_config


@pytest.fixture(scope="session")
def dumbalgo():
    """Return stab algorithm class."""
    return DumbAlgo


@pytest.fixture()
def categorical_values():
    """Return a list of all the categorical points possible for `supernaedo2` and `supernaedo3`"""
    return [
        ("rnn", "rnn"),
        ("lstm_with_attention", "rnn"),
        ("gru", "rnn"),
        ("rnn", "gru"),
        ("lstm_with_attention", "gru"),
        ("gru", "gru"),
        ("rnn", "lstm"),
        ("lstm_with_attention", "lstm"),
        ("gru", "lstm"),
    ]


@pytest.fixture()
def exp_config_file():
    """Return configuration file used for stuff"""
    return os.path.join(
        os.path.dirname(os.path.abspath(__file__)),
        "unittests",
        "core",
        "experiment.yaml",
    )


@pytest.fixture()
def exp_config():
    """Load an example database."""
    with open(
        os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "unittests",
            "core",
            "experiment.yaml",
        )
    ) as f:
        exp_config = list(yaml.safe_load_all(f))

    for i, t_dict in enumerate(exp_config[1]):
        exp_config[1][i] = Trial(**t_dict).to_dict()

    for config in exp_config[0]:
        config["metadata"]["user_script"] = os.path.join(
            os.path.dirname(__file__), config["metadata"]["user_script"]
        )
        backward.populate_space(config)
        config["version"] = 1

    return exp_config


@pytest.fixture(scope="session")
def database():
    """Return Mongo database object to test with example entries."""
    client = MongoClient(username="user", password="pass", authSource="orion_test")
    database = client.orion_test
    yield database
    client.close()


@pytest.fixture()
def mock_database():
    """
    Lightweight fixture for an empty, in-memory database using :class:`OrionState`.
    The database is automatically discarded after each test method.
    """
    storage = {"type": "legacy", "database": {"type": "EphemeralDB"}}
    with OrionState(storage=storage) as state:
        yield state


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
    Storage.instance = None
    Legacy.instance = None
    Database.instance = None
    MongoDB.instance = None
    PickledDB.instance = None


@pytest.fixture(scope="function")
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

    def fetch_metadata(user=None, user_args=None, user_script_config=None):
        metadata = non_patched_fetch_metadata(user, user_args, user_script_config)
        metadata["orion_version"] = "XYZ"
        return metadata

    monkeypatch.setattr(resolve_config, "fetch_metadata", fetch_metadata)

    non_patched_update_metadata = resolve_config.update_metadata

    def update_metadata(metadata):
        metadata = non_patched_update_metadata(metadata)
        metadata["orion_version"] = "XYZ"
        return metadata

    monkeypatch.setattr(resolve_config, "update_metadata", update_metadata)


@pytest.fixture()
def script_path():
    """Return a script path for mock"""
    return os.path.join(
        os.path.abspath(os.path.dirname(__file__)), "functional/demo/black_box.py"
    )


@pytest.fixture()
def mock_infer_versioning_metadata(monkeypatch):
    """Mock infer_versioning_metadata and create a VCS"""

    def fixed_dictionary(user_script):
        """Create VCS"""
        vcs = {}
        vcs["type"] = "git"
        vcs["is_dirty"] = False
        vcs["HEAD_sha"] = "test"
        vcs["active_branch"] = None
        vcs["diff_sha"] = "diff"
        return vcs

    monkeypatch.setattr(resolve_config, "infer_versioning_metadata", fixed_dictionary)


@pytest.fixture(scope="function")
def setup_pickleddb_database():
    """Configure the database"""
    update_singletons()
    temporary_file = tempfile.NamedTemporaryFile()

    os.environ["ORION_DB_TYPE"] = "pickleddb"
    os.environ["ORION_DB_ADDRESS"] = temporary_file.name
    yield
    temporary_file.close()
    del os.environ["ORION_DB_TYPE"]
    del os.environ["ORION_DB_ADDRESS"]


@pytest.fixture(scope="function")
def storage(setup_pickleddb_database):
    setup_storage()
    yield get_storage()


@pytest.fixture()
def with_user_userxyz(monkeypatch):
    """Make ``getpass.getuser()`` return ``'userxyz'``."""
    monkeypatch.setattr(getpass, "getuser", lambda: "userxyz")
