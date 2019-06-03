#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Collection of tests for :mod:`orion.core.worker.experiment`."""

import copy
import getpass
import random

import pytest

from orion.algo.base import BaseAlgorithm
from orion.core.io.database import Database, DuplicateKeyError
from orion.core.worker.experiment import Experiment, ExperimentView
from orion.core.worker.trial import Trial


@pytest.fixture()
def patch_sample(monkeypatch):
    """Patch ``random.sample`` to return the first one and check call."""
    def mock_sample(a_list, should_be_one):
        assert type(a_list) == list
        assert len(a_list) >= 1
        # Part of `TestReserveTrial.test_reserve_success`
        assert a_list[0].status == 'interrupted'
        assert a_list[1].status == 'new'
        assert a_list[2].status == 'new'
        assert a_list[3].status == 'suspended'
        assert should_be_one == 1
        return [a_list[0]]

    monkeypatch.setattr(random, 'sample', mock_sample)


@pytest.fixture()
def patch_sample2(monkeypatch):
    """Patch ``random.sample`` to return the first one and check call."""
    def mock_sample(a_list, should_be_one):
        assert type(a_list) == list
        assert len(a_list) >= 1
        # Part of `TestReserveTrial.test_reserve_success2`
        assert a_list[0].status == 'interrupted'
        assert a_list[1].status == 'new'
        assert a_list[2].status == 'new'
        assert a_list[3].status == 'suspended'
        assert should_be_one == 1
        return [a_list[-1]]

    monkeypatch.setattr(random, 'sample', mock_sample)


@pytest.fixture()
def patch_sample_concurrent(monkeypatch, create_db_instance, exp_config):
    """Patch ``random.sample`` to return the first one and check call."""
    def mock_sample(a_list, should_be_one):
        assert type(a_list) == list
        assert len(a_list) >= 1
        # Part of `TestReserveTrial.test_reserve_race_condition`
        if len(a_list) == 4:
            assert a_list[0].status == 'interrupted'
        if len(a_list) == 3:
            assert a_list[0].status == 'new'
        if len(a_list) == 2:
            assert a_list[0].status == 'new'
        if len(a_list) == 1:
            assert a_list[0].status == 'suspended'

        assert should_be_one == 1

        if len(a_list) > 3:
            # Set row's status as 'reserved' just like if it was reserved by
            # another process right after the call to orion_db.read()
            create_db_instance.write(
                "trials",
                data={"status": "reserved"},
                query={"_id": a_list[0].id})
            trial = create_db_instance.read("trials", {"_id": a_list[0].id})
            assert trial[0]['status'] == 'reserved'

        return [a_list[0]]

    monkeypatch.setattr(random, 'sample', mock_sample)


@pytest.fixture()
def patch_sample_concurrent2(monkeypatch, create_db_instance, exp_config):
    """Patch ``random.sample`` to return the first one and check call."""
    def mock_sample(a_list, should_be_one):
        assert type(a_list) == list
        assert len(a_list) >= 1
        # Part of `TestReserveTrial.test_reserve_dead_race_condition`
        if len(a_list) == 4:
            assert a_list[0].status == 'interrupted'
        if len(a_list) == 3:
            assert a_list[0].status == 'new'
        if len(a_list) == 2:
            assert a_list[0].status == 'new'
        if len(a_list) == 1:
            assert a_list[0].status == 'suspended'

        assert should_be_one == 1

        # Set row's status as 'reserved' just like if it was reserved by
        # another process right after the call to orion_db.read()
        create_db_instance.write(
            "trials",
            data={"status": "reserved"},
            query={"_id": a_list[0].id})
        trial = create_db_instance.read("trials", {"_id": a_list[0].id})
        assert trial[0]['status'] == 'reserved'

        return [a_list[0]]

    monkeypatch.setattr(random, 'sample', mock_sample)


@pytest.fixture()
def new_config(random_dt):
    """Create a configuration that will not hit the database."""
    new_config = dict(
        name='supernaekei',
        # refers is missing on purpose
        metadata={'user': 'tsirif',
                  'orion_version': 0.1,
                  'user_script': 'abs_path/to_yoyoy.py',
                  'user_config': 'abs_path/hereitis.yaml',
                  'user_args': ['--mini-batch~uniform(32, 256, discrete=True)'],
                  'VCS': {"type": "git",
                          "is_dirty": False,
                          "HEAD_sha": "fsa7df7a8sdf7a8s7",
                          "active_branch": None,
                          "diff_sha": "diff"}},
        pool_size=10,
        max_trials=1000,
        algorithms={'dumbalgo': {}},
        producer={'strategy': 'NoParallelStrategy'},
        # attrs starting with '_' also
        _id='fasdfasfa',
        # and in general anything which is not in Experiment's slots
        something_to_be_ignored='asdfa'
        )
    return new_config


class TestInitExperiment(object):
    """Create new Experiment instance."""

    @pytest.mark.usefixtures("with_user_tsirif")
    def test_new_experiment_due_to_name(self, create_db_instance, random_dt):
        """Hit user name, but exp_name does not hit the db, create new entry."""
        exp = Experiment('supernaekei')
        assert exp._init_done is False
        assert exp._db is create_db_instance
        assert exp._id is None
        assert exp.name == 'supernaekei'
        assert exp.refers == {}
        assert exp.metadata['user'] == 'tsirif'
        assert exp._last_fetched == random_dt
        assert len(exp.metadata) == 1
        assert exp.pool_size is None
        assert exp.max_trials is None
        assert exp.algorithms is None
        with pytest.raises(AttributeError):
            exp.this_is_not_in_config = 5

    @pytest.mark.usefixtures("with_user_bouthilx")
    def test_new_experiment_due_to_username(self, create_db_instance, random_dt):
        """Hit exp_name, but user's name does not hit the db, create new entry."""
        exp = Experiment('supernaedo2')
        assert exp._init_done is False
        assert exp._db is create_db_instance
        assert exp._id is None
        assert exp.name == 'supernaedo2'
        assert exp.refers == {}
        assert exp.metadata['user'] == 'bouthilx'
        assert exp._last_fetched == random_dt
        assert len(exp.metadata) == 1
        assert exp.pool_size is None
        assert exp.max_trials is None
        assert exp.algorithms is None
        with pytest.raises(AttributeError):
            exp.this_is_not_in_config = 5

    @pytest.mark.usefixtures("with_user_tsirif")
    def test_existing_experiment(self, create_db_instance, exp_config):
        """Hit exp_name + user's name in the db, fetch most recent entry."""
        exp = Experiment('supernaedo2')
        assert exp._init_done is False
        assert exp._db is create_db_instance
        assert exp._id == exp_config[0][0]['_id']
        assert exp.name == exp_config[0][0]['name']
        assert exp.refers == exp_config[0][0]['refers']
        assert exp.metadata == exp_config[0][0]['metadata']
        assert exp._last_fetched == exp_config[0][0]['metadata']['datetime']
        assert exp.pool_size == exp_config[0][0]['pool_size']
        assert exp.max_trials == exp_config[0][0]['max_trials']
        assert exp.algorithms == exp_config[0][0]['algorithms']
        with pytest.raises(AttributeError):
            exp.this_is_not_in_config = 5


@pytest.mark.usefixtures("create_db_instance", "with_user_tsirif")
class TestConfigProperty(object):
    """Get and set experiment's configuration, finilize initialization process."""

    def test_get_before_init_has_hit(self, exp_config, random_dt):
        """Return a configuration dict according to an experiment object.

        Assuming that experiment's (exp's name, user's name) has hit the database.
        """
        exp = Experiment('supernaedo2')
        exp_config[0][0].pop('_id')
        assert exp.configuration == exp_config[0][0]

    def test_get_before_init_no_hit(self, exp_config, random_dt):
        """Return a configuration dict according to an experiment object.

        Before initialization is done, it can be the case that the pair (`name`,
        user's name) has not hit the database. return a yaml compliant form
        of current state, to be used with :mod:`orion.core.io.resolve_config`.
        """
        exp = Experiment('supernaekei')
        cfg = exp.configuration
        assert cfg['name'] == 'supernaekei'
        assert cfg['refers'] == {}
        assert cfg['metadata']['user'] == 'tsirif'
        assert len(cfg['metadata']) == 1
        assert cfg['pool_size'] is None
        assert cfg['max_trials'] is None
        assert cfg['algorithms'] is None

    @pytest.mark.skip(reason='Interactive prompt problems')
    def test_good_set_before_init_hit_with_diffs(self, exp_config):
        """Trying to set, and differences were found from the config pulled from db.

        In this case:
        1. Force renaming of experiment, prompt user for new name.
        2. Fork from experiment with previous name. New experiments refers to the
           old one, if user wants to.
        3. Overwrite elements with the ones from input.

        .. warning:: Currently, not implemented.
        """
        new_config = copy.deepcopy(exp_config[0][1])
        new_config['metadata']['user_version'] = 1.2
        exp = Experiment('supernaedo2')

        exp.configure(new_config)

    def test_good_set_before_init_hit_no_diffs_exc_max_trials(self, exp_config):
        """Trying to set, and NO differences were found from the config pulled from db.

        Everything is normal, nothing changes. Experiment is resumed,
        perhaps with more trials to evaluate (an exception is 'max_trials').
        """
        exp = Experiment('supernaedo2')
        # Deliver an external configuration to finalize init
        exp_config[0][0]['max_trials'] = 5000
        exp.configure(exp_config[0][0])
        exp_config[0][0]['algorithms']['dumbalgo']['done'] = False
        exp_config[0][0]['algorithms']['dumbalgo']['judgement'] = None
        exp_config[0][0]['algorithms']['dumbalgo']['scoring'] = 0
        exp_config[0][0]['algorithms']['dumbalgo']['suspend'] = False
        exp_config[0][0]['algorithms']['dumbalgo']['value'] = 5
        exp_config[0][0]['algorithms']['dumbalgo']['seed'] = None
        exp_config[0][0]['producer']['strategy'] = "NoParallelStrategy"
        assert exp._id == exp_config[0][0].pop('_id')
        assert exp.configuration == exp_config[0][0]

    def test_good_set_before_init_hit_no_diffs_exc_pool_size(self, exp_config):
        """Trying to set, and NO differences were found from the config pulled from db.

        Everything is normal, nothing changes. Experiment is resumed,
        perhaps with more workers that evaluate (an exception is 'pool_size').
        """
        exp = Experiment('supernaedo2')
        # Deliver an external configuration to finalize init
        exp_config[0][0]['pool_size'] = 10
        exp.configure(exp_config[0][0])
        exp_config[0][0]['algorithms']['dumbalgo']['done'] = False
        exp_config[0][0]['algorithms']['dumbalgo']['judgement'] = None
        exp_config[0][0]['algorithms']['dumbalgo']['scoring'] = 0
        exp_config[0][0]['algorithms']['dumbalgo']['suspend'] = False
        exp_config[0][0]['algorithms']['dumbalgo']['value'] = 5
        exp_config[0][0]['algorithms']['dumbalgo']['seed'] = None
        exp_config[0][0]['producer']['strategy'] = "NoParallelStrategy"
        assert exp._id == exp_config[0][0].pop('_id')
        assert exp.configuration == exp_config[0][0]

    def test_good_set_before_init_no_hit(self, random_dt, database, new_config):
        """Trying to set, overwrite everything from input."""
        exp = Experiment(new_config['name'])
        exp.configure(new_config)
        assert exp._init_done is True
        found_config = list(database.experiments.find({'name': 'supernaekei',
                                                       'metadata.user': 'tsirif'}))

        new_config['metadata']['datetime'] = exp.metadata['datetime']

        assert len(found_config) == 1
        _id = found_config[0].pop('_id')
        assert _id != 'fasdfasfa'
        assert exp._id == _id
        new_config['refers'] = {}
        new_config.pop('_id')
        new_config.pop('something_to_be_ignored')
        new_config['algorithms']['dumbalgo']['done'] = False
        new_config['algorithms']['dumbalgo']['judgement'] = None
        new_config['algorithms']['dumbalgo']['scoring'] = 0
        new_config['algorithms']['dumbalgo']['suspend'] = False
        new_config['algorithms']['dumbalgo']['value'] = 5
        new_config['algorithms']['dumbalgo']['seed'] = None
        new_config['refers'] = {'adapter': [], 'parent_id': None, 'root_id': _id}
        assert found_config[0] == new_config
        assert exp.name == new_config['name']
        assert exp.configuration['refers'] == new_config['refers']
        assert exp.metadata == new_config['metadata']
        assert exp.pool_size == new_config['pool_size']
        assert exp.max_trials == new_config['max_trials']
        #  assert exp.algorithms == new_config['algorithms']

    def test_inconsistent_1_set_before_init_no_hit(self, random_dt, new_config):
        """Test inconsistent configuration because of name."""
        exp = Experiment(new_config['name'])
        new_config['name'] = 'asdfaa'
        with pytest.raises(ValueError) as exc_info:
            exp.configure(new_config)
        assert 'inconsistent' in str(exc_info.value)

    def test_inconsistent_2_set_before_init_no_hit(self, random_dt, new_config):
        """Test inconsistent configuration because of user."""
        exp = Experiment(new_config['name'])
        new_config['metadata']['user'] = 'asdfaa'
        with pytest.raises(ValueError) as exc_info:
            exp.configure(new_config)
        assert 'inconsistent' in str(exc_info.value)

    def test_not_inconsistent_3_set_before_init_no_hit(self, random_dt, new_config):
        """Test inconsistent configuration because of datetime."""
        exp = Experiment(new_config['name'])
        new_config['metadata']['datetime'] = 123
        exp.configure(new_config)

    def test_get_after_init_plus_hit_no_diffs(self, exp_config):
        """Return a configuration dict according to an experiment object.

        Before initialization is done, it can be the case that the pair (`name`,
        user's name) has not hit the database. return a yaml compliant form
        of current state, to be used with :mod:`orion.core.cli.esolve_config`.
        """
        exp = Experiment('supernaedo2')
        # Deliver an external configuration to finalize init
        experiment_count_before = exp._db.count("experiments")
        exp.configure(exp_config[0][0])
        assert exp._init_done is True
        exp_config[0][0]['algorithms']['dumbalgo']['done'] = False
        exp_config[0][0]['algorithms']['dumbalgo']['judgement'] = None
        exp_config[0][0]['algorithms']['dumbalgo']['scoring'] = 0
        exp_config[0][0]['algorithms']['dumbalgo']['suspend'] = False
        exp_config[0][0]['algorithms']['dumbalgo']['value'] = 5
        exp_config[0][0]['algorithms']['dumbalgo']['seed'] = None
        exp_config[0][0]['producer']['strategy'] = "NoParallelStrategy"
        assert exp._id == exp_config[0][0].pop('_id')
        assert exp.configuration == exp_config[0][0]
        assert experiment_count_before == exp._db.count("experiments")

    def test_try_set_after_init(self, exp_config):
        """Cannot set a configuration after init (currently)."""
        exp = Experiment('supernaedo2')
        # Deliver an external configuration to finalize init
        exp.configure(exp_config[0][0])
        assert exp._init_done is True
        with pytest.raises(RuntimeError) as exc_info:
            exp.configure(exp_config[0][0])
        assert 'cannot reset' in str(exc_info.value)

    def test_try_set_after_race_condition(self, exp_config, new_config):
        """Cannot set a configuration after init if it looses a race
        condition.

        The experiment from process which first writes to db is initialized
        properly. The experiment which looses the race condition cannot be
        initialized and needs to be rebuilt.
        """
        exp = Experiment(new_config['name'])
        assert exp.id is None
        # Another experiment gets configured first
        experiment_count_before = exp._db.count("experiments")
        naughty_little_exp = Experiment(new_config['name'])
        assert naughty_little_exp.id is None
        naughty_little_exp.configure(new_config)
        assert naughty_little_exp._init_done is True
        assert exp._init_done is False
        assert (experiment_count_before + 1) == exp._db.count("experiments")
        # First experiment won't be able to be configured
        with pytest.raises(DuplicateKeyError) as exc_info:
            exp.configure(new_config)
        assert 'duplicate key error' in str(exc_info.value)

        assert (experiment_count_before + 1) == exp._db.count("experiments")

    def test_try_set_after_race_condition_with_hit(self, exp_config, new_config):
        """Cannot set a configuration after init if config is built
        from no-hit (without up-to-date db info) and new exp is hit

        The experiment from process which first writes to db is initialized
        properly. The experiment which looses the race condition cannot be
        initialized and needs to be rebuilt.
        """
        # Another experiment gets configured first
        naughty_little_exp = Experiment(new_config['name'])
        assert naughty_little_exp.id is None
        experiment_count_before = naughty_little_exp._db.count("experiments")
        naughty_little_exp.configure(copy.deepcopy(new_config))
        assert naughty_little_exp._init_done is True

        exp = Experiment(new_config['name'])
        assert exp._init_done is False
        assert (experiment_count_before + 1) == exp._db.count("experiments")
        # Experiment with hit won't be able to be configured with config without db info
        with pytest.raises(DuplicateKeyError) as exc_info:
            exp.configure(new_config)
        assert 'Cannot register an existing experiment with a new config' in str(exc_info.value)

        assert (experiment_count_before + 1) == exp._db.count("experiments")

        new_config['metadata']['datetime'] = naughty_little_exp.metadata['datetime']
        exp = Experiment(new_config['name'])
        assert exp._init_done is False
        assert (experiment_count_before + 1) == exp._db.count("experiments")
        # New experiment will be able to be configured
        exp.configure(new_config)

        assert (experiment_count_before + 1) == exp._db.count("experiments")

    def test_try_reset_after_race_condition(self, exp_config, new_config):
        """Cannot set a configuration after init if it looses a race condition,
        but can set it if reloaded.

        The experiment from process which first writes to db is initialized
        properly. The experiment which looses the race condition cannot be
        initialized and needs to be rebuilt.
        """
        exp = Experiment(new_config['name'])
        # Another experiment gets configured first
        experiment_count_before = exp._db.count("experiments")
        naughty_little_exp = Experiment(new_config['name'])
        naughty_little_exp.configure(new_config)
        assert naughty_little_exp._init_done is True
        assert exp._init_done is False
        assert (experiment_count_before + 1) == exp._db.count("experiments")
        # First experiment won't be able to be configured
        with pytest.raises(DuplicateKeyError) as exc_info:
            exp.configure(new_config)
        assert 'duplicate key error' in str(exc_info.value)

        # Still not more experiment in DB
        assert (experiment_count_before + 1) == exp._db.count("experiments")

        # Retry configuring the experiment
        new_config['metadata']['datetime'] = naughty_little_exp.metadata['datetime']
        exp = Experiment(new_config['name'])
        exp.configure(new_config)
        assert exp._init_done is True
        assert (experiment_count_before + 1) == exp._db.count("experiments")
        assert exp.configuration == naughty_little_exp.configuration

    def test_after_init_algorithms_are_objects(self, exp_config):
        """Attribute exp.algorithms become objects after init."""
        exp = Experiment('supernaedo2')
        # Deliver an external configuration to finalize init
        exp.configure(exp_config[0][0])
        assert isinstance(exp.algorithms, BaseAlgorithm)

    @pytest.mark.skip(reason="To be implemented...")
    def test_after_init_refers_are_objects(self, exp_config):
        """Attribute exp.refers become objects after init."""
        pass

    def test_algorithm_config_with_just_a_string(self, exp_config):
        """Test that configuring an algorithm with just a string is OK."""
        new_config = copy.deepcopy(exp_config[0][1])
        new_config['algorithms'] = 'dumbalgo'
        exp = Experiment('supernaedo3')
        exp.configure(new_config)
        new_config['algorithms'] = dict()
        new_config['algorithms']['dumbalgo'] = dict()
        new_config['algorithms']['dumbalgo']['done'] = False
        new_config['algorithms']['dumbalgo']['judgement'] = None
        new_config['algorithms']['dumbalgo']['scoring'] = 0
        new_config['algorithms']['dumbalgo']['suspend'] = False
        new_config['algorithms']['dumbalgo']['value'] = 5
        new_config['algorithms']['dumbalgo']['seed'] = None
        assert exp._id == new_config.pop('_id')
        assert exp.configuration['algorithms'] == new_config['algorithms']

    @pytest.mark.usefixtures("trial_id_substitution")
    def test_status_is_pending_when_increase_max_trials(self, exp_config):
        """Attribute exp.algorithms become objects after init."""
        exp = Experiment('supernaedo4')

        # Deliver an external configuration to finalize init
        exp.configure(exp_config[0][2])

        assert exp.is_done

        exp = Experiment('supernaedo4')
        # Deliver an external configuration to finalize init
        exp_config[0][2]['max_trials'] = 1000
        exp.configure(exp_config[0][2])

        assert not exp.is_done


@pytest.mark.usefixtures("create_db_instance", "with_user_bouthilx")
def test_forcing_user(exp_config):
    """Trying to set by forcing user so that NO differences are found."""
    assert getpass.getuser() == 'bouthilx'
    exp = Experiment('supernaedo2')
    assert exp.metadata['user'] == 'bouthilx'
    exp = Experiment('supernaedo2', 'tsirif')
    # Deliver an external configuration to finalize init
    exp_config[0][0]['max_trials'] = 5000
    exp.configure(exp_config[0][0])
    exp_config[0][0]['algorithms']['dumbalgo']['done'] = False
    exp_config[0][0]['algorithms']['dumbalgo']['judgement'] = None
    exp_config[0][0]['algorithms']['dumbalgo']['scoring'] = 0
    exp_config[0][0]['algorithms']['dumbalgo']['suspend'] = False
    exp_config[0][0]['algorithms']['dumbalgo']['value'] = 5
    exp_config[0][0]['algorithms']['dumbalgo']['seed'] = None
    exp_config[0][0]['producer']['strategy'] = "NoParallelStrategy"
    assert exp._id == exp_config[0][0].pop('_id')
    assert exp.configuration == exp_config[0][0]


class TestReserveTrial(object):
    """Calls to interface `Experiment.reserve_trial`."""

    def test_reserve_none(self):
        """Find nothing, return None."""
        try:
            Database(of_type=('orion.core.io.database.mongodb', 'MongoDB'), name='orion_test',
                     username='user', password='pass')
        except (TypeError, ValueError):
            pass
        exp = Experiment('supernaekei')
        trial = exp.reserve_trial()
        assert trial is None

    @pytest.mark.usefixtures("patch_sample")
    def test_reserve_success(self, exp_config, hacked_exp, random_dt):
        """Successfully find new trials in db and reserve one at 'random'."""
        trial = hacked_exp.reserve_trial()
        exp_config[1][5]['status'] = 'reserved'
        assert trial.to_dict() == exp_config[1][5]

    @pytest.mark.usefixtures("patch_sample2")
    def test_reserve_success2(self, exp_config, hacked_exp):
        """Successfully find new trials in db and reserve one at 'random'.

        Version that start_time does not get written, because the selected trial
        was not a 'new' one.
        """
        trial = hacked_exp.reserve_trial()
        exp_config[1][6]['status'] = 'reserved'
        assert trial.to_dict() == exp_config[1][6]

    @pytest.mark.usefixtures("patch_sample_concurrent")
    def test_reserve_race_condition(self, exp_config, hacked_exp, random_dt):
        """Get its trials reserved before by another process once."""
        trial = hacked_exp.reserve_trial()
        exp_config[1][3]['status'] = 'reserved'
        exp_config[1][3]['start_time'] = random_dt
        assert trial.to_dict() == exp_config[1][3]

    @pytest.mark.usefixtures("patch_sample_concurrent2")
    def test_reserve_dead_race_condition(self, exp_config, hacked_exp):
        """Always get its trials reserved before by another process."""
        trial = hacked_exp.reserve_trial()
        assert trial is None

    def test_reserve_with_uncallable_score(self, hacked_exp):
        """Reserve with a score object that cannot do its job."""
        with pytest.raises(ValueError):
            hacked_exp.reserve_trial(score_handle='asfa')

    def fake_handle(self, xxx):
        """Fake score handle for testing."""
        self.times_called += 1
        return self.times_called

    def test_reserve_with_score(self, hacked_exp, exp_config):
        """Reserve with a score object that can do its job."""
        self.times_called = 0
        hacked_exp.configure(exp_config[0][3])
        trial = hacked_exp.reserve_trial(score_handle=self.fake_handle)
        exp_config[1][6]['status'] = 'reserved'
        assert trial.to_dict() == exp_config[1][6]


@pytest.mark.usefixtures("patch_sample")
def test_push_completed_trial(hacked_exp, database, random_dt):
    """Successfully push a completed trial into database."""
    trial = hacked_exp.reserve_trial()
    trial.results = []
    res = Trial.Result(name='yolo', type='objective', value='3')
    trial.results.append(res)
    hacked_exp.push_completed_trial(trial)
    yo = database.trials.find_one({'_id': trial.id})
    assert len(yo['results']) == len(trial.results)
    assert yo['results'][0] == trial.results[0].to_dict()
    assert yo['status'] == 'completed'
    assert yo['end_time'] == random_dt


@pytest.mark.usefixtures("with_user_tsirif")
def test_register_trials(database, random_dt, hacked_exp):
    """Register a list of newly proposed trials/parameters."""
    hacked_exp._id = 'lalala'  # white box hack
    trials = [
        Trial(params=[{'name': 'a', 'type': 'integer', 'value': 5}]),
        Trial(params=[{'name': 'b', 'type': 'integer', 'value': 6}]),
        ]
    for trial in trials:
        hacked_exp.register_trial(trial)
    yo = list(database.trials.find({'experiment': hacked_exp._id}))
    assert len(yo) == len(trials)
    assert yo[0]['params'] == list(map(lambda x: x.to_dict(), trials[0].params))
    assert yo[1]['params'] == list(map(lambda x: x.to_dict(), trials[1].params))
    assert yo[0]['status'] == 'new'
    assert yo[1]['status'] == 'new'
    assert yo[0]['submit_time'] == random_dt
    assert yo[1]['submit_time'] == random_dt


def test_fetch_all_trials(hacked_exp, exp_config, random_dt):
    """Fetch a list of all trials"""
    query = dict()
    trials = hacked_exp.fetch_trials(query)
    assert len(trials) == 7
    for i in range(7):
        assert trials[i].to_dict() == exp_config[1][i]


def test_fetch_completed_trials(hacked_exp, exp_config, random_dt):
    """Fetch a list of the unseen yet completed trials."""
    trials = hacked_exp.fetch_completed_trials()
    assert hacked_exp._last_fetched == random_dt
    assert len(trials) == 3
    assert trials[0].to_dict() == exp_config[1][0]
    assert trials[1].to_dict() == exp_config[1][1]
    assert trials[2].to_dict() == exp_config[1][2]


def test_fetch_non_completed_trials(hacked_exp, exp_config):
    """Fetch a list of the trials that are not completed

    trials.status in ['new', 'interrupted', 'suspended', 'broken']
    """
    # Set two of completed trials to broken and reserved to have all possible status
    query = {'status': 'completed', 'experiment': hacked_exp.id}
    database = hacked_exp._db._db
    completed_trials = database.trials.find(query)
    exp_config[1][0]['status'] = 'broken'
    database.trials.update({'_id': completed_trials[0]['_id']}, {'$set': {'status': 'broken'}})
    exp_config[1][2]['status'] = 'reserved'
    database.trials.update({'_id': completed_trials[1]['_id']}, {'$set': {'status': 'reserved'}})

    # Make sure non completed trials and completed trials are set properly for the unit-test
    query = {'status': {'$ne': 'completed'}, 'experiment': hacked_exp.id}
    non_completed_trials = list(database.trials.find(query))
    assert len(non_completed_trials) == 6
    # Make sure we have all type of status except completed
    assert (set(trial['status'] for trial in non_completed_trials) ==
            set(['new', 'reserved', 'suspended', 'interrupted', 'broken']))

    trials = hacked_exp.fetch_noncompleted_trials()
    assert len(trials) == 6

    def find_and_compare(trial_config):
        """Find the trial corresponding to given config and compare it"""
        trial = [trial for trial in trials if trial.id == trial_config['_id']]
        assert len(trial) == 1
        trial = trial[0]
        assert trial.to_dict() == trial_config

    find_and_compare(exp_config[1][0])
    find_and_compare(exp_config[1][2])
    find_and_compare(exp_config[1][3])
    find_and_compare(exp_config[1][4])
    find_and_compare(exp_config[1][5])
    find_and_compare(exp_config[1][6])


def test_is_done_property(hacked_exp):
    """Check experiment stopping conditions for maximum number of trials completed."""
    assert hacked_exp.is_done is False
    hacked_exp.max_trials = 2
    assert hacked_exp.is_done is True


def test_is_done_property_with_algo(hacked_exp):
    """Check experiment stopping conditions for algo which converged."""
    # Configure experiment to have instantiated algo
    hacked_exp.configure(hacked_exp.configuration)
    assert hacked_exp.is_done is False
    hacked_exp.algorithms.instance.done = True
    assert hacked_exp.is_done is True


def test_experiment_stats(hacked_exp, exp_config, random_dt):
    """Check that property stats is returning a proper summary of experiment's results."""
    stats = hacked_exp.stats
    assert stats['trials_completed'] == 3
    assert stats['best_trials_id'] == exp_config[1][1]['_id']
    assert stats['best_evaluation'] == 2
    assert stats['start_time'] == exp_config[0][3]['metadata']['datetime']
    assert stats['finish_time'] == exp_config[1][2]['end_time']
    assert stats['duration'] == stats['finish_time'] - stats['start_time']
    assert len(stats) == 6


class TestInitExperimentView(object):
    """Create new ExperimentView instance."""

    @pytest.mark.usefixtures("with_user_tsirif")
    def test_empty_experiment_view(self):
        """Hit user name, but exp_name does not hit the db."""
        with pytest.raises(ValueError) as exc_info:
            ExperimentView('supernaekei')
        assert ("No experiment with given name 'supernaekei' for user 'tsirif'"
                in str(exc_info.value))

    @pytest.mark.usefixtures("with_user_bouthilx")
    def test_empty_experiment_view_due_to_username(self):
        """Hit exp_name, but user's name does not hit the db, create new entry."""
        with pytest.raises(ValueError) as exc_info:
            ExperimentView('supernaedo2')
        assert ("No experiment with given name 'supernaedo2' for user 'bouthilx'"
                in str(exc_info.value))

    @pytest.mark.usefixtures("with_user_tsirif")
    def test_existing_experiment_view(self, create_db_instance, exp_config):
        """Hit exp_name + user's name in the db, fetch most recent entry."""
        exp = ExperimentView('supernaedo2')
        assert exp._experiment._init_done is True
        assert exp._experiment._db._database is create_db_instance
        assert exp._id == exp_config[0][0]['_id']
        assert exp.name == exp_config[0][0]['name']
        assert exp.configuration['refers'] == exp_config[0][0]['refers']
        assert exp.metadata == exp_config[0][0]['metadata']
        assert exp._experiment._last_fetched == exp_config[0][0]['metadata']['datetime']
        assert exp.pool_size == exp_config[0][0]['pool_size']
        assert exp.max_trials == exp_config[0][0]['max_trials']
        assert exp.algorithms.configuration == exp_config[0][0]['algorithms']

        with pytest.raises(AttributeError):
            exp.this_is_not_in_config = 5

        # Test that experiment.push_completed_trial indeed exists
        exp._experiment.push_completed_trial
        with pytest.raises(AttributeError):
            exp.push_completed_trial

        with pytest.raises(AttributeError):
            exp.register_trial

        with pytest.raises(AttributeError):
            exp.reserve_trial


def test_fetch_completed_trials_from_view(hacked_exp, exp_config, random_dt):
    """Fetch a list of the unseen yet completed trials."""
    experiment_view = ExperimentView(hacked_exp.name)
    experiment_view._experiment = hacked_exp

    trials = experiment_view.fetch_completed_trials()
    assert experiment_view._experiment._last_fetched == random_dt
    assert len(trials) == 3
    assert trials[0].to_dict() == exp_config[1][0]
    assert trials[1].to_dict() == exp_config[1][1]
    assert trials[2].to_dict() == exp_config[1][2]


def test_view_is_done_property(hacked_exp):
    """Check experiment stopping conditions accessed from view."""
    experiment_view = ExperimentView(hacked_exp.name)
    experiment_view._experiment = hacked_exp

    # Fully configure wrapper experiment (should normally occur inside ExperimentView.__init__
    # but hacked_exp has been _hacked_ inside afterwards.
    hacked_exp.configure(hacked_exp.configuration)

    assert experiment_view.is_done is False

    with pytest.raises(AttributeError):
        experiment_view.max_trials = 2

    hacked_exp.max_trials = 2

    assert experiment_view.is_done is True


def test_view_algo_is_done_property(hacked_exp):
    """Check experiment's algo stopping conditions accessed from view."""
    experiment_view = ExperimentView(hacked_exp.name)
    experiment_view._experiment = hacked_exp

    # Fully configure wrapper experiment (should normally occur inside ExperimentView.__init__
    # but hacked_exp has been _hacked_ inside afterwards.
    hacked_exp.configure(hacked_exp.configuration)

    assert experiment_view.is_done is False

    hacked_exp.algorithms.instance.done = True

    assert experiment_view.is_done is True


def test_experiment_view_stats(hacked_exp, exp_config, random_dt):
    """Check that property stats from view is consistent."""
    experiment_view = ExperimentView(hacked_exp.name)
    experiment_view._experiment = hacked_exp

    stats = experiment_view.stats
    assert stats['trials_completed'] == 3
    assert stats['best_trials_id'] == exp_config[1][1]['_id']
    assert stats['best_evaluation'] == 2
    assert stats['start_time'] == exp_config[0][3]['metadata']['datetime']
    assert stats['finish_time'] == exp_config[1][2]['end_time']
    assert stats['duration'] == stats['finish_time'] - stats['start_time']
    assert len(stats) == 6


@pytest.mark.usefixtures("with_user_tsirif")
def test_experiment_view_db_read_only():
    """Verify that wrapper experiments' database is read-only"""
    exp = ExperimentView('supernaedo2')

    # Test that database.write indeed exists
    exp._experiment._db._database.write
    with pytest.raises(AttributeError):
        exp._experiment._db.write


class TestInitExperimentWithEVC(object):
    """Create new Experiment instance with EVC."""

    @pytest.mark.usefixtures("with_user_tsirif")
    def test_new_experiment_with_parent(self, create_db_instance, random_dt, exp_config):
        """Configure a branch experiment."""
        exp = Experiment('supernaedo2.6')
        exp.metadata = exp_config[0][4]['metadata']
        exp.refers = exp_config[0][4]['refers']
        exp.algorithms = exp_config[0][4]['algorithms']
        exp.configure(exp.configuration)
        assert exp._init_done is True
        assert exp._db is create_db_instance
        assert exp._id is not None
        assert exp.name == 'supernaedo2.6'
        assert exp.configuration['refers'] == exp_config[0][4]['refers']
        exp_config[0][4]['metadata']['datetime'] = random_dt
        assert exp.metadata == exp_config[0][4]['metadata']
        assert exp._last_fetched == random_dt
        assert exp.pool_size is None
        assert exp.max_trials is None
        assert exp.configuration['algorithms'] == {'random': {'seed': None}}

    @pytest.mark.usefixtures("with_user_tsirif")
    def test_experiment_with_parent(self, create_db_instance, random_dt, exp_config):
        """Configure an existing experiment with parent."""
        exp = Experiment('supernaedo2.1')
        exp.algorithms = {'random': {'seed': None}}
        exp.configure(exp.configuration)
        assert exp._init_done is True
        assert exp._db is create_db_instance
        assert exp._id is not None
        assert exp.name == 'supernaedo2.1'
        assert exp.configuration['refers'] == exp_config[0][4]['refers']
        assert exp.metadata == exp_config[0][4]['metadata']
        assert exp.pool_size == 2
        assert exp.max_trials == 1000
        assert exp.configuration['algorithms'] == {'random': {'seed': None}}

    @pytest.mark.usefixtures("with_user_tsirif")
    def test_experiment_non_interactive_branching(self, create_db_instance, random_dt, exp_config,
                                                  monkeypatch):
        """Configure an existing experiment with parent."""
        monkeypatch.setattr('sys.__stdin__.isatty', lambda: True)
        exp = Experiment('supernaedo2.1')
        exp.algorithms = {'dumbalgo': {}}
        with pytest.raises(OSError):
            exp.configure(exp.configuration)
        monkeypatch.undo()
        with pytest.raises(ValueError) as exc_info:
            exp.configure(exp.configuration)
        assert "Configuration is different and generates a branching" in str(exc_info.value)
