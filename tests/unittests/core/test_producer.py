#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Collection of tests for :mod:`orion.core.worker.producer`."""
import copy

import pytest

from orion.core.worker.producer import Producer
from orion.core.worker.trial import Trial


class DumbParallelStrategy:
    """Mock object for parallel strategy"""

    def observe(self, points, results):
        """See BaseParallelStrategy.observe"""
        self._observed_points = points
        self._observed_results = results
        self._value = None

    def lie(self, trial):
        """See BaseParallelStrategy.lie"""
        if self._value:
            value = self._value
        else:
            value = len(self._observed_points)

        self._lie = lie = Trial.Result(name='lie', type='lie', value=value)
        return lie


@pytest.fixture()
def producer(hacked_exp, random_dt, exp_config, categorical_values):
    """Return a setup `Producer`."""
    # make init done

    # TODO: Remove this commented out if test pass
    # hacked_exp.configure(exp_config[0][3])
    # # insert fake point
    # fake_point = ('gru', 'rnn')
    # assert fake_point in hacked_exp.space
    # hacked_exp.algorithms.value = fake_point

    hacked_exp.configure(exp_config[0][3])
    hacked_exp.pool_size = 1
    hacked_exp.algorithms.instance.possible_values = categorical_values
    hacked_exp.algorithms.seed_rng(0)

    hacked_exp.producer['strategy'] = DumbParallelStrategy()

    return Producer(hacked_exp)


def test_algo_observe_completed(producer):
    """Test that algo only observes completed trials"""
    assert len(producer.experiment.fetch_trials({})) > 3
    producer.update()
    # Algorithm must have received completed points and their results
    obs_points = producer.algorithm._points
    obs_results = producer.algorithm._results
    assert len(obs_points) == 3
    assert obs_points[0] == ('lstm', 'rnn')
    assert obs_points[1] == ('rnn', 'rnn')
    assert obs_points[2] == ('gru', 'lstm_with_attention')
    assert len(obs_results) == 3
    assert obs_results[0] == {
        'objective': 3,
        'gradient': None,
        'constraint': []
        }
    assert obs_results[1] == {
        'objective': 2,
        'gradient': (-0.1, 2),
        'constraint': []
        }
    assert obs_results[2] == {
        'objective': 10,
        'gradient': (5, 3),
        'constraint': [1.2]
        }


def test_strategist_observe_completed(producer):
    """Test that strategist only observes completed trials"""
    assert len(producer.experiment.fetch_trials({})) > 3
    producer.update()
    # Algorithm must have received completed points and their results
    obs_points = producer.strategy._observed_points
    obs_results = producer.strategy._observed_results
    assert len(obs_points) == 3
    assert obs_points[0] == ('lstm', 'rnn')
    assert obs_points[1] == ('rnn', 'rnn')
    assert obs_points[2] == ('gru', 'lstm_with_attention')
    assert len(obs_results) == 3
    assert obs_results[0] == {
        'objective': 3,
        'gradient': None,
        'constraint': []
        }
    assert obs_results[1] == {
        'objective': 2,
        'gradient': (-0.1, 2),
        'constraint': []
        }
    assert obs_results[2] == {
        'objective': 10,
        'gradient': (5, 3),
        'constraint': [1.2]
        }


def test_naive_algorithm_is_producing(producer, database, random_dt):
    """Verify naive algo is used to produce, not original algo"""
    producer.experiment.pool_size = 1
    producer.algorithm.instance.possible_values = [('rnn', 'gru')]
    producer.update()
    producer.algorithm.instance.possible_values = [('gru', 'gru')]
    producer.produce()

    assert producer.naive_algorithm._num == 1  # pool size
    assert producer.algorithm._num == 1
    assert producer.naive_algorithm._suggested == [('rnn', 'gru')]
    assert producer.algorithm._suggested == [('gru', 'gru')]

    # Make sure the registered trial is the one sampled from naive not original algo
    new_trials = list(database.trials.find({'status': 'new', 'submit_time': random_dt}))
    assert len(new_trials) == 1
    assert new_trials[0]['experiment'] == producer.experiment.name
    assert new_trials[0]['start_time'] is None
    assert new_trials[0]['end_time'] is None
    assert new_trials[0]['results'] == []
    assert new_trials[0]['params'] == [
        {'name': '/encoding_layer',
         'type': 'categorical',
         'value': 'rnn'},
        {'name': '/decoding_layer',
         'type': 'categorical',
         'value': 'gru'}
        ]


def test_update_and_produce(producer, database, random_dt):
    """Test new trials are properly produced"""
    possible_values = [('rnn', 'gru')]
    producer.experiment.pool_size = 1
    producer.experiment.algorithms.instance.possible_values = possible_values

    producer.update()
    producer.produce()

    # Algorithm was ordered to suggest some trials
    num_new_points = producer.naive_algorithm._num
    assert num_new_points == 1  # pool size

    assert producer.naive_algorithm._suggested == possible_values


def test_register_new_trials(producer, database, random_dt):
    """Test new trials are properly registered"""
    trials_in_db_before = database.trials.count()
    new_trials_in_db_before = database.trials.count({'status': 'new'})

    producer.experiment.pool_size = 1
    producer.experiment.algorithms.instance.possible_values = [('rnn', 'gru')]

    producer.update()
    producer.produce()

    # Algorithm was ordered to suggest some trials
    num_new_points = producer.naive_algorithm._num
    assert num_new_points == 1  # pool size

    # `num_new_points` new trials were registered at database
    assert database.trials.count() == trials_in_db_before + 1
    assert database.trials.count({'status': 'new'}) == new_trials_in_db_before + 1
    new_trials = list(database.trials.find({'status': 'new', 'submit_time': random_dt}))
    assert new_trials[0]['experiment'] == producer.experiment.name
    assert new_trials[0]['start_time'] is None
    assert new_trials[0]['end_time'] is None
    assert new_trials[0]['results'] == []
    assert new_trials[0]['params'] == [
        {'name': '/encoding_layer',
         'type': 'categorical',
         'value': 'rnn'},
        {'name': '/decoding_layer',
         'type': 'categorical',
         'value': 'gru'}
        ]


def test_no_lies_if_all_trials_completed(producer, database, random_dt):
    """Verify that no lies are created if all trials are completed"""
    query = {'status': {'$ne': 'completed'}, 'experiment': producer.experiment.id}
    database.trials.remove(query)
    trials_in_db_before = database.trials.count({'experiment': producer.experiment.id})
    assert trials_in_db_before == 3

    producer.update()

    assert len(producer._produce_lies()) == 0


def test_lies_generation(producer, database, random_dt):
    """Verify that lies are created properly"""
    query = {'status': {'$ne': 'completed'}, 'experiment': producer.experiment.id}
    trials_non_completed = list(database.trials.find(query))
    assert len(trials_non_completed) == 4
    query = {'status': 'completed', 'experiment': producer.experiment.id}
    trials_completed = list(database.trials.find(query))
    assert len(trials_completed) == 3

    producer.update()

    lies = producer._produce_lies()
    assert len(lies) == 4

    for i in range(4):
        trials_non_completed[i]['_id'] = lies[i].id
        trials_non_completed[i]['status'] = 'completed'
        trials_non_completed[i]['end_time'] = random_dt
        trials_non_completed[i]['results'].append(producer.strategy._lie.to_dict())
        trials_non_completed[i]['parents'] = set([trial['_id'] for trial in trials_completed])
        lies_dict = lies[i].to_dict()
        lies_dict['parents'] = set(lies_dict['parents'])
        assert lies_dict == trials_non_completed[i]


def test_register_lies(producer, database, random_dt):
    """Verify that lies are registed in DB properly"""
    query = {'status': {'$ne': 'completed'}, 'experiment': producer.experiment.id}
    trials_non_completed = list(database.trials.find(query))
    assert len(trials_non_completed) == 4
    query = {'status': 'completed', 'experiment': producer.experiment.id}
    trials_completed = list(database.trials.find(query))
    assert len(trials_completed) == 3

    producer.update()
    producer._produce_lies()

    lying_trials = list(database.lying_trials.find({'experiment': producer.experiment.id}))
    assert len(lying_trials) == 4

    for i in range(4):
        trials_non_completed[i]['_id'] = lying_trials[i]['_id']
        trials_non_completed[i]['status'] = 'completed'
        trials_non_completed[i]['end_time'] = random_dt
        trials_non_completed[i]['results'].append(producer.strategy._lie.to_dict())
        trials_non_completed[i]['parents'] = set([trial['_id'] for trial in trials_completed])
        lying_trials[i]['parents'] = set(lying_trials[i]['parents'])
        assert lying_trials[i] == trials_non_completed[i]


def test_register_duplicate_lies(producer, database, random_dt):
    """Verify that duplicate lies are not registered twice in DB"""
    query = {'status': {'$ne': 'completed'}, 'experiment': producer.experiment.id}
    trials_non_completed = list(database.trials.find(query))
    assert len(trials_non_completed) == 4

    # Overwrite value of lying result of the strategist so that all lying trials have the same value
    # otherwise they would not be exact duplicates.
    producer.strategy._value = 4

    # Set specific output value for to algo to ensure successful creation of a new trial.
    producer.experiment.pool_size = 1
    producer.experiment.algorithms.instance.possible_values = [('rnn', 'gru')]

    producer.update()
    assert len(producer._produce_lies()) == 4
    lying_trials = list(database.lying_trials.find({'experiment': producer.experiment.id}))
    assert len(lying_trials) == 4

    # Create a new point to make sure additional non-completed trials increase number of lying
    # trials generated
    producer.produce()

    trials_non_completed = list(database.trials.find(query))
    assert len(trials_non_completed) == 5

    producer.update()

    assert len(producer._produce_lies()) == 5
    lying_trials = list(database.lying_trials.find({'experiment': producer.experiment.id}))
    assert len(lying_trials) == 5

    # Make sure trying to generate again does not add more fake trials since they are identical
    assert len(producer._produce_lies()) == 5
    lying_trials = list(database.lying_trials.find({'experiment': producer.experiment.id}))
    assert len(lying_trials) == 5


def test_register_duplicate_lies_with_different_results(producer, database, random_dt):
    """Verify that duplicate lies with different results are all registered in DB"""
    query = {'status': {'$ne': 'completed'}, 'experiment': producer.experiment.id}
    trials_non_completed = list(database.trials.find(query))
    assert len(trials_non_completed) == 4

    # Overwrite value of lying result to force different results.
    producer.strategy._value = 11

    assert len(producer._produce_lies()) == 4
    lying_trials = list(database.lying_trials.find({'experiment': producer.experiment.id}))
    assert len(lying_trials) == 4

    # Overwrite value of lying result to force different results.
    producer.strategy._value = new_lying_value = 12

    lying_trials = producer._produce_lies()
    assert len(lying_trials) == 4
    nb_lying_trials = database.lying_trials.count({'experiment': producer.experiment.id})
    assert nb_lying_trials == 4 + 4
    assert lying_trials[0].lie.value == new_lying_value


def test_naive_algo_not_trained_when_all_trials_completed(producer, database, random_dt):
    """Verify that naive algo is not trained on additional trials when all completed"""
    query = {'status': {'$ne': 'completed'}, 'experiment': producer.experiment.id}
    database.trials.remove(query)
    trials_in_db_before = database.trials.count({'experiment': producer.experiment.id})
    assert trials_in_db_before == 3

    producer.update()

    assert len(producer.algorithm._points) == 3
    assert len(producer.naive_algorithm._points) == 3


def test_naive_algo_trained_on_all_non_completed_trials(producer, database, random_dt):
    """Verify that naive algo is trained on additional trials"""
    # Set two of completed trials to broken and reserved to have all possible status
    query = {'status': 'completed', 'experiment': producer.experiment.id}
    completed_trials = database.trials.find(query)
    database.trials.update({'_id': completed_trials[0]['_id']}, {'$set': {'status': 'broken'}})
    database.trials.update({'_id': completed_trials[1]['_id']}, {'$set': {'status': 'reserved'}})

    # Make sure non completed trials and completed trials are set properly for the unit-test
    query = {'status': {'$ne': 'completed'}, 'experiment': producer.experiment.id}
    non_completed_trials = list(database.trials.find(query))
    assert len(non_completed_trials) == 6
    # Make sure we have all type of status except completed
    assert (set(trial['status'] for trial in non_completed_trials) ==
            set(['new', 'reserved', 'suspended', 'interrupted', 'broken']))
    query = {'status': 'completed', 'experiment': producer.experiment.id}
    assert database.trials.count(query) == 1

    # Executing the actual test
    producer.update()
    assert len(producer._produce_lies()) == 6

    assert len(producer.algorithm._points) == 1
    assert len(producer.naive_algorithm._points) == (1 + 6)


def test_naive_algo_is_discared(producer, database, monkeypatch):
    """Verify that naive algo is discarded and recopied from original algo"""
    # Get rid of the mock on datetime.datetime.utcnow() otherwise fetch_completed_trials always
    # fetch all trials since _last_fetched never changes.
    monkeypatch.undo()

    # Set values for predictions
    producer.experiment.pool_size = 1
    producer.experiment.algorithms.instance.possible_values = [('rnn', 'gru')]

    producer.update()
    assert len(producer._produce_lies()) == 4

    first_naive_algorithm = producer.naive_algorithm

    assert len(producer.algorithm._points) == 3
    assert len(first_naive_algorithm._points) == (3 + 4)

    producer.produce()

    # Only update the original algo, naive algo is still not discarded
    producer._update_algorithm()
    assert len(producer.algorithm._points) == 3
    assert first_naive_algorithm == producer.naive_algorithm
    assert len(producer.naive_algorithm._points) == (3 + 4)

    # Discard naive algo and create a new one, now trained on 5 points.
    producer._update_naive_algorithm()
    assert first_naive_algorithm != producer.naive_algorithm
    assert len(producer.naive_algorithm._points) == (3 + 5)


def test_concurent_producers(producer, database, random_dt):
    """Test concurrent production of new trials."""
    trials_in_db_before = database.trials.count()
    new_trials_in_db_before = database.trials.count({'status': 'new'})

    print(producer.experiment.fetch_trials({}))

    # Set so that first producer's algorithm generate valid point on first time
    # And second producer produce same point and thus must produce next one two.
    # Hence, we know that producer algo will have _num == 1 and
    # second producer algo will have _num == 2
    producer.algorithm.instance.possible_values = [('rnn', 'gru'), ('gru', 'gru')]
    # Make sure it starts from index 0
    producer.algorithm.seed_rng(0)

    assert producer.experiment.pool_size == 1

    second_producer = Producer(producer.experiment)
    second_producer.algorithm = copy.deepcopy(producer.algorithm)

    producer.update()
    second_producer.update()

    print(producer.algorithm._index)
    print(second_producer.algorithm._index)
    producer.produce()
    print(producer.algorithm._index)
    print(second_producer.algorithm._index)
    second_producer.produce()

    # Algorithm was required to suggest some trials
    num_new_points = producer.algorithm._num
    assert num_new_points == 1  # pool size
    num_new_points = second_producer.algorithm._num
    assert num_new_points == 2  # pool size

    # `num_new_points` new trials were registered at database
    assert database.trials.count() == trials_in_db_before + 2
    assert database.trials.count({'status': 'new'}) == new_trials_in_db_before + 2
    new_trials = list(database.trials.find({'status': 'new', 'submit_time': random_dt}))
    assert new_trials[0]['experiment'] == producer.experiment.name
    assert new_trials[0]['start_time'] is None
    assert new_trials[0]['end_time'] is None
    assert new_trials[0]['results'] == []
    assert new_trials[0]['params'] == [
        {'name': '/encoding_layer',
         'type': 'categorical',
         'value': 'rnn'},
        {'name': '/decoding_layer',
         'type': 'categorical',
         'value': 'gru'}
        ]

    assert new_trials[1]['params'] == [
        {'name': '/encoding_layer',
         'type': 'categorical',
         'value': 'gru'},
        {'name': '/decoding_layer',
         'type': 'categorical',
         'value': 'gru'}
        ]


def test_duplicate_within_pool(producer, database, random_dt):
    """Test that an algo suggesting multiple points can have a few registered even
    if one of them is a duplicate.
    """
    trials_in_db_before = database.trials.count()
    new_trials_in_db_before = database.trials.count({'status': 'new'})

    producer.experiment.pool_size = 2

    producer.experiment.algorithms.instance.possible_values = [
        ('rnn', 'gru'), ('rnn', 'gru'), ('gru', 'gru')]

    producer.update()
    producer.produce()

    # Algorithm was required to suggest some trials
    num_new_points = producer.algorithm._num
    assert num_new_points == 4  # 2 * pool size

    # `num_new_points` new trials were registered at database
    assert database.trials.count() == trials_in_db_before + 2
    assert database.trials.count({'status': 'new'}) == new_trials_in_db_before + 2
    new_trials = list(database.trials.find({'status': 'new', 'submit_time': random_dt}))
    assert new_trials[0]['experiment'] == producer.experiment.name
    assert new_trials[0]['start_time'] is None
    assert new_trials[0]['end_time'] is None
    assert new_trials[0]['results'] == []
    assert new_trials[0]['params'] == [
        {'name': '/encoding_layer',
         'type': 'categorical',
         'value': 'rnn'},
        {'name': '/decoding_layer',
         'type': 'categorical',
         'value': 'gru'}
        ]

    assert new_trials[1]['params'] == [
        {'name': '/encoding_layer',
         'type': 'categorical',
         'value': 'gru'},
        {'name': '/decoding_layer',
         'type': 'categorical',
         'value': 'gru'}
        ]


def test_duplicate_within_pool_and_db(producer, database, random_dt):
    """Test that an algo suggesting multiple points can have a few registered even
    if one of them is a duplicate with db.
    """
    trials_in_db_before = database.trials.count()
    new_trials_in_db_before = database.trials.count({'status': 'new'})

    producer.experiment.pool_size = 2

    producer.experiment.algorithms.instance.possible_values = [
        ('rnn', 'gru'), ('rnn', 'rnn'), ('gru', 'gru')]

    producer.update()
    producer.produce()

    # Algorithm was required to suggest some trials
    num_new_points = producer.algorithm._num
    assert num_new_points == 4  # pool size

    # `num_new_points` new trials were registered at database
    assert database.trials.count() == trials_in_db_before + 2
    assert database.trials.count({'status': 'new'}) == new_trials_in_db_before + 2
    new_trials = list(database.trials.find({'status': 'new', 'submit_time': random_dt}))
    assert new_trials[0]['experiment'] == producer.experiment.name
    assert new_trials[0]['start_time'] is None
    assert new_trials[0]['end_time'] is None
    assert new_trials[0]['results'] == []
    assert new_trials[0]['params'] == [
        {'name': '/encoding_layer',
         'type': 'categorical',
         'value': 'rnn'},
        {'name': '/decoding_layer',
         'type': 'categorical',
         'value': 'gru'}
        ]

    assert new_trials[1]['params'] == [
        {'name': '/encoding_layer',
         'type': 'categorical',
         'value': 'gru'},
        {'name': '/decoding_layer',
         'type': 'categorical',
         'value': 'gru'}
        ]


def test_exceed_max_attempts(producer, database, random_dt):
    """Test that RuntimeError is raised when algo keep suggesting the same points"""
    producer.max_attempts = 10  # to limit run-time, default would work as well.
    producer.experiment.algorithms.instance.possible_values = [('rnn', 'rnn')]

    assert producer.experiment.pool_size == 1

    producer.update()

    with pytest.raises(RuntimeError) as exc_info:
        producer.produce()
    assert "Looks like the algorithm keeps suggesting" in str(exc_info.value)


def test_original_seeding(producer, database):
    """Verify that rng state in original algo changes when duplicate trials is discarded"""
    assert producer.experiment.pool_size == 1

    producer.algorithm.seed_rng(0)

    assert producer.algorithm._index == 0

    producer.update()
    producer.produce()

    prev_index = producer.algorithm._index
    prev_suggested = producer.algorithm._suggested
    assert prev_index > 0

    # Force the algo back to 1 to make sure the RNG state of original algo keeps incrementing.
    # This is necessary because naive_algo is recopied from original algo and thus would always get
    # the same RNG state if the original algo RNG state would not increment.
    # See `Producer.produce` to observe the dummy `self.algorith.suggest()` used to increment
    # original algo's RNG state.
    producer.algorithm.seed_rng(0)

    producer.update()
    producer.produce()

    assert prev_suggested != producer.algorithm._suggested
    assert prev_index < producer.algorithm._index
