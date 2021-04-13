# -*- coding: utf-8 -*-
"""
Produce and register samples to try
===================================

Suggest new parameter sets which optimize the objective.

"""
from typing import Optional, List
import copy
import logging
import random
import time

import orion.core
from orion.core.io.database import DuplicateKeyError
from orion.core.utils import format_trials
from orion.core.utils.exceptions import SampleTimeout, WaitingForTrials
from orion.core.worker.trial import Trial
from orion.core.worker.trials_history import TrialsHistory

log = logging.getLogger(__name__)


class Producer(object):
    """Produce suggested sets of problem's parameter space to try out.

    It uses an `Experiment` object to poll for not yet observed trials which
    have been already evaluated and to register new suggestions (points of
    the parameter `Space`) to be evaluated.

    """

    def __init__(
        self, experiment, max_idle_time=None, knowledge_base: "KnowledgeBase" = None
    ):
        """Initialize a producer.

        :param experiment: Manager of this experiment, provides convenient
           interface for interacting with the database.
        """
        log.debug("Creating Producer object.")
        self.experiment = experiment
        self.space = experiment.space
        if self.space is None:
            raise RuntimeError(
                "Experiment object provided to Producer has not yet completed"
                " initialization."
            )
        self.algorithm = experiment.algorithms
        if max_idle_time is None:
            max_idle_time = orion.core.config.worker.max_idle_time
        self.max_idle_time = max_idle_time
        self.strategy = experiment.producer["strategy"]
        self.naive_algorithm = None
        # TODO: Move trials_history into PrimaryAlgo during the refactoring of Algorithm with
        #       Strategist and Scheduler.
        self.trials_history = TrialsHistory()
        self.params_hashes = set()
        self.naive_trials_history = None
        self.failure_count = 0

        from warmstart.new_knowledge_base import KnowledgeBase
        self.knowledge_base: Optional[KnowledgeBase] = knowledge_base
        # Indicates wether the algo has been warm-started with the knowledge base.
        self.warm_started = False

    @property
    def pool_size(self):
        """Pool-size of the experiment"""
        return self.experiment.pool_size

    def backoff(self):
        """Wait some time and update algorithm."""
        waiting_time = max(0, random.gauss(1, 0.2))
        log.info("Waiting %d seconds", waiting_time)
        time.sleep(waiting_time)
        log.info("Updating algorithm.")
        self.update()
        self.failure_count += 1

    def _sample_guard(self, start):
        """Check that the time taken sampling is less than max_idle_time"""
        if time.time() - start > self.max_idle_time:
            raise SampleTimeout(
                "Algorithm could not sample new points in less than {} seconds."
                "Failed to sample points {} times".format(
                    self.max_idle_time, self.failure_count
                )
            )

    def produce(self):
        """Create and register new trials."""
        sampled_points = 0

        # reset the number of time we failed to sample points
        self.failure_count = 0
        start = time.time()

        while sampled_points < self.pool_size and not (
            self.experiment.is_done or self.naive_algorithm.is_done
        ):
            self._sample_guard(start)

            log.debug("### Algorithm suggests new points.")
            new_points = self.naive_algorithm.suggest(self.pool_size)

            # Sync state of original algo so that state continues evolving.
            self.algorithm.set_state(self.naive_algorithm.state_dict)

            if new_points is None:
                if self.algorithm.is_done:
                    return

                raise WaitingForTrials(
                    "Algo does not have more trials to sample."
                    "Waiting for current trials to finish"
                )

            for new_point in new_points:
                sampled_points += self.register_trials(new_point)

    def register_trials(self, new_point):
        """Register a new set of sampled parameters into the DB
        guaranteeing their uniqueness

        Parameters
        ----------
        new_point: tuple
            tuple of values representing the hyperparameters values

        """
        # FIXME: Relying on DB to guarantee uniqueness
        # when the trial history will be held by that algo we can move that logic out of the DB

        log.debug("#### Convert point to `Trial` object.")
        new_trial = format_trials.tuple_to_trial(new_point, self.space)

        try:
            self._prevalidate_trial(new_trial)
            new_trial.parents = self.naive_trials_history.children
            log.debug("#### Register new trial to database: %s", new_trial)
            self.experiment.register_trial(new_trial)
            self._update_params_hashes([new_trial])
            return 1

        except DuplicateKeyError:
            log.debug("#### Duplicate sample: %s", new_trial)
            self.backoff()
            return 0

    def _prevalidate_trial(self, new_trial):
        """Verify if trial is not in parent history"""
        if (
            Trial.compute_trial_hash(new_trial, ignore_experiment=True)
            in self.params_hashes
        ):
            raise DuplicateKeyError

    def _update_params_hashes(self, trials):
        """Register locally all param hashes of trials"""
        for trial in trials:
            self.params_hashes.add(
                Trial.compute_trial_hash(trial, ignore_experiment=True, ignore_lie=True)
            )

    def update(self):
        """Pull all trials to update model with completed ones and naive model with non completed
        ones.
        """
        trials = self.experiment.fetch_trials(with_evc_tree=True)
        
        if self.knowledge_base and not self.warm_started:
            # TODO: Dont use the KB when we have enough points in the target task.
            ## Option 1:
            # Get the trials from other 'similar' experiments.
            reusable_trials = self.knowledge_base.get_reusable_trials(self.experiment)
            print(f"Reusable trials: {len(reusable_trials)}")

            if reusable_trials:
                log.debug("### Warm starting")
                self.algorithm.warm_start(reusable_trials)
                self.warm_started = True
            # ## Option 2:

            # from orion.client import ExperimentClient
            # closest_experiment_clients: List[ExperimentClient] = self.knowledge_base.get_closest_experiment_clients(self.experiment)

            # print("Closest experiment clients: ")
            # for experiment_client in closest_experiment_clients:
            #     print(f"exp client: {experiment_client}")
            #     warm_start_trials: List[Trial] = experiment_client.fetch_trials()
            #     print(f"Found {len(warm_start_trials)} trials.")

        self._update_algorithm(
            [trial for trial in trials if trial.status == "completed"]
        )
        self._update_naive_algorithm(
            [trial for trial in trials if trial.status != "completed"]
        )

    def _update_algorithm(self, completed_trials):
        """Pull newest completed trials to update local model."""
        log.debug("### Fetch completed trials to observe:")

        new_completed_trials = []
        for trial in completed_trials:
            if trial not in self.trials_history:
                new_completed_trials.append(trial)

        log.debug("### %s", new_completed_trials)

        if new_completed_trials:
            log.debug("### Convert them to list of points and their results.")
            points = list(
                map(
                    lambda trial: format_trials.trial_to_tuple(trial, self.space),
                    new_completed_trials,
                )
            )
            results = list(map(format_trials.get_trial_results, new_completed_trials))

            log.debug("### Observe them.")
            self.trials_history.update(new_completed_trials)
            self.algorithm.observe(points, results)
            self.strategy.observe(points, results)
            self._update_params_hashes(new_completed_trials)

    def _produce_lies(self, incomplete_trials):
        """Add fake objective results to incomplete trials

        Then register the trials in the db
        """
        log.debug("### Fetch active trials to observe:")
        lying_trials = []
        log.debug("### %s", incomplete_trials)

        for trial in incomplete_trials:
            log.debug("### Use defined ParallelStrategy to assign them fake results.")
            lying_result = self.strategy.lie(trial)
            if lying_result is not None:
                lying_trial = copy.deepcopy(trial)
                lying_trial.results.append(lying_result)
                lying_trials.append(lying_trial)
                log.debug("### Register lie to database: %s", lying_trial)
                lying_trial.parents = self.trials_history.children
                try:
                    self.experiment.register_lie(lying_trial)
                except DuplicateKeyError:
                    log.debug(
                        "#### Duplicate lie. No need to register a duplicate in DB."
                    )

        return lying_trials

    def _update_naive_algorithm(self, incomplete_trials):
        """Pull all non completed trials to update naive model."""
        self.naive_algorithm = copy.deepcopy(self.algorithm)
        self.naive_trials_history = copy.deepcopy(self.trials_history)
        log.debug("### Create fake trials to observe:")
        lying_trials = self._produce_lies(incomplete_trials)
        log.debug("### %s", lying_trials)
        if lying_trials:
            log.debug("### Convert them to list of points and their results.")
            points = list(
                map(
                    lambda trial: format_trials.trial_to_tuple(trial, self.space),
                    lying_trials,
                )
            )
            results = list(map(format_trials.get_trial_results, lying_trials))

            log.debug("### Observe them.")
            self.naive_trials_history.update(lying_trials)
            self.naive_algorithm.observe(points, results)
            self._update_params_hashes(lying_trials)
