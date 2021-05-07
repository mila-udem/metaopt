# -*- coding: utf-8 -*-
# pylint:disable=too-many-lines
"""
Experiment wrapper client
=========================

Wraps the core Experiment object to provide further functionalities for the user
"""
import atexit
import functools
import inspect
import logging
import sys
import traceback

from joblib import Parallel, delayed

import orion.core
import orion.core.utils.format_trials as format_trials
from orion.core.io.database import DuplicateKeyError
from orion.core.utils.exceptions import (
    BrokenExperiment,
    CompletedExperiment,
    InvalidResult,
    SampleTimeout,
    UnsupportedOperation,
    WaitingForTrials,
)
from orion.core.utils.flatten import flatten, unflatten
from orion.core.worker.trial import Trial, TrialCM
from orion.core.worker.trial_pacemaker import TrialPacemaker
from orion.plotting.base import PlotAccessor
from orion.storage.base import FailedUpdate

log = logging.getLogger(__name__)


def set_broken_trials(client):
    """Release all trials with status broken if the process exits without releasing them."""
    if sys.exc_info()[0] is KeyboardInterrupt:
        status = "interrupted"
    else:
        status = "broken"

    for trial_id in list(client._pacemakers.keys()):  # pylint: disable=protected-access
        trial = client.get_trial(uid=trial_id)
        if trial is None:
            log.warning(
                "Trial {} was not found in storage, could not set status to `broken`."
            )
            continue
        client.release(trial, status=status)


def reserve_trial(experiment, producer, _depth=1):
    """Reserve a new trial, or produce and reserve a trial if none are available."""
    log.debug("Trying to reserve a new trial to evaluate.")
    trial = experiment.reserve_trial()

    if trial is None and not experiment.is_done:

        if _depth > 10:
            raise WaitingForTrials(
                "No trials are available at the moment "
                "wait for current trials to finish"
            )

        log.debug("#### Failed to pull a new trial from database.")

        log.debug("#### Fetch most recent completed trials and update algorithm.")
        producer.update()

        log.debug("#### Produce new trials.")
        producer.produce()

        return reserve_trial(experiment, producer, _depth=_depth + 1)

    return trial


# pylint: disable=too-many-public-methods
class ExperimentClient:
    """ExperimentClient providing all functionalities for the python API

    Note that the ExperimentClient is not meant to be instantiated by the user.
    Look at `orion.client.create_experiment` to build an ExperimentClient.

    Parameters
    ----------
    experiment: `orion.core.worker.experiment.Experiment`
        Experiment object serving for interaction with storage
    producer: `orion.core.worker.producer.Producer`
        Producer object used to produce new trials.

    """

    def __init__(self, experiment, producer, heartbeat=None):
        self._experiment = experiment
        self._producer = producer
        self._pacemakers = {}
        self.set_broken_trials = functools.partial(set_broken_trials, client=self)
        if heartbeat is None:
            heartbeat = orion.core.config.worker.heartbeat
        self.heartbeat = heartbeat
        self.plot = PlotAccessor(self)
        atexit.register(self.set_broken_trials)

    ###
    # Attributes
    ###

    @property
    def name(self):
        """Return the name of the experiment in the database."""
        return self._experiment.name

    # pylint: disable=invalid-name
    @property
    def id(self):
        """Return the id of the experiment in the database."""
        return self._experiment.id

    @property
    def version(self):
        """Version of the experiment."""
        return self._experiment.version

    @property
    def max_trials(self):
        """Max-trials to execute before stopping the experiment."""
        return self._experiment.max_trials

    @property
    def max_broken(self):
        """Minimum number of broken trials before the experiment is considered broken."""
        return self._experiment.max_broken

    @property
    def metadata(self):
        """Metadata of the experiment."""
        return self._experiment.metadata

    @property
    def space(self):
        """Return problem's parameter `orion.algo.space.Space`."""
        return self._experiment.space

    @property
    def algorithms(self):
        """Algorithms of the experiment."""
        return self._experiment.algorithms

    @property
    def refers(self):
        """References to the experiment version control"""
        return self._experiment.refers

    @property
    def is_done(self):
        """Return True, if this experiment is considered to be finished.

        1. Count how many trials have been completed and compare with `max_trials`.
        2. Ask `algorithms` if they consider there is a chance for further improvement.
        """
        return self._experiment.is_done

    @property
    def is_broken(self):
        """Return True, if this experiment is considered to be broken.

        Count how many trials are broken and return True if that number has reached
        as given threshold.
        """
        return self._experiment.is_broken

    @property
    def configuration(self):
        """Return a copy of an `Experiment` configuration as a dictionary."""
        return self._experiment.configuration

    @property
    def stats(self):
        """Calculate a stats dictionary for this particular experiment.

        Returns
        -------
        stats : dict

        Stats
        -----
        trials_completed : int
           Number of completed trials
        best_trials_id : int
           Unique identifier of the :class:`orion.core.worker.trial.Trial` object in the database
           which achieved the best known objective result.
        best_evaluation : float
           Evaluation score of the best trial
        start_time : `datetime.datetime`
           When Experiment was first dispatched and started running.
        finish_time : `datetime.datetime`
           When Experiment reached terminating condition and stopped running.
        duration : `datetime.timedelta`
           Elapsed time.

        """
        return self._experiment.stats

    @property
    def node(self):
        """Node of the experiment in the version control tree."""
        return self._experiment.node

    @property
    def working_dir(self):
        """Working directory of the experiment."""
        return self._experiment.working_dir

    @property
    def producer(self):
        """Return the producer configuration of the experiment."""
        return self._experiment.producer

    @property
    def mode(self):
        """Return the access right of the experiment

        {'r': read, 'w': read/write, 'x': read/write/execute}
        """
        return self._experiment.mode

    ###
    # Rights
    ###

    def _check_if_writable(self):
        if self.mode == "r":
            calling_function = inspect.stack()[1].function
            raise UnsupportedOperation(
                f"ExperimentClient must have write rights to execute `{calling_function}()`"
            )

    def _check_if_executable(self):
        if self.mode != "x":
            calling_function = inspect.stack()[1].function
            raise UnsupportedOperation(
                f"ExperimentClient must have execution rights to execute `{calling_function}()`"
            )

    ###
    # Queries
    ###

    def to_pandas(self, with_evc_tree=False):
        """Builds a dataframe with the trials of the experiment

        Parameters
        ----------
        with_evc_tree: bool, optional
            Fetch all trials from the EVC tree.
            Default: False

        """
        return self._experiment.to_pandas(with_evc_tree=with_evc_tree)

    def fetch_trials(self, with_evc_tree=False):
        """Fetch all trials of the experiment

        Parameters
        ----------
        with_evc_tree: bool, optional
            Fetch all trials from the EVC tree.
            Default: False

        """
        return self._experiment.fetch_trials(with_evc_tree=with_evc_tree)

    def get_trial(self, trial=None, uid=None):
        """Fetch a single trial

        Parameters
        ----------
        trial: Trial, optional
           trial object to retrieve from the database

        uid: str, optional
            trial id used to retrieve the trial object

        Returns
        -------
        return none if the trial is not found,

        Raises
        ------
        UndefinedCall
            if both trial and uid are not set

        AssertionError
            if both trial and uid are provided and they do not match

        """
        return self._experiment.get_trial(trial, uid)

    def fetch_trials_by_status(self, status, with_evc_tree=False):
        """Fetch all trials with the given status

        Trials are sorted based on ``Trial.submit_time``

        :return: list of :class:`orion.core.worker.trial.Trial` objects
        """
        return self._experiment.fetch_trials_by_status(
            status, with_evc_tree=with_evc_tree
        )

    def fetch_noncompleted_trials(self, with_evc_tree=False):
        """Fetch non-completed trials of this `Experiment` instance.

        Trials are sorted based on ``Trial.submit_time``

        .. note::

            It will return all non-completed trials, including new, reserved, suspended,
            interrupted and broken ones.

        :return: list of non-completed :class:`orion.core.worker.trial.Trial` objects
        """
        return self._experiment.fetch_noncompleted_trials(with_evc_tree=with_evc_tree)

    ###
    # Actions
    ###

    # pylint: disable=unused-argument
    def insert(self, params, results=None, reserve=False):
        """Insert a new trial.

        Experiment must be in writable ('w') or executable ('x') mode.

        Parameters
        ----------
        params: dict
            Parameters of the new trial to add to the database. These parameters
            must comply with the space definition otherwise a ValueError will be raised.
        results: list, optional
            Results to be set for the new trial. Results must have the format
            {name: <str>: type: <'objective', 'constraint' or 'gradient'>, value=<float>} otherwise
            a ValueError will be raised.
            Note that passing results will mark the trial as completed and therefore cannot be
            reserved. The returned trial will have status 'completed'.
            If the results are invalid, the trial will still be inserted but reservation will be
            released.
        reserve: bool, optional
            If reserve=True, the inserted trial will be reserved. `reserve` cannot be True if
            `results` are given.
            Defaults to False.

        Returns
        -------
        `orion.core.worker.trial.Trial`
            The trial inserted in storage. If `reserve=True` and no results are given, the returned
            trial will be in a `reserved` status.

        Raises
        ------
        `ValueError`
            - If results are given and reserve=True
            - If params have invalid format
            - If results have invalid format
        `orion.core.io.database.DuplicateKeyError`
            - If a trial with identical params already exist for the current experiment.
        `orion.core.utils.exceptions.UnsupportedOperation`
            If the experiment was not loaded in writable mode.

        """
        self._check_if_writable()

        if results and reserve:
            raise ValueError(
                "Cannot observe a trial and reserve it. A trial with results has status "
                "`completed` and cannot be reserved."
            )
        trial = format_trials.dict_to_trial(params, self.space)
        try:
            self._experiment.register_trial(trial, status="reserved")
            self._maintain_reservation(trial)
        except DuplicateKeyError as e:
            message = (
                "A trial with params {} already exist for experiment {}-v{}".format(
                    params, self.name, self.version
                )
            )
            raise DuplicateKeyError(message) from e

        if results:
            try:
                self.observe(trial, results)
            except ValueError:
                self._release_reservation(trial)
                raise

            return trial

        if not reserve:
            self.release(trial)

        return trial

    def reserve(self, trial):
        """Reserve a trial.

        Experiment must be in executable ('x') mode.

        Set a trial status to reserve to ensure that concurrent process cannot work on it.
        Trials can only be reserved with status 'new', 'interrupted' or 'suspended'.

        Parameters
        ----------
        trial: `orion.core.worker.trial.Trial`
            Trial to reserve.

        Raises
        ------
        `RuntimeError`
            If trial is reserved by another process
        `ValueError`
            If the trial does not exist in storage.
        `orion.core.utils.exceptions.UnsupportedOperation`
            If the experiment was not loaded in executable mode.

        Notes
        -----
        When reserved, a :class:`TrialPacemaker <orion.core.worker.trial_pacemaker.TrialPacemaker>`
        is started to update an heartbeat in storage. The frequency of the heartbeat is configurable
        at creation of experiment or with ``orion.core.config.worker.heartbeat``.
        If the process terminates unexpectedly, the heartbeat will cease and remote processes
        may reset the status of the trial to 'interrupted' when the heartbeat has not been updated
        since twice the value of ``heartbeat``.

        """
        self._check_if_executable()

        if trial.status == "reserved" and trial.id in self._pacemakers:
            log.warning("Trial %s is already reserved.", trial.id)
            return
        elif trial.status == "reserved" and trial.id not in self._pacemakers:
            raise RuntimeError(
                "Trial {} is already reserved by another process.".format(trial.id)
            )
        try:
            self._experiment.set_trial_status(
                trial, "reserved", heartbeat=self.heartbeat
            )
        except FailedUpdate as e:
            if self.get_trial(trial) is None:
                raise ValueError(
                    "Trial {} does not exist in database.".format(trial.id)
                ) from e
            raise RuntimeError("Could not reserve trial {}.".format(trial.id)) from e

        self._maintain_reservation(trial)

    def release(self, trial, status="interrupted"):
        """Release a trial.

        Release the reservation and stop the heartbeat.

        Experiment must be in writable ('w') or executable ('x') mode.

        Parameters
        ----------
        trial: `orion.core.worker.trial.Trial`
            Trial to reserve.
        status: str, optional
            Set the trial to given status while releasing the reservation.
            Defaults to 'interrupted'.

        Raises
        ------
        `RuntimeError`
            If reservation of the trial has been lost prior to releasing it.
        `ValueError`
            If the trial does not exist in storage.
        `orion.core.utils.exceptions.UnsupportedOperation`
            If the experiment was not loaded in writable mode.

        """
        self._check_if_writable()

        current_status = trial.status
        raise_if_unreserved = True
        try:
            self._experiment.set_trial_status(trial, status, was="reserved")
        except FailedUpdate as e:
            if self.get_trial(trial) is None:
                raise ValueError(
                    "Trial {} does not exist in database.".format(trial.id)
                ) from e
            if current_status != "reserved":
                raise_if_unreserved = False
                raise RuntimeError(
                    "Trial {} was already released locally.".format(trial.id)
                ) from e

            raise RuntimeError(
                "Reservation for trial {} has been lost before release.".format(
                    trial.id
                )
            ) from e
        finally:
            self._release_reservation(trial, raise_if_unreserved=raise_if_unreserved)

    def suggest(self):
        """Suggest a trial to execute.

        Experiment must be in executable ('x') mode.

        If any trial is available (new or interrupted), it selects one and reserves it.
        Otherwise, the algorithm is used to generate a new trial that is registered in storage and
        reserved.

        Returns
        -------
        `orior.core.worker.trial.Trial`
            Reserved trial for execution.

        Raises
        ------
        :class:`orion.core.utils.exceptions.WaitingForTrials`
            if the experiment is not completed and algorithm needs to wait for some
            trials to complete before it can suggest new trials.

        :class:`orion.core.utils.exceptions.BrokenExperiment`
            if too many trials failed to run and the experiment cannot continue.
            This is determined by ``max_broken`` in the configuration of the experiment.

        :class:`orion.core.utils.exceptions.SampleTimeout`
            if the algorithm of the experiment could not sample new unique trials.

        :class:`orion.core.utils.exceptions.CompletedExperiment`
            if the experiment was completed and algorithm could not sample new trials.

        :class:`orion.core.utils.exceptions.UnsupportedOperation`
            If the experiment was not loaded in executable mode.

        """
        self._check_if_executable()

        if self.is_broken:
            raise BrokenExperiment("Trials failed too many times")

        if self.is_done:
            raise CompletedExperiment("Experiment is done, cannot sample more trials.")

        try:
            trial = reserve_trial(self._experiment, self._producer)

        except (WaitingForTrials, SampleTimeout) as e:
            if self.is_broken:
                raise BrokenExperiment("Trials failed too many times") from e

            raise e

        # This is to handle cases where experiment was completed during call to `reserve_trial`
        if trial is None:
            raise CompletedExperiment("Experiment is done, cannot sample more trials.")

        self._maintain_reservation(trial)
        return TrialCM(self, trial)

    def observe(self, trial, results):
        """Observe trial results

        Experiment must be in executable ('x') mode.

        Parameters
        ----------
        trial: `orion.core.worker.trial.Trial`
            Reserved trial to observe.
        results: list
            Results to be set for the new trial. Results must have the format
            {name: <str>: type: <'objective', 'constraint' or 'gradient'>, value=<float>} otherwise
            a ValueError will be raised. If the results are invalid, the trial will not be released.

        Returns
        -------
        `orion.core.worker.trial.Trial`
            The trial inserted in storage. If `reserve=True` and no results are given, the returned
            trial will be in a `reserved` status.

        Raises
        ------
        `ValueError`
            - If results have invalid format
            - If the trial does not exist in storage.
        `RuntimeError`
            If reservation of the trial has been lost prior to releasing it.
        `orion.core.utils.exceptions.UnsupportedOperation`
            If the experiment was not loaded in executable mode.

        """
        self._check_if_executable()

        trial.results += [Trial.Result(**result) for result in results]
        raise_if_unreserved = True
        try:
            self._experiment.update_completed_trial(trial)
        except FailedUpdate as e:
            if self.get_trial(trial) is None:
                raise_if_unreserved = False
                raise ValueError(
                    "Trial {} does not exist in database.".format(trial.id)
                ) from e

            raise RuntimeError(
                "Reservation for trial {} has been lost.".format(trial.id)
            ) from e
        finally:
            self._release_reservation(trial, raise_if_unreserved=raise_if_unreserved)

    def workon(
        self,
        fct,
        n_workers=1,
        max_trials=None,
        max_trials_per_worker=None,
        max_broken=None,
        trial_arg=None,
        on_error=None,
        **kwargs,
    ):
        """Optimize a given function

        Experiment must be in executable ('x') mode.

        Parameters
        ----------
        fct: callable
            Function to optimize. Must take arguments provided by trial.params. Additional constant
            parameter can be passed as ``**kwargs`` to `workon`. Function must return the final
            objective.
        n_workers: int, optional
            Number of workers to run in parallel.
        max_trials: int, optional
            Maximum number of trials to execute within ``workon``. If the experiment or algorithm
            reach status is_done before, the execution of ``workon`` terminates.
            Defaults to experiment's max trial. If ``max_trials`` is larger than
            ``experiment.max_trials``, the experiment will stop when reaching
            ``experiment.max_trials``.
        max_trials_per_worker: int, optional
            Maximum number of trials to execute within each worker. ``max_trials`` and
            ``experiment.max_trials`` have precedence. Defaults to global config.
        max_broken: int, optional
            Maximum number of broken trials to accept during `workon`. When this threshold is
            reached the function will raise :class:`orion.core.utils.exceptions.BrokenExperiment`.
            Defaults to value of global config.
        trial_alg: str, optional
            The function ``fct`` may support receiving the trial as an argument. This argument name
            can be specified with ``trial_arg``. If not defined (``None``), then only the
            hyperparameters will be passed to `fct`.
        on_error: callable, optional
            Callback that is executed if an error occur during the execution of ``fct``.
            The signature of the callback must be
            ``foo(ExperimentClient, Trial, Error, nb_of_worker_broken_trials)``.
            If the callblack returns False, the error will be ignored, otherwise it is counted
            for the threshold `max_broken`. In case of critical errors, you may also directly
            raise an error and force break out of ``workon``.
        **kwargs
            Constant argument to pass to `fct` in addition to trial.params. If values in kwargs are
            present in trial.params, the latter takes precedence.

        Raises
        ------
        :class:`orion.core.utils.exceptions.InvalidResult`
             If results returned by `fct` have invalid format.

        :class:`orion.core.utils.exceptions.WaitingForTrials`
            if the experiment is not completed and algorithm needs to wait for some
            trials to complete before it can suggest new trials.

        :class:`orion.core.utils.exceptions.BrokenExperiment`
            if too many trials failed to run and the experiment cannot continue.
            This is determined by ``max_broken`` in the configuration of the experiment.

        :class:`orion.core.utils.exceptions.SampleTimeout`
            if the algorithm of the experiment could not sample new unique points.

        :class:`orion.core.utils.exceptions.UnsupportedOperation`
            If the experiment was not loaded in executable mode.

        """
        self._check_if_executable()

        if max_trials is None:
            max_trials = self.max_trials

        if max_trials_per_worker is None:
            max_trials_per_worker = orion.core.config.worker.max_trials

        if max_broken is None:
            max_broken = orion.core.config.worker.max_broken

        # Use worker's max_trials inside `exp.is_done` to reduce chance of
        # race condition for trials creation
        if self.max_trials > max_trials:
            self._experiment.max_trials = max_trials
            self._experiment.algorithms.algorithm.max_trials = max_trials

        with Parallel(n_jobs=n_workers) as parallel:
            trials = parallel(
                delayed(self._optimize)(
                    fct,
                    max_trials_per_worker,
                    max_broken,
                    trial_arg,
                    on_error,
                    **kwargs,
                )
                for _ in range(n_workers)
            )

        return sum(trials)

    def _optimize(self, fct, max_trials, max_broken, trial_arg, on_error, **kwargs):
        worker_broken_trials = 0
        trials = 0
        kwargs = flatten(kwargs)
        max_trials = min(max_trials, self.max_trials)
        while not self.is_done and trials - worker_broken_trials < max_trials:
            try:
                with self.suggest() as trial:

                    kwargs.update(flatten(trial.params))

                    if trial_arg:
                        kwargs[trial_arg] = trial

                    try:
                        results = fct(**unflatten(kwargs))
                        self.observe(trial, results=results)
                    except (KeyboardInterrupt, InvalidResult):
                        raise
                    except BaseException as e:
                        if on_error is None or on_error(
                            self, trial, e, worker_broken_trials
                        ):
                            log.error(traceback.format_exc())
                            worker_broken_trials += 1
                        else:
                            log.error(str(e))
                            log.debug(traceback.format_exc())

                        if worker_broken_trials >= max_broken:
                            raise BrokenExperiment(
                                "Worker has reached broken trials threshold"
                            )
                        else:
                            self.release(trial, status="broken")
            except CompletedExperiment as e:
                log.warning(e)
                break

            trials += 1

        return trials

    def close(self):
        """Verify that no reserved trials are remaining and unregister atexit().

        Experiment must be in executable ('x') mode.

        Raises
        ------
        `orion.core.utils.exceptions.UnsupportedOperation`
            If the experiment was not loaded in executable mode.

        """
        self._check_if_executable()

        if self._pacemakers:
            raise RuntimeError(
                "There is still reserved trials: {}\nRelease all trials before "
                "closing the client, using "
                "client.release(trial).".format(self._pacemakers.keys())
            )

        atexit.unregister(self.set_broken_trials)

    ###
    # Private
    ###

    def __repr__(self):
        """Represent the object as a string."""
        return "Experiment(name=%s, version=%s)" % (self.name, self.version)

    def _verify_reservation(self, trial):
        if trial.id not in self._pacemakers:
            raise RuntimeError(
                "Trial {} had no pacemakers. Was it reserved properly?".format(trial.id)
            )

        if self.get_trial(trial).status != "reserved":
            self._release_reservation(trial)
            raise RuntimeError(
                "Reservation for trial {} has been lost.".format(trial.id)
            )

    def _maintain_reservation(self, trial):
        self._pacemakers[trial.id] = TrialPacemaker(trial)
        self._pacemakers[trial.id].start()

    def _release_reservation(self, trial, raise_if_unreserved=True):
        if trial.id not in self._pacemakers:
            if raise_if_unreserved:
                raise RuntimeError(
                    "Trial {} had no pacemakers. Was it reserved properly?".format(
                        trial.id
                    )
                )
            else:
                return

        self._pacemakers.pop(trial.id).stop()
