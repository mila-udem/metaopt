# -*- coding: utf-8 -*-
"""
Python API
==========

Provides functions for communicating with `orion.core`.

"""
import orion.core.io.experiment_builder as experiment_builder
from orion.client.cli import (
    interrupt_trial,
    report_bad_trial,
    report_objective,
    report_results,
)
from orion.client.experiment import ExperimentClient
from orion.core.utils.exceptions import RaceCondition
from orion.core.utils.singleton import update_singletons
from orion.core.worker.producer import Producer
from orion.storage.base import setup_storage

__all__ = [
    "interrupt_trial",
    "report_bad_trial",
    "report_objective",
    "report_results",
    "create_experiment",
    "build_experiment",
    "get_experiment",
    "workon",
]


def create_experiment(name, **config):
    """Build an experiment to be executable

    This function is deprecated and will be removed in v0.3.0. Use `build_experiment`
    instead.
    """
    return build_experiment(name, **config)


# pylint: disable=too-many-arguments
def build_experiment(
    name,
    version=None,
    space=None,
    algorithms=None,
    strategy=None,
    max_trials=None,
    max_broken=None,
    storage=None,
    branching=None,
    max_idle_time=None,
    heartbeat=None,
    working_dir=None,
    debug=False,
    knowledge_base: "KnowledgeBase" = None,
):
    """Build an experiment to be executable

    Building the experiment can result in branching if there are any changes in the environment.
    This is required to ensure coherence between execution of trials. For an experiment
    in read/write mode without execution rights, see `get_experiment`.

    There is 2 main scenarios

    1) The experiment is new

    ``name`` and ``space`` arguments are required, otherwise ``NoConfigurationError`` will be
    raised.

    All other arguments (``algorithms``, ``strategy``, ``max_trials``, ``storage``, ``branching``
    and ``working_dir``) will be replaced by system's defaults if ommited. The system's defaults can
    also be overriden in global configuration file as described for the database in
    :ref:`Database Configuration`. We do not recommand overriding the algorithm configuration using
    system's default, but overriding the storage configuration can be very convenient if the same
    storage is used for all your experiments.

    2) The experiment exist in the database.

    We can break down this scenario in two sub-scenarios for clarity.

    2.1) Only experiment name is given.

    The configuration will be fetched from database.

    2.2) Some other arguments than the name are given.

    The configuration will be fetched from database and given arguments will override them.
    ``max_trials`` and ``max_broken`` may be overwritten in DB, but any other changes will lead to a
    branching. Instead of creating the experiment ``(name, version)``, it will create a new
    experiment ``(name, version+1)`` which will have the same configuration than ``(name, version)``
    except for the differing arguments given by user. This new experiment will have access to trials
    of ``(name, version)``, adapted according to the differences between ``version`` and
    ``version+1``.  A previous version can be accessed by specifying the ``version`` argument.

    Causes of experiment branching are:

    - Change of search space

        - New dimension

        - Different prior

        - Missing dimension

    - Change of algorithm

    - Change of strategy (Not implemented yet)

    - Change of code version (Only supported by commandline API for now)

    - Change of orion version

    Parameters
    ----------
    name: str
        Name of the experiment
    version: int, optional
        Version of the experiment. Defaults to last existing version for a given ``name``
        or 1 for new experiment.
    space: dict, optional
        Optimization space of the algorithm. Should have the form ``dict(name='<prior>(args)')``.
    algorithms: str or dict, optional
        Algorithm used for optimization.
    strategy: str or dict, optional
        Parallel strategy to use to parallelize the algorithm.
    max_trials: int, optional
        Maximum number or trials before the experiment is considered done.
    max_broken: int, optional
        Number of broken trials for the experiment to be considered broken.
    storage: dict, optional
        Configuration of the storage backend.
    working_dir: str, optional
        Working directory created for the experiment inside which a unique folder will be created
        for each trial. Defaults to a temporary directory that is deleted at end of execution.
    max_idle_time: int, optional
        Maximum time the producer can spend trying to generate a new suggestion.
        Such timeout are generally caused by slow database, large number of
        concurrent workers leading to many race conditions or small search spaces
        with integer/categorical dimensions that may be fully explored.
        Defaults to ``orion.core.config.worker.max_idle_time``.
    heartbeat: int, optional
        Frequency (seconds) at which the heartbeat of the trial is updated.
        If the heartbeat of a `reserved` trial is larger than twice the configured
        heartbeat, Oríon will reset the status of the trial to `interrupted`.
        This allows restoring lost trials (ex: due to killed worker).
        Defaults to ``orion.core.config.worker.max_idle_time``.
    debug: bool, optional
        If using in debug mode, the storage config is overrided with legacy:EphemeralDB.
        Defaults to False.
    branching: dict, optional
        Arguments to control the branching.

        branch_to: str, optional
            Name of the experiment to branch to. The parent experiment will be the one specified by
            ``(name, version)``, and the child will be ``(branch_to, 1)``.
        branch_from: str, optional
            Name of the experiment to branch from.
            The parent experiment will be the one specified by
            ``(branch_from, last version)``, and the child will be ``(name, 1)``.
        manual_resolution: bool, optional
            Starts the prompt to resolve manually the conflicts. Defaults to False.
        algorithm_change: bool, optional
            Whether to automatically solve the algorithm conflict (change of algo config).
            Defaults to True.
        orion_version_change: bool, optional
            Whether to automatically solve the orion version conflict.
            Defaults to True.
        code_change_type: str, optional
            How to resolve code change automatically. Must be one of 'noeffect', 'unsure' or
            'break'.  Defaults to 'break'.
        cli_change_type: str, optional
            How to resolve cli change automatically. Must be one of 'noeffect', 'unsure' or 'break'.
            Defaults to 'break'.
        config_change_type: str, optional
            How to resolve config change automatically. Must be one of 'noeffect', 'unsure' or
            'break'.  Defaults to 'break'.

    Raises
    ------
    :class:`orion.core.utils.singleton.SingletonAlreadyInstantiatedError`
        If the storage is already instantiated and given configuration is different.
        Storage is a singleton, you may only use one instance per process.
    :class:`orion.core.utils.exceptions.NoConfigurationError`
        The experiment is not in database and no space is provided by the user.
    :class:`orion.core.utils.exceptions.RaceCondition`
        There was a race condition during branching and new version cannot be infered because of
        that. Single race conditions are normally handled seemlessly. If this error gets raised, it
        means that different modifications occured during each race condition resolution. This is
        likely due to quick code change during experiment creation. Make sure your script is not
        generating files within your code repository.
    :class:`orion.core.utils.exceptions.BranchingEvent`
        The configuration is different than the corresponding one in DB and the branching cannot be
        solved automatically. This usually happens if the version=x is specified but the experiment
        ``(name, x)`` already has a child ``(name, x+1)``. If you really need to branch from version
        ``x``, give it a new name to branch to with ``branching={'branch_to': <new_name>}``.
    `NotImplementedError`
        If the algorithm, storage or strategy specified is not properly installed.

    """
    setup_storage(storage=storage, debug=debug)

    try:
        experiment = experiment_builder.build(
            name,
            version=version,
            space=space,
            algorithms=algorithms,
            strategy=strategy,
            max_trials=max_trials,
            max_broken=max_broken,
            branching=branching,
            working_dir=working_dir,
        )
    except RaceCondition:
        # Try again, but if it fails again, raise. Race conditions due to version increment should
        # only occur once in a short window of time unless code version is changing at a crazy pace.
        try:
            experiment = experiment_builder.build(
                name,
                version=version,
                space=space,
                algorithms=algorithms,
                strategy=strategy,
                max_trials=max_trials,
                max_broken=max_broken,
                branching=branching,
                working_dir=working_dir,
            )
        except RaceCondition as e:
            raise RaceCondition(
                "There was a race condition during branching and new version cannot be infered "
                "because of that. Single race conditions are normally handled seemlessly. If this "
                "error gets raised, it means that different modifications occured during each race "
                "condition resolution. This is likely due to quick code change during experiment "
                "creation. Make sure your script is not generating files within your code "
                "repository."
            ) from e

    producer = Producer(experiment, max_idle_time, knowledge_base=knowledge_base)

    return ExperimentClient(experiment, producer, heartbeat)


def get_experiment(name, version=None, mode="r", storage=None):
    """
    Retrieve an existing experiment as :class:`orion.client.experiment.ExperimentClient`.

    Parameters
    ----------
    name: str
        The name of the experiment.
    version: int, optional
        Version to select. If None, last version will be selected. If version given is larger than
        largest version available, the largest version will be selected.
    mode: str, optional
        The access rights of the experiment on the database.
        'r': read access only
        'w': can read and write to database
        Default is 'r'

    storage: dict, optional
        Configuration of the storage backend.

    Returns
    -------
    An instance of :class:`orion.client.experiment.ExperimentClient` representing the experiment.

    Raises
    ------
    `orion.core.utils.exceptions.NoConfigurationError`
        The experiment is not in the database provided by the user.
    """
    setup_storage(storage)
    assert mode in set("rw")
    experiment = experiment_builder.load(name, version, mode)
    return ExperimentClient(experiment, None)


def workon(
    function, space, name="loop", algorithms=None, max_trials=None, max_broken=None
):
    """Optimize a function over a given search space

    This will create a new experiment with an in-memory storage and optimize the given function
    until `max_trials` is reached or the `algorithm` is done
    (some algorithms like random search are never done).

    For informations on how to fetch results, see
    :py:class:`orion.client.experiment.ExperimentClient`.

    .. note::

        Each call to this function will create a separate in-memory storage.

    Parameters
    ----------
    name: str
        Name of the experiment
    version: int, optional
        Version of the experiment. Defaults to last existing version for a given `name`
        or 1 for new experiment.
    space: dict, optional
        Optimization space of the algorithm. Should have the form `dict(name='<prior>(args)')`.
    algorithms: str or dict, optional
        Algorithm used for optimization.
    max_trials: int, optional
        Maximum number or trials before the experiment is considered done.
    max_broken: int, optional
        Number of broken trials for the experiment to be considered broken.

    Raises
    ------
    `NotImplementedError`
        If the algorithm specified is not properly installed.

    """
    # Clear singletons and keep pointers to restore them.
    singletons = update_singletons()

    try:
        setup_storage(storage={"type": "legacy", "database": {"type": "EphemeralDB"}})

        experiment = experiment_builder.build(
            name,
            version=1,
            space=space,
            algorithms=algorithms,
            strategy="NoParallelStrategy",
            max_trials=max_trials,
            max_broken=max_broken,
        )

        producer = Producer(experiment)

        experiment_client = ExperimentClient(experiment, producer)
        experiment_client.workon(function, max_trials=max_trials)
    finally:
        # Restore singletons
        update_singletons(singletons)

    return experiment_client
