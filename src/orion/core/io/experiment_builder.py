# pylint:disable=too-many-lines
"""
Create experiment from user options
===================================

Functions which build :class:`orion.core.worker.experiment.Experiment` objects based on user
configuration.

The instantiation of an :class:`orion.core.worker.experiment.Experiment` is not a trivial process
when the user request an experiment with specific options. One can easily create a new experiment
with ``Experiment('some_experiment_name')``, but the configuration of a _writable_ experiment is
less straighforward. This is because there is many sources of configuration and they have a strict
hierarchy. From the more global to the more specific, there is:

1. Global configuration:

  Defined by ``orion.core.DEF_CONFIG_FILES_PATHS``.
  Can be scattered in user file system, defaults could look like:

    - ``/some/path/to/.virtualenvs/orion/share/orion.core``
    - ``/etc/xdg/xdg-ubuntu/orion.core``
    - ``/home/${USER}/.config/orion.core``

  Note that most variables have default value even if user do not defined them in global
  configuration. These are defined in ``orion.core.__init__``.

2. Oríon specific environment variables:

   Environment variables which can override global configuration

    - Database specific:

      * ``ORION_DB_NAME``
      * ``ORION_DB_TYPE``
      * ``ORION_DB_ADDRESS``

3. Experiment configuration inside the database

  Configuration of the experiment if present in the database.
  Making this part of the configuration of the experiment makes it possible
  for the user to execute an experiment by only specifying partial configuration. The rest of the
  configuration is fetched from the database.

  For example, a user could:

    1. Rerun the same experiment

      Only providing the name is sufficient to rebuild the entire configuration of the
      experiment.

    2. Make a modification to an existing experiment

      The user can provide the name of the existing experiment and only provide the changes to
      apply on it. Here is an minimal example where we fully initialize a first experiment with a
      config file and then branch from it with minimal information.

      .. code-block:: bash

          # Initialize root experiment
          orion hunt --init-only --config previous_exeriment.yaml ./userscript -x~'uniform(0, 10)'
          # Branch a new experiment
          orion hunt -n previous_experiment ./userscript -x~'uniform(0, 100)'

4. Configuration file

  This configuration file is meant to overwrite the configuration coming from the database.
  If this configuration file was interpreted as part of the global configuration, a user could
  only modify an experiment using command line arguments.

5. Command-line arguments

  Those are the arguments provided to ``orion`` for any method (hunt, insert, etc). It includes the
  argument to ``orion`` itself as well as the user's script name and its arguments.

"""
import copy
import datetime
import getpass
import logging
import pprint
import sys

import orion.core
import orion.core.utils.backward as backward
from orion.algo.space import Space
from orion.core.evc.adapters import Adapter
from orion.core.evc.conflicts import ExperimentNameConflict, detect_conflicts
from orion.core.io import resolve_config
from orion.core.io.database import DuplicateKeyError
from orion.core.io.experiment_branch_builder import ExperimentBranchBuilder
from orion.core.io.interactive_commands.branching_prompt import BranchingPrompt
from orion.core.io.space_builder import SpaceBuilder
from orion.core.utils.exceptions import (
    BranchingEvent,
    NoConfigurationError,
    NoNameError,
    RaceCondition,
)
from orion.core.worker.experiment import Experiment
from orion.core.worker.primary_algo import PrimaryAlgo
from orion.core.worker.strategy import Strategy
from orion.storage.base import get_storage, setup_storage

log = logging.getLogger(__name__)


##
# Functions to build experiments
##


def build(name, version=None, branching=None, **config):
    """Build an experiment object

    If new, ``space`` argument must be provided, else all arguments are fetched from the database
    based on (name, version). If any argument given does not match the corresponding ones in the
    database for given (name, version), than the version is incremented and the experiment will be a
    child of the previous version.

    Parameters
    ----------
    name: str
        Name of the experiment to build
    version: int, optional
        Version to select. If None, last version will be selected. If version given is larger than
        largest version available, the largest version will be selected.
    space: dict, optional
        Optimization space of the algorithm. Should have the form ``dict(name='<prior>(args)')``.
    algorithms: str or dict, optional
        Algorithm used for optimization.
    strategy: str or dict, optional
        Parallel strategy to use to parallelize the algorithm.
    max_trials: int, optional
        Maximum number of trials before the experiment is considered done.
    max_broken: int, optional
        Number of broken trials for the experiment to be considered broken.
    storage: dict, optional
        Configuration of the storage backend.

    branching: dict, optional
        Arguments to control the branching.

        branch_from: str, optional
            Name of the experiment to branch from.
        manual_resolution: bool, optional
            Starts the prompt to resolve manually the conflicts. Defaults to False.
        non_monitored_arguments: list of str, optional
            Will ignore these arguments while looking for differences. Defaults to [].
        ignore_code_changes: bool, optional
            Will ignore code changes while looking for differences. Defaults to False.
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

    """
    log.debug(f"Building experiment {name} with {version}")
    log.debug("    Passed experiment config:\n%s", pprint.pformat(config))
    log.debug("    Branching config:\n%s", pprint.pformat(branching))

    name, config, branching = clean_config(name, config, branching)

    config = consolidate_config(name, version, config)

    if "space" not in config:
        raise NoConfigurationError(
            "Experiment {} does not exist in DB and space was not defined.".format(name)
        )

    if len(config["space"]) == 0:
        raise NoConfigurationError("No prior found. Please include at least one.")

    experiment = create_experiment(mode="x", **copy.deepcopy(config))
    if experiment.id is None:
        log.debug("Experiment not found in DB. Now attempting registration in DB.")
        try:
            _register_experiment(experiment)
            log.debug("Experiment successfully registered in DB.")
        except DuplicateKeyError:
            log.debug(
                "Experiment registration failed. This is likely due to a race condition. "
                "Now rolling back and re-attempting building it."
            )
            experiment = build(branching=branching, **config)

        return experiment

    log.debug(f"Experiment {config['name']}-v{config['version']} already existed.")

    conflicts = _get_conflicts(experiment, branching)
    must_branch = len(conflicts.get()) > 1 or branching.get("branch_to")
    if must_branch:
        if len(conflicts.get()) > 1:
            log.debug("Experiment must branch because of conflicts")
        else:
            assert branching.get("branch_to")
            log.debug("Experiment branching forced with ``branch_to``")
        branched_experiment = _branch_experiment(
            experiment, conflicts, version, branching
        )
        log.debug("Now attempting registration of branched experiment in DB.")
        try:
            _register_experiment(branched_experiment)
            log.debug("Branched experiment successfully registered in DB.")
        except DuplicateKeyError as e:
            log.debug(
                "Experiment registration failed. This is likely due to a race condition "
                "during branching. Now rolling back and re-attempting building "
                "the branched experiment."
            )
            raise RaceCondition("There was a race condition during branching.") from e

        return branched_experiment

    log.debug("No branching required.")

    _update_experiment(experiment)
    return experiment


def clean_config(name, config, branching):
    """Clean configuration from hidden fields (ex: ``_id``) and update branching if necessary"""
    log.debug("Cleaning config")

    config = copy.deepcopy(config)
    for key, value in list(config.items()):
        if key.startswith("_") or value is None:
            log.debug(f"Ignoring field {key}")
            config.pop(key)

    if "strategy" in config:
        config["producer"] = {"strategy": config.pop("strategy")}

    if branching is None:
        branching = {}

    if branching.get("branch_from"):
        branching.setdefault("branch_to", name)
        name = branching["branch_from"]

    log.debug("Cleaned experiment config")
    log.debug("    Experiment config:\n%s", pprint.pformat(config))
    log.debug("    Branching config:\n%s", pprint.pformat(branching))

    return name, config, branching


def consolidate_config(name, version, config):
    """Merge together given configuration with db configuration matching
    for experiment (``name``, ``version``)
    """
    db_config = fetch_config_from_db(name, version)

    # Do not merge spaces, the new definition overrides it.
    if "space" in config:
        db_config.pop("space", None)

    log.debug("Merging user and db configs:")
    log.debug("    config from user:\n%s", pprint.pformat(config))
    log.debug("    config from DB:\n%s", pprint.pformat(db_config))

    new_config = config
    config = resolve_config.merge_configs(db_config, config)

    config.setdefault("metadata", {})
    resolve_config.update_metadata(config["metadata"])

    merge_algorithm_config(config, new_config)
    merge_producer_config(config, new_config)

    config.setdefault("name", name)
    config.setdefault("version", version)

    log.debug("    Merged config:\n%s", pprint.pformat(config))

    return config


def merge_algorithm_config(config, new_config):
    """Merge given algorithm configuration with db config"""
    # TODO: Find a better solution
    if isinstance(config.get("algorithms"), dict) and len(config["algorithms"]) > 1:
        log.debug("Overriding algo config with new one.")
        log.debug("    Old config:\n%s", pprint.pformat(config["algorithms"]))
        log.debug("    New config:\n%s", pprint.pformat(new_config["algorithms"]))
        config["algorithms"] = new_config["algorithms"]


def merge_producer_config(config, new_config):
    """Merge given producer configuration with db config"""
    # TODO: Find a better solution
    if (
        isinstance(config.get("producer", {}).get("strategy"), dict)
        and len(config["producer"]["strategy"]) > 1
    ):
        log.debug("Overriding strategy config with new one.")
        log.debug("    Old config:\n%s", pprint.pformat(config["producer"]["strategy"]))
        log.debug(
            "    New config:\n%s", pprint.pformat(new_config["producer"]["strategy"])
        )

        config["producer"]["strategy"] = new_config["producer"]["strategy"]


def build_view(name, version=None):
    """Load experiment from database

    This function is deprecated and will be remove in v0.3.0. Use `load()` instead.
    """
    return load(name, version=version, mode="r")


def load(name, version=None, mode="r"):
    """Load experiment from database

    An experiment view provides all reading operations of standard experiment but prevents the
    modification of the experiment and its trials.

    Parameters
    ----------
    name: str
        Name of the experiment to build
    version: int, optional
        Version to select. If None, last version will be selected. If version given is larger than
        largest version available, the largest version will be selected.
    mode: str, optional
        The access rights of the experiment on the database.
        'r': read access only
        'w': can read and write to database
        Default is 'r'

    """
    assert mode in set("rw")

    log.debug(
        f"Loading experiment {name} (version={version}) from database in mode `{mode}`"
    )
    db_config = fetch_config_from_db(name, version)

    if not db_config:
        message = (
            "No experiment with given name '%s' and version '%s' inside database, "
            "no view can be created." % (name, version if version else "*")
        )
        raise NoConfigurationError(message)

    db_config.setdefault("version", 1)

    return create_experiment(mode=mode, **db_config)


def create_experiment(name, version, mode, space, **kwargs):
    """Instantiate the experiment and its attribute objects

    All unspecified arguments will be replaced by system's defaults (orion.core.config.*).

    Parameters
    ----------
    name: str
        Name of the experiment.
    version: int
        Version of the experiment.
    mode: str
        The access rights of the experiment on the database.
        'r': read access only
        'w': can read and write to database
        'x': can read and write to database, algo is instantiated and can execute optimization
    space: dict or Space object
        Optimization space of the algorithm. If dict, should have the form
        `dict(name='<prior>(args)')`.
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

    """
    experiment = Experiment(name=name, version=version, mode=mode)
    experiment._id = kwargs.get("_id", None)  # pylint:disable=protected-access
    experiment.pool_size = kwargs.get("pool_size")
    if experiment.pool_size is None:
        experiment.pool_size = orion.core.config.experiment.get(
            "pool_size", deprecated="ignore"
        )
    experiment.max_trials = kwargs.get(
        "max_trials", orion.core.config.experiment.max_trials
    )
    experiment.max_broken = kwargs.get(
        "max_broken", orion.core.config.experiment.max_broken
    )
    experiment.space = _instantiate_space(space)
    experiment.algorithms = _instantiate_algo(
        experiment.space,
        experiment.max_trials,
        kwargs.get("algorithms"),
        ignore_unavailable=mode != "x",
    )
    experiment.producer = kwargs.get("producer", {})
    experiment.producer["strategy"] = _instantiate_strategy(
        experiment.producer.get("strategy"), ignore_unavailable=mode != "x"
    )
    experiment.working_dir = kwargs.get(
        "working_dir", orion.core.config.experiment.working_dir
    )
    experiment.metadata = kwargs.get(
        "metadata", {"user": kwargs.get("user", getpass.getuser())}
    )
    experiment.refers = kwargs.get(
        "refers", {"parent_id": None, "root_id": None, "adapter": []}
    )
    experiment.refers["adapter"] = _instantiate_adapters(
        experiment.refers.get("adapter", [])
    )

    log.debug(
        "Created experiment with config:\n%s", pprint.pformat(experiment.configuration)
    )

    return experiment


def fetch_config_from_db(name, version=None):
    """Fetch configuration from database

    Parameters
    ----------
    name: str
        Name of the experiment to fetch
    version: int, optional
        Version to select. If None, last version will be selected. If version given is larger than
        largest version available, the largest version will be selected.

    """
    configs = get_storage().fetch_experiments({"name": name})

    if not configs:
        return {}

    config = _fetch_config_version(configs, version)

    if len(configs) > 1 and version is None:
        log.info(
            "Many versions for experiment %s have been found. Using latest "
            "version %s.",
            name,
            config["version"],
        )

    log.debug("Config found in DB:\n%s", pprint.pformat(config))

    backward.populate_space(config, force_update=False)
    backward.update_max_broken(config)

    return config


##
# Private helper functions to build experiments
##


def _instantiate_adapters(config):
    """Instantiate the adapter object

    Parameters
    ----------
    config: list
         List of adapter configurations to build a CompositeAdapter for the EVC.

    """
    return Adapter.build(config)


def _instantiate_space(config):
    """Instantiate the space object

    Build the Space object if argument is a dictionary, else return the Space object as is.

    Parameters
    ----------
    config: dict or Space object
        Dictionary of priors or already built Space object.

    """
    if isinstance(config, Space):
        return config

    return SpaceBuilder().build(config)


def _instantiate_algo(space, max_trials, config=None, ignore_unavailable=False):
    """Instantiate the algorithm object

    Parameters
    ----------
    config: dict, optional
        Configuration of the algorithm. If None or empty, system's defaults are used
        (orion.core.config.experiment.algorithms).
    ignore_unavailable: bool, optional
        If True and algorithm is not available (plugin not installed), return the configuration.
        Otherwise, raise Factory error from PrimaryAlgo

    """
    if not config:
        config = orion.core.config.experiment.algorithms

    try:
        algo = PrimaryAlgo(space, config)
        algo.algorithm.max_trials = max_trials
        # from orion.core.worker.multi_task_algo import MultiTaskAlgo
        # algo = MultiTaskAlgo(space, config)
        # algo.unwrapped.max_trials = max_trials
    except NotImplementedError as e:
        if not ignore_unavailable:
            raise e
        log.warning(str(e))
        log.warning("Algorithm will not be instantiated.")
        algo = config

    return algo


def _instantiate_strategy(config=None, ignore_unavailable=False):
    """Instantiate the strategy object

    Parameters
    ----------
    config: dict, optional
        Configuration of the strategy. If None of empty, system's defaults are used
        (orion.core.config.producer.strategy).
    ignore_unavailable: bool, optional
        If True and algorithm is not available (plugin not installed), return the configuration.
        Otherwise, raise Factory error from PrimaryAlgo


    """
    if not config:
        config = orion.core.config.experiment.strategy

    if isinstance(config, str):
        strategy_type = config
        config = {}
    else:
        config = copy.deepcopy(config)
        strategy_type, config = next(iter(config.items()))

    try:
        strategy = Strategy(of_type=strategy_type, **config)
    except NotImplementedError as e:
        if not ignore_unavailable:
            raise e
        log.warning(str(e))
        log.warning("Strategy will not be instantiated.")
        strategy = {strategy_type: config}

    return strategy


def _register_experiment(experiment):
    """Register a new experiment in the database"""
    experiment.metadata["datetime"] = datetime.datetime.utcnow()
    config = experiment.configuration
    # This will raise DuplicateKeyError if a concurrent experiment with
    # identical (name, metadata.user) is written first in the database.

    get_storage().create_experiment(config)

    # XXX: Reminder for future DB implementations:
    # MongoDB, updates an inserted dict with _id, so should you :P
    experiment._id = config["_id"]  # pylint:disable=protected-access

    # Update refers in db if experiment is root
    if experiment.refers.get("parent_id") is None:
        log.debug("update refers (name: %s)", experiment.name)
        experiment.refers["root_id"] = experiment.id
        get_storage().update_experiment(
            experiment, refers=experiment.configuration["refers"]
        )


def _update_experiment(experiment):
    """Update experiment configuration in database"""
    log.debug("Updating experiment (name: %s)", experiment.name)
    config = experiment.configuration

    # TODO: Remove since this should not occur anymore without metadata.user in the indices?
    # Writing the final config to an already existing experiment raises
    # a DuplicatKeyError because of the embedding id `metadata.user`.
    # To avoid this `final_config["name"]` is popped out before
    # `db.write()`, thus seamingly breaking  the compound index
    # `(name, metadata.user)`
    config.pop("name")

    get_storage().update_experiment(experiment, **config)

    log.debug("Experiment configuration successfully updated in DB.")


def _branch_experiment(experiment, conflicts, version, branching_arguments):
    """Create a new branch experiment with adapters for the given conflicts"""
    experiment_brancher = ExperimentBranchBuilder(conflicts, **branching_arguments)

    needs_manual_resolution = (
        not experiment_brancher.is_resolved or experiment_brancher.manual_resolution
    )

    if not experiment_brancher.is_resolved:
        name_conflict = conflicts.get([ExperimentNameConflict])[0]
        if not name_conflict.is_resolved and not version:
            log.debug(
                "A race condition likely occured during conflicts resolutions. "
                "Now rolling back and attempting re-building the branched experiment."
            )
            raise RaceCondition(
                "There was likely a race condition during version increment."
            )

    if needs_manual_resolution:
        log.debug("Some conflicts cannot be solved automatically.")

        # TODO: This should only be possible when using cmdline API
        branching_prompt = BranchingPrompt(experiment_brancher)

        if not sys.__stdin__.isatty():
            log.debug("No interactive prompt available to manually resolve conflicts.")
            raise BranchingEvent(branching_prompt.get_status())

        branching_prompt.cmdloop()

        if branching_prompt.abort or not experiment_brancher.is_resolved:
            sys.exit()

    log.debug("Creating new branched configuration")
    config = experiment_brancher.conflicting_config
    config["refers"]["adapter"] = experiment_brancher.create_adapters().configuration
    config["refers"]["parent_id"] = experiment.id

    config.pop("_id")

    return create_experiment(mode="x", **config)


def _get_conflicts(experiment, branching):
    """Get conflicts between current experiment and corresponding configuration in database"""
    log.debug("Looking for conflicts in new configuration.")
    db_experiment = load(experiment.name, experiment.version, mode="r")
    conflicts = detect_conflicts(
        db_experiment.configuration, experiment.configuration, branching
    )

    log.debug(f"{len(conflicts.get())} conflicts detected:\n {conflicts.get()}")

    # elif must_branch and not enable_branching:
    #     raise ValueError("Configuration is different and generate a branching event")

    return conflicts


def _fetch_config_version(configs, version=None):
    """Fetch the experiment configuration corresponding to the given version

    Parameters
    ----------
    configs: list
        List of configurations fetched from storoge.
    version: int, optional
        Version to select. If None, last version will be selected. If version given is larger than
        largest version available, the largest version will be selected.

    """
    max_version = max(configs, key=lambda exp: exp.get("version", 1)).get("version", 1)

    if version is None:
        version = max_version
    else:
        version = version

    if version > max_version:
        log.warning(
            "Version %s was specified but most recent version is only %s. " "Using %s.",
            version,
            max_version,
            max_version,
        )

    version = min(version, max_version)

    configs = filter(lambda exp: exp.get("version", 1) == version, configs)

    return next(iter(configs))


###
# Functions for commandline API
###


def build_from_args(cmdargs):
    """Build an experiment based on commandline arguments.

    Options provided in commandline and configuration file (--config) will overwrite system's
    default values and configuration from database if experiment already exits.
    Commandline arguments have precedence over configuration file options.

    .. seealso::

        :func:`orion.core.io.experiment_builder.build` for more information on experiment creation.

    """
    cmd_config = get_cmd_config(cmdargs)

    if "name" not in cmd_config:
        raise NoNameError()

    setup_storage(cmd_config["storage"], debug=cmd_config.get("debug"))

    return build(**cmd_config)


def get_from_args(cmdargs, mode="r"):
    """Build an experiment view based on commandline arguments

    .. seealso::

        :func:`orion.core.io.experiment_builder.load` for more information on creation of read-only
        experiments.

    """
    cmd_config = get_cmd_config(cmdargs)

    if "name" not in cmd_config:
        raise NoNameError()

    setup_storage(cmd_config["storage"], debug=cmd_config.get("debug"))

    name = cmd_config.get("name")
    version = cmd_config.get("version")

    return load(name, version, mode=mode)


def get_cmd_config(cmdargs):
    """Fetch configuration defined by commandline and local configuration file.

    Arguments of commandline have priority over options in configuration file.
    """
    cmdargs = resolve_config.fetch_config_from_cmdargs(cmdargs)
    cmd_config = resolve_config.fetch_config(cmdargs)
    cmd_config = resolve_config.merge_configs(cmd_config, cmdargs)

    cmd_config.update(cmd_config.pop("experiment", {}))
    cmd_config["user_script_config"] = cmd_config.get("worker", {}).get(
        "user_script_config", None
    )

    cmd_config["branching"] = cmd_config.pop("evc", {})

    # TODO: We should move branching specific stuff below in a centralized place for EVC stuff.
    if (
        cmd_config["branching"].get("auto_resolution", False)
        and cmdargs.get("manual_resolution", None) is None
    ):
        cmd_config["branching"]["manual_resolution"] = False

    non_monitored_arguments = cmdargs.get("non_monitored_arguments")
    if non_monitored_arguments:
        cmd_config["branching"][
            "non_monitored_arguments"
        ] = non_monitored_arguments.split(":")

    # TODO: user_args won't be defined if reading from DB only (`orion hunt -n <exp> ` alone)
    metadata = resolve_config.fetch_metadata(
        cmd_config.get("user"),
        cmd_config.get("user_args"),
        cmd_config.get("user_script_config"),
    )
    cmd_config["metadata"] = metadata
    cmd_config.pop("config", None)

    cmd_config["space"] = cmd_config["metadata"].get("priors", None)

    backward.update_db_config(cmd_config)

    return cmd_config
