# -*- coding: utf-8 -*-
"""
Oríon is an asynchronous distributed framework for black-box function optimization.

Its purpose is to serve as a hyperparameter optimizer for
machine learning models and training, as well as a flexible experimentation
platform for large scale asynchronous optimization procedures.

It has been designed firstly to disrupt a user's workflow at minimum, allowing
fast and efficient hyperparameter tuning, and secondly to provide secondary APIs
for more advanced features, such as dynamically reporting validation scores on
training time for automatic early stopping or on-the-fly reconfiguration.

Start by having a look here: https://github.com/epistimio/orion
"""
import logging
import os
import socket

from appdirs import AppDirs

from orion.core.io.config import Configuration

from ._version import get_versions

logger = logging.getLogger(__name__)


VERSIONS = get_versions()
del get_versions

__descr__ = "Asynchronous [black-box] Optimization"
__version__ = VERSIONS["version"]
__license__ = "BSD-3-Clause"
__author__ = u"Epistímio"
__author_short__ = u"Epistímio"
__author_email__ = "xavier.bouthillier@umontreal.ca"
__copyright__ = u"2017-2020, Epistímio"
__url__ = "https://github.com/epistimio/orion"

DIRS = AppDirs(__name__, __author_short__)
del AppDirs

DEF_CONFIG_FILES_PATHS = [
    os.path.join(DIRS.site_data_dir, "orion_config.yaml.example"),
    os.path.join(DIRS.site_config_dir, "orion_config.yaml"),
    os.path.join(DIRS.user_config_dir, "orion_config.yaml"),
]


def define_config():
    """Create and define the fields of the configuration object."""
    config = Configuration()
    define_storage_config(config)
    define_experiment_config(config)
    define_worker_config(config)
    define_evc_config(config)

    config.add_option(
        "user_script_config",
        option_type=str,
        default="config",
        deprecate=dict(version="v0.3", alternative="worker.user_script_config"),
    )

    config.add_option(
        "debug",
        option_type=bool,
        default=False,
        help="Turn Oríon into debug mode. Storage will be overriden to in-memory EphemeralDB.",
    )

    return config


def define_storage_config(config):
    """Create and define the fields of the storage configuration."""
    storage_config = Configuration()

    storage_config.add_option(
        "type", option_type=str, default="legacy", env_var="ORION_STORAGE_TYPE"
    )

    config.storage = storage_config

    define_database_config(config.storage)
    # Backward compatibility, should be removed in v0.3.0, or not?
    config.database = config.storage.database


def define_database_config(config):
    """Create and define the fields of the database configuration."""
    database_config = Configuration()

    database_config.add_option(
        "name",
        option_type=str,
        default="orion",
        env_var="ORION_DB_NAME",
        help="Name of the database.",
    )
    database_config.add_option(
        "type",
        option_type=str,
        default="PickledDB",
        env_var="ORION_DB_TYPE",
        help=(
            "Type of database. Builtin backends are ``mongodb``, "
            "``pickleddb`` and ``ephemeraldb``."
        ),
    )
    database_config.add_option(
        "host",
        option_type=str,
        default="",
        env_var="ORION_DB_ADDRESS",
        help="URI for ``mongodb``, or file path for ``pickleddb``.",
    )
    database_config.add_option(
        "port",
        option_type=int,
        default=27017,
        env_var="ORION_DB_PORT",
        help="Port address for ``mongodb``.",
    )

    config.database = database_config


def define_experiment_config(config):
    """Create and define the fields of generic experiment configuration."""
    experiment_config = Configuration()

    experiment_config.add_option(
        "max_trials",
        option_type=int,
        default=1000,
        env_var="ORION_EXP_MAX_TRIALS",
        help="number of trials to be completed for the experiment. This value "
        "will be saved within the experiment configuration and reused "
        "across all workers to determine experiment's completion. ",
    )

    experiment_config.add_option(
        "worker_trials",
        option_type=int,
        default=1000,
        deprecate=dict(
            version="v0.3",
            alternative="worker.max_trials",
            name="experiment.worker_trials",
        ),
        help="This argument will be removed in v0.3. Use --worker-max-trials instead.",
    )

    experiment_config.add_option(
        "max_broken",
        option_type=int,
        default=3,
        env_var="ORION_EXP_MAX_BROKEN",
        help=("Maximum number of broken trials before experiment stops."),
    )

    experiment_config.add_option(
        "working_dir",
        option_type=str,
        default="",
        env_var="ORION_WORKING_DIR",
        help="Set working directory for running experiment.",
    )

    experiment_config.add_option(
        "pool_size",
        option_type=int,
        default=1,
        deprecate=dict(version="v0.3", alternative=None, name="experiment.pool_size"),
        help="This argument will be removed in v0.3.",
    )

    experiment_config.add_option(
        "algorithms",
        option_type=dict,
        default={"random": {"seed": None}},
        help="Algorithm configuration for the experiment.",
    )

    experiment_config.add_option(
        "strategy",
        option_type=dict,
        default={"MaxParallelStrategy": {}},
        help="Parallel strategy to use with the algorithm.",
    )

    config.experiment = experiment_config


def define_worker_config(config):
    """Create and define the fields of the worker configuration."""
    worker_config = Configuration()

    worker_config.add_option(
        "heartbeat",
        option_type=int,
        default=120,
        env_var="ORION_HEARTBEAT",
        help=(
            "Frequency (seconds) at which the heartbeat of the trial is updated. "
            "If the heartbeat of a `reserved` trial is larger than twice the configured "
            "heartbeat, Oríon will reset the status of the trial to `interrupted`. "
            "This allows restoring lost trials (ex: due to killed worker)."
        ),
    )

    worker_config.add_option(
        "max_trials",
        option_type=int,
        default=int(10e8),
        env_var="ORION_WORKER_MAX_TRIALS",
        help="number of trials to be completed for this worker. "
        "If the experiment is completed, the worker will die even if it "
        "did not reach its maximum number of trials ",
    )

    worker_config.add_option(
        "max_broken",
        option_type=int,
        default=3,
        env_var="ORION_WORKER_MAX_BROKEN",
        help=("Maximum number of broken trials before worker stops."),
    )

    worker_config.add_option(
        "max_idle_time",
        option_type=int,
        default=60,
        env_var="ORION_MAX_IDLE_TIME",
        help=(
            "Maximum time the producer can spend trying to generate a new suggestion."
            "Such timeout are generally caused by slow database, large number of "
            "concurrent workers leading to many race conditions or small search spaces "
            "with integer/categorical dimensions that may be fully explored."
        ),
    )

    worker_config.add_option(
        "interrupt_signal_code",
        option_type=int,
        default=130,
        env_var="ORION_INTERRUPT_CODE",
        help="Signal returned by user script to signal to Oríon that it was interrupted.",
    )

    # TODO: Will this support -config as well, or only --config?
    worker_config.add_option(
        "user_script_config",
        option_type=str,
        default="config",
        env_var="ORION_USER_SCRIPT_CONFIG",
        help="Config argument name of user's script (--config).",
    )

    config.worker = worker_config


def define_evc_config(config):
    """Create and define the fields of the evc configuration."""
    evc_config = Configuration()

    # TODO: This should be built automatically like get_branching_args_group
    #       After this, the cmdline parser should be built based on config.

    evc_config.add_option(
        "auto_resolution",
        option_type=bool,
        default=True,
        deprecate=dict(
            version="v0.3",
            alternative="evc.manual_resolution",
            name="evc.auto_resolution",
        ),
        help="This argument will be removed in v0.3. "
        "Conflicts are now resolved automatically by default. "
        "See --manual-resolution to avoid auto-resolution.",
    )

    evc_config.add_option(
        "manual_resolution",
        option_type=bool,
        default=False,
        env_var="ORION_EVC_MANUAL_RESOLUTION",
        help=(
            "If ``True``, enter experiment version control conflict resolver for "
            "manual resolution on branching events. Otherwise, auto-resolution is "
            "attempted."
        ),
    )

    evc_config.add_option(
        "non_monitored_arguments",
        option_type=list,
        default=[],
        env_var="ORION_EVC_NON_MONITORED_ARGUMENTS",
        help=(
            "Ignore these commandline arguments when looking for differences in "
            "user's commandline call. "
            "Environment variable and commandline only supports one argument. "
            "Use global config or local config to pass a list of arguments to ignore."
        ),
    )

    evc_config.add_option(
        "ignore_code_changes",
        option_type=bool,
        default=False,
        env_var="ORION_EVC_IGNORE_CODE_CHANGES",
        help=("If ``True``, ignore code changes when looking for differences."),
    )

    evc_config.add_option(
        "algorithm_change",
        option_type=bool,
        default=False,
        env_var="ORION_EVC_ALGO_CHANGE",
        help=(
            "Set algorithm change as resolved if a branching event occur. "
            "Child and parent experiment have access to all trials from each other "
            "when the only difference between them is the algorithm configuration."
        ),
    )

    evc_config.add_option(
        "code_change_type",
        option_type=str,
        default="break",
        env_var="ORION_EVC_CODE_CHANGE",
        help=(
            "One of ``break``, ``unsure`` or ``noeffect``. "
            "Defines how trials should be filtered in Experiment Version Control tree "
            "if there is a change in the user's code repository. "
            "If the effect of the change is ``unsure``, "
            "the child experiment will access the trials of the parent but not "
            "the other way around. "
            "This is to ensure parent experiment does not get corrupted with possibly "
            "incompatible results. "
            "The child cannot access the trials from parent if ``code_change_type`` "
            "is ``break``. The parent cannot access trials from child if "
            "``code_change_type`` is ``unsure`` or ``break``."
        ),
    )

    evc_config.add_option(
        "cli_change_type",
        option_type=str,
        default="break",
        env_var="ORION_EVC_CMDLINE_CHANGE",
        help=(
            "One of ``break``, ``unsure`` or ``noeffect``. "
            "Defines how trials should be filtered in Experiment Version Control tree "
            "if there is a change in the user's commandline call. "
            "If the effect of the change is ``unsure``, "
            "the child experiment will access the trials of the parent but not "
            "the other way around. "
            "This is to ensure parent experiment does not get corrupted with possibly "
            "incompatible results. "
            "The child cannot access the trials from parent if ``cli_change_type`` "
            "is ``break``. The parent cannot access trials from child if "
            "``cli_change_type`` is ``unsure`` or ``break``."
        ),
    )

    evc_config.add_option(
        "config_change_type",
        option_type=str,
        default="break",
        env_var="ORION_EVC_CONFIG_CHANGE",
        help=(
            "One of ``break``, ``unsure`` or ``noeffect``. "
            "Defines how trials should be filtered in Experiment Version Control tree "
            "if there is a change in the user's script. "
            "If the effect of the change is ``unsure``, "
            "the child experiment will access the trials of the parent but not "
            "the other way around. "
            "This is to ensure parent experiment does not get corrupted with possibly "
            "incompatible results. "
            "The child cannot access the trials from parent if ``config_change_type`` "
            "is ``break``. The parent cannot access trials from child if "
            "``config_change_type`` is ``unsure`` or ``break``."
        ),
    )

    evc_config.add_option(
        "orion_version_change",
        option_type=bool,
        default=False,
        env_var="ORION_EVC_ORION_VERSION_CHANGE",
        help=(
            "Set orion version change as resolved if branching event occured"
            "Child and parent experiment have access to all trials from each other "
            "when the only difference between them is the orion version used during execution."
        ),
    )

    config.evc = evc_config


def build_config():
    """Define the config and fill it based on global configuration files."""
    config = define_config()
    for file_path in DEF_CONFIG_FILES_PATHS:
        if not os.path.exists(file_path):
            logger.debug("Config file not found: %s", file_path)
            continue

        config.load_yaml(file_path)

    return config


config = build_config()
