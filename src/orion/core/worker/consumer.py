# -*- coding: utf-8 -*-
"""
Evaluate objective on a set of parameters
=========================================

Call user's script as a black box process to evaluate a trial.

"""
import copy
import json
import logging
import os
import signal
import subprocess
import tempfile

import orion.core
from orion.core.io.convert import JSONConverter
from orion.core.io.orion_cmdline_parser import OrionCmdlineParser
from orion.core.io.resolve_config import infer_versioning_metadata
from orion.core.utils.exceptions import (
    BranchingEvent,
    InexecutableUserScript,
    MissingResultFile,
)
from orion.core.utils.working_dir import WorkingDir

log = logging.getLogger(__name__)


# pylint: disable = unused-argument
def _handler(signum, frame):
    log.error("Oríon has been interrupted.")
    raise KeyboardInterrupt


class ExecutionError(Exception):
    """Error raised when Orion is unable to execute the user's script without errors."""

    pass


class Consumer(object):
    """Consume a trial by using it to initialize a black-box box to evaluate it.

    It uses an `Experiment` object to push an evaluated trial, if results are
    delivered to the worker process successfully.

    It forks another process which executes user's script with the suggested
    options. It expects results to be written in a **JSON** file, whose path
    has been defined in a special orion environmental variable which is set
    into the child process' environment.

    Parameters
    ----------
    experiment: `orion.core.worker.experiment.Experiment`
        Manager of this experiment, provides convenient interface for interacting with
        the database.

    heartbeat: int, optional
        Frequency (seconds) at which the heartbeat of the trial is updated.
        If the heartbeat of a `reserved` trial is larger than twice the configured
        heartbeat, Oríon will reset the status of the trial to `interrupted`.
        This allows restoring lost trials (ex: due to killed worker).
        Defaults to ``orion.core.config.worker.heartbeat``.

    user_script_config: str, optional
        Config argument name of user's script (--config).
        Defaults to ``orion.core.config.worker.user_script_config``.

    interrupt_signal_code: int, optional
        Signal returned by user script to signal to Oríon that it was interrupted.
        Defaults to ``orion.core.config.worker.interrupt_signal_code``.

    """

    def __init__(
        self,
        experiment,
        user_script_config=None,
        interrupt_signal_code=None,
        ignore_code_changes=None,
    ):
        log.debug("Creating Consumer object.")
        self.experiment = experiment
        self.space = experiment.space
        if self.space is None:
            raise RuntimeError(
                "Experiment object provided to Consumer has not yet completed"
                " initialization."
            )

        if user_script_config is None:
            user_script_config = orion.core.config.worker.user_script_config

        if interrupt_signal_code is None:
            interrupt_signal_code = orion.core.config.worker.interrupt_signal_code

        if ignore_code_changes is None:
            ignore_code_changes = orion.core.config.evc.ignore_code_changes

        self.interrupt_signal_code = interrupt_signal_code
        self.ignore_code_changes = ignore_code_changes

        # Fetch space builder
        self.template_builder = OrionCmdlineParser(user_script_config)
        self.template_builder.set_state_dict(experiment.metadata["parser"])
        # Get path to user's script and infer trial configuration directory
        if experiment.working_dir:
            self.working_dir = os.path.abspath(experiment.working_dir)
        else:
            self.working_dir = os.path.join(tempfile.gettempdir(), "orion")

        self.pacemaker = None

    def __call__(self, trial, **kwargs):
        """Execute user's script as a block box using the options contained
        within ``trial``.

        Parameters
        ----------
        trial: `orion.core.worker.trial.Trial`
            Orion trial to execute.

        Returns
        -------
        bool
            True if the trial was successfully executed. False if the trial is broken.

        """
        signal.signal(signal.SIGTERM, _handler)

        log.debug("Creating new directory at '%s':", self.working_dir)
        temp_dir = not bool(self.experiment.working_dir)
        prefix = self.experiment.name + "_"
        suffix = trial.id

        with WorkingDir(
            self.working_dir, temp_dir, prefix=prefix, suffix=suffix
        ) as workdirname:
            log.debug("New consumer context: %s", workdirname)
            trial.working_dir = workdirname

            results_file = self._consume(trial, workdirname)

            log.debug("Parsing results from file and fill corresponding Trial object.")
            results = self.retrieve_results(results_file)

        return results

    def retrieve_results(self, results_file):
        """Retrive the results from the file"""
        try:
            results = JSONConverter().parse(results_file.name)
        except json.decoder.JSONDecodeError:
            raise MissingResultFile()

        return results

    def get_execution_environment(self, trial, results_file="results.log"):
        """Set a few environment variables to allow users and
        underlying processes to know if they are running under orion.

        Parameters
        ----------
        results_file: str
           file used to store results, this is only used by the legacy protocol

        trial: Trial
           reference to the trial object that is going to be run

        Notes
        -----
        This function defines the environment variables described below

        .. envvar:: ORION_EXPERIMENT_ID
           :noindex:

           Current experiment that is being ran.

        .. envvar::  ORION_EXPERIMENT_NAME
           :noindex:

           Name of the experiment the worker is currently working on.

        .. envvar::  ORION_EXPERIMENT_VERSION
           :noindex:

           Version of the experiment the worker is currently working on.

        .. envvar:: ORION_TRIAL_ID
           :noindex:

           Current trial id that is currently being executed in this process.

        .. envvar:: ORION_WORKING_DIRECTORY
           :noindex:

           Trial's current working directory.

        .. envvar:: ORION_RESULTS_PATH
           :noindex:

           Trial's results file that is read by the legacy protocol to get the results of the trial
           after a successful run.

        """
        env = dict(os.environ)
        env["ORION_EXPERIMENT_ID"] = str(self.experiment.id)
        env["ORION_EXPERIMENT_NAME"] = str(self.experiment.name)
        env["ORION_EXPERIMENT_VERSION"] = str(self.experiment.version)
        env["ORION_TRIAL_ID"] = str(trial.id)

        env["ORION_WORKING_DIR"] = str(trial.working_dir)
        env["ORION_RESULTS_PATH"] = str(results_file)
        env["ORION_INTERRUPT_CODE"] = str(self.interrupt_signal_code)

        return env

    def _consume(self, trial, workdirname):
        config_file = tempfile.NamedTemporaryFile(
            mode="w", prefix="trial_", suffix=".conf", dir=workdirname, delete=False
        )
        config_file.close()
        log.debug("New temp config file: %s", config_file.name)
        results_file = tempfile.NamedTemporaryFile(
            mode="w", prefix="results_", suffix=".log", dir=workdirname, delete=False
        )
        results_file.close()
        log.debug("New temp results file: %s", results_file.name)

        log.debug("Building command line argument and configuration for trial.")
        env = self.get_execution_environment(trial, results_file.name)
        cmd_args = self.template_builder.format(
            config_file.name, trial, self.experiment
        )

        log.debug("Launch user's script as a subprocess and wait for finish.")

        self._validate_code_version()

        self.execute_process(cmd_args, env)

        return results_file

    def _validate_code_version(self):
        if self.ignore_code_changes:
            return

        old_config = self.experiment.configuration
        new_config = copy.deepcopy(old_config)
        new_config["metadata"]["VCS"] = infer_versioning_metadata(
            old_config["metadata"]["user_script"]
        )

        # Circular import
        from orion.core.evc.conflicts import CodeConflict

        conflicts = list(CodeConflict.detect(old_config, new_config))
        if conflicts:
            raise BranchingEvent(
                f"Code changed between execution of 2 trials:\n{conflicts[0]}"
            )

    # pylint: disable = no-self-use
    def execute_process(self, cmd_args, environ):
        """Facilitate launching a black-box trial."""
        command = cmd_args

        try:
            process = subprocess.Popen(command, env=environ)
        except PermissionError:
            log.debug("Script is not executable")
            raise InexecutableUserScript(" ".join(cmd_args))

        return_code = process.wait()

        if return_code == self.interrupt_signal_code:
            raise KeyboardInterrupt()
        elif return_code != 0:
            raise ExecutionError(
                "Something went wrong. Check logs. Process "
                "returned with code {} !".format(return_code)
            )
