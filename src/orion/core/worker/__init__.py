# -*- coding: utf-8 -*-
"""
Coordination of the optimization procedure
==========================================

Executes optimization steps and runs training experiment with parameter values suggested.

"""
import itertools
import logging

from orion.core.utils.exceptions import WaitingForTrials
from orion.core.utils.format_terminal import format_stats
from orion.core.worker.consumer import Consumer
from orion.core.worker.producer import Producer

log = logging.getLogger(__name__)


def reserve_trial(experiment, producer, _depth=1):
    """Reserve a new trial, or produce and reserve a trial if none are available."""
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


COMPLETION_MESSAGE = """\
Hints
=====

Info
----

To get more information on the experiment, run the command

orion info --name {experiment.name} --version {experiment.version}

"""


NONCOMPLETED_MESSAGE = """\
Status
------

To get the status of the trials, run the command

orion status --name {experiment.name} --version {experiment.version}


For a detailed view with status of each trial listed, use the argument `--all`

orion status --name {experiment.name} --version {experiment.version} --all

"""


def workon(
    experiment,
    max_trials=None,
    max_broken=None,
    max_idle_time=None,
    heartbeat=None,
    user_script_config=None,
    interrupt_signal_code=None,
    ignore_code_changes=None,
    knowledge_base=None,
):
    """Try to find solution to the search problem defined in `experiment`."""
    producer = Producer(experiment, max_idle_time, knowledge_base=knowledge_base)
    consumer = Consumer(
        experiment,
        heartbeat,
        user_script_config,
        interrupt_signal_code,
        ignore_code_changes,
    )

    log.debug("#####  Init Experiment  #####")
    try:
        iterator = range(int(max_trials))
    except (OverflowError, TypeError):
        # When worker_trials is inf
        iterator = itertools.count()

    worker_broken_trials = 0
    for _ in iterator:
        log.debug("#### Poll for experiment termination.")
        if experiment.is_broken:
            print("#### Experiment has reached broken trials threshold, terminating.")
            break

        if experiment.is_done:
            print("#####  Search finished successfully  #####")
            break

        log.debug("#### Try to reserve a new trial to evaluate.")
        try:
            trial = reserve_trial(experiment, producer)
        except WaitingForTrials as ex:
            print(
                "### Experiment failed to reserve new trials: {reason}, terminating. ".format(
                    reason=str(ex)
                )
            )
            break

        if trial is not None:
            log.info("#### Successfully reserved %s to evaluate. Consuming...", trial)
            success = consumer.consume(trial)
            if not success:
                worker_broken_trials += 1

        if worker_broken_trials >= max_broken:
            print("#### Worker has reached broken trials threshold, terminating.")
            print(worker_broken_trials, max_broken)
            break

    print("\n" + format_stats(experiment))

    print("\n" + COMPLETION_MESSAGE.format(experiment=experiment))
    if not experiment.is_done:
        print(NONCOMPLETED_MESSAGE.format(experiment=experiment))
