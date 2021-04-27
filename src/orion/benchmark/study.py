#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Benchmark definition
======================
"""
import copy
import itertools

from tabulate import tabulate

from orion.client import create_experiment

class Study:
    """
    A study is one assessment and task combination in the `Benchmark` targets. It will
    build and run experiments for all the algorithms for that task.

    Parameters
    ----------
    benchmark: A Benchmark instance
    algorithms: list
        Algorithms used for benchmark, each algorithm can be a string or dict, with same format as `Benchmark` algorithms.
    assessment: list
        `Assessment` instance
    task: list
        `Task` instance
    """

    class _StudyAlgorithm:
        """
        Represent user input json format algorithm as a Study algorithm object for easy to use.
        Parameters
        ----------
        algorithm: one algorithm in the `Study` algorithms list.
        """

        def __init__(self, algorithm):
            parameters = None
            deterministic = False

            if isinstance(algorithm, dict):
                if len(algorithm) > 1 or algorithm.get("algorithm"):
                    deterministic = algorithm.get("deterministic", False)
                    experiment_algorithm = algorithm["algorithm"]

                    if isinstance(experiment_algorithm, dict):
                        name, parameters = list(experiment_algorithm.items())[0]
                    else:
                        name = experiment_algorithm
                else:
                    name, parameters = list(algorithm.items())[0]
            else:
                name = algorithm

            self.algo_name = name
            self.parameters = parameters
            self.deterministic = deterministic

        @property
        def name(self):
            return self.algo_name

        @property
        def experiment_algorithm(self):
            if self.parameters:
                return {self.algo_name: self.parameters}
            else:
                return self.algo_name

        @property
        def is_deterministic(self):
            return self.deterministic

    def __init__(self, benchmark, algorithms, assessment, task):
        self.algorithms = self._build_benchmark_algorithms(algorithms)
        self.assessment = assessment
        self.task = task
        self.benchmark = benchmark

        self.assess_name = type(self.assessment).__name__
        self.task_name = type(self.task).__name__
        self.experiments_info = []

    def _build_benchmark_algorithms(self, algorithms):
        benchmark_algorithms = list()
        for algorithm in algorithms:
            benchmark_algorithm = self._StudyAlgorithm(algorithm)
            benchmark_algorithms.append(benchmark_algorithm)
        return benchmark_algorithms

    def setup_experiments(self):
        """Setup experiments to run of the study"""
        max_trials = self.task.max_trials
        task_num = self.assessment.task_num
        space = self.task.get_search_space()

        for task_index in range(task_num):

            for algo_index, algorithm in enumerate(self.algorithms):

                # Run only 1 experiment for deterministic algorithm
                if algorithm.is_deterministic and task_index > 0:
                    continue

                experiment_name = "_".join([
                    self.benchmark.name,
                    self.assess_name,
                    self.task_name,
                    str(task_index),
                    str(algo_index),
                ])

                experiment = create_experiment(
                    experiment_name,
                    space=space,
                    algorithms=algorithm.experiment_algorithm,
                    max_trials=max_trials,
                )
                self.experiments_info.append((task_index, experiment))

    def execute(self):
        """Execute all the experiments of the study"""
        max_trials = self.task.max_trials

        for _, experiment in self.experiments_info:
            # TODO: it is a blocking call
            experiment.workon(self.task, max_trials)

    def status(self):
        """Return status of the study"""
        algorithm_tasks = {}

        for _, experiment in self.experiments_info:
            trials = experiment.fetch_trials()

            algorithm_name = list(experiment.configuration["algorithms"].keys())[0]

            if algorithm_tasks.get(algorithm_name, None) is None:
                task_state = {
                    "algorithm": algorithm_name,
                    "experiments": 0,
                    "assessment": self.assess_name,
                    "task": self.task_name,
                    "completed": 0,
                    "trials": 0,
                }
            else:
                task_state = algorithm_tasks[algorithm_name]

            task_state["experiments"] += 1
            task_state["trials"] += len(trials)
            if experiment.is_done:
                task_state["completed"] += 1

            algorithm_tasks[algorithm_name] = task_state

        return list(algorithm_tasks.values())

    def analysis(self):
        """Return assessment figure"""
        return self.assessment.analysis(self.task_name, self.experiments_info)

    def experiments(self):
        """Return all the experiments of the study"""
        exps = []
        for _, experiment in self.experiments_info:
            exps.append(experiment)
        return exps

    def __repr__(self):
        """Represent the object as a string."""
        algorithms_list = [algorithm.name for algorithm in self.algorithms]

        return "Study(assessment=%s, task=%s, algorithms=[%s])" % (
            self.assess_name,
            self.task_name,
            ",".join(algorithms_list),
        )
