#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Benchmark definition
======================
"""
import copy
import itertools

from tabulate import tabulate


class Benchmark:
    """
    Benchmark definition

    Parameters
    ----------
    name: str
        Name of the benchmark
    algorithms: list, optional
        Algorithms used for benchmark, and for each algorithm, it can be formats as below:

        - A `str` of the algorithm name
        - A `dict`, with only one key and one value, where key is the algorithm name and value is a dict for the algorithm config.
        - A `dict`, with two keys.

            algorithm: str or dict
                Algorithm name in string or a dict with algorithm configure.
            deterministic: bool, optional
                True if it is a deterministic algorithm, then for each assessment, only one experiment
                will be run for this algorithm.

        Examples:

        >>> ["random", "tpe"]
        >>> ["random", {"tpe": {"seed": 1}}]
        >>> [{"algorithm": "random"}, {"algorithm": {"gridsearch": {"n_values": 50}}, "deterministic": True}]

    targets: list, optional
        Targets for the benchmark, each target will be a dict with two keys.

        assess: list
            Assessment objects
        task: list
            Task objects
    """

    def __init__(self, name, algorithms, targets):
        self._id = None
        self.name = name
        self.algorithms = algorithms
        self.targets = targets
        self.metadata = {}

        self.studies = []

    def setup_studies(self):
        """Setup studies to run for the benchmark.
        Benchmark `algorithms`, together with each `task` and `assessment` combination
        define a study.
        """
        for target in self.targets:
            assessments = target["assess"]
            tasks = target["task"]

            for assess, task in itertools.product(*[assessments, tasks]):
                study = Study(self, self.algorithms, assess, task)
                study.setup_experiments()
                self.studies.append(study)

    def process(self):
        """Run studies experiment"""
        for study in self.studies:
            study.execute()

    def status(self, silent=True):
        """Display benchmark status"""
        total_exp_num = 0
        complete_exp_num = 0
        total_trials = 0
        benchmark_status = []
        for study in self.studies:
            for status in study.status():
                column = dict()
                column["Algorithms"] = status["algorithm"]
                column["Assessments"] = status["assessment"]
                column["Tasks"] = status["task"]
                column["Total Experiments"] = status["experiments"]
                total_exp_num += status["experiments"]
                column["Completed Experiments"] = status["completed"]
                complete_exp_num += status["completed"]
                column["Submitted Trials"] = status["trials"]
                total_trials += status["trials"]
                benchmark_status.append(column)

        if not silent:
            print(
                "Benchmark name: {}, Experiments: {}/{}, Submitted trials: {}".format(
                    self.name, complete_exp_num, total_exp_num, total_trials
                )
            )
            self._pretty_table(benchmark_status)

        return benchmark_status

    def analysis(self):
        """Return all the assessment figures"""
        figures = []
        for study in self.studies:
            figure = study.analysis()
            figures.append(figure)
        return figures

    def experiments(self, silent=True):
        """Return all the experiments submitted in benchmark"""
        experiment_table = []
        for study in self.studies:
            for exp in study.experiments():
                exp_column = dict()
                stats = exp.stats
                exp_column["Algorithm"] = list(exp.configuration["algorithms"].keys())[
                    0
                ]
                exp_column["Experiment Name"] = exp.name
                exp_column["Number Trial"] = len(exp.fetch_trials())
                exp_column["Best Evaluation"] = stats["best_evaluation"]
                experiment_table.append(exp_column)

        if not silent:
            print("Total Experiments: {}".format(len(experiment_table)))
            self._pretty_table(experiment_table)

        return experiment_table

    def _pretty_table(self, dict_list):
        """
        Print a list of same format dict as pretty table with IPython disaply(notebook) or tablute
        :param dict_list: a list of dict where each dict has the same keys.
        """
        try:
            from IPython.display import HTML, display

            display(
                HTML(
                    tabulate(
                        dict_list,
                        headers="keys",
                        tablefmt="html",
                        stralign="center",
                        numalign="center",
                    )
                )
            )
        except ImportError:
            table = tabulate(
                dict_list,
                headers="keys",
                tablefmt="grid",
                stralign="center",
                numalign="center",
            )
            print(table)

    # pylint: disable=invalid-name
    @property
    def id(self):
        """Id of the benchmark in the database if configured.

        Value is `None` if the benchmark is not configured.
        """
        return self._id

    @property
    def configuration(self):
        """Return a copy of an `Benchmark` configuration as a dictionary."""
        config = dict()

        config["name"] = self.name
        config["algorithms"] = self.algorithms

        targets = []
        for target in self.targets:
            str_target = {}
            assessments = target["assess"]
            str_assessments = dict()
            for assessment in assessments:
                str_assessments.update(assessment.configuration)
            str_target["assess"] = str_assessments

            tasks = target["task"]
            str_tasks = dict()
            for task in tasks:
                str_tasks.update(task.configuration)
            str_target["task"] = str_tasks

        targets.append(str_target)
        config["targets"] = targets

        if self.id is not None:
            config["_id"] = self.id

        return copy.deepcopy(config)
