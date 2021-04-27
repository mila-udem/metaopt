import copy
import itertools

from tabulate import tabulate

from orion.client import create_experiment, ExperimentClient
from typing import List, Union, Dict, Tuple
from .study import Study
from .assessment.base import BaseAssess
from .task.base import BaseTask
from .benchmark import Benchmark


class WarmStartStudy(Study):
    """
    A study is one assessment and task combination in the `Benchmark` targets. It will
    build and run experiments for all the algorithms for that task.
    """

    def __init__(
        self,
        benchmark: Benchmark,
        algorithms: List[Union[str, Dict]],
        assessment: BaseAssess,
        task: BaseTask,
    ):
        """
        [extended_summary]

        Parameters
        ----------
        - benchmark : Benchmark
            [description]
        - algorithms : List[Union[str]]
            Algorithms used for benchmark, each algorithm can be a string or dict, with
            same format as `Benchmark` algorithms.
        - assessment : BaseAssess
        - task : BaseTask
        """
        super().__init__(benchmark=benchmark, algorithms=algorithms,
                         assessment=assessment, task=task)
        self.algorithms: List[Study._StudyAlgorithm]
        self.assessment: BaseAssess
        self.task: BaseTask
        self.benchmark: Benchmark
        self.assess_name = type(self.assessment).__name__
        self.task_name = type(self.task).__name__
        self.experiments_info: List[Tuple[int, ExperimentClient]] = []

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

                experiment_name = "_".join(
                    [
                        self.benchmark.name,
                        self.assess_name,
                        self.task_name,
                        str(task_index),
                        str(algo_index),
                    ]
                )
                # TODO: Here we need to, instead, create the cold, warm, and hot-start
                # experiments?
                # TODO: IDEA: Need to also create, for each existing experiment, two other
                # experiments:
                # - One for warm-start, which need to reuse the same 'storage' as the
                # previous exp somehow
                # - One for hot-start, which need to reuse the results of the first exp?
                # Or start from a number of random points from the first experiment?
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
