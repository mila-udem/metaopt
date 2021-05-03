# TODO: Not sure if this is totally necessary, but the idea is that we need to swap out
# the type of Study to use.

import itertools
import copy
from typing import Dict, List, Type, Union

from orion.benchmark.assessment.base import BaseAssess
from orion.benchmark.task.base import BaseTask
from orion.core.worker.multi_task_algo import AbstractKnowledgeBase

from .assessment.warm_start_efficiency import WarmStartEfficiency
from .benchmark import Benchmark
from .study import Study
from .warm_start_study import WarmStartStudy

from typing_extensions import TypedDict, runtime_checkable, Protocol


class TargetsDict(TypedDict):
    assess: List[BaseAssess]
    task: List[BaseTask]


class WarmStartTargetsDict(TypedDict):
    assess: List[WarmStartEfficiency]
    source_tasks: List[Union[BaseTask, List[BaseTask]]]
    target_tasks: List[BaseTask]


class WarmStartBenchmark(Benchmark):
    def __init__(
        self,
        name: str,
        algorithms: List[Union[str, Dict[str, Union[str, Dict]]]],
        targets: List[Union[TargetsDict, WarmStartTargetsDict]],
        knowledge_base_type: Type[AbstractKnowledgeBase],
    ):
        super().__init__(name, algorithms, targets)
        self.knowledge_base_type = knowledge_base_type

    def setup_studies(self):
        """Setup studies to run for the benchmark.
        Benchmark `algorithms`, together with each `task` and `assessment` combination
        define a study.
        """
        for target in self.targets:
            if "task" in target:
                assessments = target["assess"]
                tasks = target["task"]
                for assess, task in itertools.product(assessments, tasks):
                    # TODO: This isn't great.
                    study = Study(self, self.algorithms, assess, task,)
                    study.setup_experiments()
                    self.studies.append(study)
            elif "source_tasks" in target:
                assessments = target["assess"]
                source_tasks: List[Union[BaseTask, List[BaseTask]]] = target["source_tasks"]
                target_tasks = target["target_tasks"]

                for assessment in assessments:
                    assert isinstance(assessment, WarmStartEfficiency)
                    for source_task_or_tasks, target_task in zip(source_tasks, target_tasks):
                        study = WarmStartStudy(
                            benchmark=self,
                            algorithms=self.algorithms,
                            assessment=assessment,
                            source_tasks=source_task_or_tasks,
                            target_task=target_task,
                            knowledge_base_type=self.knowledge_base_type,
                            warm_start_seed=123,
                        )
                        study.setup_experiments()
                        self.studies.append(study)

    @property
    def configuration(self) -> Dict:
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

            if "task" in target:
                tasks = target["task"]
                str_tasks = dict()
                for task in tasks:
                    str_tasks.update(task.configuration)
                str_target["task"] = str_tasks
            elif "source_tasks" in target:
                source_tasks = target["source_tasks"]
                str_tasks = dict()
                for task in source_tasks:
                    str_tasks.update(task.configuration)
                str_target["source_tasks"] = str_tasks

                target_tasks = target["target_tasks"]
                str_tasks = dict()
                for task in target_tasks:
                    str_tasks.update(task.configuration)
                str_target["target_tasks"] = str_tasks
            else:
                raise NotImplementedError(target)

        targets.append(str_target)
        config["targets"] = targets

        if self.id is not None:
            config["_id"] = self.id

        return copy.deepcopy(config)
