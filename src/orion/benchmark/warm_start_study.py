import copy
import itertools
import logging
from typing import Any, Dict, List, Optional, Tuple, Type, Union

from orion.algo.base import BaseAlgorithm
from orion.client import ExperimentClient, create_experiment
from orion.core.utils.singleton import update_singletons
from orion.core.worker.multi_task_algo import AbstractKnowledgeBase
from orion.core.worker.producer import Producer
from tabulate import tabulate

from .assessment.base import BaseAssess
from .benchmark import Benchmark
from .study import Study
from .task.base import BaseTask

logger = logging.getLogger(__file__)


def is_deterministic(algorithm: Union[str, Dict[str, Any], BaseAlgorithm]) -> bool:
    """Returns wether the given algorithm or algorithm config is deterministic.

    Parameters
    ----------
    algorithm : Union[str, Dict]
        The algorithm name, or the algorithm config.

    Returns
    -------
    bool
        Wether the algorithm is deterministic or not.
    """
    if isinstance(algorithm, dict):
        if len(algorithm) > 1 or algorithm.get("algorithm"):
            return algorithm.get("deterministic", False)
    return getattr(algorithm, "deterministic", False)


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
        source_tasks: List[BaseTask],
        target_task: BaseTask,
        knowledge_base_type: Type[AbstractKnowledgeBase],
        warm_start_seed: int = None,
        debug: bool = True,
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
        - source_tasks : List[BaseTask]
            The tasks from which to sample data when warm-starting the target task.
        - target_task: BaseTask
            The task which is to be warm-started using the knowledge base.
        - knowledge_base_type: Type[AbstractKnowledgeBase]
            The type of knowledge base to use for this study.
        - warm_start_seed: Optional[int]
            The seed to use for the sampling of the warm-starting points.
        """
        super().__init__(
            benchmark=benchmark,
            algorithms=algorithms,
            assessment=assessment,
            task=target_task,
        )
        # NOTE: Bypassing the '_StudyAlgorithm' wrapper from Study.
        # self.algorithms: List[Study._StudyAlgorithm]
        self.algorithms: List[Union[str, Dict[str, Any]]] = algorithms
        self.assessment: BaseAssess
        self.task: BaseTask
        self.benchmark: Benchmark
        self.assess_name = type(self.assessment).__name__
        self.task_name = type(self.task).__name__
        self.source_tasks: List[BaseTask]
        if isinstance(source_tasks, list):
            self.source_tasks = source_tasks
        else:
            self.source_tasks = [source_tasks]
        self.target_task = target_task
        self.knowledge_base_type = knowledge_base_type
        self.warm_start_seed: Optional[int] = warm_start_seed
        self.debug = debug

        # Dict mapping from some kind of key (TODO) to a list of the (cold, warm, hot)
        # experiment clients for each repetition.
        self.experiments_info: Dict[
            str, List[Tuple[ExperimentClient, ExperimentClient, ExperimentClient]]
        ] = {}

    def setup_experiments(self):
        """Setup experiments to run of the study"""
        n_required_tasks = self.assessment.task_num
        target_task_name = getattr(
            self.target_task, "name", type(self.target_task).__name__
        )
        self.experiments_info.clear()
        for algo_index, algorithm in enumerate(self.algorithms):
            self.experiments_info[algo_index] = []

            for task_repetition_index in range(n_required_tasks):
                # Create the Cold / Warm / Hot Knowledge bases:
                # Cold start KB: Start with no previous trials.
                cold_start_kb = self.knowledge_base_type()
                assert cold_start_kb.n_stored_experiments == 0

                warm_start_kb = self.knowledge_base_type()
                for source_task in self.source_tasks:
                    # Add randomly sampled trials from each source task to the warm KB.
                    source_task_name = getattr(
                        source_task, "name", type(source_task).__name__
                    )
                    dummy_hot_start_exp_name = "_".join(
                        [
                            self.benchmark.name,
                            self.assess_name,
                            source_task_name,
                            str(task_repetition_index),
                            str(algo_index),
                            "dummy_warm",
                        ]
                    )
                    # Use a different seed for the sampling of the warm-start trials for
                    # each repetition.
                    seed = (self.warm_start_seed or 0) + task_repetition_index
                    dummy_warm_start_experiment = create_experiment(
                        name=dummy_hot_start_exp_name,
                        space=source_task.get_search_space(),
                        algorithms={"random": {"seed": seed}},
                        max_trials=source_task.max_trials,
                        debug=True,
                    )
                    # TODO: Do we want to run this dummy experiment now? or later?
                    logger.info(
                        f"Sampling trials for the dummy 'warm-start' experiment "
                        f"{dummy_warm_start_experiment}"
                    )
                    dummy_warm_start_experiment.workon(
                        source_task, max_trials=source_task.max_trials
                    )
                    warm_start_kb.add_experiment(dummy_warm_start_experiment)
                assert warm_start_kb.n_stored_experiments == len(self.source_tasks)

                # Add randomly sampled trials from the target task to the hot-start KB.
                hot_start_kb = self.knowledge_base_type()
                # Create a 'source' experiment for the Hot-start experiment.
                dummy_hot_start_exp_name = "_".join(
                    [
                        self.benchmark.name,
                        self.assess_name,
                        target_task_name,
                        str(task_repetition_index),
                        str(algo_index),
                        "dummy_hot",
                    ]
                )
                seed = (self.warm_start_seed or 0) + task_repetition_index
                dummy_hot_start_experiment = create_experiment(
                    name=dummy_hot_start_exp_name,
                    space=self.target_task.get_search_space(),
                    algorithms={"random": {"seed": seed}},
                    max_trials=self.target_task.max_trials,
                    debug=True,
                )
                # Gather some random trials from the target experiment.
                # TODO: (same as above)
                logger.info(
                    f"Sampling trials for the dummy 'hot-start' experiment "
                    f"{dummy_hot_start_experiment}"
                )
                dummy_hot_start_experiment.workon(
                    self.target_task, max_trials=self.target_task.max_trials
                )
                hot_start_kb.add_experiment(dummy_hot_start_experiment)
                assert hot_start_kb.n_stored_experiments == 1

                # Create the Cold / Warm / Hot Experiments, using the corresponding
                # knowledge bases created above.

                base_experiment_name = "_".join(
                    [
                        self.benchmark.name,
                        self.assess_name,
                        target_task_name,
                        str(task_repetition_index),
                        str(algo_index),
                    ]
                )
                import numpy as np

                logger.info(f"Creating the cold start experiment.")
                cold_start_experiment = create_experiment(
                    name=f"{base_experiment_name}_cold",
                    space=self.target_task.get_search_space(),
                    algorithms=algorithm,
                    # # Huuh? Why isn't this just `algorithm`?
                    # algorithms=algorithm.experiment_algorithm,
                    max_trials=self.target_task.max_trials,
                    # NOTE: Could also pass None here, which wouldn't use a KB, but this
                    # could be considered more explicit, in that it prevents us from
                    # ever deciding to create a KB with the current storage, or
                    # something like that, which might contain trials from other
                    # experiments, which we don't want.
                    knowledge_base=None,
                    debug=self.debug,  # ? TODO: Should we set the debug flag?
                )
                logger.info(f"Creating the warm start experiment.")
                warm_start_experiment = create_experiment(
                    name=f"{base_experiment_name}_warm",
                    space=self.target_task.get_search_space(),
                    algorithms=algorithm,
                    # Huuh? Why isn't this just `algorithm`?
                    # algorithms=algorithm.experiment_algorithm,
                    max_trials=self.target_task.max_trials,
                    knowledge_base=warm_start_kb,
                    debug=self.debug,
                )
                logger.info(f"Creating the hot start experiment.")
                hot_start_experiment = create_experiment(
                    name=f"{base_experiment_name}_hot",
                    space=self.target_task.get_search_space(),
                    algorithms=algorithm,
                    max_trials=self.target_task.max_trials,
                    knowledge_base=hot_start_kb,
                    debug=self.debug,
                )
                self.experiments_info[algo_index].append(
                    (cold_start_experiment, warm_start_experiment, hot_start_experiment)
                )

    def execute(self):
        """Execute all the experiments of the study"""
        for _, experiments_to_run in self.experiments_info.items():
            for cold_start_exp, warm_start_exp, hot_start_exp in experiments_to_run:
                # Actually run the cold / warm / hot experiments.
                logger.info("Starting cold start experiment.")
                cold_start_exp.workon(self.target_task, self.target_task.max_trials)

                logger.info("Starting warm start experiment.")
                warm_start_exp.workon(self.target_task, self.target_task.max_trials)

                logger.info("Starting hot start experiment.")
                hot_start_exp.workon(self.target_task, self.target_task.max_trials)

    def status(self):
        """Return status of the study"""
        algorithm_tasks = {}

        for _, experiment_tuples_list in self.experiments_info.items():
            for experiment_tuple in experiment_tuples_list:
                for experiment in experiment_tuple:
                    trials = experiment.fetch_trials()

                    algorithm_name = list(experiment.configuration["algorithms"].keys())[0]

                    task_state = algorithm_tasks.setdefault(
                        algorithm_name,
                        {
                            "algorithm": algorithm_name,
                            "experiments": 0,
                            "assessment": self.assess_name,
                            "task": self.task_name,
                            "completed": 0,
                            "trials": 0,
                        },
                    )

                    task_state["experiments"] += 1
                    task_state["trials"] += len(trials)
                    if experiment.is_done:
                        task_state["completed"] += 1

        return list(algorithm_tasks.values())

