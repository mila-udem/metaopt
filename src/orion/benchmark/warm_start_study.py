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
from .assessment.warm_start_efficiency import WarmStartEfficiency
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


def get_algorithm_name(algorithm: Union[str, Dict, BaseAlgorithm]) -> str:
    """Get the name of the given algorithm or algo config or algo instance.

    Parameters
    ----------
    algorithm : Union[str, Dict, BaseAlgorithm]
        Algorithm or config.

    Returns
    -------
    str
        Its name.
    """
    if isinstance(algorithm, str):
        return algorithm
    if isinstance(algorithm, dict):
        if len(algorithm) == 1:
            first_key = list(algorithm.keys())[0]
            return first_key
    if isinstance(algorithm, BaseAlgorithm):
        return getattr(algorithm, "name", type(algorithm).__name__)
    return NotImplementedError(algorithm)


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

        self.target_task_name: str = getattr(
            self.target_task, "name", type(self.target_task).__name__
        )
        # Number of repetitions to perform for each algo / task combination.
        # Each repetition will use a different seed for the warm-start trials.
        # TODO: Also use a different seed for the algo?
        self.repetitions: int = self.assessment.task_num

        # Dict mapping from some kind of key (TODO) to a list of the (cold, warm, hot)
        # experiment clients for each repetition.
        # self.experiments_info: Dict[
        #     int, List[Tuple[ExperimentClient, ExperimentClient, ExperimentClient]]
        # ] = {}

        # self.cold_start_kbs: List[AbstractKnowledgeBase] = []
        self.warm_start_kbs: List[AbstractKnowledgeBase] = []
        self.hot_start_kbs: List[AbstractKnowledgeBase] = []

        # Lists containing the 'source' experiments for the warm-start and hot-start
        # experiments. In the case of warm-start it is a list of lists, since there may
        # be more than one source experiment to warm-start from.
        self.dummy_warm_experiments: List[List[ExperimentClient]] = []
        self.dummy_hot_experiments: List[ExperimentClient] = []

        # Lists that hold the cold / warm / hot experiments, for each algorithm, for
        # each repetition.
        self.cold_start_experiments: List[List[ExperimentClient]] = []
        self.warm_start_experiments: List[List[ExperimentClient]] = []
        self.hot_start_experiments: List[List[ExperimentClient]] = []

    def _setup_source_experiments(self):
        """ Create the experiments containing the 'previous' trials to be used to
        warm-start and hot-start the algorithms.

        The trials are sampled using random search. All algorithms share the same
        knowledge base, for a given repetition index.
        The random search algorithm uses a different seed for each run.

        NOTE: This registers these warm and hot "source" experiments into the
        corresponding knowledge base, but doesn't actually call 'workon'. However, since
        the experiments are registered in the KB, calling workon on the experiment
        client will make those trials available in the KB as well.
        """
        for task_repetition_index in range(self.repetitions):
            # Get the Cold / Warm / Hot Knowledge bases:
            # NOTE: Not actually creating a Knowledge base for cold-start, see note
            # below.
            # Cold start KB: Start with no previous trials.
            # cold_start_kb = self.knowledge_base_type()
            # assert cold_start_kb.n_stored_experiments == 0
            warm_start_kb = self.warm_start_kbs[task_repetition_index]
            hot_start_kb = self.hot_start_kbs[task_repetition_index]

            dummy_warm_start_experiments = []
            for source_task_index, source_task in enumerate(self.source_tasks):
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
                        "dummy_warm",
                    ]
                )
                # Use a different seed for the sampling of the warm-start trials for
                # each repetition.
                seed = (
                    (self.warm_start_seed or 0)
                    + task_repetition_index
                    + source_task_index
                )
                dummy_warm_start_experiment = create_experiment(
                    name=dummy_hot_start_exp_name,
                    space=source_task.get_search_space(),
                    algorithms={"random": {"seed": seed}},
                    max_trials=source_task.max_trials,
                    debug=True,
                )
                dummy_warm_start_experiments.append(dummy_warm_start_experiment)
                warm_start_kb.add_experiment(dummy_warm_start_experiment)
            assert warm_start_kb.n_stored_experiments == len(self.source_tasks), (
                warm_start_kb.n_stored_experiments,
                len(self.source_tasks),
            )

            # Create a 'source' experiment for the Hot-start experiment, which will
            # contain the same number of trials as in the "warm-start" case, but all
            # points are from the target task.
            n_hot_start_trials = sum(task.max_trials for task in self.source_tasks)
            dummy_hot_start_exp_name = "_".join(
                [
                    self.benchmark.name,
                    self.assess_name,
                    self.target_task_name,
                    str(task_repetition_index),
                    "dummy_hot",
                ]
            )
            seed = (self.warm_start_seed or 0) + task_repetition_index
            dummy_hot_start_experiment = create_experiment(
                name=dummy_hot_start_exp_name,
                space=self.target_task.get_search_space(),
                algorithms={"random": {"seed": seed}},
                max_trials=n_hot_start_trials,
                debug=True,
            )
            hot_start_kb.add_experiment(dummy_hot_start_experiment)
            assert hot_start_kb.n_stored_experiments == 1

            self.dummy_warm_experiments.append(dummy_warm_start_experiments)
            self.dummy_hot_experiments.append(dummy_hot_start_experiment)
            # Store the knowledge bases for later.
            # self.cold_start_kbs.append(cold_start_kb)
            self.warm_start_kbs.append(warm_start_kb)
            self.hot_start_kbs.append(hot_start_kb)

    def _fill_knowledge_bases(self) -> None:
        """
        """

    def _clear(self) -> None:
        # Clear everything, for no real reason (this shouldn't really be used twice
        # anyway).
        self.dummy_warm_experiments.clear()
        self.dummy_hot_experiments.clear()

        self.cold_start_experiments.clear()
        self.warm_start_experiments.clear()
        self.hot_start_experiments.clear()

        # self.cold_start_kbs.clear()
        self.warm_start_kbs.clear()
        self.hot_start_kbs.clear()

        self.cold_start_experiments.clear()
        self.warm_start_experiments.clear()
        self.hot_start_experiments.clear()

    def setup_experiments(self):
        """Setup experiments to run of the study"""
        repetitions = self.assessment.task_num
        target_task_name = getattr(
            self.target_task, "name", type(self.target_task).__name__
        )
        self._clear()

        self.warm_start_kbs = [self.knowledge_base_type() for _ in range(repetitions)]
        self.hot_start_kbs = [self.knowledge_base_type() for _ in range(repetitions)]

        self._setup_source_experiments()

        self.cold_start_experiments = [[] for _ in self.algorithms]
        self.warm_start_experiments = [[] for _ in self.algorithms]
        self.hot_start_experiments = [[] for _ in self.algorithms]

        for algo_index, algorithm in enumerate(self.algorithms):
            # Create the Cold / Warm / Hot Experiments, using the corresponding
            # knowledge bases created above.
            for repetition_id, warm_start_kb, hot_start_kb in zip(
                range(repetitions), self.warm_start_kbs, self.hot_start_kbs
            ):
                base_experiment_name = "_".join(
                    [
                        self.benchmark.name,
                        self.assess_name,
                        target_task_name,
                        str(repetition_id),
                        str(algo_index),
                    ]
                )

                logger.info("Creating the cold start experiment.")
                cold_start_experiment = create_experiment(
                    name=f"{base_experiment_name}_cold",
                    space=self.target_task.get_search_space(),
                    algorithms=algorithm,
                    # # Huuh? Why isn't this just `algorithm`?
                    # algorithms=algorithm.experiment_algorithm,
                    max_trials=self.target_task.max_trials,
                    # TODO: Passing None here, because if we pass an empty KB, the
                    # MultiTaskAlgo wrapper will be enabled, and there will be an unused
                    # task-id dimension which might reduce the performance of the algo.
                    knowledge_base=None,
                    debug=self.debug,  # ? TODO: Should we set the debug flag?
                )
                logger.info("Creating the warm start experiment.")
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

                self.cold_start_experiments[algo_index].append(cold_start_experiment)
                self.warm_start_experiments[algo_index].append(warm_start_experiment)
                self.hot_start_experiments[algo_index].append(hot_start_experiment)

    def execute(self):
        """Execute all the experiments of the study"""
        # Actually fill the knowledge bases, by calling `workon` on the source
        # experiments, which are already registered in the corresponding knowledge base.
        for warm_start_experiments in self.dummy_warm_experiments:
            for dummy_experiment, source_task in zip(
                warm_start_experiments, self.source_tasks
            ):
                logger.info(
                    f"Sampling a maximum of {source_task.max_trials} trials for the "
                    f"dummy 'hot-start' experiment {dummy_experiment.name}"
                )
                dummy_experiment.workon(source_task, max_trials=source_task.max_trials)

        for dummy_experiment in self.dummy_hot_experiments:
            # Use the same total number of points as warm-starting, but from the target
            # task.
            hot_start_trials = sum(task.max_trials for task in self.source_tasks)
            logger.info(
                f"Sampling a maximum of {hot_start_trials} trials for the dummy "
                f"'hot-start' experiment {dummy_experiment.name}"
            )
            dummy_experiment.workon(self.target_task, max_trials=hot_start_trials)

        # Run the main experiments:
        for algo_index, algorithm in enumerate(self.algorithms):
            for run_id in range(self.repetitions):
                cold_start_exp = self.cold_start_experiments[algo_index][run_id]
                warm_start_exp = self.warm_start_experiments[algo_index][run_id]
                hot_start_exp = self.hot_start_experiments[algo_index][run_id]
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

        for experiment in self.experiments():
            trials = experiment.fetch_trials()

            algorithm_name = get_algorithm_name(experiment.algorithms)
            # TODO: Double-check that this also makes sense here:
            task_state = algorithm_tasks.setdefault(
                algorithm_name,
                {
                    "algorithm": algorithm_name,
                    "experiments": 0,
                    "assessment": self.assess_name,
                    "task": self.target_task_name,
                    "completed": 0,
                    "trials": 0,
                },
            )

            task_state["experiments"] += 1
            task_state["trials"] += len(trials)
            if experiment.is_done:
                task_state["completed"] += 1

        return list(algorithm_tasks.values())

    def analysis(self):
        """Return assessment figure"""
        assert isinstance(self.assessment, WarmStartEfficiency)
        experiment_infos = {}
        for i, _ in enumerate(self.algorithms):
            tuples = list(
                zip(
                    self.cold_start_experiments[i],
                    self.warm_start_experiments[i],
                    self.hot_start_experiments[i],
                )
            )
            experiment_infos[i] = tuples
        return self.assessment.analysis(self.task_name, experiment_infos)

    def __repr__(self):
        """Represent the object as a string."""
        algorithms_list = [
            get_algorithm_name(algorithm) for algorithm in self.algorithms
        ]

        return (
            f"WarmStartStudy(assessment={self.assess_name}, task={self.task_name}, "
            f"algorithms={algorithms_list})"
        )

    def experiments(
        self, include_source_experiments: bool = False
    ) -> List[ExperimentClient]:
        """Return all the experiments of the study
        
        When `include_source_experiments` is True, also includes the 'dummy' experiments
        used to gather the warm-start trials.
        """
        exps = []
        # TODO: Should we include the source experiments here?
        if include_source_experiments:
            for experiments in self.dummy_warm_experiments:
                exps.extend(experiments)
            for experiments in self.dummy_hot_experiments:
                exps.extend(experiments)
        for experiments in self.cold_start_experiments:
            exps.extend(experiments)
        for experiments in self.warm_start_experiments:
            exps.extend(experiments)
        for experiments in self.hot_start_experiments:
            exps.extend(experiments)
        return exps
