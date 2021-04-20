# -*- coding: utf-8 -*-
"""
Sanitizing wrapper of main algorithm
====================================

Performs checks and organizes required transformations of points.

"""
import textwrap
from typing import List, Dict, Tuple, Type, Union
from orion.core.worker.trial import Trial
import orion.core.utils.backward as backward
from orion.algo.base import BaseAlgorithm, infer_trial_id
from orion.core.worker.transformer import build_required_space
from orion.algo.space import Space, Categorical
from orion.core.utils.config import ExperimentInfo
from orion.core.utils.format_trials import trial_to_tuple
from orion.algo.random import Random
import inspect
import logging

log = logging.getLogger(__name__)


# pylint: disable=too-many-public-methods
class PrimaryAlgo(BaseAlgorithm):
    """Perform checks on points and transformations. Wrap the primary algorithm.

    1. Checks requirements on the parameter space from algorithms and create the
    appropriate transformations. Apply transformations before and after methods
    of the primary algorithm.
    2. Checks whether incoming and outcoming points are compliant with a space.

    """

    def __init__(
        self,
        space: Space,
        algorithm_config: Union[
            str, Type[BaseAlgorithm], Dict[Union[str, Type[BaseAlgorithm]], Dict]
        ],
    ):
        """
        Initialize the primary algorithm.

        Parameters
        ----------
        space : `orion.algo.space.Space`
           The original definition of a problem's parameters space.
        algorithm_config : dict
           Configuration for the algorithm.

        """
        self.algorithm = None
        _space_without_task_ids = Space(space.copy())
        _space_without_task_ids.pop("task_id", None)
        log.info(f"Creating PrimaryAlgo for space {space}, algorithm_config={algorithm_config}")

        # TODO: Figure out the number of potential tasks using the KB.
        # from warmstart.new_knowledge_base import KnowledgeBase
        # from orion.storage.base import get_storage
        # kb = KnowledgeBase(get_storage())
        # if kb.n_stored_experiments == 1:
        #     use_kb = False
        max_number_of_tasks = 5
        task_label_dimension = Categorical(
            "task_id", list(range(0, max_number_of_tasks)), default_value=0,
        )
        # BUG: This space might already contain the `task_id` key, for isntance when
        # loading from storage?
        # BUG: For some algos like TPE, it seems like we need to pass the space with
        # the task IDS to the constructor, while for the random algo we need to NOT pass
        # it like that, because there's some weird kind of recursion going on.
        # Add the task-id to the space so it is used when creating the algorithm inside
        # super().__init__()
        space_with_task_ids = Space(space.copy())
        assert "task_id" not in space
        space_with_task_ids.register(task_label_dimension)
        assert "task_id" not in _space_without_task_ids

        # --------------------------------------------------
        # TODO: Reworking super().__init__() 
        # super().__init__(space, algorithm=algorithm_config)

        self._trials_info = {}  # Stores Unique Trial -> Result
        self._warm_start_trials = {}  # Stores unique warm-star trials and their results
        # self._space = space_with_task_ids
        # self._param_names = list(kwargs.keys())

        # Create the algorithm:
        # - algorithm_config = {"tpe": {"seed" : 123, ...}}
        # - algorithm_config = {TPE: {"seed" : 123, ...}}
        # - algorithm_config = TPE
        # - algorithm_config = "tpe"
        algo_name: str = "random"
        algo_type: Type[BaseAlgorithm] = Random
        algo_kwargs: Dict = {}
        available_algos = {
            c.__name__.lower(): c for c in list(BaseAlgorithm.__subclasses__())
        }
        if isinstance(algorithm_config, dict):
            if len(algorithm_config) != 1:
                raise RuntimeError(f"Algorithm config dict needs to have only one key.")
            key = list(algorithm_config.keys())[0]
            if isinstance(key, str):
                algo_name = key
                if algo_name.lower() not in available_algos:
                    raise NotImplementedError(
                        f"Couldn't find an algorithm matching name {algo_name}! "
                        f"Available algorithms: {available_algos}"
                    )
                algo_type = available_algos[key.lower()]
                algo_kwargs = algorithm_config[key]
            else:
                if not (inspect.isclass(key) and issubclass(key, BaseAlgorithm)):
                    raise RuntimeError(
                        "Key of the dict needs to either be a str or a type of "
                        "Algorithm to use (a subclass of BaseAlgorithm)."
                    )
                algo_name = key.__name__.lower()
                algo_type = key
                algo_kwargs = algorithm_config[key]
        elif isinstance(algorithm_config, str):
            algo_name = algorithm_config
            if algo_name.lower() not in available_algos:
                raise NotImplementedError(
                    f"Couldn't find an algorithm matching name {algo_name}! "
                    f"Available algorithms: {available_algos}"
                )
            algo_type = available_algos[algo_name.lower()]
            algo_kwargs = {}
        elif inspect.isclass(algorithm_config) and issubclass(algorithm_config, BaseAlgorithm):
            algo_type = algorithm_config
            algo_name = algo_type.__name__.lower()
            algo_kwargs = {}
        else:
            raise RuntimeError(
                f"Algorithm config must either be a string (algo name), a type (algo "
                f"type), or a dict with a single key (either algo name or algo type) "
                f"mapping to the kwargs of the algorithm. Got {algorithm_config}."
            )

        # assert "space" not in algo_kwargs
        # self.algorithm = algo_type(space=space_with_task_ids, **algo_kwargs)
        # --------------------------------------------------------
        self._space = _space_without_task_ids
        requirements = backward.get_algo_requirements(algo_type)
        self.transformed_space = build_required_space(space_with_task_ids, **requirements)

        assert "space" not in algo_kwargs
        self.algorithm = algo_type(space=self.transformed_space, **algo_kwargs)

        self.current_task_id = 0
        assert "task_id" not in self._space
        assert "task_id" not in self.space
        assert "task_id" in self.transformed_space
        assert "task_id" in self.algorithm.space

    def seed_rng(self, seed):
        """Seed the state of the algorithm's random number generator."""
        self.algorithm.seed_rng(seed)

    @property
    def state_dict(self):
        """Return a state dict that can be used to reset the state of the algorithm."""
        return self.algorithm.state_dict

    def set_state(self, state_dict):
        """Reset the state of the algorithm based on the given state_dict

        :param state_dict: Dictionary representing state of an algorithm
        """
        self.algorithm.set_state(state_dict)

    def suggest(self, num=1):
        """Suggest a `num` of new sets of parameters.

        :param num: how many sets to be suggested.

        .. note:: New parameters must be compliant with the problem's domain
           `orion.algo.space.Space`.
        """
        points = self.algorithm.suggest(num)
        if points is None:
            return None

        # Check that the algorithm suggested valid points (with task ids)
        for point in points:
            if point not in self.transformed_space:
                raise ValueError(
                    textwrap.dedent(
                        f"""\
                        Point is not contained in space:
                        Point: {point}
                        Space: {self.transformed_space}
                        """
                    )
                )

        transformed_points = [self.transformed_space.reverse(point) for point in points]

        # Remove the task ids from these points:
        # NOTE: These predicted task ids aren't currently used.
        points_without_task_ids, _ = zip(
            *[(point[:-1], point[-1]) for point in transformed_points]
        )

        for point in points_without_task_ids:
            if point not in self.space:
                raise ValueError(
                    textwrap.dedent(
                        f"""\
                        Point is not contained in space:
                        Point: {point}
                        Space: {self.space}
                        """
                    )
                )
        return points_without_task_ids

    def observe(self, points, results):
        """Observe evaluation `results` corresponding to list of `points` in
        space.

        .. seealso:: `orion.algo.base.BaseAlgorithm.observe`
        """
        assert len(points) == len(results)
        tpoints = []
        # Add the task ids to the points, because we assume that the points passed to
        # `observe` are from the current task.
        points = [point + (self.current_task_id,) for point in points]

        for point in points:
            assert point in self.space
            tpoints.append(self.transformed_space.transform(point))
        self.algorithm.observe(tpoints, results)

    def warm_start(self, warm_start_trials: Dict[ExperimentInfo, List[Trial]]) -> None:
        # Being asked to warm-start the algorithm.
        log.debug(f"Warm starting the algo with trials {warm_start_trials}")
        try:
            self.algorithm.warm_start(warm_start_trials)
        except NotImplementedError:
            # Perform warm-starting using only the supported trials.
            log.info(
                "Will warm-start using contextual information, since the algo "
                "isn't warm-starteable."
            )
        else:
            # Algo was successfully warm-started, return.
            log.info("Algorithm was successfully warm-started, returning.")
            return

        compatible_trials: List[Trial] = []
        compatible_points: List[Tuple] = []

        for i, (experiment_info, trials) in enumerate(warm_start_trials.items()):
            # exp_space = experiment_info.space
            # from warmstart.utils.api_config import hparam_class_from_orion_space_dict
            # exp_hparam_class = hparam_class_from_orion_space_dict(exp_space)
            log.debug(f"Experiment {experiment_info.name} has {len(trials)} trials.")

            for trial in trials:
                try:
                    point = trial_to_tuple(trial=trial, space=self.space)
                    # Drop the point if it doesn't fit inside the current space.
                    # TODO: Do we want to 'translate' the point in this case?
                    if point not in self.space:
                        continue
                    # Add the task id to the point:
                    point = point + (i,)
                    compatible_trials.append(trial)
                    compatible_points.append(point)
                except ValueError as e:
                    log.error(f"Can't reuse trial {trial}: {e}")

        with self.warm_start_mode():
            # Only keep trials that are new.
            new_trials, new_points = zip(
                *[
                    (trial, point)
                    for trial, point in zip(compatible_trials, compatible_points)
                    if infer_trial_id(point) not in self._warm_start_trials
                ]
            )
            results = [trial.objective for trial in new_trials]
            log.info(f"About to observe {len(new_points)} warm-starting points!")
            self.algorithm.observe(new_points, results)

    @property
    def unwrapped(self) -> "BaseAlgorithm":
        return self.algorithm.unwrapped

    @property
    def is_done(self):
        """Return True, if an algorithm holds that there can be no further improvement."""
        return self.algorithm.is_done

    def score(self, point):
        """Allow algorithm to evaluate `point` based on a prediction about
        this parameter set's performance. Return a subjective measure of expected
        performance.

        By default, return the same score any parameter (no preference).
        """
        assert point in self.space
        return self.algorithm.score(self.transformed_space.transform(point))

    def judge(self, point, measurements):
        """Inform an algorithm about online `measurements` of a running trial.

        The algorithm can return a dictionary of data which will be provided
        as a response to the running environment. Default is None response.

        """
        assert point in self._space
        return self.algorithm.judge(
            self.transformed_space.transform(point), measurements
        )

    @property
    def should_suspend(self):
        """Allow algorithm to decide whether a particular running trial is still
        worth to complete its evaluation, based on information provided by the
        `judge` method.

        """
        return self.algorithm.should_suspend

    @property
    def configuration(self):
        """Return tunable elements of this algorithm in a dictionary form
        appropriate for saving.
        """
        return self.algorithm.configuration

    @property
    def space(self):
        """Domain of problem associated with this algorithm's instance.

        .. note:: Redefining property here without setter, denies base class' setter.
        """
        return self._space
