import copy
import numpy as np
from typing import List, Dict

import orion.core.utils.backward as backward
from orion.algo.base import BaseAlgorithm
from orion.core.worker.transformer import build_required_space
from orion.core.worker.trial import Trial

from orion.core.worker.primary_algo import PrimaryAlgo, BaseAlgorithm
from warmstart.new_knowledge_base import KnowledgeBase
from orion.algo.space import Space, Dimension, Integer
from orion.core.worker.transformer import TransformedSpace

from orion.storage.base import Storage, get_storage


# pylint: disable=too-many-public-methods
class MultiTaskAlgo(BaseAlgorithm):
    """Wrapper that makes the algo "multi-task" by concatenating the task ids to the inputs.
    """

    def __init__(self, space: Space, algorithm_config: Dict):
        print(f"Creating a MultiTaskAlgo wrapper around space {space}, algo config: {algorithm_config}")
        super().__init__(space)
        self.transformed_space = copy.deepcopy(space)
        todo = 1234  # Need to determine the max number of tasks (using the KB) 
        task_label_dimension = Integer(name="task_id", prior="uniform", low=0, high=todo)
        # BUG: How come the space can sometimes already have the task id?
        assert "task_id" not in self.space
        assert "task_id" not in self.transformed_space
        self.transformed_space.register(task_label_dimension)

        # Pass the 'transformed space' to the PrimaryAlgo
        self.algorithm = PrimaryAlgo(self.transformed_space, algorithm_config)  # <----- Crash!
        self.warm_started: bool = False
        self.current_task_id: int = 0

        # self.knowledge_base: KnowledgeBase = KnowledgeBase()
        # try:
        #     self.knowledge_base.add_storage(get_storage())
        # except:
        #     pass

        # if self.knowledge_base.n_stored_experiments:
        #     # TODO: Fetch the current experiment, pass it to the knowledge base, and
        #     # obtain the reusable trials, and whenever possible use them to warm-start
        #     # the algorithm?
        #     from orion.client import get_experiment
        #     assert False, self.knowledge_base.experiment_infos
        #     reusable_trials = self.knowledge_base.get_reusable_trials(experiment)
        #     # Maybe do the warm-starting here?
        #     self.algorithm.warm_start(reusable_trials)

    @property
    def unwrapped(self) -> "BaseAlgorithm":
        return self.algorithm.unwrapped

    def warm_start(self, warm_start_trials: List[Trial]) -> None:
        print(f"Warm starting the algo with trials {warm_start_trials}")
        assert False, "hey?"
        try:
            self.algorithm.warm_start(warm_start_trials)
        except NotImplementedError:
            # Perform warm-starting using only the supported trials.
            print(f"Will warm-start using contextual information, since the algo "
                  f"isn't warm-starteable.")
        else:
            # Algo was successfully warm-started, return.
            print(f"Algorithm was successfully warm-started, returning.")
            return

        compatible_trials: List[Trial] = []
        from orion.core.utils.format_trials import trial_to_tuple

        for trial in warm_start_trials:
            print(trial)
            try:
                point = trial_to_tuple(trial=trial, space=self.space)
                if point in self.space:
                    compatible_trials.append(trial)
            except ValueError:
                print(f"Can't reuse point {point}")

        with self.warm_start_mode():
            # TODO: Only keep points that fit within our space.
            # Only keep trials we haven't already warm-started with? Or leave that
            # responsability to the `observe` method?
            # TODO: Convert trials to points?
            from orion.core.utils.format_trials import trial_to_tuple
            points = [
                trial_to_tuple(trial=trial, space=self.space) for trial in compatible_trials
            ]
            new_trials = [
                trial for trial, point in zip(compatible_trials, points)
                if infer_trial_id(point) not in self._warm_start_trials
            ]
            results = [
                trial.objective for trial in new_trials
            ]
            print(f"About to observe {len(points)} warm-starting points.")
            self.observe(points, results)

    def suggest(self, num=1):
        """Suggest a `num` of new sets of parameters.

        :param num: how many sets to be suggested.

        .. note:: New parameters must be compliant with the problem's domain
           `orion.algo.space.Space`.
        """
        points = self.algorithm.suggest(num=num)
        # Remove the 'task-id' dimension.
        return [
            point[:-1] for point in points
        ]

    def observe(self, points, results):
        """Observe evaluation `results` corresponding to list of `points` in
        space.

        .. seealso:: `orion.algo.base.BaseAlgorithm.observe`
        """
        assert len(points) == len(results)
        transformed_points = []

        for point in points:
            # NOTE: Assumes that points served to `observe` don't have the task id
            assert point in self.space
            transformed_point = point + (self.current_task_id,)
            transformed_points.append(transformed_point)
        self.algorithm.observe(transformed_points, results)
