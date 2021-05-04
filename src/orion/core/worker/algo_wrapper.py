""" ABC for a wrapper around an Algorithm. """
from abc import ABC
from typing import Optional, List, Dict, Tuple, Union, Any
import numpy as np
from orion.algo.base import BaseAlgorithm
from orion.algo.space import Space


Point = Union[Tuple, List[float], np.ndarray]


class AlgoWrapper(BaseAlgorithm, ABC):
    """Base class for a Wrapper around an algorithm.
    """
    def __init__(self, space: Space, **kwargs):
        self.algorithm: BaseAlgorithm
        super().__init__(space, **kwargs)

    def seed_rng(self, seed: Optional[int]) -> None:
        """Seed the state of the algorithm's random number generator."""
        self.algorithm.seed_rng(seed)

    @property
    def state_dict(self) -> Dict:
        """Return a state dict that can be used to reset the state of the algorithm."""
        return self.algorithm.state_dict

    def set_state(self, state_dict: Dict) -> None:
        """Reset the state of the algorithm based on the given state_dict

        :param state_dict: Dictionary representing state of an algorithm
        """
        self.algorithm.set_state(state_dict)

    def suggest(self, num: int = 1) -> List[Point]:
        """Suggest a `num` of new sets of parameters.

        :param num: how many sets to be suggested.

        .. note:: New parameters must be compliant with the problem's domain
           `orion.algo.space.Space`.
        """
        return self.algorithm.suggest(num)

    def observe(self, points: List[Union[List, Tuple, np.ndarray]], results: List[Dict]):
        """Observe evaluation `results` corresponding to list of `points` in
        space.

        .. seealso:: `orion.algo.base.BaseAlgorithm.observe`
        """
        self.algorithm.observe(points, results)

    @property
    def is_done(self) -> bool:
        """Return True, if an algorithm holds that there can be no further improvement."""
        return self.algorithm.is_done

    def score(self, point: Point) -> float:
        """Allow algorithm to evaluate `point` based on a prediction about
        this parameter set's performance. Return a subjective measure of expected
        performance.

        By default, return the same score any parameter (no preference).
        """
        return self.algorithm.score(point)

    def judge(self, point: Point, measurements: Any) -> Optional[Dict]:
        """Inform an algorithm about online `measurements` of a running trial.

        The algorithm can return a dictionary of data which will be provided
        as a response to the running environment. Default is None response.

        """
        return self.algorithm.judge(point, measurements)

    @property
    def should_suspend(self) -> bool:
        """Allow algorithm to decide whether a particular running trial is still
        worth to complete its evaluation, based on information provided by the
        `judge` method.

        """
        return self.algorithm.should_suspend

    @property
    def configuration(self) -> Dict:
        """Return tunable elements of this algorithm in a dictionary form
        appropriate for saving.
        """
        return self.algorithm.configuration

    @property
    def unwrapped(self) -> BaseAlgorithm:
        """Returns the unwrapped algorithm (the root).

        Returns
        -------
        BaseAlgorithm
            The unwrapped `BaseAlgorithm` instance.
        """
        return self.algorithm
