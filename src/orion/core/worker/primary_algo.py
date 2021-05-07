# -*- coding: utf-8 -*-
"""
Sanitizing wrapper of main algorithm
====================================

Performs checks and organizes required transformations of points.

"""
import orion.core.utils.backward as backward
from orion.algo.base import BaseAlgorithm
from orion.core.worker.transformer import build_required_space


# pylint: disable=too-many-public-methods
class PrimaryAlgo(BaseAlgorithm):
    """Perform checks on points and transformations. Wrap the primary algorithm.

    1. Checks requirements on the parameter space from algorithms and create the
    appropriate transformations. Apply transformations before and after methods
    of the primary algorithm.
    2. Checks whether incoming and outcoming points are compliant with a space.

    """

    def __init__(self, space, algorithm_config):
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
        super(PrimaryAlgo, self).__init__(space, algorithm=algorithm_config)
        requirements = backward.get_algo_requirements(self.algorithm)
        self.transformed_space = build_required_space(self.space, **requirements)
        self.algorithm.space = self.transformed_space

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

    def suggest(self, num):
        """Suggest a `num` of new sets of parameters.

        :param num: how many sets to be suggested.

        .. note:: New parameters must be compliant with the problem's domain
           `orion.algo.space.Space`.
        """
        points = self.algorithm.suggest(num)

        if points is None:
            return None

        for point in points:
            if point not in self.transformed_space:
                raise ValueError(
                    """
Point is not contained in space:
Point: {}
Space: {}""".format(
                        point, self.transformed_space
                    )
                )

        rpoints = []
        for point in points:
            rpoints.append(self.transformed_space.reverse(point))

        return rpoints

    def observe(self, points, results):
        """Observe evaluation `results` corresponding to list of `points` in
        space.

        .. seealso:: `orion.algo.base.BaseAlgorithm.observe`
        """
        assert len(points) == len(results)
        tpoints = []
        for point in points:
            assert point in self.space
            tpoints.append(self.transformed_space.transform(point))
        self.algorithm.observe(tpoints, results)

    def has_suggested(self, point):
        """Whether the algorithm has suggested a given point.

        .. seealso:: `orion.algo.base.BaseAlgorithm.has_suggested`

        """
        return self.algorithm.has_suggested(self.transformed_space.transform(point))

    def has_observed(self, point):
        """Whether the algorithm has observed a given point.

        .. seealso:: `orion.algo.base.BaseAlgorithm.has_observed`

        """
        return self.algorithm.has_observed(self.transformed_space.transform(point))

    @property
    def n_suggested(self):
        """Number of trials suggested by the algorithm"""
        return self.algorithm.n_suggested

    @property
    def n_observed(self):
        """Number of completed trials observed by the algorithm"""
        return self.algorithm.n_observed

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
