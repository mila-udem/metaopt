# -*- coding: utf-8 -*-
"""
Sanitizing wrapper of main algorithm
====================================

Performs checks and organizes required transformations of points.

"""
import textwrap
import orion.core.utils.backward as backward
from orion.core.worker.transformer import build_required_space
from .algo_wrapper import AlgoWrapper


class PrimaryAlgo(AlgoWrapper):
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
        super().__init__(space, algorithm=algorithm_config)
        requirements = backward.get_algo_requirements(self.algorithm)
        self.transformed_space = build_required_space(self.space, **requirements)
        self.algorithm.space = self.transformed_space

    def suggest(self, num=1):
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
                    textwrap.dedent(
                        f"""\
                        Point is not contained in space:
                        Point: {point}
                        Space: {self.transformed_space}
                        """
                    )
                )

        return [self.transformed_space.reverse(point) for point in points]

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
    def space(self):
        """Domain of problem associated with this algorithm's instance.

        .. note:: Redefining property here without setter, denies base class' setter.
        """
        return self._space
