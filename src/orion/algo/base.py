# -*- coding: utf-8 -*-
"""
:mod:`orion.algo.base` -- What is a search algorithm, optimizer of a process
==============================================================================

.. module:: base
   :platform: Unix
   :synopsis: Formulation of a general search algorithm with respect to some
      objective.

"""
from abc import (ABCMeta, abstractmethod)
import logging

from orion.core.utils import (Concept, Wrapper)
from orion.core.worker.transformer import build_required_space


log = logging.getLogger(__name__)


class BaseAlgorithm(Concept, metaclass=ABCMeta):
    """Base class describing what an algorithm can do.

    Notes
    -----
    We are using the No Free Lunch theorem's [1]_[3]_ formulation of an
    `BaseAlgorithm`.
    We treat it as a part of a procedure which in each iteration suggests a
    sample of the parameter space of the problem as a candidate solution and observes
    the results of its evaluation.

    **Developer Note**: Each algorithm's complete specification, i.e.
    implementation of its methods and parameters of its own, lies in a
    separate concrete algorithm class, which must be an **immediate** subclass of
    `BaseAlgorithm`. [The reason for this is current implementation of `Factory`
    metaclass which uses `BaseAlgorithm.__subclasses__()`.] Second, one must
    declare an algorithm's own parameters (tunable elements which could be set
    by configuration). This is done by passing them to `BaseAlgorithm.__init__`
    by calling Python's super with a `Space` object as a positional argument plus
    algorithm's own parameters as keyword arguments. The keys of the keyword
    arguments passed to `BaseAlgorithm.__init__` are interpreted as the algorithm's
    parameter names. So for example, a subclass could be as simple as this
    (regarding the logistics, not an actual algorithm's implementation):

    Examples
    --------
    .. code-block:: python
       :linenos:
       :emphasize-lines: 7

       from orion.algo.base import BaseAlgorithm
       from orion.algo.space import (Integer, Space)

       class MySimpleAlgo(BaseAlgorithm):

           def __init__(self, space, multiplier=1, another_param="a string param"):
               super().__init__(space, multiplier=multiplier, another_param=another_param)

           def suggest(self, num=1):
               print(self.another_param)
               return list(map(lambda x: tuple(map(lambda y: self.multiplier * y, x)),
                               self.space.sample(num)))

           def observe(self, points, results):
               pass

       dim = Integer('named_param', 'norm', 3, 2, shape=(2, 3))
       s = Space()
       s.register(dim)

       algo = MySimpleAlgo(s, 2, "I am just sampling!")
       algo.suggest()

    References
    ----------
    .. [1] D. H. Wolpert and W. G. Macready, “No Free Lunch Theorems for Optimization,”
       IEEE Transactions on Evolutionary Computation, vol. 1, no. 1, pp. 67–82, Apr. 1997.
    .. [2] W. G. Macready and D. H. Wolpert, “What Makes An Optimization Problem Hard?,”
       Complexity, vol. 1, no. 5, pp. 40–46, 1996.
    .. [3] D. H. Wolpert and W. G. Macready, “No Free Lunch Theorems for Search,”
       Technical Report SFI-TR-95-02-010, Santa Fe Institute, 1995.

    """

    requires = []
    name = "Algorithm"
    implementation_module = "orion.algo"

    def __init__(self, space, seed=None, **kwargs):
        """Declare problem's parameter space and set up algo's hyperparameters.

        Parameters
        ----------
        space : `orion.algo.space.Space`
           Definition of a problem's parameter space.
        kwargs : dict
           Tunable elements of a particular algorithm, a dictionary from
           hyperparameter names to values.

        """
        self._space = space
        self._param_names = ['seed'] + list(kwargs.keys())

        super(BaseAlgorithm, self).__init__(space, seed=seed, **kwargs)

        self.seed_rng(seed)

    def seed_rng(self, seed):
        """Seed the state of the random number generator.

        :param seed: Integer seed for the random number generator.

        .. note:: This methods does nothing if the algorithm is deterministic.
        """
        pass

    @abstractmethod
    def suggest(self, num=1):
        """Suggest a `num` of new sets of parameters.

        :param num: how many sets to be suggested.

        .. note:: New parameters must be compliant with the problem's domain
           `orion.algo.space.Space`.
        """
        pass

    @abstractmethod
    def observe(self, points, results):
        """Observe the `results` of the evaluation of the `points` in the
        process defined in user's script.

        Parameters
        ----------
        points : list of tuples of array-likes
           Points from a `orion.algo.space.Space`.
           Evaluated problem parameters by a consumer.
        results : list of dicts
           Contains the result of an evaluation; partial information about the
           black-box function at each point in `params`.

        Result
        ------
        objective : numeric
           Evaluation of this problem's objective function.
        gradient : 1D array-like, optional
           Contains values of the derivatives of the `objective` function
           with respect to `params`.
        constraint : list of numeric, optional
           List of constraints expression evaluation which must be greater
           or equal to zero by the problem's definition.

        """
        pass

    @property
    def is_done(self):
        """Return True, if an algorithm holds that there can be no further improvement."""
        return False

    def score(self, point):  # pylint:disable=no-self-use,unused-argument
        """Allow algorithm to evaluate `point` based on a prediction about
        this parameter set's performance.

        By default, return the same score any parameter (no preference).

        :returns: A subjective measure of expected perfomance.
        :rtype: float

        """
        return 0

    def judge(self, point, measurements):  # pylint:disable=no-self-use,unused-argument
        """Inform an algorithm about online `measurements` of a running trial.

        :param point: A tuple which specifies the values of the (hyper)parameters
           used to execute user's script with.

        This method is to be used as a callback in a client-server communication
        between user's script and a orion's worker using a `BaseAlgorithm`.
        Data returned from this method must be serializable and will be used as
        a response to the running environment. Default response is None.

        .. note:: Calling algorithm to `judge` a `point` based on its online
           `measurements` will effectively change a state in the algorithm (like
           a reinforcement learning agent's hidden state or an automatic early
           stopping mechanism's regression), which it may change the value of
           the property `should_suspend`.

        :returns: None or a serializable dictionary containing named data

        """
        return None

    @property
    def should_suspend(self):
        """Allow algorithm to decide whether a particular running trial is still
        worth to complete its evaluation, based on information provided by the
        `judge` method.

        """
        return False

    @property
    def configuration(self):
        """Return tunable elements of this algorithm in a dictionary form
        appropriate for saving.

        """
        dict_form = dict()
        for attrname in self._param_names:
            if attrname.startswith('_'):  # Do not log _space or others in conf
                continue
            attr = getattr(self, attrname)
            if isinstance(attr, BaseAlgorithm):
                attr = attr.configuration
            dict_form[attrname] = attr

        return {self.__class__.__name__.lower(): dict_form}

    @property
    def space(self):
        """Domain of problem associated with this algorithm's instance."""
        return self._space

    @space.setter
    def space(self, space_):
        """Propagate changes in defined space to possibly nested algorithms."""
        self._space = space_
        for attr in self.__dict__.values():
            if isinstance(attr, BaseAlgorithm):
                attr.space = space_


class PrimaryAlgo(Wrapper):
    """Perform checks on points and transformations. Wrap the primary algorithm.

    1. Checks requirements on the parameter space from algorithms and create the
    appropriate transformations. Apply transformations before and after methods
    of the primary algorithm.
    2. Checks whether incoming and outcoming points are compliant with a space.

    """

    implementation_module = "orion.algo"

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
        self._space = space
        super(PrimaryAlgo, self).__init__(space, instance=algorithm_config)
        requirements = self.instance.requires
        self.transformed_space = build_required_space(requirements, self.space)
        self.instance.space = self.transformed_space

    @property
    def wraps(self):
        """Return the type of object this wrapper wraps"""
        return BaseAlgorithm

    def seed_rng(self, seed):
        """Seed the state of the algorithm's random number generator."""
        self.instance.seed_rng(seed)

    def suggest(self, num=1):
        """Suggest a `num` of new sets of parameters.

        :param num: how many sets to be suggested.

        .. note:: New parameters must be compliant with the problem's domain
           `orion.algo.space.Space`.

        """
        points = self.instance.suggest(num)
        for point in points:
            assert point in self.transformed_space
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
        self.instance.observe(tpoints, results)

    @property
    def is_done(self):
        """Return True, if an algorithm holds that there can be no further improvement."""
        return self.instance.is_done

    def score(self, point):
        """Allow algorithm to evaluate `point` based on a prediction about
        this parameter set's performance. Return a subjective measure of expected
        performance.
        By default, return the same score any parameter (no preference).
        """
        assert point in self.space
        return self.instance.score(self.transformed_space.transform(point))

    def judge(self, point, measurements):
        """Inform an algorithm about online `measurements` of a running trial.
        The algorithm can return a dictionary of data which will be provided
        as a response to the running environment. Default is None response.
        """
        assert point in self._space
        return self.instance.judge(self.transformed_space.transform(point), measurements)

    @property
    def should_suspend(self):
        """Allow algorithm to decide whether a particular running trial is still
        worth to complete its evaluation, based on information provided by the
        `judge` method.
        """
        return self.instance.should_suspend

    @property
    def configuration(self):
        """Return tunable elements of this algorithm in a dictionary form
        appropriate for saving.
        """
        return self.instance.configuration

    @property
    def space(self):
        """Domain of problem associated with this algorithm's instance.
        .. note:: Redefining property here without setter, denies base class' setter.
        """
        return self._space
