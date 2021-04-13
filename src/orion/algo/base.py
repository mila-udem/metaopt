# -*- coding: utf-8 -*-
"""
Base Search Algorithm
=====================

Formulation of a general search algorithm with respect to some objective.
Algorithm implementations must inherit from `orion.algo.base.OptimizationAlgorithm`.

"""
import hashlib
import logging
from abc import ABCMeta, abstractmethod
from contextlib import contextmanager
from orion.core.utils import Factory
from orion.core.worker.trial import Trial
from typing import List

log = logging.getLogger(__name__)


def infer_trial_id(point):
    """Compute a hashing of a point"""
    return hashlib.md5(str(list(point)).encode("utf-8")).hexdigest()


# pylint: disable=too-many-public-methods
class BaseAlgorithm(object, metaclass=ABCMeta):
    """Base class describing what an algorithm can do.

    Parameters
    ----------
    space : `orion.algo.space.Space`
       Definition of a problem's parameter space.
    kwargs : dict
       Tunable elements of a particular algorithm, a dictionary from
       hyperparameter names to values.

    Notes
    -----
    We are using the No Free Lunch theorem's [1]_[3]_ formulation of an
    `BaseAlgorithm`.
    We treat it as a part of a procedure which in each iteration suggests a
    sample of the parameter space of the problem as a candidate solution and observes
    the results of its evaluation.

    **Developer Note**: Each algorithm's complete specification, i.e.  implementation of its methods
    and parameters of its own, lies in a separate concrete algorithm class, which must be an
    **immediate** subclass of `BaseAlgorithm`. [The reason for this is current implementation of
    `orion.core.utils.Factory` metaclass which uses `BaseAlgorithm.__subclasses__()`.] Second, one
    must declare an algorithm's own parameters (tunable elements which could be set by
    configuration). This is done by passing them to `BaseAlgorithm.__init__()` by calling Python's
    super with a `Space` object as a positional argument plus algorithm's own parameters as keyword
    arguments. The keys of the keyword arguments passed to `BaseAlgorithm.__init__()` are interpreted
    as the algorithm's parameter names. So for example, a subclass could be as simple as this
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

    requires_type = None
    requires_shape = None
    requires_dist = None

    def __init__(self, space, **kwargs):
        log.debug(
            "Creating Algorithm object of %s type with parameters:\n%s",
            type(self).__name__,
            kwargs,
        )
        self._trials_info = {}  # Stores Unique Trial -> Result
        self._warm_start_trials = {} # Stores unique warm-star trials and their results
        self._space = space
        self._param_names = list(kwargs.keys())
        # Instantiate tunable parameters of an algorithm
        for varname, param in kwargs.items():
            # Check if tunable element is another algorithm
            if isinstance(param, dict) and len(param) == 1:
                subalgo_type = list(param)[0]
                subalgo_kwargs = param[subalgo_type]
                if isinstance(subalgo_kwargs, dict):
                    param = OptimizationAlgorithm(subalgo_type, space, **subalgo_kwargs)
            elif (
                isinstance(param, str)
                and param.lower() in OptimizationAlgorithm.typenames
            ):
                # pylint: disable=too-many-function-args
                param = OptimizationAlgorithm(param, space)
            elif varname == "seed":
                self.seed_rng(param)

            setattr(self, varname, param)

    def seed_rng(self, seed):
        """Seed the state of the random number generator.

        :param seed: Integer seed for the random number generator.

        .. note:: This methods does nothing if the algorithm is deterministic.
        """
        pass

    @property
    def state_dict(self):
        """Return a state dict that can be used to reset the state of the algorithm."""
        return {"_trials_info": self._trials_info}

    def set_state(self, state_dict):
        """Reset the state of the algorithm based on the given state_dict

        :param state_dict: Dictionary representing state of an algorithm
        """
        self._trials_info = state_dict.get("_trials_info")

    @abstractmethod
    def suggest(self, num=1):
        """Suggest a `num` of new sets of parameters.

        Parameters
        ----------
        num: int, optional
            Number of points to suggest. Defaults to 1.

        Returns
        -------
        list of points or None
            A list of lists representing points suggested by the algorithm. The algorithm may opt
            out if it cannot make a good suggestion at the moment (it may be waiting for other
            trials to complete), in which case it will return None.

        Notes
        -----
        New parameters must be compliant with the problem's domain `orion.algo.space.Space`.

        """
        pass

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
        for point, result in zip(points, results):
            point_id = infer_trial_id(point)

            if point_id not in self._trials_info:
                self._trials_info[point_id] = (point, result)

    def warm_start(self, warm_start_trials: List[Trial]) -> None:
        
        compatible_trials: List[Trial] = []
        from orion.core.utils.format_trials import trial_to_tuple

        for trial in warm_start_trials:
            print(trial)
            try:
                point = trial_to_tuple(trial=trial, space=self.space)
                if point in self.space:
                    compatible_trials.append(trial)
            except ValueError:
                log.debug(f"Can't reuse point {point}")

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
            log.debug(f"About to observe {len(points)} warm-starting points.")
            self.observe(points, results)

    @property
    def unwrapped(self) -> "BaseAlgorithm":
        return self

    @contextmanager
    def warm_start_mode(self):
        """ TODO: Context manager that prevents the algo from modifying its state
        while observing points from warm-starting.
        """
        trials_info_copy = self._trials_info.copy()
        yield
        # Store any new entries into the 'warm start' dict.
        self._warm_start_trials.update({
            k: v for k, v in self._trials_info.items() if k not in trials_info_copy
        })
        # TODO: Could also maybe restore the "number of used trials"?
        self._trials_info = trials_info_copy

    @property
    def is_done(self):
        """Return True, if an algorithm holds that there can be no further improvement.
        By default, the cardinality of the specified search space will be used to check
        if all possible sets of parameters has been tried.
        """
        if len(self._trials_info) >= self.space.cardinality:
            return True

        if len(self._trials_info) >= getattr(self, "max_trials", float("inf")):
            return True

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
            if attrname.startswith("_"):  # Do not log _space or others in conf
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


# pylint: disable=too-few-public-methods,abstract-method
class OptimizationAlgorithm(BaseAlgorithm, metaclass=Factory):
    """Class used to inject dependency on an algorithm implementation."""

    pass
