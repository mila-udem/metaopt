# -*- coding: utf-8 -*-
"""
Asynchronous Successive Halving Algorithm
=========================================
"""
from collections import defaultdict
import copy
import hashlib
import logging

import numpy

from orion.algo.base import BaseAlgorithm
from orion.algo.hyperband import Hyperband, HyperbandBracket, display_budgets
from orion.algo.space import Fidelity

logger = logging.getLogger(__name__)


REGISTRATION_ERROR = """
Bad fidelity level {fidelity}. Should be in {budgets}.
Params: {params}
"""

SPACE_ERROR = """
ASHA can only be used if there is one fidelity dimension.
For more information on the configuration and usage of ASHA, see
https://orion.readthedocs.io/en/develop/user/algorithms.html#asha
"""

BUDGET_ERROR = """
Cannot build budgets below max_resources;
(max: {}) - (min: {}) > (num_rungs: {})
"""


def compute_budgets(
    min_resources, max_resources, reduction_factor, num_rungs, num_brackets
):
    """Compute the budgets used for ASHA"""

    # Tracks state for new trial add
    if num_brackets > num_rungs:
        logger.warning(
            "The input num_brackets %i is larger than the number of rungs %i, "
            "set num_brackets as %i",
            num_brackets,
            num_rungs,
            num_rungs,
        )
        num_brackets = num_rungs

    budgets = numpy.logspace(
        numpy.log(min_resources) / numpy.log(reduction_factor),
        numpy.log(max_resources) / numpy.log(reduction_factor),
        num_rungs,
        base=reduction_factor,
    )
    budgets = (budgets + 0.5).astype(int)

    for i in range(num_rungs - 1):
        if budgets[i] >= budgets[i + 1]:
            budgets[i + 1] = budgets[i] + 1

    if budgets[-1] > max_resources:
        raise ValueError(BUDGET_ERROR.format(min_resources, max_resources, num_rungs))

    budgets = [budgets[bracket_index:] for bracket_index in range(num_brackets)]

    ressources = budgets
    budgets = []
    budgets_tab = defaultdict(list)
    for bracket_ressources in ressources:
        bracket_budgets = []
        for i, min_ressources in enumerate(bracket_ressources[::-1]):
            budget = (reduction_factor ** i, min_ressources)
            bracket_budgets.append(budget)
            budgets_tab[len(bracket_ressources) - i - 1].append(budget)
        budgets.append(bracket_budgets[::-1])

    display_budgets(budgets_tab, max_resources, reduction_factor)

    return list(budgets)


class ASHA(Hyperband):
    """Asynchronous Successive Halving Algorithm

    `A simple and robust hyperparameter tuning algorithm with solid theoretical underpinnings
    that exploits parallelism and aggressive early-stopping.`

    For more information on the algorithm, see original paper at https://arxiv.org/abs/1810.05934.

    Li, Liam, et al. "Massively parallel hyperparameter tuning."
    arXiv preprint arXiv:1810.05934 (2018)

    Parameters
    ----------
    space: `orion.algo.space.Space`
        Optimisation space with priors for each dimension.
    seed: None, int or sequence of int
        Seed for the random number generator used to sample new trials.
        Default: ``None``
    num_rungs: int, optional
        Number of rungs for the largest bracket. If not defined, it will be equal to (base + 1) of
        the fidelity dimension. In the original paper,
        num_rungs == log(fidelity.high/fidelity.low) / log(fidelity.base) + 1.
        Default: log(fidelity.high/fidelity.low) / log(fidelity.base) + 1
    num_brackets: int
        Using a grace period that is too small may bias ASHA too strongly towards
        fast converging trials that do not lead to best results at convergence (stagglers). To
        overcome this, you can increase the number of brackets, which increases the amount of
        resource required for optimisation but decreases the bias towards stragglers.
        Default: 1
    repetitions: int
        Number of execution of ASHA. Default is numpy.inf which means to
        run ASHA until no new trials can be suggested.

    """

    def __init__(
        self,
        space,
        seed=None,
        num_rungs=None,
        num_brackets=1,
        repetitions=numpy.inf,
    ):
        super(ASHA, self).__init__(space, seed=seed, repetitions=repetitions)

        self.num_rungs = num_rungs
        self.num_brackets = num_brackets

        self._param_names += ["num_rungs", "num_brackets"]

        if self.reduction_factor < 2:
            raise AttributeError("Reduction factor for ASHA needs to be at least 2.")

        if num_rungs is None:
            num_rungs = int(
                numpy.log(self.max_resources / self.min_resources)
                / numpy.log(self.reduction_factor)
                + 1
            )

        self.num_rungs = num_rungs

        self.budgets = compute_budgets(
            self.min_resources,
            self.max_resources,
            self.reduction_factor,
            self.num_rungs,
            self.num_brackets,
        )

        self.brackets = self.create_brackets()

    def compute_bracket_idx(self, num):
        def assign_resources(n, remainings, totals):
            if n == 0 or remainings.sum() == 0:
                return remainings

            ratios = remainings / totals
            fractions = numpy.nan_to_num(ratios / ratios.sum(), 0)
            index = numpy.argmax(fractions)
            remainings[index] -= 1
            n -= 1
            if n > 0:
                return assign_resources(n, remainings, totals)

            return remainings

        remainings = numpy.array([bracket.remainings for bracket in self.brackets])
        totals = numpy.array(
            [bracket.rungs[0]["n_trials"] for bracket in self.brackets]
        )
        remainings_after = copy.deepcopy(remainings)
        assign_resources(num, remainings_after, totals)
        allocations = remainings - remainings_after

        return allocations

    def sample(self, num):
        samples = []
        bracket_nums = self.compute_bracket_idx(num)
        for idx, bracket_num in enumerate(bracket_nums):
            bracket = self.brackets[idx]
            bracket_samples = self.sample_from_bracket(bracket, bracket_num)
            self.register_samples(bracket, bracket_samples)
            samples += bracket_samples

        return samples

    def suggest(self, num=None):
        if num is None:
            num = 1

        return super(ASHA, self).suggest(num)

    def create_bracket(self, i, budgets, iteration):
        return ASHABracket(self, budgets, iteration)


class ASHABracket(HyperbandBracket):
    """Bracket of rungs for the algorithm ASHA.

    Parameters
    ----------
    asha: `ASHA` algorithm
        The asha algorithm object which this bracket will be part of.
    budgets: list of tuple
        Each tuple gives the (n_trials, resource_budget) for the respective rung.
    repetition_id: int
        The id of hyperband execution this bracket belongs to

    """

    def sample(self, num):
        """Sample a new trial with lowest fidelity"""
        should_have_n_trials = self.rungs[0]["n_trials"]
        return self.hyperband.sample_for_bracket(num, self)

    def get_candidate(self, rung_id):
        """Get a candidate for promotion"""
        rung = self.rungs[rung_id]["results"]
        next_rung = self.rungs[rung_id + 1]["results"]

        rung = list(
            sorted(
                (objective, point)
                for objective, point in rung.values()
                if objective is not None
            )
        )
        k = len(rung) // self.hyperband.reduction_factor
        k = min(k, len(rung))

        for i in range(k):
            point = rung[i][1]
            _id = self.hyperband.get_id(point, ignore_fidelity=True)
            if _id not in next_rung:
                return point

        return None

    @property
    def is_filled(self):
        """ASHA's first rung can always sample new points"""
        return False

    def is_ready(self, rung_id=None):
        """ASHA's always ready for promotions"""
        return True

    def promote(self, num):
        """Promote the first candidate that is found and return it

        The rungs are iterated over in reversed order, so that high rungs
        are prioritised for promotions. When a candidate is promoted, the loop is broken and
        the method returns the promoted point.

        .. note ::

            All trials are part of the rungs, for any state. Only completed trials
            are eligible for promotion, i.e., only completed trials can be part of top-k.
            Lookup for promotion in rung l + 1 contains trials of any status.

        """
        if num < 1 or self.is_done:
            return []

        for rung_id in range(len(self.rungs) - 2, -1, -1):
            candidate = self.get_candidate(rung_id)
            if candidate:

                # pylint: disable=logging-format-interpolation
                logger.debug(
                    "Promoting {point} from rung {past_rung} with fidelity {past_fidelity} to "
                    "rung {new_rung} with fidelity {new_fidelity}".format(
                        point=candidate,
                        past_rung=rung_id,
                        past_fidelity=candidate[self.hyperband.fidelity_index],
                        new_rung=rung_id + 1,
                        new_fidelity=self.rungs[rung_id + 1]["resources"],
                    )
                )

                candidate = list(copy.deepcopy(candidate))
                candidate[self.hyperband.fidelity_index] = self.rungs[rung_id + 1][
                    "resources"
                ]

                return [tuple(candidate)]

        return []
