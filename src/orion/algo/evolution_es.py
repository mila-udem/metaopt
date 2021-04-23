# -*- coding: utf-8 -*-
"""
The Evolved Transformer and large-scale evolution of image classifiers
======================================================================

Implement evolution to exploit configurations with fixed resource efficiently

"""
import copy
import importlib
import logging

import numpy as np

from orion.algo.hyperband import HyperbandBracket, Hyperband

logger = logging.getLogger(__name__)

REGISTRATION_ERROR = """
Bad fidelity level {fidelity}. Should be in {budgets}.
Params: {params}
"""

SPACE_ERROR = """
EvolutionES cannot be used if space does not contain a fidelity dimension.
"""

BUDGET_ERROR = """
Cannot build budgets below max_resources;
(max: {}) - (min: {}) > (num_rungs: {})
"""


def compute_budgets(
    min_resources, max_resources, reduction_factor, nums_population, pairs
):
    """Compute the budgets used for each execution of hyperband"""
    budgets_eves = []
    if reduction_factor == 1:
        for i in range(min_resources, max_resources + 1):
            if i == min_resources:
                budgets_eves.append([(nums_population, i)])
            else:
                budgets_eves[0].append((pairs * 2, i))
    else:
        num_brackets = int(np.log(max_resources) / np.log(reduction_factor))
        budgets = []
        budgets_tab = {}  # just for display consideration
        for bracket_id in range(0, num_brackets + 1):
            bracket_budgets = []
            num_trials = int(
                np.ceil(
                    int((num_brackets + 1) / (num_brackets - bracket_id + 1))
                    * (reduction_factor ** (num_brackets - bracket_id))
                )
            )

            min_resources = max_resources / reduction_factor ** (
                num_brackets - bracket_id
            )
            for i in range(0, num_brackets - bracket_id + 1):
                n_i = int(num_trials / reduction_factor ** i)
                min_i = int(min_resources * reduction_factor ** i)
                bracket_budgets.append((n_i, min_i))

                if budgets_tab.get(i):
                    budgets_tab[i].append((n_i, min_i))
                else:
                    budgets_tab[i] = [(n_i, min_i)]

            budgets.append(bracket_budgets)

        for i in range(len(budgets[0])):
            if i == 0:
                budgets_eves.append([(nums_population, budgets[0][i][1])])
            else:
                budgets_eves[0].append((pairs * 2, budgets[0][i][1]))

    return budgets_eves


class EvolutionES(Hyperband):
    """EvolutionES formulates hyperparameter optimization as an evolution.

    For more information on the algorithm,
    see original paper at
    https://arxiv.org/pdf/1703.01041.pdf and
    https://arxiv.org/pdf/1901.11117.pdf

    Real et al. "Large-Scale Evolution of Image Classifiers"
    So et all. "The Evolved Transformer"

    Parameters
    ----------
    space: `orion.algo.space.Space`
        Optimisation space with priors for each dimension.
    seed: None, int or sequence of int
        Seed for the random number generator used to sample new trials.
        Default: ``None``
    repetitions: int
        Number of execution of Hyperband. Default is numpy.inf which means to
        run Hyperband until no new trials can be suggested.

    """

    requires_type = None
    requires_dist = None
    requires_shape = "flattened"

    def __init__(
        self,
        space,
        seed=None,
        repetitions=np.inf,
        nums_population=20,
        mutate=None,
        max_retries=1000,
    ):
        super(EvolutionES, self).__init__(space, seed=seed, repetitions=repetitions)
        pair = nums_population // 2
        mutate_ratio = 0.3
        self.nums_population = nums_population
        self.nums_comp_pairs = pair
        self.max_retries = max_retries
        self.mutate_ratio = mutate_ratio
        self.mutate = mutate
        self.nums_mutate_gene = (
            int((len(self.space.values()) - 1) * mutate_ratio)
            if int((len(self.space.values()) - 1) * mutate_ratio) > 0
            else 1
        )

        self._param_names += ["nums_population", "mutate", "max_retries"]

        self.hurdles = []

        self.population = {}
        for key in range(len(self.space)):
            if not key == self.fidelity_index:
                self.population[key] = [-1] * nums_population

        self.performance = np.inf * np.ones(nums_population)

        self.budgets = compute_budgets(
            self.min_resources,
            self.max_resources,
            self.reduction_factor,
            nums_population,
            pair,
        )

        self.brackets = [
            BracketEVES(self, bracket_budgets, 1) for bracket_budgets in self.budgets
        ]
        self.seed_rng(seed)

    @property
    def state_dict(self):
        """Return a state dict that can be used to reset the state of the algorithm."""
        state_dict = super(EvolutionES, self).state_dict
        state_dict["population"] = copy.deepcopy(self.population)
        state_dict["performance"] = copy.deepcopy(self.performance)
        state_dict["hurdles"] = copy.deepcopy(self.hurdles)
        return state_dict

    def set_state(self, state_dict):
        """Reset the state of the algorithm based on the given state_dict"""
        super(EvolutionES, self).set_state(state_dict)
        self.population = state_dict["population"]
        self.performance = state_dict["performance"]
        self.hurdles = state_dict["hurdles"]

    def _get_bracket(self, point):
        """Get the bracket of a point during observe"""
        return self.brackets[0]


class BracketEVES(HyperbandBracket):
    """Bracket of rungs for the algorithm Hyperband.

    Parameters
    ----------
    evolutiones: `evolutiones` algorithm
        The evolutiones algorithm object which this bracket will be part of.
    budgets: list of tuple
        Each tuple gives the (n_trials, resource_budget) for the respective rung.
    repetition_id: int
        The id of hyperband execution this bracket belongs to

    """

    def __init__(self, evolution_es, budgets, repetition_id):
        super(BracketEVES, self).__init__(evolution_es, budgets, repetition_id)
        self.eves = self.hyperband
        self.search_space_remove_fidelity = []
        self._candidates = {}

        if evolution_es.mutate:
            self.mutate_attr = copy.deepcopy(evolution_es.mutate)
        else:
            self.mutate_attr = {}

        function_string = self.mutate_attr.pop(
            "function", "orion.algo.mutate_functions.default_mutate"
        )
        mod_name, func_name = function_string.rsplit(".", 1)
        mod = importlib.import_module(mod_name)
        self.mutate_func = getattr(mod, func_name)

        for i in range(len(self.space.values())):
            if not i == self.eves.fidelity_index:
                self.search_space_remove_fidelity.append(i)

    @property
    def space(self):
        return self.eves.space

    @property
    def state_dict(self):
        state_dict = super(BracketEVES, self).state_dict
        state_dict["candidates"] = copy.deepcopy(self._candidates)
        return state_dict

    def set_state(self, state_dict):
        super(BracketEVES, self).set_state(state_dict)
        self._candidates = state_dict["candidates"]

    def _get_teams(self, rung_id):
        """Get the red team and blue team"""
        if self.has_rung_filled(rung_id + 1):
            return []

        rung = self.rungs[rung_id]["results"]

        population_range = (
            self.eves.nums_population
            if len(list(rung.values())) > self.eves.nums_population
            else len(list(rung.values()))
        )

        for i in range(population_range):
            for j in self.search_space_remove_fidelity:
                self.eves.population[j][i] = list(rung.values())[i][1][j]
            self.eves.performance[i] = list(rung.values())[i][0]

        population_index = list(range(self.eves.nums_population))
        red_team = self.eves.rng.choice(
            population_index, self.eves.nums_comp_pairs, replace=False
        )
        diff_list = list(set(population_index).difference(set(red_team)))
        blue_team = self.eves.rng.choice(
            diff_list, self.eves.nums_comp_pairs, replace=False
        )

        return rung, population_range, red_team, blue_team

    def _mutate_population(self, red_team, blue_team, rung, population_range, fidelity):
        """Get the mutated population and hurdles"""
        winner_list = []
        loser_list = []

        if set(red_team) != set(blue_team):
            hurdles = 0
            for i, _ in enumerate(red_team):
                winner, loser = (
                    (red_team, blue_team)
                    if self.eves.performance[red_team[i]]
                    < self.eves.performance[blue_team[i]]
                    else (blue_team, red_team)
                )

                winner_list.append(winner[i])
                loser_list.append(loser[i])
                hurdles += self.eves.performance[winner[i]]
                self._mutate(winner[i], loser[i])

            hurdles /= len(red_team)
            self.eves.hurdles.append(hurdles)

            logger.debug("Evolution hurdles are: %s", str(self.eves.hurdles))

        points = []
        nums_all_equal = [0] * population_range
        for i in range(population_range):
            point = [0] * len(self.space)
            while True:
                point = list(point)
                point[self.eves.fidelity_index] = fidelity

                for j in self.search_space_remove_fidelity:
                    point[j] = self.eves.population[j][i]

                point = self.eves.format_point(point)

                if point in points:
                    nums_all_equal[i] += 1
                    logger.debug("find equal one, continue to mutate.")
                    self._mutate(i, i)
                elif self.eves.has_suggested(point):
                    nums_all_equal[i] += 1
                    logger.debug("find one already suggested, continue to mutate.")
                    self._mutate(i, i)
                else:
                    break
                if nums_all_equal[i] > self.eves.max_retries:
                    logger.warning(
                        "Can not Evolve any more. You can make an early stop."
                    )
                    break

            if nums_all_equal[i] < self.eves.max_retries:
                points.append(point)
            else:
                logger.debug("Dropping point %s", point)

        return points, np.array(nums_all_equal)

    def get_candidates(self, rung_id):
        """Get a candidate for promotion"""
        if rung_id not in self._candidates:
            rung, population_range, red_team, blue_team = self._get_teams(rung_id)
            fidelity = self.rungs[rung_id + 1]["resources"]
            self._candidates[rung_id] = self._mutate_population(
                red_team, blue_team, rung, population_range, fidelity
            )[0]

        candidates = []
        for candidate in self._candidates[rung_id]:
            if not self.eves.has_suggested(candidate):
                candidates.append(candidate)
        return candidates

    def _mutate(self, winner_id, loser_id):
        select_genes_key_list = self.eves.rng.choice(
            self.search_space_remove_fidelity, self.eves.nums_mutate_gene, replace=False
        )
        self.copy_winner(winner_id, loser_id)
        kwargs = copy.deepcopy(self.mutate_attr)

        for i, _ in enumerate(select_genes_key_list):
            space = self.space.values()[select_genes_key_list[i]]
            old = self.eves.population[select_genes_key_list[i]][loser_id]
            new = self.mutate_func(space, self.eves.rng, old, **kwargs)
            self.eves.population[select_genes_key_list[i]][loser_id] = new

        self.eves.performance[loser_id] = -1

    def copy_winner(self, winner_id, loser_id):
        """Copy winner to loser"""
        for key in self.search_space_remove_fidelity:
            self.eves.population[key][loser_id] = self.eves.population[key][winner_id]
