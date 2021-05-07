# -*- coding: utf-8 -*-
"""Tests for :mod:`orion.algo.asha`."""
import hashlib
import itertools
import logging

import numpy as np
import pytest

from orion.algo.asha import ASHA, ASHABracket, compute_budgets
from orion.algo.space import Fidelity, Integer, Real, Space
from orion.testing.algo import BaseAlgoTests


@pytest.fixture
def space():
    """Create a Space with a real dimension and a fidelity value."""
    space = Space()
    space.register(Real("lr", "uniform", 0, 1))
    space.register(Fidelity("epoch", 1, 9, 3))
    return space


@pytest.fixture
def b_config(space):
    """Return a configuration for a bracket."""
    fidelity_dim = space.values()[0]
    num_rungs = 3
    budgets = np.logspace(
        np.log(fidelity_dim.low) / np.log(fidelity_dim.base),
        np.log(fidelity_dim.high) / np.log(fidelity_dim.base),
        num_rungs,
        base=fidelity_dim.base,
    )
    return {
        "reduction_factor": fidelity_dim.base,
        "budgets": list(zip(list(map(int, budgets[::-1])), budgets)),
    }


@pytest.fixture
def asha(b_config, space):
    """Return an instance of ASHA."""
    return ASHA(space)


@pytest.fixture
def bracket(b_config, asha):
    """Return a `ASHABracket` instance configured with `b_config`."""
    return ASHABracket(asha, b_config["budgets"], 1)


@pytest.fixture
def rung_0():
    """Create fake points and objectives for rung 0."""
    points = np.linspace(0, 1, 9)
    return dict(
        n_trials=9,
        resources=1,
        results={
            hashlib.md5(str([point]).encode("utf-8")).hexdigest(): (point, (1, point))
            for point in points
        },
    )


@pytest.fixture
def rung_1(rung_0):
    """Create fake points and objectives for rung 1."""
    return dict(
        n_trials=9,
        resources=3,
        results={
            hashlib.md5(str([value[0]]).encode("utf-8")).hexdigest(): value
            for value in map(
                lambda v: (v[0], (3, v[0])), sorted(rung_0["results"].values())
            )
        },
    )


@pytest.fixture
def rung_2(rung_1):
    """Create fake points and objectives for rung 1."""
    return dict(
        n_trials=9,
        resources=9,
        results={
            hashlib.md5(str([value[0]]).encode("utf-8")).hexdigest(): value
            for value in map(
                lambda v: (v[0], (9, v[0])), sorted(rung_1["results"].values())
            )
        },
    )


def force_observe(asha, point, results):

    full_id = asha.get_id(point, ignore_fidelity=False)
    asha.register(point, results)

    bracket = asha._get_bracket(point)
    id_wo_fidelity = asha.get_id(point, ignore_fidelity=True)
    asha.trial_to_brackets[id_wo_fidelity] = bracket

    asha.observe([point], [results])


def test_compute_budgets():
    """Verify proper computation of budgets on a logarithmic scale"""
    # Check typical values
    assert compute_budgets(1, 16, 4, 3, 1) == [[(16, 1), (4, 4), (1, 16)]]
    # Check rounding (max_resources is not a multiple of reduction_factor)
    assert compute_budgets(1, 30, 4, 3, 1) == [[(16, 1), (4, 5), (1, 30)]]
    # Check rounding (min_resources may be rounded below its actual value)
    assert compute_budgets(25, 1000, 2, 6, 1) == [
        [(32, 25), (16, 52), (8, 109), (4, 229), (2, 478), (1, 1000)]
    ]
    # Check min_resources
    assert compute_budgets(5, 125, 5, 3, 1) == [[(25, 5), (5, 25), (1, 125)]]
    # Check num_rungs
    assert compute_budgets(1, 16, 2, 5, 1) == [
        [(16, 1), (8, 2), (4, 4), (2, 8), (1, 16)]
    ]


def test_compute_compressed_budgets():
    """Verify proper computation of budgets when scale is small and integer rounding creates
    duplicates
    """
    assert compute_budgets(1, 16, 2, 10, 1) == [
        [
            (512, 1),
            (256, 2),
            (128, 3),
            (64, 4),
            (32, 5),
            (16, 6),
            (8, 7),
            (4, 9),
            (2, 12),
            (1, 16),
        ]
    ]

    with pytest.raises(ValueError) as exc:
        compute_budgets(1, 2, 2, 10, 1)

    assert "Cannot build budgets below max_resources" in str(exc.value)


class TestASHABracket:
    """Tests for the `ASHABracket` class."""

    def test_rungs_creation(self, bracket):
        """Test the creation of rungs for bracket 0."""
        assert len(bracket.rungs) == 3
        assert bracket.rungs[0]["resources"] == 1
        assert bracket.rungs[1]["resources"] == 3
        assert bracket.rungs[2]["resources"] == 9

    def test_register(self, asha, bracket):
        """Check that a point is correctly registered inside a bracket."""
        bracket.asha = asha
        point = (1, 0.0)
        point_hash = hashlib.md5(str([0.0]).encode("utf-8")).hexdigest()

        bracket.register(point, 0.0)

        assert len(bracket.rungs[0])
        assert point_hash in bracket.rungs[0]["results"]
        assert (0.0, point) == bracket.rungs[0]["results"][point_hash]

    def test_bad_register(self, asha, bracket):
        """Check that a non-valid point is not registered."""
        bracket.asha = asha

        with pytest.raises(IndexError) as ex:
            bracket.register((55, 0.0), 0.0)

        assert "Bad fidelity level 55" in str(ex.value)

    def test_candidate_promotion(self, asha, bracket, rung_0):
        """Test that correct point is promoted."""
        bracket.asha = asha
        bracket.rungs[0] = rung_0

        point = bracket.get_candidate(0)

        assert point == (1, 0.0)

    def test_promotion_with_rung_1_hit(self, asha, bracket, rung_0):
        """Test that get_candidate gives us the next best thing if point is already in rung 1."""
        point = (1, 0.0)
        point_hash = hashlib.md5(str([0.0]).encode("utf-8")).hexdigest()
        bracket.asha = asha
        bracket.rungs[0] = rung_0
        bracket.rungs[1]["results"][point_hash] = (0.0, point)

        point = bracket.get_candidate(0)

        assert point == (1, 0.125)

    def test_no_promotion_when_rung_full(self, asha, bracket, rung_0, rung_1):
        """Test that get_candidate returns `None` if rung 1 is full."""
        bracket.asha = asha
        bracket.rungs[0] = rung_0
        bracket.rungs[1] = rung_1

        point = bracket.get_candidate(0)

        assert point is None

    def test_no_promotion_if_not_enough_points(self, asha, bracket):
        """Test the get_candidate return None if there is not enough points ready."""
        bracket.asha = asha
        bracket.rungs[0] = dict(
            n_trials=1,
            resources=1,
            results={
                hashlib.md5(str([0.0]).encode("utf-8")).hexdigest(): (0.0, (1, 0.0))
            },
        )

        point = bracket.get_candidate(0)

        assert point is None

    def test_no_promotion_if_not_completed(self, asha, bracket, rung_0):
        """Test the get_candidate return None if trials are not completed."""
        bracket.asha = asha
        bracket.rungs[0] = rung_0
        rung = bracket.rungs[0]["results"]

        point = bracket.get_candidate(0)

        for p_id in rung.keys():
            rung[p_id] = (None, rung[p_id][1])

        point = bracket.get_candidate(0)

        assert point is None

    def test_is_done(self, bracket, rung_0):
        """Test that the `is_done` property works."""
        assert not bracket.is_done

        # Actual value of the point is not important here
        bracket.rungs[2] = dict(n_trials=1, resources=9, results={"1": (1, 0.0)})

        assert bracket.is_done

    def test_update_rungs_return_candidate(self, asha, bracket, rung_1):
        """Check if a valid modified candidate is returned by update_rungs."""
        bracket.asha = asha
        bracket.rungs[1] = rung_1
        point_hash = hashlib.md5(str([0.0]).encode("utf-8")).hexdigest()

        candidate = bracket.promote(1)[0]

        assert point_hash in bracket.rungs[1]["results"]
        assert bracket.rungs[1]["results"][point_hash] == (0.0, (3, 0.0))
        assert candidate[0] == 9

    def test_update_rungs_return_no_candidate(self, asha, bracket, rung_1):
        """Check if no candidate is returned by update_rungs."""
        bracket.asha = asha

        candidate = bracket.promote(1)

        assert candidate == []

    def test_repr(self, bracket, rung_0, rung_1, rung_2):
        """Test the string representation of ASHABracket"""
        bracket.rungs[0] = rung_0
        bracket.rungs[1] = rung_1
        bracket.rungs[2] = rung_2

        assert str(bracket) == "ASHABracket(resource=[1, 3, 9], repetition id=1)"


class TestASHA:
    """Tests for the algo ASHA."""

    def test_register(self, asha, bracket, rung_0, rung_1):
        """Check that a point is registered inside the bracket."""
        asha.brackets = [bracket]
        bracket.asha = asha
        bracket.rungs = [rung_0, rung_1]
        point = (1, 0.0)
        point_hash = hashlib.md5(str([0.0]).encode("utf-8")).hexdigest()

        asha.observe([point], [{"objective": 0.0}])

        assert len(bracket.rungs[0])
        assert point_hash in bracket.rungs[0]["results"]
        assert (0.0, point) == bracket.rungs[0]["results"][point_hash]

    def test_register_bracket_multi_fidelity(self, space, b_config):
        """Check that a point is registered inside the same bracket for diff fidelity."""
        asha = ASHA(space, num_brackets=3)

        value = 50
        fidelity = 1
        point = (fidelity, value)
        point_hash = hashlib.md5(str([value]).encode("utf-8")).hexdigest()

        force_observe(asha, point, {"objective": 0.0})

        bracket = asha.brackets[0]

        assert len(bracket.rungs[0])
        assert point_hash in bracket.rungs[0]["results"]
        assert (0.0, point) == bracket.rungs[0]["results"][point_hash]

        fidelity = 3
        point = [fidelity, value]
        point_hash = hashlib.md5(str([value]).encode("utf-8")).hexdigest()

        force_observe(asha, point, {"objective": 0.0})

        assert len(bracket.rungs[0])
        assert point_hash in bracket.rungs[1]["results"]
        assert (0.0, point) != bracket.rungs[0]["results"][point_hash]
        assert (0.0, point) == bracket.rungs[1]["results"][point_hash]

    def test_register_next_bracket(self, space, b_config):
        """Check that a point is registered inside the good bracket when higher fidelity."""
        asha = ASHA(space, num_brackets=3)

        value = 50
        fidelity = 3
        point = (fidelity, value)
        point_hash = hashlib.md5(str([value]).encode("utf-8")).hexdigest()

        force_observe(asha, point, {"objective": 0.0})

        assert sum(len(rung["results"]) for rung in asha.brackets[0].rungs) == 0
        assert sum(len(rung["results"]) for rung in asha.brackets[1].rungs) == 1
        assert sum(len(rung["results"]) for rung in asha.brackets[2].rungs) == 0
        assert point_hash in asha.brackets[1].rungs[0]["results"]
        assert (0.0, point) == asha.brackets[1].rungs[0]["results"][point_hash]

        value = 51
        fidelity = 9
        point = (fidelity, value)
        point_hash = hashlib.md5(str([value]).encode("utf-8")).hexdigest()

        force_observe(asha, point, {"objective": 0.0})

        assert sum(len(rung["results"]) for rung in asha.brackets[0].rungs) == 0
        assert sum(len(rung["results"]) for rung in asha.brackets[1].rungs) == 1
        assert sum(len(rung["results"]) for rung in asha.brackets[2].rungs) == 1
        assert point_hash in asha.brackets[2].rungs[0]["results"]
        assert (0.0, point) == asha.brackets[2].rungs[0]["results"][point_hash]

    def test_register_invalid_fidelity(self, space, b_config):
        """Check that a point cannot registered if fidelity is invalid."""
        asha = ASHA(space, num_brackets=3)

        value = 50
        fidelity = 2
        point = (fidelity, value)

        with pytest.raises(ValueError) as ex:
            force_observe(asha, point, {"objective": 0.0})

        assert "No bracket found for point" in str(ex.value)

    def test_register_not_sampled(self, space, b_config, caplog):
        """Check that a point cannot registered if not sampled."""
        asha = ASHA(space, num_brackets=3)

        value = 50
        fidelity = 2
        point = (fidelity, value)

        with caplog.at_level(logging.INFO, logger="orion.algo.hyperband"):
            asha.observe([point], [{"objective": 0.0}])

        assert len(caplog.records) == 1
        assert "Ignoring point" in caplog.records[0].msg

    def test_register_corrupted_db(self, caplog, space, b_config):
        """Check that a point cannot registered if passed in order diff than fidelity."""
        asha = ASHA(space, num_brackets=3)

        value = 50
        fidelity = 3
        point = (fidelity, value)

        force_observe(asha, point, {"objective": 0.0})
        assert "Point registered to wrong bracket" not in caplog.text

        fidelity = 1
        point = [fidelity, value]

        caplog.clear()
        force_observe(asha, point, {"objective": 0.0})
        assert "Point registered to wrong bracket" in caplog.text

    def test_suggest_new(self, monkeypatch, asha, bracket, rung_0, rung_1, rung_2):
        """Test that a new point is sampled."""
        asha.brackets = [bracket]
        bracket.asha = asha

        def sample(num=1, seed=None):
            return [("fidelity", 0.5)]

        monkeypatch.setattr(asha.space, "sample", sample)

        points = asha.suggest(1)

        assert points == [(1, 0.5)]

    def test_suggest_duplicates(
        self, monkeypatch, asha, bracket, rung_0, rung_1, rung_2
    ):
        """Test that sampling collisions are handled."""
        asha.brackets = [bracket]
        bracket.asha = asha

        duplicate_point = ("fidelity", 0.0)
        new_point = ("fidelity", 0.5)

        duplicate_id_wo_fidelity = asha.get_id(duplicate_point, ignore_fidelity=True)
        bracket.rungs[0] = dict(
            n_trials=2,
            resources=1,
            results={duplicate_id_wo_fidelity: (0.0, duplicate_point)},
        )
        asha.trial_to_brackets[duplicate_id_wo_fidelity] = bracket

        asha.register(duplicate_point, 0.0)

        points = [duplicate_point, new_point]

        def sample(num=1, seed=None):
            return points

        monkeypatch.setattr(asha.space, "sample", sample)

        assert asha.suggest(1)[0][1] == new_point[1]

    def test_suggest_inf_duplicates(
        self, monkeypatch, asha, bracket, rung_0, rung_1, rung_2
    ):
        """Test that sampling inf collisions returns None."""
        asha.brackets = [bracket]
        bracket.asha = asha

        zhe_point = ("fidelity", 0.0)
        asha.trial_to_brackets[asha.get_id(zhe_point, ignore_fidelity=True)] = bracket

        def sample(num=1, seed=None):
            return [zhe_point]

        monkeypatch.setattr(asha.space, "sample", sample)

        assert asha.suggest(1) == []

    def test_suggest_in_finite_cardinality(self):
        """Test that suggest None when search space is empty"""
        space = Space()
        space.register(Integer("yolo1", "uniform", 0, 5))
        space.register(Fidelity("epoch", 1, 9, 3))

        asha = ASHA(space)
        for i in range(6):
            force_observe(asha, (1, i), {"objective": i})

        for i in range(2):
            force_observe(asha, (3, i), {"objective": i})

        assert asha.suggest(1) == []

    def test_suggest_promote(self, asha, bracket, rung_0):
        """Test that correct point is promoted and returned."""
        asha.brackets = [bracket]
        bracket.asha = asha
        bracket.rungs[0] = rung_0

        points = asha.suggest(1)

        assert points == [(3, 0.0)]


class TestGenericASHA(BaseAlgoTests):
    algo_name = "asha"
    config = {
        "seed": 123456,
        "num_rungs": 5,
        "num_brackets": 2,
        "repetitions": 3,
    }
    space = {"x": "uniform(0, 1)", "f": "fidelity(1, 10, base=2)"}

    def test_suggest_n(self, mocker, num, attr):
        algo = self.create_algo()
        spy = self.spy_phase(mocker, num, algo, attr)
        points = algo.suggest(5)
        assert len(points) == 1

    @pytest.mark.skip(reason="See https://github.com/Epistimio/orion/issues/598")
    def test_is_done_cardinality(self):
        space = self.update_space(
            {
                "x": "uniform(0, 4, discrete=True)",
                "y": "choices(['a', 'b', 'c'])",
                "z": "loguniform(1, 6, discrete=True)",
            }
        )
        space = self.create_space(space)
        assert space.cardinality == 5 * 3 * 6

        algo = self.create_algo(num_brackets=2, space=space)

        for rung in range(algo.algorithm.num_rungs):
            assert not algo.is_done

            points = []
            while True:
                assert not algo.is_done
                n_sampled = len(algo.algorithm.sampled)
                n_trials = len(algo.algorithm.trial_to_brackets)
                new_points = algo.suggest(1)
                if new_points is None:
                    break
                points += new_points
                if rung == 0:
                    assert len(algo.algorithm.sampled) == n_sampled + 1
                else:
                    assert len(algo.algorithm.sampled) == n_sampled
                assert len(algo.algorithm.trial_to_brackets) == n_trials + 1

            assert not algo.is_done

            for i, point in enumerate(points):
                algo.observe([point], [dict(objective=i)])

        assert algo.is_done

    def test_is_done_max_trials(self):
        space = self.create_space()

        MAX_TRIALS = 10
        algo = self.create_algo(space=space)
        algo.algorithm.max_trials = MAX_TRIALS

        objective = 0
        while not algo.is_done:
            points = algo.suggest(1)
            assert points is not None
            if points:
                self.observe_points(points, algo, objective)
                objective += len(points)

        # ASHA should ignore max trials.
        assert algo.n_observed > MAX_TRIALS
        assert algo.is_done

    @pytest.mark.skip(reason="See https://github.com/Epistimio/orion/issues/599")
    def test_optimize_branin(self):
        pass

    def infer_repetition_and_rung(self, num):
        budgets = BUDGETS
        if num > budgets[-1] * 2:
            return 3, 0
        elif num > budgets[-1]:
            return 2, 0

        if num == 0:
            return 1, -1

        return 1, budgets.index(num)

    def assert_callbacks(self, spy, num, algo):

        if num == 0:
            return

        repetition_id, rung_id = self.infer_repetition_and_rung(num)

        brackets = algo.algorithm.brackets

        assert len(brackets) == repetition_id * 2

        for j in range(0, rung_id + 1):
            for bracket in brackets:
                assert len(bracket.rungs[j]["results"]) > 0, (bracket, j)


BUDGETS = [
    4 + 2,  # rung 0
    (8 + 4 + 2 + 4),  # rung 1 (first bracket 8 4 2, second bracket 4)
    (16 + 8 + 4 + 2 + 1 + 8 + 4 + 2),  #  rung 2
]

TestGenericASHA.set_phases(
    [("random", 0, "space.sample")]
    + [(f"rung{i}", budget, "suggest") for i, budget in enumerate(BUDGETS)]
    + [("rep1-rung1", BUDGETS[-1] + 16, "suggest")]
    + [("rep2-rung1", BUDGETS[-1] * 2 + 16, "suggest")]
)
