#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for :mod:`orion.algo.hyperband`."""
import copy
import hashlib
import logging

import numpy as np
import pytest

from orion.algo.hyperband import Hyperband, HyperbandBracket, compute_budgets
from orion.algo.space import Fidelity, Integer, Real, Space
from orion.testing.algo import BaseAlgoTests, phase


@pytest.fixture
def space():
    """Create a Space with a real dimension and a fidelity value."""
    space = Space()
    space.register(Real("lr", "uniform", 0, 1))
    space.register(Fidelity("epoch", 1, 9, 3))
    return space


@pytest.fixture
def budgets():
    """Return a configuration for a bracket."""
    return [(9, 1), (3, 3), (1, 9)]


@pytest.fixture
def hyperband(space):
    """Return an instance of Hyperband."""
    return Hyperband(space, repetitions=1)


@pytest.fixture
def bracket(budgets, hyperband):
    """Return a `HyperbandBracket` instance configured with `b_config`."""
    return HyperbandBracket(hyperband, budgets, 1)


@pytest.fixture
def rung_0():
    """Create fake points and objectives for rung 0."""
    points = np.linspace(0, 8, 9)
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
    values = map(
        lambda v: (v[0], (3, v[0])), list(sorted(rung_0["results"].values()))[:3]
    )
    return dict(
        n_trials=3,
        resources=3,
        results={
            hashlib.md5(str([value[0]]).encode("utf-8")).hexdigest(): value
            for value in values
        },
    )


@pytest.fixture
def rung_2(rung_1):
    """Create fake points and objectives for rung 1."""
    values = map(
        lambda v: (v[0], (9, v[0])), list(sorted(rung_1["results"].values()))[:1]
    )
    return dict(
        n_trials=1,
        resources=9,
        results={
            hashlib.md5(str([value[0]]).encode("utf-8")).hexdigest(): value
            for value in values
        },
    )


def test_compute_budgets():
    """Verify proper computation of budgets on a logarithmic scale"""
    # Check typical values
    assert compute_budgets(81, 3) == [
        [(81, 1), (27, 3), (9, 9), (3, 27), (1, 81)],
        [(27, 3), (9, 9), (3, 27), (1, 81)],
        [(9, 9), (3, 27), (1, 81)],
        [(6, 27), (2, 81)],
        [(5, 81)],
    ]
    assert compute_budgets(16, 4) == [
        [(16, 1), (4, 4), (1, 16)],
        [(4, 4), (1, 16)],
        [(3, 16)],
    ]
    assert compute_budgets(16, 5) == [[(5, 3), (1, 16)], [(2, 16)]]


def force_observe(hyperband, point, results):
    # hyperband.sampled.add(hashlib.md5(str(list(point)).encode("utf-8")).hexdigest())

    hyperband.register(point, results)

    bracket = hyperband._get_bracket(point)
    id_wo_fidelity = hyperband.get_id(point, ignore_fidelity=True)
    hyperband.trial_to_brackets[id_wo_fidelity] = bracket

    hyperband.observe([point], [results])


def mock_samples(hyperband, samples):
    for bracket in hyperband.brackets:
        bracket._samples = samples


class TestHyperbandBracket:
    """Tests for the `HyperbandBracket` class."""

    def test_rungs_creation(self, bracket):
        """Test the creation of rungs for bracket 0."""
        assert len(bracket.rungs) == 3
        assert bracket.rungs[0] == dict(n_trials=9, resources=1, results=dict())
        assert bracket.rungs[1] == dict(n_trials=3, resources=3, results=dict())
        assert bracket.rungs[2] == dict(n_trials=1, resources=9, results=dict())

    def test_register(self, hyperband, bracket):
        """Check that a point is correctly registered inside a bracket."""
        bracket.hyperband = hyperband
        point = (1, 0.0)
        point_hash = hashlib.md5(str([0.0]).encode("utf-8")).hexdigest()

        bracket.register(point, 0.0)

        assert len(bracket.rungs[0])
        assert point_hash in bracket.rungs[0]["results"]
        assert (0.0, point) == bracket.rungs[0]["results"][point_hash]

    def test_bad_register(self, hyperband, bracket):
        """Check that a non-valid point is not registered."""
        bracket.hyperband = hyperband

        with pytest.raises(IndexError) as ex:
            bracket.register((55, 0.0), 0.0)

        assert "Bad fidelity level 55" in str(ex.value)

    def test_candidate_promotion(self, hyperband, bracket, rung_0):
        """Test that correct point is promoted."""
        bracket.hyperband = hyperband
        bracket.rungs[0] = rung_0

        points = bracket.get_candidates(0)

        assert points[0] == (1, 0.0)

    def test_promotion_with_rung_1_hit(self, hyperband, bracket, rung_0):
        """Test that get_candidate gives us the next best thing if point is already in rung 1."""
        point = (1, 0.0)
        point_hash = hashlib.md5(str([0.0]).encode("utf-8")).hexdigest()
        bracket.hyperband = hyperband
        bracket.rungs[0] = rung_0
        bracket.rungs[1]["results"][point_hash] = (0.0, point)

        points = bracket.get_candidates(0)

        assert points[0] == (1, 1)

    def test_no_promotion_when_rung_full(self, hyperband, bracket, rung_0, rung_1):
        """Test that get_candidate returns `None` if rung 1 is full."""
        bracket.hyperband = hyperband
        bracket.rungs[0] = rung_0
        bracket.rungs[1] = rung_1

        points = bracket.get_candidates(0)

        assert points == []

    def test_no_promotion_if_not_completed(self, hyperband, bracket, rung_0):
        """Test the get_candidate return None if trials are not completed."""
        bracket.hyperband = hyperband
        bracket.rungs[0] = rung_0
        rung = bracket.rungs[0]["results"]

        # points = bracket.get_candidates(0)

        for p_id in rung.keys():
            rung[p_id] = (None, rung[p_id][1])

        with pytest.raises(AssertionError):
            bracket.get_candidates(0)

    def test_is_done(self, bracket, rung_0):
        """Test that the `is_done` property works."""
        assert not bracket.is_done

        # Actual value of the point is not important here
        bracket.rungs[2]["results"] = {"1": (1, 0.0), "2": (1, 0.0), "3": (1, 0.0)}

        assert bracket.is_done

    def test_update_rungs_return_candidate(self, hyperband, bracket, rung_1):
        """Check if a valid modified candidate is returned by update_rungs."""
        bracket.hyperband = hyperband
        bracket.rungs[1] = rung_1
        point_hash = hashlib.md5(str([0.0]).encode("utf-8")).hexdigest()

        candidates = bracket.promote(1)

        assert point_hash in bracket.rungs[1]["results"]
        assert bracket.rungs[1]["results"][point_hash] == (0.0, (3, 0.0))
        assert candidates[0][0] == 9

    def test_update_rungs_return_no_candidate(self, hyperband, bracket, rung_1):
        """Check if no candidate is returned by update_rungs."""
        bracket.hyperband = hyperband

        candidates = bracket.promote(1)

        assert candidates == []

    def test_get_point_max_resource(self, hyperband, bracket, rung_0, rung_1, rung_2):
        """Test to get the max resource R for a particular point"""
        bracket.hyperband = hyperband
        bracket.rungs[0] = rung_0

        assert bracket.get_point_max_resource(point=(1, 0.0)) == 1
        assert bracket.get_point_max_resource(point=(1, 8.0)) == 1

        bracket.rungs[1] = rung_1
        assert bracket.get_point_max_resource(point=(1, 0.0)) == 3
        assert bracket.get_point_max_resource(point=(1, 8.0)) == 1

        bracket.rungs[2] = rung_2
        assert bracket.get_point_max_resource(point=(1, 0.0)) == 9
        assert bracket.get_point_max_resource(point=(1, 8.0)) == 1

    def test_repr(self, bracket, rung_0, rung_1, rung_2):
        """Test the string representation of HyperbandBracket"""
        bracket.rungs[0] = rung_0
        bracket.rungs[1] = rung_1
        bracket.rungs[2] = rung_2

        assert str(bracket) == "HyperbandBracket(resource=[1, 3, 9], repetition id=1)"


class TestHyperband:
    """Tests for the algo Hyperband."""

    def test_register(self, hyperband, bracket, rung_0, rung_1):
        """Check that a point is registered inside the bracket."""
        hyperband.brackets = [bracket]
        bracket.hyperband = hyperband
        bracket.rungs = [rung_0, rung_1]
        point = (1, 0.0)
        point_hash = hashlib.md5(str([0.0]).encode("utf-8")).hexdigest()

        hyperband.observe([point], [{"objective": 0.0}])

        assert len(bracket.rungs[0])
        assert point_hash in bracket.rungs[0]["results"]
        assert (0.0, point) == bracket.rungs[0]["results"][point_hash]

    def test_register_bracket_multi_fidelity(self, space):
        """Check that a point is registered inside the same bracket for diff fidelity."""
        hyperband = Hyperband(space)

        value = 50
        fidelity = 1
        point = (fidelity, value)
        point_hash = hashlib.md5(str([value]).encode("utf-8")).hexdigest()

        force_observe(hyperband, point, {"objective": 0.0})

        bracket = hyperband.brackets[0]

        assert len(bracket.rungs[0])
        assert point_hash in bracket.rungs[0]["results"]
        assert (0.0, point) == bracket.rungs[0]["results"][point_hash]

        fidelity = 3
        point = [fidelity, value]
        point_hash = hashlib.md5(str([value]).encode("utf-8")).hexdigest()

        force_observe(hyperband, point, {"objective": 0.0})

        assert len(bracket.rungs[0])
        assert point_hash in bracket.rungs[1]["results"]
        assert (0.0, point) != bracket.rungs[0]["results"][point_hash]
        assert (0.0, point) == bracket.rungs[1]["results"][point_hash]

    def test_register_next_bracket(self, space):
        """Check that a point is registered inside the good bracket when higher fidelity."""
        hyperband = Hyperband(space)

        value = 50
        fidelity = 3
        point = (fidelity, value)
        point_hash = hashlib.md5(str([value]).encode("utf-8")).hexdigest()

        force_observe(hyperband, point, {"objective": 0.0})

        assert sum(len(rung["results"]) for rung in hyperband.brackets[0].rungs) == 0
        assert sum(len(rung["results"]) for rung in hyperband.brackets[1].rungs) == 1
        assert sum(len(rung["results"]) for rung in hyperband.brackets[2].rungs) == 0
        assert point_hash in hyperband.brackets[1].rungs[0]["results"]
        assert (0.0, point) == hyperband.brackets[1].rungs[0]["results"][point_hash]

        value = 51
        fidelity = 9
        point = (fidelity, value)
        point_hash = hashlib.md5(str([value]).encode("utf-8")).hexdigest()

        force_observe(hyperband, point, {"objective": 0.0})

        assert sum(len(rung["results"]) for rung in hyperband.brackets[0].rungs) == 0
        assert sum(len(rung["results"]) for rung in hyperband.brackets[1].rungs) == 1
        assert sum(len(rung["results"]) for rung in hyperband.brackets[2].rungs) == 1
        assert point_hash in hyperband.brackets[2].rungs[0]["results"]
        assert (0.0, point) == hyperband.brackets[2].rungs[0]["results"][point_hash]

    def test_register_invalid_fidelity(self, space):
        """Check that a point cannot registered if fidelity is invalid."""
        hyperband = Hyperband(space)

        value = 50
        fidelity = 2
        point = (fidelity, value)

        with pytest.raises(ValueError) as ex:
            force_observe(hyperband, point, {"objective": 0.0})

        assert "No bracket found for point" in str(ex.value)

    def test_register_not_sampled(self, space, caplog):
        """Check that a point cannot registered if not sampled."""
        hyperband = Hyperband(space)

        value = 50
        fidelity = 2
        point = (fidelity, value)

        with caplog.at_level(logging.INFO, logger="orion.algo.hyperband"):
            hyperband.observe([point], [{"objective": 0.0}])

        assert len(caplog.records) == 1
        assert "Ignoring point" in caplog.records[0].msg

    def test_register_corrupted_db(self, caplog, space):
        """Check that a point cannot registered if passed in order diff than fidelity."""
        hyperband = Hyperband(space)

        value = 50
        fidelity = 3
        point = (fidelity, value)

        force_observe(hyperband, point, {"objective": 0.0})
        assert "Point registered to wrong bracket" not in caplog.text

        fidelity = 1
        point = [fidelity, value]

        caplog.clear()
        force_observe(hyperband, point, {"objective": 0.0})
        assert "Point registered to wrong bracket" in caplog.text

    def test_suggest_new(self, monkeypatch, hyperband, bracket, rung_0, rung_1, rung_2):
        """Test that a new point is sampled."""
        hyperband.brackets = [bracket]
        bracket.hyperband = hyperband

        mock_samples(hyperband, [("fidelity", i) for i in range(10)])

        points = hyperband.suggest(100)

        assert points[0] == (1.0, 0)
        assert points[1] == (1.0, 1)

    def test_suggest_duplicates_between_calls(self, monkeypatch, hyperband, bracket):
        """Test that same points are not allowed in different suggest call of
        the same hyperband execution.
        """
        hyperband.brackets = [bracket]
        bracket.hyperband = hyperband

        duplicate_point = ("fidelity", 0.0)
        new_point = ("fidelity", 0.5)

        duplicate_id = hashlib.md5(str([duplicate_point]).encode("utf-8")).hexdigest()
        bracket.rungs[0]["results"] = {duplicate_id: (0.0, duplicate_point)}

        hyperband.trial_to_brackets[
            hyperband.get_id(duplicate_point, ignore_fidelity=True)
        ] = bracket

        points = [duplicate_point, new_point]

        mock_samples(hyperband, points + [("fidelity", i) for i in range(10 - 2)])

        assert hyperband.suggest(100)[0][1] == new_point[1]

    def test_suggest_duplicates_one_call(self, monkeypatch, hyperband, bracket):
        """Test that same points are not allowed in the same suggest call ofxs
        the same hyperband execution.
        """
        hyperband.brackets = [bracket]
        bracket.hyperband = hyperband

        zhe_point = [(1, 0.0), (1, 1.0), (1, 1.0), (1, 2.0)]

        mock_samples(hyperband, zhe_point * 2)
        zhe_samples = hyperband.suggest(100)

        assert zhe_samples[0][1] == 0.0
        assert zhe_samples[1][1] == 1.0
        assert zhe_samples[2][1] == 2.0

        # zhe_point =
        mock_samples(
            hyperband,
            [
                (3, 0.0),
                (3, 1.0),
                (3, 1.0),
                (3, 2.0),
                (3, 5.0),
                (3, 4.0),
            ],
        )
        hyperband.trial_to_brackets[
            hyperband.get_id((1, 0.0), ignore_fidelity=True)
        ] = bracket
        hyperband.trial_to_brackets[
            hyperband.get_id((1, 1.0), ignore_fidelity=True)
        ] = bracket
        zhe_samples = hyperband.suggest(100)
        assert zhe_samples[0][1] == 5.0
        assert zhe_samples[1][1] == 4.0

    def test_suggest_duplicates_between_execution(
        self, monkeypatch, hyperband, budgets
    ):
        """Test that sampling collisions are handled between different hyperband execution."""
        hyperband.repetitions = 2
        bracket = HyperbandBracket(hyperband, budgets, 1)
        hyperband.brackets = [bracket]
        bracket.hyperband = hyperband

        for i in range(9):
            force_observe(hyperband, (1, i), {"objective": i})

        for i in range(3):
            force_observe(hyperband, (3, i), {"objective": i})

        force_observe(hyperband, (9, 0), {"objective": 0})

        assert not hyperband.is_done

        zhe_point = [(9, 0), (9, 1), (9, 2)]

        hyperband._refresh_brackets()
        mock_samples(hyperband, zhe_point * 2)
        zhe_samples = hyperband.suggest(100)
        assert zhe_samples == [(9, 1), (9, 2)]

    def test_suggest_inf_duplicates(
        self, monkeypatch, hyperband, bracket, rung_0, rung_1, rung_2
    ):
        """Test that sampling inf collisions will return None."""
        hyperband.brackets = [bracket]
        bracket.hyperband = hyperband

        zhe_point = ("fidelity", 0.0)
        hyperband.trial_to_brackets[
            hyperband.get_id(zhe_point, ignore_fidelity=True)
        ] = bracket

        mock_samples(hyperband, [zhe_point] * 2)

        assert hyperband.suggest(100) == []

    def test_suggest_in_finite_cardinality(self):
        """Test that suggest None when search space is empty"""
        space = Space()
        space.register(Integer("yolo1", "uniform", 0, 5))
        space.register(Fidelity("epoch", 1, 9, 3))

        hyperband = Hyperband(space, repetitions=1)
        for i in range(6):
            force_observe(hyperband, (1, i), {"objective": i})

        assert hyperband.suggest(100) == []

    def test_suggest_promote(self, hyperband, bracket, rung_0):
        """Test that correct point is promoted and returned."""
        hyperband.brackets = [bracket]
        bracket.hyperband = hyperband
        bracket.rungs[0] = rung_0

        points = hyperband.suggest(100)

        assert points == [(3, i) for i in range(3)]

    def test_is_filled(self, hyperband, bracket, rung_0, rung_1, rung_2):
        """Test that Hyperband bracket detects when rung is filled."""
        hyperband.brackets = [bracket]
        bracket.hyperband = hyperband
        bracket.rungs[0] = rung_0

        rung = bracket.rungs[0]["results"]
        trial_id = next(iter(rung.keys()))
        objective, point = rung.pop(trial_id)

        assert not bracket.is_filled
        assert not bracket.has_rung_filled(0)

        rung[trial_id] = (objective, point)

        assert bracket.is_filled
        assert bracket.has_rung_filled(0)
        assert not bracket.has_rung_filled(1)
        assert not bracket.has_rung_filled(2)

        bracket.rungs[1] = rung_1

        rung = bracket.rungs[1]["results"]
        trial_id = next(iter(rung.keys()))
        objective, point = rung.pop(trial_id)

        assert bracket.is_filled  # Should depend first rung only
        assert bracket.has_rung_filled(0)
        assert not bracket.has_rung_filled(1)

        rung[trial_id] = (objective, point)

        assert bracket.is_filled  # Should depend first rung only
        assert bracket.has_rung_filled(0)
        assert bracket.has_rung_filled(1)
        assert not bracket.has_rung_filled(2)

        bracket.rungs[2] = rung_2

        rung = bracket.rungs[2]["results"]
        trial_id = next(iter(rung.keys()))
        objective, point = rung.pop(trial_id)

        assert bracket.is_filled  # Should depend first rung only
        assert bracket.has_rung_filled(0)
        assert bracket.has_rung_filled(1)
        assert not bracket.has_rung_filled(2)

        rung[trial_id] = (objective, point)

        assert bracket.is_filled  # Should depend first rung only
        assert bracket.has_rung_filled(0)
        assert bracket.has_rung_filled(1)
        assert bracket.has_rung_filled(2)

    def test_is_ready(self, hyperband, bracket, rung_0, rung_1, rung_2):
        """Test that Hyperband bracket detects when rung is ready."""
        hyperband.brackets = [bracket]
        bracket.hyperband = hyperband
        bracket.rungs[0] = rung_0

        rung = bracket.rungs[0]["results"]
        trial_id = next(iter(rung.keys()))
        objective, point = rung[trial_id]
        rung[trial_id] = (None, point)

        assert not bracket.is_ready()
        assert not bracket.is_ready(0)

        rung[trial_id] = (objective, point)

        assert bracket.is_ready()
        assert bracket.is_ready(0)
        assert not bracket.is_ready(1)
        assert not bracket.is_ready(2)

        bracket.rungs[1] = rung_1

        rung = bracket.rungs[1]["results"]
        trial_id = next(iter(rung.keys()))
        objective, point = rung[trial_id]
        rung[trial_id] = (None, point)

        assert not bracket.is_ready()  # Should depend on last rung that contains trials
        assert bracket.is_ready(0)
        assert not bracket.is_ready(1)
        assert not bracket.is_ready(2)

        rung[trial_id] = (objective, point)

        assert bracket.is_ready()  # Should depend on last rung that contains trials
        assert bracket.is_ready(0)
        assert bracket.is_ready(1)
        assert not bracket.is_ready(2)

        bracket.rungs[2] = rung_2

        rung = bracket.rungs[2]["results"]
        trial_id = next(iter(rung.keys()))
        objective, point = rung[trial_id]
        rung[trial_id] = (None, point)

        assert not bracket.is_ready()  # Should depend on last rung that contains trials
        assert bracket.is_ready(0)
        assert bracket.is_ready(1)
        assert not bracket.is_ready(2)

        rung[trial_id] = (objective, point)

        assert bracket.is_ready()  # Should depend on last rung that contains trials
        assert bracket.is_ready(0)
        assert bracket.is_ready(1)
        assert bracket.is_ready(2)

    def test_suggest_opt_out(self, hyperband, bracket, rung_0, rung_1, rung_2):
        """Test that Hyperband opts out when rungs are not ready."""
        hyperband.brackets = [bracket]
        bracket.hyperband = hyperband

        bracket.rungs[0] = rung_0

        trial_id = next(iter(rung_0["results"].keys()))
        objective, point = rung_0["results"][trial_id]
        rung_0["results"][trial_id] = (None, point)

        assert hyperband.suggest(100) == []

    def test_full_process(self, monkeypatch, hyperband):
        """Test Hyperband full process."""
        sample_points = [("fidelity", i) for i in range(100)]

        hyperband._refresh_brackets()
        mock_samples(hyperband, copy.deepcopy(sample_points))

        # Fill all brackets' first rung

        points = hyperband.suggest(100)
        from orion.algo.hyperband import tabulate_status

        print(tabulate_status(hyperband.brackets))
        assert points[:3] == [(9, i) for i in range(3)]
        assert points[3:6] == [(3, i) for i in range(3, 6)]
        assert points[6:] == [(1, i) for i in range(6, 15)]

        assert hyperband.brackets[0].has_rung_filled(0)
        assert not hyperband.brackets[0].is_ready()
        assert hyperband.suggest(100) == []
        assert hyperband.suggest(100) == []

        # Observe first bracket first rung

        for i in range(9):
            hyperband.observe([(1, i + 3 + 3)], [{"objective": 16 - i}])

        assert hyperband.brackets[0].is_ready()
        assert not hyperband.brackets[1].is_ready()
        assert not hyperband.brackets[2].is_ready()

        # Promote first bracket first rung
        points = hyperband.suggest(100)
        assert points == [(3, 3 + 3 + 9 - 1 - i) for i in range(3)]

        assert hyperband.brackets[0].has_rung_filled(1)
        assert not hyperband.brackets[0].is_ready()
        assert not hyperband.brackets[1].is_ready()
        assert not hyperband.brackets[2].is_ready()

        # Observe first bracket second rung
        for i in range(3):
            hyperband.observe([(3, 3 + 3 + 9 - 1 - i)], [{"objective": 8 - i}])

        assert hyperband.brackets[0].is_ready()
        assert not hyperband.brackets[1].is_ready()
        assert not hyperband.brackets[2].is_ready()

        # Promote first bracket second rung
        points = hyperband.suggest(100)
        assert points == [(9, 12)]

        assert hyperband.brackets[0].has_rung_filled(2)
        assert not hyperband.brackets[0].is_ready()
        assert not hyperband.brackets[1].is_ready()
        assert not hyperband.brackets[2].is_ready()

        # Observe second bracket first rung
        for i in range(3):
            hyperband.observe([(3, i + 3)], [{"objective": 8 - i}])

        assert not hyperband.brackets[0].is_ready()
        assert hyperband.brackets[1].is_ready()
        assert not hyperband.brackets[2].is_ready()

        # Promote second bracket first rung
        points = hyperband.suggest(100)
        assert points == [(9, 5)]

        assert not hyperband.brackets[0].is_ready()
        assert hyperband.brackets[1].has_rung_filled(1)
        assert not hyperband.brackets[1].is_ready()
        assert not hyperband.brackets[2].is_ready()

        # Observe third bracket first rung
        for i in range(3):
            hyperband.observe([(9, i)], [{"objective": 3 - i}])

        assert not hyperband.brackets[0].is_ready(2)
        assert not hyperband.brackets[1].is_ready(1)
        assert hyperband.brackets[2].is_ready(0)
        assert hyperband.brackets[2].is_done

        # Observe second bracket second rung
        for i in range(1):
            hyperband.observe([(9, 3 + 3 - 1 - i)], [{"objective": 5 - i}])

        assert not hyperband.brackets[0].is_ready(2)
        assert hyperband.brackets[1].is_ready(1)
        assert hyperband.brackets[1].is_done

        # Observe first bracket third rung
        hyperband.observe(points, [{"objective": 3 - i}])

        assert hyperband.is_done
        assert hyperband.brackets[0].is_done
        assert hyperband.suggest(100) == []

        # Refresh repeat and execution times property
        monkeypatch.setattr(hyperband, "repetitions", 2)
        # monkeypatch.setattr(hyperband.brackets[0], "repetition_id", 0)
        # hyperband.observe([(9, 12)], [{"objective": 3 - i}])
        hyperband._refresh_brackets()
        mock_samples(hyperband, copy.deepcopy(sample_points))
        points = hyperband.suggest(100)
        assert not hyperband.is_done
        assert not hyperband.brackets[0].is_ready(2)
        assert not hyperband.brackets[0].is_done
        assert points[:3] == [(9, 3), (9, 4), (9, 6)]
        assert points[3:6] == [(3, 7), (3, 8), (3, 9)]
        assert points[6:] == [(1, i) for i in range(15, 24)]


class TestGenericHyperband(BaseAlgoTests):
    algo_name = "hyperband"
    config = {
        "seed": 123456,
        "repetitions": 3,
    }
    space = {"x": "uniform(0, 1)", "f": "fidelity(1, 10, base=2)"}

    @phase
    def test_suggest_lots(self, mocker, num, attr):
        """Test that hyperband returns whole rungs when requesting large `num`"""
        algo = self.create_algo()
        spy = self.spy_phase(mocker, num, algo, attr)
        points = algo.suggest(10000)
        repetition_id, rung_id = self.infer_repetition_and_rung(num)
        assert len(points) == BUDGETS[rung_id + 1 if rung_id < 3 else 0]

    def test_suggest_n(self, mocker, num, attr):
        algo = self.create_algo()
        spy = self.spy_phase(mocker, num, algo, attr)
        points = algo.suggest(5)
        repetition_id, rung_id = self.infer_repetition_and_rung(num)
        assert len(points) == min(BUDGETS[rung_id + 1 if rung_id < 3 else 0], 5)

    @pytest.mark.skip(reason="See https://github.com/Epistimio/orion/issues/598")
    def test_is_done_cardinality(self):
        space = self.update_space(
            {
                # Increase fidelity to increase number of trials in first rungs
                "f": "fidelity(1, 100, base=2)",
                "x": "uniform(0, 4, discrete=True)",
                "y": "choices(['a', 'b', 'c'])",
                "z": "loguniform(1, 6, discrete=True)",
            }
        )
        space = self.create_space(space)
        assert space.cardinality == 5 * 3 * 6

        algo = self.create_algo(space=space)

        for rung in range(len(algo.algorithm.brackets[0].rungs)):
            assert not algo.is_done

            points = []
            assert not algo.is_done
            n_sampled = algo.n_suggested
            n_trials = len(algo.algorithm.trial_to_brackets)
            new_points = algo.suggest(100)
            if not new_points:
                break
            points += new_points
            assert algo.n_suggested == n_sampled + len(new_points)
            assert len(algo.algorithm.trial_to_brackets) == space.cardinality

            # We reached max number of trials we can suggest before observing any.
            assert algo.suggest(100) == []

            assert not algo.is_done

            for i, point in enumerate(points):
                algo.observe([point], [dict(objective=i)])

        assert algo.is_done

    @pytest.mark.parametrize("num", [100000, 1])
    def test_is_done_max_trials(self, num):
        space = self.create_space()

        MAX_TRIALS = 10
        algo = self.create_algo(space=space)
        algo.algorithm.max_trials = MAX_TRIALS

        objective = 0
        while not algo.is_done:
            points = algo.suggest(num)
            assert points is not None
            if points:
                self.observe_points(points, algo, objective)
                objective += len(points)

        # Hyperband should ignore max trials.
        assert algo.n_observed > MAX_TRIALS
        assert algo.is_done

    @pytest.mark.skip(reason="See https://github.com/Epistimio/orion/issues/599")
    def test_optimize_branin(self):
        pass

    def infer_repetition_and_rung(self, num):
        budgets = list(np.cumsum(BUDGETS))
        if num >= budgets[-1] * 2:
            return 3, -1
        elif num >= budgets[-1]:
            return 2, -1

        if num <= 1:
            return 1, -1

        return 1, budgets.index(num)

    def assert_callbacks(self, spy, num, algo):
        if num == 0:
            return

        repetition_id, rung_id = self.infer_repetition_and_rung(num)

        brackets = algo.algorithm.brackets

        assert len(brackets) == repetition_id * len(brackets[0].rungs)

        for j in range(0, rung_id + 1):
            for bracket in brackets:
                if len(bracket.rungs) > j:
                    assert len(bracket.rungs[j]["results"]) > 0, (bracket, j)


BUDGETS = [20, 8, 3, 1]

TestGenericHyperband.set_phases(
    [("random", 0, "space.sample")]
    + [
        (f"rung{i}", budget, "suggest")
        for i, budget in enumerate(np.cumsum(BUDGETS[:-1]))
    ]
    + [("rep1-rung1", sum(BUDGETS), "suggest")]
    + [("rep2-rung1", sum(BUDGETS) * 2, "suggest")]
)
