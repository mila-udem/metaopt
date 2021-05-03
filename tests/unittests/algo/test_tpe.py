#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Tests for :mod:`orion.algo.tpe`."""
import itertools

import numpy
import pytest
from scipy.stats import norm

from orion.algo.space import Categorical, Fidelity, Integer, Real, Space
from orion.algo.tpe import (
    TPE,
    CategoricalSampler,
    GMMSampler,
    adaptive_parzen_estimator,
    compute_max_ei_point,
    ramp_up_weights,
)
from orion.core.worker.transformer import build_required_space
from orion.testing.algo import BaseAlgoTests


@pytest.fixture()
def space():
    """Return an optimization space"""
    space = Space()

    dim1 = Real("yolo1", "uniform", -10, 20)
    space.register(dim1)

    dim2 = Integer("yolo2", "uniform", -5, 10)
    space.register(dim2)

    categories = ["a", 0.1, 2, "c"]
    dim3 = Categorical("yolo3", categories)
    space.register(dim3)

    return space


@pytest.fixture
def tpe(space):
    """Return an instance of TPE."""
    return TPE(space, seed=1)


def test_compute_max_ei_point():
    """Test that max ei point is computed correctly"""
    points = numpy.linspace(-3, 3, num=10)
    below_likelis = numpy.linspace(0.5, 0.9, num=10)
    above_likes = numpy.linspace(0.2, 0.5, num=10)

    numpy.random.shuffle(below_likelis)
    numpy.random.shuffle(above_likes)
    max_ei_index = (below_likelis - above_likes).argmax()

    max_ei_point = compute_max_ei_point(points, below_likelis, above_likes)
    assert max_ei_point == points[max_ei_index]


def test_ramp_up_weights():
    """Test TPE adjust observed points correctly"""
    weights = ramp_up_weights(25, 15, True)
    assert len(weights) == 25
    assert numpy.all(weights == 1.0)

    weights = ramp_up_weights(25, 15, False)
    assert len(weights) == 25
    assert numpy.all(weights[:10] == (numpy.linspace(1.0 / 25, 1.0, num=10)))
    assert numpy.all(weights[10:] == 1.0)

    weights = ramp_up_weights(10, 15, False)
    assert len(weights) == 10
    assert numpy.all(weights == 1.0)

    weights = ramp_up_weights(25, 0, False)
    assert len(weights) == 25
    assert numpy.all(weights == (numpy.linspace(1.0 / 25, 1.0, num=25)))


def test_adaptive_parzen_normal_estimator():
    """Test adaptive parzen estimator"""
    low = -1
    high = 5

    obs_mus = [1.2]
    mus, sigmas, weights = adaptive_parzen_estimator(
        obs_mus, low, high, prior_weight=1.0, equal_weight=False, flat_num=25
    )
    assert list(mus) == [1.2, 2]
    assert list(sigmas) == [3, 6]
    assert list(weights) == [1.0 / 2, 1.0 / 2]

    obs_mus = [3.4]
    mus, sigmas, weights = adaptive_parzen_estimator(
        obs_mus, low, high, prior_weight=0.5, equal_weight=False, flat_num=25
    )
    assert list(mus) == [2, 3.4]
    assert list(sigmas) == [6, 3]
    assert list(weights) == [0.5 / 1.5, 1.0 / 1.5]

    obs_mus = numpy.linspace(-1, 5, num=30, endpoint=False)
    mus, sigmas, weights = adaptive_parzen_estimator(
        obs_mus, low, high, prior_weight=1.0, equal_weight=False, flat_num=25
    )

    ramp = numpy.linspace(1.0 / 30, 1.0, num=30 - 25)
    full = numpy.ones(25 + 1)
    all_weights = numpy.concatenate([ramp, full])

    assert len(mus) == len(sigmas) == len(weights) == 30 + 1
    assert numpy.all(weights[: 30 - 25] == ramp / all_weights.sum())
    assert numpy.all(weights[30 - 25 :] == 1 / all_weights.sum())
    assert numpy.all(sigmas == 6 / 10)


def test_adaptive_parzen_normal_estimator_weight():
    """Test the weight for the normal components"""
    obs_mus = numpy.linspace(-1, 5, num=30, endpoint=False)
    low = -1
    high = 5

    # equal weight
    mus, sigmas, weights = adaptive_parzen_estimator(
        obs_mus, low, high, prior_weight=1.0, equal_weight=True, flat_num=25
    )
    assert numpy.all(weights == 1 / 31)
    assert numpy.all(sigmas == 6 / 10)

    # prior weight
    mus, sigmas, weights = adaptive_parzen_estimator(
        obs_mus, low, high, prior_weight=0.5, equal_weight=False, flat_num=25
    )

    ramp = numpy.linspace(1.0 / 30, 1.0, num=30 - 25)
    full = numpy.ones(25 + 1)
    all_weights = numpy.concatenate([ramp, full])
    prior_pos = numpy.searchsorted(mus, 2)
    all_weights[prior_pos] = 0.5

    assert numpy.all(
        weights[: 30 - 25]
        == (numpy.linspace(1.0 / 30, 1.0, num=30 - 25) / all_weights.sum())
    )
    assert numpy.all(weights[33 - 25 : prior_pos] == 1 / all_weights.sum())
    assert weights[prior_pos] == 0.5 / all_weights.sum()
    assert numpy.all(weights[prior_pos + 1 :] == 1 / all_weights.sum())
    assert numpy.all(sigmas == 6 / 10)

    # full weights number
    mus, sigmas, weights = adaptive_parzen_estimator(
        obs_mus, low, high, prior_weight=1.0, equal_weight=False, flat_num=15
    )

    ramp = numpy.linspace(1.0 / 30, 1.0, num=30 - 15)
    full = numpy.ones(15 + 1)
    all_weights = numpy.concatenate([ramp, full])
    prior_pos = numpy.searchsorted(mus, 2)
    all_weights[prior_pos] = 1.0

    assert numpy.all(
        weights[: 30 - 15]
        == (numpy.linspace(1.0 / 30, 1.0, num=30 - 15) / all_weights.sum())
    )
    assert numpy.all(weights[30 - 15 :] == 1 / all_weights.sum())
    assert numpy.all(sigmas == 6 / 10)


def test_adaptive_parzen_normal_estimator_sigma_clip():
    """Test that the magic clip of sigmas for parzen estimator"""
    low = -1
    high = 5

    obs_mus = numpy.linspace(-1, 5, num=8, endpoint=False)
    mus, sigmas, weights = adaptive_parzen_estimator(
        obs_mus, low, high, prior_weight=1.0, equal_weight=False, flat_num=25
    )
    assert len(mus) == len(sigmas) == len(weights) == 8 + 1
    assert numpy.all(weights == 1 / 9)
    assert numpy.all(sigmas == 6 / 8)

    obs_mus = numpy.random.uniform(-1, 5, 30)
    mus, sigmas, weights = adaptive_parzen_estimator(
        obs_mus, low, high, prior_weight=1.0, equal_weight=False, flat_num=25
    )

    assert len(mus) == len(sigmas) == len(weights) == 30 + 1
    assert numpy.all(weights[-25:] == weights[-1])
    assert numpy.all(sigmas <= 6) and numpy.all(sigmas >= 6 / 10)

    obs_mus = numpy.random.uniform(-1, 5, 400)
    mus, sigmas, weights = adaptive_parzen_estimator(
        obs_mus, low, high, prior_weight=1.0, equal_weight=False, flat_num=25
    )

    assert len(mus) == len(sigmas) == len(weights) == 400 + 1
    assert numpy.all(weights[-25:] == weights[-1])
    assert numpy.all(sigmas <= 6) and numpy.all(sigmas >= 6 / 20)

    obs_mus = numpy.random.uniform(-1, 5, 10000)
    mus, sigmas, weights = adaptive_parzen_estimator(
        obs_mus, low, high, prior_weight=1.0, equal_weight=False, flat_num=25
    )

    assert len(mus) == len(sigmas) == len(weights) == 10000 + 1
    assert numpy.all(weights[-25:] == weights[-1])
    assert numpy.all(sigmas <= 6) and numpy.all(sigmas >= 6 / 100)


class TestCategoricalSampler:
    """Tests for TPE Categorical Sampler"""

    def test_cat_sampler_creation(self, tpe):
        """Test CategoricalSampler creation"""
        obs = [0, 3, 9]
        choices = list(range(-5, 5))
        cat_sampler = CategoricalSampler(tpe, obs, choices)
        assert len(cat_sampler.weights) == len(choices)

        obs = [0, 3, 9]
        choices = ["a", "b", 11, 15, 17, 18, 19, 20, 25, "c"]
        cat_sampler = CategoricalSampler(tpe, obs, choices)

        assert len(cat_sampler.weights) == len(choices)

        tpe.equal_weight = True
        tpe.prior_weight = 1.0
        obs = numpy.random.randint(0, 10, 100)
        cat_sampler = CategoricalSampler(tpe, obs, choices)
        counts_obs = numpy.bincount(obs) + 1.0
        weights = counts_obs / counts_obs.sum()

        assert numpy.all(cat_sampler.weights == weights)

        tpe.equal_weight = False
        tpe.prior_weight = 0.5
        tpe.full_weight_num = 30
        obs = numpy.random.randint(0, 10, 100)

        cat_sampler = CategoricalSampler(tpe, obs, choices)

        ramp = numpy.linspace(1.0 / 100, 1.0, num=100 - 30)
        full = numpy.ones(30)
        ramp_weights = numpy.concatenate([ramp, full])

        counts_obs = numpy.bincount(obs, weights=ramp_weights) + 0.5
        weights = counts_obs / counts_obs.sum()

        assert numpy.all(cat_sampler.weights == weights)

    def test_sample(self, tpe):
        """Test CategoricalSampler sample function"""
        obs = numpy.random.randint(0, 10, 100)
        choices = ["a", "b", 11, 15, 17, 18, 19, 20, 25, "c"]
        cat_sampler = CategoricalSampler(tpe, obs, choices)

        points = cat_sampler.sample(25)

        assert len(points) == 25
        assert numpy.all(points >= 0)
        assert numpy.all(points < 10)

        weights = numpy.linspace(1, 10, num=10) ** 3
        numpy.random.shuffle(weights)
        weights = weights / weights.sum()
        cat_sampler = CategoricalSampler(tpe, obs, choices)
        cat_sampler.weights = weights

        points = cat_sampler.sample(10000)
        points = numpy.array(points)
        hist = numpy.bincount(points)

        assert numpy.all(hist.argsort() == weights.argsort())
        assert len(points) == 10000
        assert numpy.all(points >= 0)
        assert numpy.all(points < 10)

    def test_get_loglikelis(self, tpe):
        """Test to get log likelis of points"""
        obs = numpy.random.randint(0, 10, 100)
        choices = ["a", "b", 11, 15, 17, 18, 19, 20, 25, "c"]
        cat_sampler = CategoricalSampler(tpe, obs, choices)

        points = cat_sampler.sample(25)

        likelis = cat_sampler.get_loglikelis(points)

        assert numpy.all(
            likelis == numpy.log(numpy.asarray(cat_sampler.weights)[points])
        )


class TestGMMSampler:
    """Tests for TPE GMM Sampler"""

    def test_gmm_sampler_creation(self, tpe):
        """Test GMMSampler creation"""
        mus = numpy.linspace(-3, 3, num=12, endpoint=False)
        sigmas = [0.5] * 12

        gmm_sampler = GMMSampler(tpe, mus, sigmas, -3, 3)

        assert len(gmm_sampler.weights) == 12
        assert len(gmm_sampler.pdfs) == 12

    def test_sample(self, tpe):
        """Test GMMSampler sample function"""
        mus = numpy.linspace(-3, 3, num=12, endpoint=False)
        sigmas = [0.5] * 12

        gmm_sampler = GMMSampler(tpe, mus, sigmas, -3, 3)
        points = gmm_sampler.sample(25)
        points = numpy.array(points)

        assert len(points) <= 25
        assert numpy.all(points >= -3)
        assert numpy.all(points < 3)

        mus = numpy.linspace(-10, 10, num=10, endpoint=False)
        sigmas = [0.00001] * 10
        weights = numpy.linspace(1, 10, num=10) ** 3
        numpy.random.shuffle(weights)
        weights = weights / weights.sum()

        gmm_sampler = GMMSampler(tpe, mus, sigmas, -11, 9, weights)
        points = gmm_sampler.sample(10000)
        points = numpy.array(points)
        hist = numpy.histogram(points, bins=[-11, -9, -7, -5, -3, -1, 1, 3, 5, 7, 9])

        assert numpy.all(hist[0].argsort() == numpy.array(weights).argsort())
        assert numpy.all(points >= -11)
        assert numpy.all(points < 9)

    def test_get_loglikelis(self):
        """Test to get log likelis of points"""
        mus = numpy.linspace(-10, 10, num=10, endpoint=False)
        weights = numpy.linspace(1, 10, num=10) ** 3
        numpy.random.shuffle(weights)
        weights = weights / weights.sum()

        sigmas = [0.00001] * 10
        gmm_sampler = GMMSampler(tpe, mus, sigmas, -11, 9, weights)

        points = [mus[7]]
        pdf = norm(mus[7], sigmas[7])
        point_likeli = numpy.log(pdf.pdf(mus[7]) * weights[7])
        likelis = gmm_sampler.get_loglikelis(points)

        assert list(likelis) == point_likeli
        assert likelis[0] == point_likeli

        sigmas = [2] * 10
        gmm_sampler = GMMSampler(tpe, mus, sigmas, -11, 9, weights)

        log_pdf = []
        pdfs = []
        for i in range(10):
            pdfs.append(norm(mus[i], sigmas[i]))
        for pdf, weight in zip(pdfs, weights):
            log_pdf.append(numpy.log(pdf.pdf(0) * weight))
        point_likeli = numpy.log(numpy.sum(numpy.exp(log_pdf)))

        points = numpy.random.uniform(-11, 9, 30)
        points = numpy.insert(points, 10, 0)
        likelis = gmm_sampler.get_loglikelis(points)

        point_likeli = numpy.format_float_scientific(point_likeli, precision=10)
        gmm_likeli = numpy.format_float_scientific(likelis[10], precision=10)
        assert point_likeli == gmm_likeli
        assert len(likelis) == len(points)


class TestTPE:
    """Tests for the algo TPE."""

    def test_seed_rng(self, tpe):
        """Test that algo is seeded properly"""
        tpe.seed_rng(1)
        a = tpe.suggest(1)[0]
        assert not numpy.allclose(a, tpe.suggest(1)[0])

        tpe.seed_rng(1)
        assert numpy.allclose(a, tpe.suggest(1)[0])

    def test_set_state(self, tpe):
        """Test that state is reset properly"""
        tpe.seed_rng(1)
        state = tpe.state_dict
        a = tpe.suggest(1)[0]
        assert not numpy.allclose(a, tpe.suggest(1)[0])

        tpe.set_state(state)
        assert numpy.allclose(a, tpe.suggest(1)[0])

    def test_unsupported_space(self):
        """Test tpe only work for supported search space"""
        space = Space()
        dim1 = Real("yolo1", "uniform", -10, 10)
        space.register(dim1)
        dim2 = Real("yolo2", "reciprocal", 10, 20)
        space.register(dim2)
        categories = ["a", 0.1, 2, "c"]
        dim3 = Categorical("yolo3", categories)
        space.register(dim3)
        dim4 = Fidelity("epoch", 1, 9, 3)
        space.register(dim4)
        TPE(space)

        space = Space()
        dim = Real("yolo1", "norm", 0.9)
        space.register(dim)

        with pytest.raises(ValueError) as ex:
            tpe = TPE(space)
            tpe.space = build_required_space(
                space, shape_requirement=TPE.requires_shape
            )

        assert (
            "TPE now only supports uniform, loguniform, uniform discrete and choices"
            in str(ex.value)
        )

    def test_split_trials(self, tpe):
        """Test observed trials can be split based on TPE gamma"""
        space = Space()
        dim1 = Real("yolo1", "uniform", -3, 6)
        space.register(dim1)

        tpe.space = space

        points = numpy.linspace(-3, 3, num=10, endpoint=False)
        results = numpy.linspace(0, 1, num=10, endpoint=False)
        points_results = list(zip(points, results))
        numpy.random.shuffle(points_results)
        points, results = zip(*points_results)
        for point, result in zip(points, results):
            tpe.observe([[point]], [{"objective": result}])

        tpe.gamma = 0.25
        below_points, above_points = tpe.split_trials()

        assert below_points == [[-3.0], [-2.4], [-1.8]]
        assert len(above_points) == 7

        tpe.gamma = 0.2
        below_points, above_points = tpe.split_trials()

        assert below_points == [[-3.0], [-2.4]]
        assert len(above_points) == 8

    def test_sample_int_dimension(self):
        """Test sample values for a integer dimension"""
        space = Space()
        dim1 = Integer("yolo1", "uniform", -10, 20)
        space.register(dim1)

        dim2 = Integer("yolo2", "uniform", -5, 10, shape=(2))
        space.register(dim2)

        tpe = TPE(space)

        obs_points = numpy.random.randint(-10, 10, 100)
        below_points = [obs_points[:25]]
        above_points = [obs_points[25:]]
        points = tpe.sample_one_dimension(
            dim1, 1, below_points, above_points, tpe._sample_int_point
        )
        points = numpy.asarray(points)
        assert len(points) == 1
        assert all(points >= -10)
        assert all(points < 10)

        obs_points_below = numpy.random.randint(-10, 0, 25).reshape(1, 25)
        obs_points_above = numpy.random.randint(0, 10, 75).reshape(1, 75)
        points = tpe.sample_one_dimension(
            dim1, 1, obs_points_below, obs_points_above, tpe._sample_int_point
        )
        points = numpy.asarray(points)
        assert len(points) == 1
        assert all(points >= -10)
        assert all(points < 0)

        obs_points = numpy.random.randint(-5, 5, 100)
        below_points = [obs_points[:25], obs_points[25:50]]
        above_points = [obs_points[50:75], obs_points[75:]]
        points = tpe.sample_one_dimension(
            dim2, 2, below_points, above_points, tpe._sample_int_point
        )
        points = numpy.asarray(points)
        assert len(points) == 2
        assert all(points >= -10)
        assert all(points < 10)

        tpe.n_ei_candidates = 0
        points = tpe.sample_one_dimension(
            dim2, 2, below_points, above_points, tpe._sample_int_point
        )
        assert len(points) == 0

    def test_sample_categorical_dimension(self):
        """Test sample values for a categorical dimension"""
        space = Space()
        categories = ["a", "b", 11, 15, 17, 18, 19, 20, 25, "c"]
        dim1 = Categorical("yolo1", categories)
        space.register(dim1)
        dim2 = Categorical("yolo2", categories, shape=(2))
        space.register(dim2)

        tpe = TPE(space)

        obs_points = numpy.random.randint(0, 10, 100)
        obs_points = [categories[point] for point in obs_points]
        below_points = [obs_points[:25]]
        above_points = [obs_points[25:]]
        points = tpe.sample_one_dimension(
            dim1, 1, below_points, above_points, tpe._sample_categorical_point
        )
        assert len(points) == 1
        assert points[0] in categories

        obs_points_below = numpy.random.randint(0, 3, 25)
        obs_points_above = numpy.random.randint(3, 10, 75)
        below_points = [[categories[point] for point in obs_points_below]]
        above_points = [[categories[point] for point in obs_points_above]]
        points = tpe.sample_one_dimension(
            dim1, 1, below_points, above_points, tpe._sample_categorical_point
        )
        assert len(points) == 1
        assert points[0] in categories[:3]

        obs_points = numpy.random.randint(0, 10, 100)
        obs_points = [categories[point] for point in obs_points]
        below_points = [obs_points[:25], obs_points[25:50]]
        above_points = [obs_points[50:75], obs_points[75:]]

        points = tpe.sample_one_dimension(
            dim2, 2, below_points, above_points, tpe._sample_categorical_point
        )
        assert len(points) == 2
        assert points[0] in categories
        assert points[1] in categories

        tpe.n_ei_candidates = 0
        points = tpe.sample_one_dimension(
            dim2, 2, below_points, above_points, tpe._sample_categorical_point
        )
        assert len(points) == 0

    def test_sample_real_dimension(self):
        """Test sample values for a real dimension"""
        space = Space()
        dim1 = Real("yolo1", "uniform", -10, 20)
        space.register(dim1)
        dim2 = Real("yolo2", "uniform", -5, 10, shape=(2))
        space.register(dim2)
        dim3 = Real("yolo3", "reciprocal", 1, 20)
        space.register(dim3)

        tpe = TPE(space)
        points = numpy.random.uniform(-10, 10, 20)
        below_points = [points[:8]]
        above_points = [points[8:]]
        points = tpe._sample_real_dimension(dim1, 1, below_points, above_points)
        points = numpy.asarray(points)
        assert len(points) == 1
        assert all(points >= -10)
        assert all(points < 10)

        points = numpy.random.uniform(1, 20, 20)
        below_points = [points[:8]]
        above_points = [points[8:]]
        points = tpe._sample_real_dimension(dim3, 1, below_points, above_points)
        points = numpy.asarray(points)
        assert len(points) == 1
        assert all(points >= 1)
        assert all(points < 20)

        below_points = numpy.random.uniform(-10, 0, 25).reshape(1, 25)
        above_points = numpy.random.uniform(0, 10, 75).reshape(1, 75)
        points = tpe._sample_real_dimension(dim1, 1, below_points, above_points)
        points = numpy.asarray(points)
        assert len(points) == 1
        assert all(points >= -10)
        assert all(points < 0)

        points = numpy.random.uniform(-5, 5, 32)
        below_points = [points[:8], points[8:16]]
        above_points = [points[16:24], points[24:]]
        points = tpe._sample_real_dimension(dim2, 2, below_points, above_points)
        points = numpy.asarray(points)
        assert len(points) == 2
        assert all(points >= -10)
        assert all(points < 10)

        tpe.n_ei_candidates = 0
        points = tpe._sample_real_dimension(dim2, 2, below_points, above_points)
        assert len(points) == 0

    def test_suggest(self, tpe):
        """Test suggest with no shape dimensions"""
        tpe.n_initial_points = 10
        results = numpy.random.random(10)
        for i in range(10):
            point = tpe.suggest(1)
            assert len(point) == 1
            assert len(point[0]) == 3
            assert not isinstance(point[0][0], tuple)
            tpe.observe(point, [{"objective": results[i]}])

        point = tpe.suggest(1)
        assert len(point) == 1
        assert len(point[0]) == 3
        assert not isinstance(point[0][0], tuple)

    def test_1d_shape(self, tpe):
        """Test suggest with 1D shape dimensions"""
        space = Space()
        dim1 = Real("yolo1", "uniform", -3, 6, shape=(2))
        space.register(dim1)
        dim2 = Real("yolo2", "uniform", -2, 4)
        space.register(dim2)

        tpe.space = space

        tpe.n_initial_points = 10
        results = numpy.random.random(10)
        for i in range(10):
            point = tpe.suggest(1)
            assert len(point) == 1
            assert len(point[0]) == 2
            assert len(point[0][0]) == 2
            tpe.observe(point, [{"objective": results[i]}])

        point = tpe.suggest(1)
        assert len(point) == 1
        assert len(point[0]) == 2
        assert len(point[0][0]) == 2

    def test_suggest_initial_points(self, tpe, monkeypatch):
        """Test that initial points can be sampled correctly"""
        points = [(i, i - 6, "c") for i in range(1, 12)]

        global index
        index = 0

        def sample(num=1, seed=None):
            global index
            pts = points[index : index + num]
            index += num
            return pts

        monkeypatch.setattr(tpe.space, "sample", sample)

        tpe.n_initial_points = 10
        results = numpy.random.random(10)
        for i in range(1, 11):
            point = tpe.suggest(1)[0]
            assert point == (i, i - 6, "c")
            tpe.observe([point], [{"objective": results[i - 1]}])

        point = tpe.suggest(1)[0]
        assert point != (11, 5, "c")

    def test_suggest_ei_candidates(self, tpe):
        """Test suggest with no shape dimensions"""
        tpe.n_initial_points = 2
        tpe.n_ei_candidates = 0

        results = numpy.random.random(2)
        for i in range(2):
            point = tpe.suggest(1)
            assert len(point) == 1
            assert len(point[0]) == 3
            assert not isinstance(point[0][0], tuple)
            tpe.observe(point, [{"objective": results[i]}])

        point = tpe.suggest(1)
        assert not point

        tpe.n_ei_candidates = 24
        point = tpe.suggest(1)
        assert len(point) > 0


N_INIT = 10


class TestTPE(BaseAlgoTests):
    algo_name = "tpe"
    config = {
        "seed": 123456,
        "n_initial_points": N_INIT,
        "n_ei_candidates": 4,
        "gamma": 0.5,
        "equal_weight": True,
        "prior_weight": 0.8,
        "full_weight_num": 10,
    }

    def test_suggest(self, mocker, num, attr):
        algo = self.create_algo()
        spy = self.spy_phase(mocker, num, algo, attr)
        points = algo.suggest()
        if num == 0:
            assert len(points) == N_INIT
        else:
            assert len(points) == 1

    def test_int_data(self, mocker, num, attr):
        if num > 0:
            pytest.skip("See https://github.com/Epistimio/orion/issues/600")
        super(TestTPE, self).test_int_data(mocker, num, attr)

    def test_is_done_cardinality(self):
        # TODO: Support correctly loguniform(discrete=True)
        #       See https://github.com/Epistimio/orion/issues/566
        space = self.update_space(
            {
                "x": "uniform(0, 4, discrete=True)",
                "y": "choices(['a', 'b', 'c'])",
                "z": "uniform(1, 6, discrete=True)",
            }
        )
        space = self.create_space(space)
        assert space.cardinality == 5 * 3 * 6

        algo = self.create_algo(space=space)
        for i, (x, y, z) in enumerate(itertools.product(range(5), "abc", range(1, 7))):
            assert not algo.is_done
            n = algo.n_suggested
            algo.observe([[x, y, z]], [dict(objective=i)])
            assert algo.n_suggested == n + 1

        assert i + 1 == space.cardinality

        assert algo.is_done


TestTPE.set_phases([("random", 0, "space.sample"), ("bo", N_INIT + 1, "_suggest_bo")])
