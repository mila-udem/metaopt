# -*- coding: utf-8 -*-
"""
Tree-structured Parzen Estimator Approach
=========================================
"""
import logging

import numpy
from scipy.stats import norm

from orion.algo.base import BaseAlgorithm
from orion.core.utils.points import flatten_dims, regroup_dims

logger = logging.getLogger(__name__)


def compute_max_ei_point(points, below_likelis, above_likelis):
    """Compute ei among points based on their log likelihood and return the point with max ei.

    :param points: list of point with real values.
    :param below_likelis: list of log likelihood for each point in the good GMM.
    :param above_likelis: list of log likelihood for each point in the bad GMM.
    """
    max_ei = -numpy.inf
    point_index = 0
    for i, (lik_b, lik_a) in enumerate(zip(below_likelis, above_likelis)):
        ei = lik_b - lik_a
        if ei > max_ei:
            max_ei = ei
            point_index = i
    return points[point_index]


def ramp_up_weights(total_num, flat_num, equal_weight):
    """Adjust weights of observed trials.

    :param total_num: total number of observed trials.
    :param flat_num: the number of the most recent trials which
        get the full weight where the others will be applied with a linear ramp
        from 0 to 1.0. It will only take effect if equal_weight is False.
    :param equal_weight: whether all the observed trails share the same weights.
    """
    if total_num < flat_num or equal_weight:
        return numpy.ones(total_num)

    ramp_weights = numpy.linspace(1.0 / total_num, 1.0, num=total_num - flat_num)
    flat_weights = numpy.ones(flat_num)
    return numpy.concatenate([ramp_weights, flat_weights])


# pylint:disable=assignment-from-no-return
def adaptive_parzen_estimator(
    mus, low, high, prior_weight=1.0, equal_weight=False, flat_num=25
):
    """Return the sorted mus, the corresponding sigmas and weights with adaptive kernel estimator.

    This adaptive parzen window estimator is based on the original papers and also refer the use of
    prior mean in `this implementation
    <https://github.com/hyperopt/hyperopt/blob/40bd617a0ca47e09368fb65919464ba0bfa85962/hyperopt/tpe.py#L400-L468>`_.

    :param mus: list of real values for observed mus.
    :param low: real value for lower bound of points.
    :param high: real value for upper bound of points.
    :param prior_weight: real value for the weight of the prior mean.
    :param equal_weight: bool value indicating if all points with equal weights.
    :param flat_num: int value indicating the number of the most recent trials which
        get the full weight where the others will be applied with a linear ramp
        from 0 to 1.0. It will only take effect if equal_weight is False.
    """
    mus = numpy.asarray(mus)

    prior_mu = (low + high) * 0.5
    prior_sigma = (high - low) * 1.0

    size = len(mus)
    if size > 1:
        order = numpy.argsort(mus)
        sorted_mus = mus[order]
        prior_mu_pos = numpy.searchsorted(sorted_mus, prior_mu)

        weights = ramp_up_weights(size, flat_num, equal_weight)

        mixture_mus = numpy.zeros(size + 1)
        mixture_mus[:prior_mu_pos] = sorted_mus[:prior_mu_pos]
        mixture_mus[prior_mu_pos] = prior_mu
        mixture_mus[prior_mu_pos + 1 :] = sorted_mus[prior_mu_pos:]

        mixture_weights = numpy.ones(size + 1)
        mixture_weights[:prior_mu_pos] = weights[:prior_mu_pos]
        mixture_weights[prior_mu_pos] = prior_weight
        mixture_weights[prior_mu_pos + 1 :] = weights[prior_mu_pos:]

        sigmas = numpy.ones(size + 1)
        sigmas[0] = mixture_mus[1] - mixture_mus[0]
        sigmas[-1] = mixture_mus[-1] - mixture_mus[-2]
        sigmas[1:-1] = numpy.maximum(
            (mixture_mus[1:-1] - mixture_mus[0:-2]),
            (mixture_mus[2:] - mixture_mus[1:-1]),
        )
        sigmas = numpy.clip(
            sigmas, prior_sigma / max(10, numpy.sqrt(size)), prior_sigma
        )

    else:
        if prior_mu < mus[0]:

            mixture_mus = numpy.array([prior_mu, mus[0]])
            sigmas = numpy.array([prior_sigma, prior_sigma * 0.5])
            mixture_weights = numpy.array([prior_weight, 1.0])
        else:
            mixture_mus = numpy.array([mus[0], prior_mu])
            sigmas = numpy.array([prior_sigma * 0.5, prior_sigma])
            mixture_weights = numpy.array([1.0, prior_weight])

    weights = mixture_weights / mixture_weights.sum()

    return mixture_mus, sigmas, weights


class TPE(BaseAlgorithm):
    """Tree-structured Parzen Estimator (TPE) algorithm is one of Sequential Model-Based
    Global Optimization (SMBO) algorithms, which will build models to propose new points based
    on the historical observed trials.

    Instead of modeling p(y|x) like other SMBO algorithms, TPE models p(x|y) and p(y),
    and p(x|y) is modeled by transforming that generative process, replacing the distributions of
    the configuration prior with non-parametric densities.

    The TPE defines p(x|y) using two such densities l(x) and g(x) while l(x) is distribution of
    good points and g(x) is the distribution of bad points. New point candidates will be sampled
    with l(x) and Expected Improvement (EI) optimization scheme will be used to find the most
    promising point among the candidates.

    For more information on the algorithm, see original papers at:

    - `Algorithms for Hyper-Parameter Optimization
      <https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf>`_
    - `Making a Science of Model Search: Hyperparameter Optimizationin Hundreds of Dimensions
      for Vision Architectures <http://proceedings.mlr.press/v28/bergstra13.pdf>`_

    Parameters
    ----------
    space: `orion.algo.space.Space`
        Optimisation space with priors for each dimension.
    seed: None, int or sequence of int
        Seed to sample initial points and candidates points.
        Default: ``None``
    n_initial_points: int
        Number of initial points randomly sampled. If new points
        are requested and less than `n_initial_points` are observed,
        the next points will also be sampled randomly instead of being
        sampled from the parzen estimators.
        Default: ``20``
    n_ei_candidates: int
        Number of candidates points sampled for ei compute.
        Default: ``24``
    gamma: real
        Ratio to split the observed trials into good and bad distributions.
        Default: ``0.25``
    equal_weight: bool
        True to set equal weights for observed points.
        Default: ``False``
    prior_weight: int
        The weight given to the prior point of the input space.
        Default: ``1.0``
    full_weight_num: int
        The number of the most recent trials which get the full weight where the others will be
        applied with a linear ramp from 0 to 1.0. It will only take effect if equal_weight
        is False.

    """

    requires_type = None
    requires_dist = "linear"
    requires_shape = "flattened"

    # pylint:disable=too-many-arguments
    def __init__(
        self,
        space,
        seed=None,
        n_initial_points=20,
        n_ei_candidates=24,
        gamma=0.25,
        equal_weight=False,
        prior_weight=1.0,
        full_weight_num=25,
    ):

        if n_initial_points < 2:
            n_initial_points = 2
            logger.warning(
                "n_initial_points %s is not valid, set n_initial_points = 2",
                str(n_initial_points),
            )

        if n_ei_candidates < 1:
            n_ei_candidates = 1
            logger.warning(
                "n_ei_candidates %s is not valid, set n_ei_candidates = 1",
                str(n_ei_candidates),
            )

        super(TPE, self).__init__(
            space,
            seed=seed,
            n_initial_points=n_initial_points,
            n_ei_candidates=n_ei_candidates,
            gamma=gamma,
            equal_weight=equal_weight,
            prior_weight=prior_weight,
            full_weight_num=full_weight_num,
        )

    @property
    def space(self):
        """Return transformed space of TPE"""
        return self._space

    @space.setter
    def space(self, space):
        """Set the space of TPE and initialize it"""
        self._space = space
        self._initialize()

    def _initialize(self):
        """Initialize TPE once the space is transformed"""

        for dimension in self.space.values():

            if dimension.type != "fidelity" and dimension.prior_name not in [
                "uniform",
                "reciprocal",
                "int_uniform",
                "int_reciprocal",
                "choices",
            ]:
                raise ValueError(
                    "TPE now only supports uniform, loguniform, uniform discrete "
                    f"and choices as prior: {dimension.prior_name}"
                )

            shape = dimension.shape
            if shape and len(shape) != 1:
                raise ValueError("TPE now only supports 1D shape.")

    def seed_rng(self, seed):
        """Seed the state of the random number generator.

        :param seed: Integer seed for the random number generator.
        """
        self.rng = numpy.random.RandomState(seed)

    @property
    def state_dict(self):
        """Return a state dict that can be used to reset the state of the algorithm."""
        _state_dict = super(TPE, self).state_dict

        _state_dict["rng_state"] = self.rng.get_state()
        _state_dict["seed"] = self.seed
        return _state_dict

    def set_state(self, state_dict):
        """Reset the state of the algorithm based on the given state_dict

        :param state_dict: Dictionary representing state of an algorithm
        """
        super(TPE, self).set_state(state_dict)

        self.seed_rng(state_dict["seed"])
        self.rng.set_state(state_dict["rng_state"])

    def _n_suggested(self):
        return len(self._trials_info)

    def _n_observed(self):
        return sum(bool(point[1] is not None) for point in self._trials_info.values())

    def suggest(self, num=None):
        """Suggest a `num` of new sets of parameters. Randomly draw samples
        from the import space and return them.

        Parameters
        ----------
        num: int, optional
            Number of points to sample. If None, TPE will sample all random points at once, or a
            single point if it is at the Bayesian Optimization stage.
        :param num: how many sets to be suggested.

        .. note:: New parameters must be compliant with the problem's domain
           `orion.algo.space.Space`.
        """
        if num is None:
            num = max(self.n_initial_points - self._n_observed(), 1)
        samples = []
        candidates = []
        while len(samples) < num and self.n_suggested < self.space.cardinality:
            if candidates:
                candidate = candidates.pop(0)
                if candidate:
                    self._trials_info[self.get_id(candidate)] = (candidate, None)
                    samples.append(candidate)
            elif self._n_observed() < self.n_initial_points:
                candidates = self._suggest_random(num)
            else:
                candidates = self._suggest_bo(max(num - len(samples), 0))

            if not candidates:
                break

        if samples:
            return samples

        return None

    def _suggest(self, num, function):
        points = []

        ids = set(self._trials_info.keys())
        while len(points) < num:
            for candidate in function(num - len(points)):
                candidate_id = self.get_id(candidate)
                if candidate_id not in ids:
                    ids.add(candidate_id)
                    points.append(candidate)

                if len(ids) >= self.space.cardinality:
                    return points

        return points

    def _suggest_random(self, num):
        def sample(num):
            return self.space.sample(
                num, seed=tuple(self.rng.randint(0, 1000000, size=3))
            )

        return self._suggest(num, sample)

    def _suggest_bo(self, num):
        def suggest_bo(num):
            return [self._suggest_one_bo() for _ in range(num)]

        return self._suggest(num, suggest_bo)

    def _suggest_one_bo(self):

        point = []
        below_points, above_points = self.split_trials()

        below_points = list(map(list, zip(*below_points)))
        above_points = list(map(list, zip(*above_points)))

        idx = 0
        for i, dimension in enumerate(self.space.values()):

            shape = dimension.shape
            if not shape:
                shape = (1,)

            if dimension.type == "real":
                dim_samples = self._sample_real_dimension(
                    dimension,
                    shape[0],
                    below_points[idx : idx + shape[0]],
                    above_points[idx : idx + shape[0]],
                )
            elif dimension.type == "integer" and dimension.prior_name in [
                "int_uniform",
                "int_reciprocal",
            ]:
                dim_samples = self.sample_one_dimension(
                    dimension,
                    shape[0],
                    below_points[idx : idx + shape[0]],
                    above_points[idx : idx + shape[0]],
                    self._sample_int_point,
                )
            elif dimension.type == "categorical" and dimension.prior_name == "choices":
                dim_samples = self.sample_one_dimension(
                    dimension,
                    shape[0],
                    below_points[idx : idx + shape[0]],
                    above_points[idx : idx + shape[0]],
                    self._sample_categorical_point,
                )
            elif dimension.type == "fidelity":
                # fidelity dimension
                dim_samples = [point[i] for point in self.space.sample(1)]
            else:
                raise NotImplementedError()

            idx += shape[0]
            point += dim_samples

        return self.format_point(point)

    # pylint:disable=no-self-use
    def sample_one_dimension(
        self, dimension, shape_size, below_points, above_points, sampler
    ):
        """Sample values for a dimension

        :param dimension: Dimension.
        :param shape_size: 1D Shape Size of the Real Dimension.
        :param below_points: good points with shape (m, n), m=shape_size.
        :param above_points: bad points with shape (m, n), m=shape_size.
        :param sampler: method to sample one value for upon the dimension.
        """
        points = []

        for j in range(shape_size):
            new_point = sampler(dimension, below_points[j], above_points[j])
            if new_point is not None:
                points.append(new_point)

        return points

    def _sample_real_dimension(self, dimension, shape_size, below_points, above_points):
        """Sample values for real dimension"""
        if dimension.prior_name in ["uniform", "reciprocal"]:
            return self.sample_one_dimension(
                dimension,
                shape_size,
                below_points,
                above_points,
                self._sample_real_point,
            )
        else:
            raise NotImplementedError()

    def _sample_loguniform_real_point(self, dimension, below_points, above_points):
        """Sample one value for real dimension in a loguniform way"""
        return self._sample_real_point(
            dimension, below_points, above_points, is_log=True
        )

    def _sample_real_point(self, dimension, below_points, above_points, is_log=False):
        """Sample one value for real dimension based on the observed good and bad points"""
        low, high = dimension.interval()
        if is_log:
            low = numpy.log(low)
            high = numpy.log(high)
            below_points = numpy.log(below_points)
            above_points = numpy.log(above_points)

        below_mus, below_sigmas, below_weights = adaptive_parzen_estimator(
            below_points,
            low,
            high,
            self.prior_weight,
            self.equal_weight,
            flat_num=self.full_weight_num,
        )
        above_mus, above_sigmas, above_weights = adaptive_parzen_estimator(
            above_points,
            low,
            high,
            self.prior_weight,
            self.equal_weight,
            flat_num=self.full_weight_num,
        )

        gmm_sampler_below = GMMSampler(
            self, below_mus, below_sigmas, low, high, below_weights
        )
        gmm_sampler_above = GMMSampler(
            self, above_mus, above_sigmas, low, high, above_weights
        )

        candidate_points = gmm_sampler_below.sample(self.n_ei_candidates)
        if candidate_points:
            lik_blow = gmm_sampler_below.get_loglikelis(candidate_points)
            lik_above = gmm_sampler_above.get_loglikelis(candidate_points)
            new_point = compute_max_ei_point(candidate_points, lik_blow, lik_above)

            if is_log:
                new_point = numpy.exp(new_point)

            return new_point

        return None

    def _sample_int_point(self, dimension, below_points, above_points):
        """Sample one value for integer dimension based on the observed good and bad points"""
        low, high = dimension.interval()
        choices = range(low, high)

        below_points = numpy.array(below_points).astype(int) - low
        above_points = numpy.array(above_points).astype(int) - low

        sampler_below = CategoricalSampler(self, below_points, choices)
        candidate_points = sampler_below.sample(self.n_ei_candidates)

        if list(candidate_points):
            sampler_above = CategoricalSampler(self, above_points, choices)

            lik_below = sampler_below.get_loglikelis(candidate_points)
            lik_above = sampler_above.get_loglikelis(candidate_points)

            new_point = compute_max_ei_point(candidate_points, lik_below, lik_above)
            new_point = new_point + low
            return new_point

        return None

    def _sample_categorical_point(self, dimension, below_points, above_points):
        """Sample one value for categorical dimension based on the observed good and bad points"""
        choices = dimension.interval()

        below_points = [choices.index(point) for point in below_points]
        above_points = [choices.index(point) for point in above_points]

        sampler_below = CategoricalSampler(self, below_points, choices)
        candidate_points = sampler_below.sample(self.n_ei_candidates)

        if list(candidate_points):
            sampler_above = CategoricalSampler(self, above_points, choices)

            lik_below = sampler_below.get_loglikelis(candidate_points)
            lik_above = sampler_above.get_loglikelis(candidate_points)

            new_point_index = compute_max_ei_point(
                candidate_points, lik_below, lik_above
            )
            new_point = choices[new_point_index]

            return new_point
        return None

    def split_trials(self):
        """Split the observed trials into good and bad ones based on the ratio `gamma``"""
        sorted_trials = sorted(
            (point for point in self._trials_info.values() if point[1] is not None),
            key=lambda x: x[1]["objective"],
        )
        sorted_points = [list(points) for points, results in sorted_trials]

        split_index = int(numpy.ceil(self.gamma * len(sorted_points)))

        below = sorted_points[:split_index]
        above = sorted_points[split_index:]

        return below, above

    def observe(self, points, results):
        """Observe evaluation `results` corresponding to list of `points` in
        space.

        A simple random sampler though does not take anything into account.
        """
        super(TPE, self).observe(points, results)


class GMMSampler:
    """Gaussian Mixture Model Sampler for TPE algorithm

    Parameters
    ----------
    tpe: `TPE` algorithm
        The tpe algorithm object which this sampler will be part of.
    mus: list
        mus for each Gaussian components in the GMM.
        Default: ``None``
    sigmas: list
        sigmas for each Gaussian components in the GMM.
    low: real
        Lower bound of the sampled points.
    high: real
        Upper bound of the sampled points.
    weights: list
        Weights for each Gaussian components in the GMM
        Default: ``None``

    """

    def __init__(self, tpe, mus, sigmas, low, high, weights=None):
        self.tpe = tpe

        self.mus = mus
        self.sigmas = sigmas
        self.low = low
        self.high = high
        self.weights = weights if weights is not None else len(mus) * [1.0 / len(mus)]

        self.pdfs = []
        self._build_mixture()

    def _build_mixture(self):
        """Build the Gaussian components in the GMM"""
        for mu, sigma in zip(self.mus, self.sigmas):
            self.pdfs.append(norm(mu, sigma))

    def sample(self, num=1):
        """Sample required number of points"""
        point = []
        for _ in range(num):
            pdf = numpy.argmax(self.tpe.rng.multinomial(1, self.weights))
            new_points = self.pdfs[pdf].rvs(size=5, random_state=self.tpe.rng)
            for pt in new_points:
                if self.low <= pt < self.high:
                    point.append(pt)
                    break

        return point

    def get_loglikelis(self, points):
        """Return the log likelihood for the points"""
        points = numpy.array(points)
        weight_likelis = [
            numpy.log(self.weights[i] * pdf.pdf(points))
            for i, pdf in enumerate(self.pdfs)
        ]
        weight_likelis = numpy.array(weight_likelis)
        # (num_weights, num_points) => (num_points, num_weights)
        weight_likelis = weight_likelis.transpose()

        # log-sum-exp trick
        max_likeli = numpy.nanmax(weight_likelis, axis=1)
        point_likeli = max_likeli + numpy.log(
            numpy.nansum(numpy.exp(weight_likelis - max_likeli[:, None]), axis=1)
        )

        return point_likeli


class CategoricalSampler:
    """Categorical Sampler for discrete integer and categorical choices

    Parameters
    ----------
    tpe: `TPE` algorithm
        The tpe algorithm object which this sampler will be part of.
    observations: list
        Observed values in the dimension
    choices: list
        Candidate values for the dimension

    """

    def __init__(self, tpe, observations, choices):
        self.tpe = tpe
        self.obs = observations
        self.choices = choices

        self._build_multinomial_weights()

    def _build_multinomial_weights(self):
        """Build weights for categorical distribution based on observations"""
        weights_obs = ramp_up_weights(
            len(self.obs), self.tpe.full_weight_num, self.tpe.equal_weight
        )
        counts_obs = numpy.bincount(
            self.obs, minlength=len(self.choices), weights=weights_obs
        )
        counts_obs = counts_obs + self.tpe.prior_weight
        self.weights = counts_obs / counts_obs.sum()

    def sample(self, num=1):
        """Sample required number of points"""
        samples = self.tpe.rng.multinomial(n=1, pvals=self.weights, size=num)

        assert samples.shape == (num,) + (len(self.weights),)

        samples_index = samples.argmax(-1)
        assert samples_index.shape == (num,)

        return samples_index

    def get_loglikelis(self, points):
        """Return the log likelihood for the points"""
        return numpy.log(numpy.asarray(self.weights)[points])
