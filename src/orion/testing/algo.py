# -*- coding: utf-8 -*-
"""Generic tests for Algorithms"""
import copy
import functools
import inspect
import itertools

import numpy
import pytest

import orion.algo.base
from orion.algo.asha import ASHA
from orion.algo.gridsearch import GridSearch
from orion.algo.hyperband import Hyperband
from orion.algo.random import Random
from orion.algo.tpe import TPE
from orion.benchmark.task.branin import Branin
from orion.core.io.space_builder import SpaceBuilder
from orion.core.worker.primary_algo import PrimaryAlgo
from orion.testing.space import build_space

algorithms = {
    "evolutionaryes": "what is the config?",
}


def recursive_getattr(obj, attribute):
    """Get attribute recursively based on str with pattern sub.(sub.)*name"""
    attributes = attribute.split(".")

    attribute = getattr(obj, attributes[0])
    if len(attributes) == 1:
        return attribute

    return recursive_getattr(attribute, ".".join(attributes[1:]))


def spy_attr(mocker, algo, attribute):
    """Return a mocker.spy object on the algorithms's given `attribute`"""
    attributes = attribute.split(".")
    attr_to_mock = attributes[-1]
    if len(attributes) > 1:
        obj = recursive_getattr(algo, ".".join(attributes[:-1]))
    else:
        obj = algo
    return mocker.spy(obj, attr_to_mock)


methods_with_phase = set()


def phase(method):
    """Decorator to mark methods that must be parametrized with phases."""
    methods_with_phase.add(method.__name__)
    method.__doc__ += (
        "\n\nThis test is parametrizable with phases. See "
        "``orion.testing.algo.BaseAlgoTests.set_phases``."
    )

    return method


def parametrize_this(cls, method_name, attrs, ids):
    """Parametrize a method with phases.

    Notes
    -----
    We need to replace the method to avoid parametrizing the base one.
    This is a fugly hack because of pytest limitations with inheritance for test classes.
    """

    base_method = getattr(cls, method_name)

    def method(self, mocker, num, attr):
        print(method_name, base_method)
        return base_method(self, mocker, num, attr)

    setattr(cls, method_name, method)

    pytest.mark.parametrize("num,attr", attrs, ids=ids)(method)


def customized_mutate_example(search_space, rng, old_value, **kwargs):
    """Define a customized mutate function example"""
    multiply_factor = kwargs.pop("multiply_factor", 3.0)
    add_factor = kwargs.pop("add_factor", 1)
    if search_space.type == "real":
        new_value = old_value / multiply_factor
    elif search_space.type == "integer":
        new_value = int(old_value + add_factor)
    else:
        new_value = old_value
    return new_value


class BaseAlgoTests:
    """Generic Test-suite for HPO algorithms.

    This test-suite covers all typical cases for HPO algorithms. To use it for a new algorithm,
    the class inheriting from this one must redefine the attributes ``algo_name`` with
    the name of the algorithm used to create it with the algorithm factory
    ``orion.core.worker.primary_algo.PrimaryAlgo`` and ``config`` with a base configuration for the
    algorithm that contains all its arguments. The base space can be redefine if needed
    with the attribute ``space``.

    Many algorithms have different phases that should be tested. For instance
    TPE have a first phase of random search and a second of Bayesian Optimization.
    The random search and Bayesian optimization are different implementations and both should be
    tested. For this reason, the class method ``orion.testing.algo.BaseAlgoTests.set_phases`` must
    be called to parametrize the tests with phases. Failure to doing so will causes the tests to
    crash. See ``tests/unittests/algo/test_tpe.py`` for an example.
    """

    algo_name = None
    config = {}
    space = {"x": "uniform(0, 1)", "y": "uniform(0, 1)"}

    @classmethod
    def set_phases(cls, phases):
        """Parametrize the tests with different phases.

        Some algorithms have different phases that should be tested. For instance
        TPE have a first phase of random search and a second of Bayesian Optimization.
        The random search and Bayesian optimization are different implementations and both should be
        tested.

        Parameters
        ----------
        phases: list of tuples
            The different phases to test. The format of the tuples should be
            (str(id of the test),
            int(number of trials before the phase begins),
            str(name of the algorithm's attribute to spy (ex: "space.sample"))
            )
        """
        ids = [phase[0] for phase in phases]
        attrs = [phase[1:] for phase in phases]

        for method_name in sorted(methods_with_phase):
            parametrize_this(cls, method_name, attrs, ids)

    def create_algo(self, config=None, space=None, **kwargs):
        """Create the algorithm based on config.

        Parameters
        ----------
        config: dict, optional
            The configuration for the algorithm. ``self.config`` will be used
            if ``config`` is ``None``.
        space: ``orion.algo.space.Space``, optional
            Space object to pass to algo. The output of ``self.create_space()``
            will be used if ``space`` is ``None``.
        kwargs: dict
            Values to override algorithm configuration.
        """
        config = copy.deepcopy(config or self.config)
        config.update(kwargs)
        return PrimaryAlgo(
            space or self.create_space(),
            {self.algo_name: config},
        )

    def update_space(self, test_space):
        """Get complete space configuration with partial overwrite

        The values passed in ``test_space`` will override the default values
        in ``self.config``.

        Parameters
        ----------
        test_space: dic
            The configuration for the space.
        """
        space = copy.deepcopy(self.space)
        space.update(test_space)
        return space

    def create_space(self, space=None):
        """Create the space object

        Parameters
        ----------
        space: dict, optional
            Configuration of the search space. The default ``self.space`` will be used
            if ``space`` is ``None``.

        """
        return SpaceBuilder().build(space if space is not None else self.space)

    def observe_points(self, points, algo, objective=0):
        """Make the algorithm observe points

        Parameters
        ----------
        points: list of points
            Trials formatted as tuples of values
        algo: ``orion.algo.base.OptimizationAlgorithm``
            The algorithm used to observe points.
        objective: int, optional
            The base objective for the trials. All objectives
            will have value ``objective + i``. Defaults to 0.
        """
        algo.observe(
            points, [dict(objective=objective + i) for i in range(len(points))]
        )

    def get_num(self, num):
        """Force number of trials to suggest

        Some algorithms must be tested with specific number of suggests at a time (ex: ASHA).
        This method can be overriden to change ``num`` based on the special needs.
        """
        return num

    def force_observe(self, num, algo):
        """Force observe ``num`` points.

        Parameters
        ----------
        num: int
            Number of trials to suggest and observe.
        algo: ``orion.algo.base.OptimizationAlgorithm``
            The algorithm that must suggest and observe.

        Raises
        ------
        RuntimeError
            - If the algorithm returns duplicates. Algorithms may return duplicates across workers,
              but in sequential scenarios as here, it should not happen.
            - If the algorithm fails to sample any point at least 5 times.
        """
        objective = 0
        failed = 0
        MAX_FAILED = 5
        ids = set()
        while not algo.is_done and algo.n_observed < num and failed < MAX_FAILED:
            points = algo.suggest(self.get_num(num - algo.n_observed))
            if len(points) == 0:
                failed += 1
                continue
            for point in points:
                point_id = algo.get_id(point)
                if point_id in ids:
                    raise RuntimeError(f"algo suggested a duplicate: {point}")
                ids.add(point_id)
            ids |= set(algo.get_id(point) for point in points)
            self.observe_points(points, algo, objective)
            objective += len(points)

        if failed >= MAX_FAILED:
            raise RuntimeError(
                f"Algorithm cannot sample more than {algo.n_observed} points. Is it normal?"
            )

    def spy_phase(self, mocker, num, algo, attribute):
        """Force observe ``num`` points and then mock a given method to count calls.

        Parameters
        ----------
        mocker: ``pytest_mock.mocker``
            Mocker from ``pytest_mock``. Should be given by fixtures of the tests.
        num: int
            Number of trials to suggest and observe
        algo: ``orion.algo.base.OptimizationAlgorithm``
            The algorithm to test
        attribute: str
            The algorithm attribute or method to mock. The path is respective to the
            algorithm object. For example, a valid value would be `'space.sample'`
            which will mock ``algo.algorithm.space.sample``.
        """
        self.force_observe(num, algo)
        spy = spy_attr(mocker, algo.algorithm, attribute)
        return spy

    def assert_callbacks(self, spy, num, algo):
        """Callback to make special asserts at end of tests

        Override this method in algorithm test-suite to customize verifications done at end
        of tests.

        Parameters
        ----------
        spy: Mocked object
            Object mocked by ``BaseAlgoTests.spy_phase``.
        num: int
            number of points of the phase.
        algo: ``orion.algo.base.OptimizationAlgorithm``
            The algorithm being tested.
        """
        pass

    def assert_dim_type_supported(self, mocker, num, attr, test_space):
        """Test that a given dimension type is properly supported by the algorithm

        This will test that the algorithm sample points valid for the given type
        and that the algorithm can observe these points.

        Parameters
        ----------
        mocker: ``pytest_mock.mocker``
            Mocker from ``pytest_mock``. Should be given by fixtures of the tests.
        num: int
            Number of trials to suggest and observe
        algo: ``orion.algo.base.OptimizationAlgorithm``
            The algorithm to test
        attribute: str
            The algorithm attribute or method to mock. The path is respective to the
            algorithm object. For example, a valid value would be `'space.sample'`
            which will mock ``algo.algorithm.space.sample``.

        """
        space = self.create_space(self.update_space(test_space))
        algo = self.create_algo(space=space)

        spy = self.spy_phase(mocker, num, algo, attr)

        points = algo.suggest()
        assert points[0] in space
        spy.call_count == 1
        self.observe_points(points, algo, 1)
        self.assert_callbacks(spy, num, algo)

    def test_configuration(self):
        """Test that configuration property attribute contains all class arguments."""
        algo = self.create_algo()
        assert algo.configuration != self.create_algo(config={})
        assert algo.configuration == {self.algo_name: self.config}

    def test_get_id(self):
        """Test that the id hashing is valid"""
        space = self.create_space(
            space=self.update_space({"f": "fidelity(1, 10, base=2)"})
        )

        algo = self.create_algo(space=space)

        assert algo.get_id(["is here", 1]) == algo.get_id(["is here", 1])
        assert algo.get_id(["is here", 1]) != algo.get_id(["is here", 2])
        assert algo.get_id(["matters", 1]) != algo.get_id(["is here", 1])

        assert algo.get_id(["is here", 1], ignore_fidelity=False) == algo.get_id(
            ["is here", 1], ignore_fidelity=False
        )
        assert algo.get_id(["is here", 1], ignore_fidelity=False) != algo.get_id(
            ["is here", 2], ignore_fidelity=False
        )
        assert algo.get_id(["matters", 1], ignore_fidelity=False) != algo.get_id(
            ["is here", 1], ignore_fidelity=False
        )

        assert algo.get_id(["is here", 1], ignore_fidelity=True) == algo.get_id(
            ["is here", 1], ignore_fidelity=True
        )
        assert algo.get_id(["is here", 1], ignore_fidelity=True) != algo.get_id(
            ["is here", 2], ignore_fidelity=True
        )
        assert algo.get_id(["whatever", 1], ignore_fidelity=True) == algo.get_id(
            ["is here", 1], ignore_fidelity=True
        )

    @phase
    def test_seed_rng(self, mocker, num, attr):
        """Test that the seeding gives reproducibile results."""
        algo = self.create_algo()

        seed = numpy.random.randint(10000)
        algo.seed_rng(seed)
        spy = self.spy_phase(mocker, num, algo, attr)
        points = algo.suggest(1)
        with pytest.raises(AssertionError):
            numpy.testing.assert_equal(points, algo.suggest(1))

        new_algo = self.create_algo()
        new_algo.seed_rng(seed)
        self.force_observe(algo.n_observed, new_algo)
        numpy.testing.assert_equal(points, new_algo.suggest(1))

        self.assert_callbacks(spy, num, new_algo)

    @phase
    def test_state_dict(self, mocker, num, attr):
        """Verify that resetting state makes sampling deterministic"""
        algo = self.create_algo()

        seed = numpy.random.randint(10000)
        algo.seed_rng(seed)
        spy = self.spy_phase(mocker, max(num, 1), algo, attr)
        state = algo.state_dict
        a = algo.suggest(1)[0]

        new_algo = self.create_algo()
        with pytest.raises(AssertionError):
            numpy.testing.assert_equal(a, new_algo.suggest(1)[0])

        new_algo.set_state(state)
        numpy.testing.assert_equal(a, new_algo.suggest(1)[0])

        self.assert_callbacks(spy, num, algo)

    @phase
    def test_suggest(self, mocker, num, attr):
        """Verify that suggest returns correct number of points.

        This method will likely require to be overriden based on the behavior of the algorith
        for ``suggest(num=None)``.
        """
        algo = self.create_algo()
        spy = self.spy_phase(mocker, num, algo, attr)
        points = algo.suggest()
        assert len(points) == 1

    @phase
    def test_suggest_n(self, mocker, num, attr):
        """Verify that suggest returns correct number of trials if ``num`` is specified in ``suggest``."""
        algo = self.create_algo()
        spy = self.spy_phase(mocker, num, algo, attr)
        points = algo.suggest(5)
        assert len(points) == 5

    @phase
    def test_has_suggested(self, mocker, num, attr):
        """Verify that algorithm detects correctly if a point was suggested"""
        algo = self.create_algo()
        spy = self.spy_phase(mocker, num, algo, attr)
        a = algo.suggest(1)[0]
        assert algo.has_suggested(a)
        # NOTE: not algo.has_suggested(some random point) is tested in test_has_suggested_statedict

    @phase
    def test_has_suggested_statedict(self, mocker, num, attr):
        """Verify that algorithm detects correctly if a point was suggested even when state was restored."""
        algo = self.create_algo()
        spy = self.spy_phase(mocker, num, algo, attr)

        a = algo.suggest(1)[0]
        state = algo.state_dict
        assert algo.has_suggested(a)

        algo = self.create_algo()
        assert not algo.has_suggested(a)

        algo.set_state(state)
        assert algo.has_suggested(a)

    @phase
    def test_observe(self, mocker, num, attr):
        """Verify that algorithm observes point without any issues"""
        algo = self.create_algo()
        spy = self.spy_phase(mocker, num, algo, attr)

        a = algo.space.sample()[0]
        algo.observe([a], [dict(objective=1)])

        b = algo.suggest(1)[0]
        algo.observe([b], [dict(objective=2)])

    @phase
    def test_has_observed(self, mocker, num, attr):
        """Verify that algorithm detects correctly if a point was observed"""
        algo = self.create_algo()
        spy = self.spy_phase(mocker, num, algo, attr)

        a = algo.suggest(1)[0]
        assert not algo.has_observed(a)
        algo.observe([a], [dict(objective=1)])
        assert algo.has_observed(a)

        b = algo.suggest(1)[0]
        assert not algo.has_observed(b)
        algo.observe([b], [dict(objective=2)])
        assert algo.has_observed(b)

    @phase
    def test_has_observed_statedict(self, mocker, num, attr):
        """Verify that algorithm detects correctly if a point was observed even when state was restored."""
        algo = self.create_algo()
        spy = self.spy_phase(mocker, num, algo, attr)

        a = algo.suggest(1)[0]
        algo.observe([a], [dict(objective=1)])
        state = algo.state_dict

        algo = self.create_algo()
        assert not algo.has_observed(a)
        algo.set_state(state)
        assert algo.has_observed(a)

        b = algo.suggest(1)[0]
        algo.observe([b], [dict(objective=2)])
        state = algo.state_dict

        algo = self.create_algo()
        assert not algo.has_observed(b)
        algo.set_state(state)
        assert algo.has_observed(b)

    @phase
    def test_n_suggested(self, mocker, num, attr):
        """Verify that algorithm returns correct number of suggested points"""
        algo = self.create_algo()
        spy = self.spy_phase(mocker, num, algo, attr)
        assert algo.n_suggested == num
        algo.suggest(1)
        assert algo.n_suggested == num + 1

    @phase
    def test_n_observed(self, mocker, num, attr):
        """Verify that algorithm returns correct number of observed points"""
        algo = self.create_algo()
        spy = self.spy_phase(mocker, num, algo, attr)
        assert algo.n_observed == num
        points = algo.suggest(1)
        assert algo.n_observed == num
        self.observe_points(points, algo)
        assert algo.n_observed == num + 1

    @phase
    def test_real_data(self, mocker, num, attr):
        """Test that algorithm supports real dimesions"""
        self.assert_dim_type_supported(mocker, num, attr, {"x": "uniform(0, 5)"})

    @phase
    def test_int_data(self, mocker, num, attr):
        """Test that algorithm supports integer dimesions"""
        self.assert_dim_type_supported(
            mocker, num, attr, {"x": "uniform(0, 5000, discrete=True)"}
        )

    @phase
    def test_cat_data(self, mocker, num, attr):
        """Test that algorithm supports categorical dimesions"""
        self.assert_dim_type_supported(
            mocker,
            num,
            attr,
            {  # Add 3 dims so that there exists many possible trials for the test
                "x": "choices(['a', 'b', 'c', 'd'])",
                "y": "choices(['a', 'b', 'c', 'd'])",
                "z": "choices(['a', 'b', 'c', 'd'])",
            },
        )

    @phase
    def test_logreal_data(self, mocker, num, attr):
        """Test that algorithm supports logreal dimesions"""
        self.assert_dim_type_supported(mocker, num, attr, {"x": "loguniform(1, 5)"})

    @phase
    def test_logint_data(self, mocker, num, attr):
        """Test that algorithm supports loginteger dimesions"""
        self.assert_dim_type_supported(
            mocker, num, attr, {"x": "loguniform(1, 100, discrete=True)"}
        )

    @phase
    def test_shape_data(self, mocker, num, attr):
        """Test that algorithm supports dimesions with shape"""
        self.assert_dim_type_supported(
            mocker, num, attr, {"x": "uniform(0, 5, shape=(3, 2))"}
        )

    def test_is_done_cardinality(self):
        """Test that algorithm will stop when cardinality is reached"""
        space = self.update_space(
            {
                "x": "uniform(0, 4, discrete=True)",
                "y": "choices(['a', 'b', 'c'])",
                "z": "loguniform(1, 6, discrete=True)",
            }
        )
        space = self.create_space(space)
        assert space.cardinality == 5 * 3 * 6

        algo = self.create_algo(space=space)
        for i, (x, y, z) in enumerate(itertools.product(range(5), "abc", range(1, 7))):
            assert not algo.is_done
            n = len(algo.algorithm._trials_info)
            algo.observe([[x, y, z]], [dict(objective=i)])
            assert len(algo.algorithm._trials_info) == n + 1

        assert i + 1 == space.cardinality

        assert algo.is_done

    def test_is_done_max_trials(self):
        """Test that algorithm will stop when max trials is reached"""
        MAX_TRIALS = 10
        algo = self.create_algo()
        algo.algorithm.max_trials = MAX_TRIALS

        self.force_observe(MAX_TRIALS, algo)

        assert algo.is_done

    def test_optimize_rosenbrock(self):
        """Test that algorithm optimizes somehow (this is on-par with random search)"""
        MAX_TRIALS = 20
        task = Branin()
        space = self.create_space(task.get_search_space())
        algo = self.create_algo(config={}, space=space)
        algo.algorithm.max_trials = MAX_TRIALS
        safe_guard = 0
        points = []
        objectives = []
        while points or not algo.is_done:
            if safe_guard >= MAX_TRIALS:
                break

            if not points:
                points = algo.suggest()

            point = points.pop(0)
            results = task(*point)
            objectives.append(results[0]["value"])
            algo.observe([point], [dict(objective=objectives[-1])])
            safe_guard += 1

        assert algo.is_done
        assert min(objectives) <= 10
