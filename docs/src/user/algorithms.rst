.. _Setup Algorithms:

**********
Algorithms
**********

.. contents::
   :depth: 2
   :local:

Default algorithm is a random search based on the probability
distribution given to a search parameter's definition.

Selecting and Configuring
=========================

In a Oríon configuration YAML, define:

.. code-block:: yaml

   experiment:
       algorithms:
           gradient_descent:
               learning_rate: 0.1

In this particular example, the name of the algorithm extension class to be
imported and instantiated is ``Gradient_Descent``, so the lower-case identifier
corresponds to it.

All algorithms have default arguments that should work reasonably well in general.
To tune the algorithm for a specific problem, you can set those arguments in the
yaml file as shown above with ``learning_rate``.

Included Algorithms
===================

.. contents::
   :depth: 1
   :local:

.. _random-search:

Random Search
-------------

Random search is the most simple algorithm. It samples from given priors. That's it.

Configuration
~~~~~~~~~~~~~

.. code-block:: yaml

    experiment:
        algorithms:
            random:
                seed: null


``seed``

Seed for the random number generator used to sample new trials. Default is ``None``.

.. _grid-search:

Grid Search
-----------

Grid search is one of the simplest algorithm. It can work reasonably well for small search spaces of
one or two dimensions but should be avoided for larger search spaces. The search space can be
configured in three different ways.

1. Default for the lazy. You can set a very large ``n_values`` (ex: 100) and the grid will be
   adjusted so that it results in less than ``max_trials`` as defined in the experiment
   configuration.
2. You can set ``n_values`` to the number of points desired by dimension. Note that if this
   leads to too many trials the grid will be shrunken down to fit below ``max_trials``.
3. You can pass a dictionary to ``n_values`` specifying the number of points for each dimensions.
   Ex: ``n_values: {'dim1': 3, 'dim2': 4}``.

.. note::

   For categorical dimensions (``choices()``) all values are used to build the grid. This means
   ``n_values`` will not be honored. A warning is printed when this happens. Accordingly,
   if too many options are provided for the categorical dimensions the grid may lead to more trials
   than ``max_trials``. A ``ValueError`` will be raised in such scenario.

Configuration
~~~~~~~~~~~~~

.. code-block:: yaml

    experiment:
        algorithms:
            gridsearch:
                n_values: 100


``n_values``

Number of different values to use for each dimensions to build the grid. Can be either
1. An integer. The same number will be used for all dimensions
2. A dictionary many dimension names to integers. Each dimension will have its own number of values.

.. _hyperband-algorithm:

Hyperband
---------

`Hyperband`_ extends the `SuccessiveHalving`_ algorithm by providing a way to exploit a
fixed budget with different number of configurations for ``SuccessiveHalving`` algorithm to
evaluate. Each run of ``SuccessiveHalving`` will be defined as a ``bracket`` in Hyperband.
Hyperband requires two inputs (1) ``R``, the maximum amount of resource that can be allocated
to a single configuration, and (2) ``eta``, an input that controls the proportion of
configurations discarded in each round of SuccessiveHalving.

To use Hyperband in Oríon, you must specify one parameter with ``fidelity(low, high, base)``
as the prior, ``low`` will be ignored, ``high`` will be taken as the maximum resource ``R``
and ``base`` will be taken as the reduction factor ``eta``.

Number of epochs usually can be used as the resource but the algorithm is generic and can be
applied to any multi-fidelity setting. That is, you can use training time, specifying the
fidelity with ``--epochs~fidelity(low=1, high=81, base=3)``
(assuming your script takes this argument in commandline),
but you could also use other fidelity
such as dataset size ``--dataset-size~fidelity(low=500, high=50000)``
(assuming your script takes this argument and adapt dataset size accordingly).


.. _SuccessiveHalving: https://arxiv.org/abs/1502.07943

.. note::

   Current implementation does not support more than one fidelity dimension.

Configuration
~~~~~~~~~~~~~

.. code-block:: yaml

    experiment:
        algorithms:
            hyperband:
                seed: null
                repetitions: 1

        strategy: StubParallelStrategy


.. note::

   Notice the additional ``strategy`` in configuration which is not mandatory for most other
   algorithms. See :ref:`StubParallelStrategy` for more information.


``seed``

Seed for the random number generator used to sample new trials. Default is ``None``.

``repetitions``

Number of executions for Hyperband. A single execution of Hyperband takes a finite
budget of ``(log(R)/log(eta) + 1) * (log(R)/log(eta) + 1) * R``, and ``repetitions`` allows you
to run multiple executions of Hyperband. Default is ``numpy.inf`` which means to run Hyperband
until no new trials can be suggested.


.. _ASHA:

ASHA
----

`Asynchronous Successive Halving Algorithm`_, the asynchronous version of
`Hyperband`_, can be roughly interpreted as a sophisticated random search that leverages
partial information of the trial execution to concentrate resources on the
most promising ones.

The main idea of the algorithm is the following. Given a fidelity dimension, such as
the number of epochs to train or the size of the dataset, ASHA samples trials
with low-fidelity and promotes the most promising ones to the next fidelity level.
This makes it possible to only execute one trial with full fidelity, leading
to very optimal resource usage.

The most common way of using ASHA is to reduce the number of epochs,
but the algorithm is generic and can be applied to any multi-fidelity setting.
That is, you can use training time, specifying the fidelity with
``--epochs~fidelity(low=1, high=100)``
(assuming your script takes this argument in commandline),
but you could also use other fidelity
such as dataset size ``--dataset-size~fidelity(low=500, high=50000)``
(assuming your script takes this argument and
adapt dataset size accordingly). The placeholder ``fidelity(low, high)`` is a special prior for
multi-fidelity algorithms.


.. _asynchronous successive halving algorithm: https://arxiv.org/abs/1810.05934
.. _Hyperband: https://arxiv.org/abs/1603.06560

.. note::

   Current implementation does not support more than one fidelity dimension.

Configuration
~~~~~~~~~~~~~

.. code-block:: yaml

    experiment:
        algorithms:
            asha:
                seed: null
                num_rungs: null
                num_brackets: 1
                repetitions: 1

        strategy: StubParallelStrategy


.. note::

   Notice the additional ``strategy`` in configuration which is not mandatory for most other
   algorithms. See :ref:`StubParallelStrategy` for more information.


``seed``

Seed for the random number generator used to sample new trials. Default is ``None``.


``num_rungs``

Number of rungs for the largest bracket. If not defined, it will be equal to ``(base + 1)`` of the
fidelity dimension. In the original paper,
``num_rungs == log(fidelity.high/fidelity.low) / log(fidelity.base) + 1``.

``num_brackets``

Using a grace period that is too small may bias ASHA too strongly towards fast
converging trials that do not lead to best results at convergence (stragglers).
To overcome this, you can increase the number of brackets, which increases the amount of resources
required for optimisation but decreases the bias towards stragglers. Default is 1.


``repetitions``

Number of execution of ASHA. Default is ``numpy.inf`` which means to
run ASHA until no new trials can be suggested.


.. _tpe-algorithm:

TPE
---------

`Tree-structured Parzen Estimator`_ (TPE) algorithm is one of Sequential Model-Based
Global Optimization (SMBO) algorithms, which will build models to propose new points based
on the historical observed trials.

Instead of modeling p(y|x) like other SMBO algorithms, TPE models p(x|y) and p(y),
and p(x|y) is modeled by transforming that generative process, replacing the distributions of
the configuration prior with non-parametric densities.

The TPE defines p(x|y) using two such densities l(x) and g(x) where l(x) is distribution of
good points and g(x) is the distribution of bad points. Good and bad points are split from observed
points so far with a parameter `gamma` which defines the ratio of good points. New point candidates
will be sampled with l(x) and Expected Improvement (EI) optimization scheme will be used to find
the most promising point among the candidates.


.. _Tree-structured Parzen Estimator:
    https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf

.. note::

   Current implementation only supports uniform, loguniform, uniform discrete and choices as prior.
   As for choices prior, the probabilities if any given will be ignored.

Configuration
~~~~~~~~~~~~~

.. code-block:: yaml

    experiment:
        algorithms:
            tpe:
                seed: null
                n_initial_points: 20
                n_ei_candidates: 25
                gamma: 0.25
                equal_weight: False
                prior_weight: 1.0
                full_weight_num: 25


``seed``

Seed to sample initial points and candidates points. Default is ``None``.

``n_initial_points``

Number of initial points randomly sampled. Default is ``20``.

``n_ei_candidates``

Number of candidates points sampled for ei compute. Default is ``24``.

``gamma``

Ratio to split the observed trials into good and bad distributions. Default is ``0.25``.

``equal_weight``

True to set equal weights for observed points. Default is ``False``.

``prior_weight``

The weight given to the prior point of the input space. Default is ``1.0``.

``full_weight_num``

The number of the most recent trials which get the full weight where the others will be
applied with a linear ramp from 0 to 1.0. It will only take effect if ``equal_weight``
is ``False``. Default is ``25``.

.. _evolution-es algorithm:

Evolution-ES
-------------

`Evolution-ES`_, the evolution algorithm with early stop version.
Here is an implementation of `Evolution-ES`_.
In the evolution algorithm, we follow the tournament selection algorithm
as `Large-Scale-Evolution`_.
Tournament selection evolutionary hyper-parameter search is conducted
by first defining a gene encoding
that describes a hyper-parameter combination,
and then creating the initial population by randomly
sampling from the space of gene encodings to create individuals,
which are trained and assigned fitnesses.
The population is then repeatedly sampled from to produce groups,
and the parent is selected by the individual
with the highest fitness. Selected parents have their gene encodings
mutated to produce child models.
Individual in the group with the lowest fitness is killed,
while the newly evaluated child model is added to
the population, taking the killed individual’s place.
This process is repeated and results
in a population with high fitness individuals can
represent the good hyper-parameter combination.
`Evolution-ES`_ also formulated a method to dynamically allocate
resources to more promising individual
according to their fitness, which is referred to as
Progressive Dynamic Hurdles (PDH),
allows individuals that are consistently performing well to train for more steps.
It can be roughly interpreted as a sophisticated random search
that leverages partial information of the trial execution to concentrate resources
on the most promising ones.

The implementation follows the process and use way of Hyperband.
Additionally, The fidelity base in Evolution-ES can be
extended to support ``fidelity(low, high, base=1)``,
which is the same as ``linspace(low, high)``.

.. _Evolution-ES: https://arxiv.org/abs/1901.11117
.. _Large-Scale-Evolution: https://arxiv.org/abs/1703.01041


Configuration
~~~~~~~~~~~~~

.. code-block:: yaml

    experiment:
        algorithms:
            EvolutionES:
                seed: null
                repetitions: 1
                nums_population: 20
                mutate:
                    function: orion.algo.mutate_functions.default_mutate
                    multiply_factor: 3.0
                    add_factor: 1

        strategy: StubParallelStrategy

``seed``

Seed for the random number generator used to sample new trials. Default is ``None``.

``repetitions``

Number of executions for Hyperband. A single execution of Hyperband takes a finite
budget of ``(log(R)/log(eta) + 1) * (log(R)/log(eta) + 1) * R``, and ``repetitions`` allows you
to run multiple executions of Hyperband. Default is ``numpy.inf`` which means to run Hyperband
until no new trials can be suggested.

``nums_population``

Number of population for EvolutionES. Larger number of population often gets better performance
but causes more computation. So there is a trade-off according to
the search space and required budget of your problems.

``mutate``

In the mutate part, one can define the customized mutate function with its mutate factors,
such as multiply factor (times/divides by a multiply factor) and add factor
(add/subtract by a multiply factor). We support the default mutate function.


Algorithm Plugins
=================

.. _scikit-bayesopt:

Scikit Bayesian Optimizer
-------------------------

``orion.algo.skopt`` provides a wrapper for `Bayesian optimizer`_ using Gaussian process implemented
in `scikit optimize`_.

.. _scikit optimize: https://scikit-optimize.github.io/
.. _bayesian optimizer: https://scikit-optimize.github.io/#skopt.Optimizer

Installation
~~~~~~~~~~~~

.. code-block:: sh

   pip install orion.algo.skopt

Configuration
~~~~~~~~~~~~~

.. code-block:: yaml

    experiment:
        algorithms:
            BayesianOptimizer:
                seed: null
                n_initial_points: 10
                acq_func: gp_hedge
                alpha: 1.0e-10
                n_restarts_optimizer: 0
                noise: "gaussian"
                normalize_y: False

``seed``

``n_initial_points``

Number of evaluations of ``func`` with initialization points
before approximating it with ``base_estimator``. Points provided as
``x0`` count as initialization points. If len(x0) < n_initial_points
additional points are sampled at random.

``acq_func``

Function to minimize over the posterior distribution. Can be:
``["LCB", "EI", "PI", "gp_hedge", "EIps", "PIps"]``. Check skopt
docs for details.

``alpha``

Value added to the diagonal of the kernel matrix during fitting.
Larger values correspond to increased noise level in the observations
and reduce potential numerical issues during fitting. If an array is
passed, it must have the same number of entries as the data used for
fitting and is used as datapoint-dependent noise level. Note that this
is equivalent to adding a WhiteKernel with c=alpha. Allowing to specify
the noise level directly as a parameter is mainly for convenience and
for consistency with Ridge.

``n_restarts_optimizer``

The number of restarts of the optimizer for finding the kernel's
parameters which maximize the log-marginal likelihood. The first run
of the optimizer is performed from the kernel's initial parameters,
the remaining ones (if any) from thetas sampled log-uniform randomly
from the space of allowed theta-values. If greater than 0, all bounds
must be finite. Note that n_restarts_optimizer == 0 implies that one
run is performed.

``noise``

If set to "gaussian", then it is assumed that y is a noisy estimate of f(x) where the
noise is gaussian.

``normalize_y``

Whether the target values y are normalized, i.e., the mean of the
observed target values become zero. This parameter should be set to
True if the target values' mean is expected to differ considerable from
zero. When enabled, the normalization effectively modifies the GP's
prior based on the data, which contradicts the likelihood principle;
normalization is thus disabled per default.

.. _parallel-strategies:

Parallel Strategies
===================

A parallel strategy is a method to improve parallel optimization
for sequential algorithms. Such algorithms can only observe
trials that are completed and have a corresponding objective.
To get around this, parallel strategies produces *lies*,
noncompleted trials with fake objectives, which are then
passed to a temporary copy of the algorithm that will suggest
a new point. The temporary algorithm is then discarded.
The original algorithm never obverses lies, and
the temporary copy always observes lies that are based on
most up-to-date data.
The strategies will differ in how they assign objectives
to the *lies*.

By default, the strategy used is :ref:`MaxParallelStrategy`

NoParallelStrategy
------------------

Does not return any lie. This is useful to benchmark parallel
strategies and measure how they can help compared to no
strategy.

.. _StubParallelStrategy:

StubParallelStrategy
--------------------

Assign to *lies* an objective of ``None`` so that
non-completed trials are observed and identifiable by algorithms
that can leverage parallel optimization.

The value of the objective is customizable with ``stub_value``.

.. code-block:: yaml

    experiment:
        strategy:
            StubParallelStrategy:
                stub_value: 'custom value'

.. _MaxParallelStrategy:

MaxParallelStrategy
-------------------

Assigns to *lies* the best objective observed so far.

The default value assigned to objective when less than 1 trial
is completed is configurable with ``default_result``. It
is ``float('inf')`` by default.

.. code-block:: yaml

    experiment:
        strategy:
            MaxParallelStrategy:
                default_result: 10000


MeanParallelStrategy
--------------------

Assigns to *lies* the mean of all objectives observed so far.

The default value assigned to objective when less than 2 trials
are completed is configurable with ``default_result``. It
is ``float('inf')`` by default.

.. code-block:: yaml

    experiment:
        strategy:
            MeanParallelStrategy:
                default_result: 0.5
