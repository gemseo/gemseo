..
   Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

   This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
   International License. To view a copy of this license, visit
   http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
   Commons, PO Box 1866, Mountain View, CA 94042, USA.

.. _doe:

Design of experiments
=====================

Design of experiments (DOE) is a branch of applied statistics
to plan, conduct and analyze real or numerical experiments.
It consists in selecting input values in a methodical way (sampling)
and then performing the experiments to obtain output values (measurement or evaluation).

.. note::
   "DOE" may also refer to the sampling method itself, e.g. Latin hypercube sampling.

A DOE can be used to:

- determine whether an input or an interaction between inputs has an effect on an output (sensitivity analysis),
- model the relationship between inputs and outputs (surrogate modeling),
- optimize an output with respect to inputs while satisfying some constraints (trade-off).

API
---

In |g|,
a :class:`.DOELibrary` contains one or several DOE algorithms.

As any :class:`.DriverLib`,
a :class:`.DOELibrary` executes an algorithm from an :class:`.OptimizationProblem` and options.
Most of the DOE algorithms also need the number of samples when calling :meth:`~.DOELibrary.execute`:

.. code::

    >>> from gemseo.algos.doe.lib_pydoe import PyDOE
    >>> pydoe_library = PyDOE()
    >>> optimization_result = pydoe_library.execute(problem, "lhs", n_samples=100)

In the presence of an :class:`.OptimizationProblem`,
it is advisable to apply DOE algorithms with the function :func:`.execute_algo`
which returns an :class:`.OptimizationResult`:

.. code::

    >>> from gemseo.api import execute_algo
    >>> optimization_result = execute_algo(problem, "lhs", algo_type="doe", n_samples=100)

In the presence of an :class:`.MDODiscipline`,
it is advisable to create a :class:`.DOEScenario` with the function :func:`.create_scenario`
and pass the DOE algorithm to :meth:`.DOEScenario.execute`:

.. code::

    >>> doe_scenario.execute({"algo": "lhs", "n_samples": 100})

Algorithms
----------

|g| wraps different kinds of DOE algorithms
from the libraries `PyDOE <https://github.com/clicumu/pyDOE2>`__ and `OpenTURNS <https://openturns.github.io/www/>`__.

.. note::

   The names of the algorithms coming from OpenTURNS starts with ``"OT_"``, e.g. ``"OT_OPT_LHS"``.
   You need to :ref:`install <installation>` the full features of |g| in order to use them.

.. seealso::
   All the DOE algorithms and their settings are listed on :ref:`this page <gen_doe_algos>`.

These DOE algorithms can be classified into categories:

- the Monte Carlo sampling generates values in the input space
  distributed as a multivariate uniform probability distribution with stochastically independent components;
  the algorithm is ``"OT_MONTE_CARLO"``,
- the `low-discrepancy sequences <https://en.wikipedia.org/wiki/Low-discrepancy_sequence>`__
  are sequences of input values designed to be distributed as uniformly as possible
  (the deviation from uniform distribution is called *discrepancy*);
  the algorithms are ``"OT_FAURE"``, ``"OT_HALTON"``, ``"OT_HASELGROVE"``, ``"OT_SOBOL"`` and ``"OT_REVERSE_HALTON"``,
- the Latin hypercube sampling (LHS) is an algorithm generating :math:`N` points in the input space
  based on the generalization of the `Latin square <https://en.wikipedia.org/wiki/Latin_square>`__:
  the range of each input is partitioned into :math:`N` equal intervals and,
  for each interval,
  one and only one of the points has its corresponding input value inside the interval;
  the algorithms are ``"lhs"``, ``"OT_LHS"`` and ``"OT_LHSC"``,
- the optimized LHS is an LHS optimized by Monte Carlo replicates or simulated annealing;
  the algorithm is ``"OT_OPT_LHS"``,
- the stratified DOEs makes the inputs, also called *factors*, vary by level;

  - a full factorial DOE considers all the possible combinations of these levels across all the inputs;
    the algorithms are ``"ff2n"``, ``"fullfact"`` and ``"OT_FULLFACT"``;
  - a factorial DOE samples the diagonals of the input space, symmetrically with respect to its center;
    the algorithm is ``"OT_FACTORIAL"``;
  - an axial DOE samples the axes of the input space, symmetrically with respect to its center;
    the algorithm is ``"OT_AXIAL"``;
  - a central composite DOE combines a factorial and an axial DOEs;
    the algorithms are ``"OT_COMPOSITE"`` and ``"ccdesign"``;
  - Box–Behnken and Plackett-Burman DOEs for response surface methodology;
    the algorithms are ``"bbdesign"`` and ``"pbdesign"``.

|g| also offers a :class:`.CustomDOE` to set its own input values,
either as a CSV file or a two-dimensional NumPy array.

Advanced use
------------

Once the functions of the :class:`.OptimizationProblem` have been evaluated,
the input samples can be accessed with :attr:`~.DOELibrary.samples`.

.. note::
   |g| applies a DOE algorithm over a unit hypercube of the same dimension as the input space
   and then project the :attr:`~.DOELibrary.unit_samples` onto the input space
   using either the probability distributions of the inputs, if the latter are random variables,
   or their lower and upper bounds.

If we do not want to evaluate the functions but only obtain the input samples,
we can use the method :meth:`~gemseo.api.compute_doe` which returns the samples as a two-dimensional NumPy array.

The quality of the input samples can be assessed with a :class:`.DOEQuality`
computing the :math:`\varphi_p`, minimum-distance and discrepancy criteria.
The smaller these quality measures, the better,
except for the minimum-distance criterion for which the larger it is the better.
The qualities can be compared with logical operations,
with ``DOEQuality(doe_1) > DOEQuality(doe_2)`` meaning that ``doe_1`` is better than ``doe_2``.

.. note::
   When numerical metrics are not sufficient to compare two input samples sets,
   graphical indicators (e.g. :class:`.ScatterMatrix`) could be considered.

Lastly,
a :class:`.DOELibrary` has a :attr:`~.DOELibrary.seed` initialized at 0
and each call to :meth:`~.DOELibrary.execute` increments it before using it.
Thus,
two executions generate two distinct set of input-output samples.
For the sake of reproducibility,
you can pass your own seed to :meth:`~.DOELibrary.execute` as a DOE option.
