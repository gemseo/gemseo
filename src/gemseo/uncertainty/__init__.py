# Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License version 3 as published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program; if not, write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
"""Uncertainty quantification and management.

The package :mod:`~gemseo.uncertainty` provides several functionalities
to quantify and manage uncertainties.
Most of them can be used from the high-level functions provided by this module.

The sub-package :mod:`~gemseo.uncertainty.distributions` offers an abstract level
for probability distributions, as well as interfaces to the OpenTURNS and SciPy ones.
It is also possible to fit a probability distribution from data
or select the most likely one from a list of candidates.
These distributions can be used to define random variables in a :class:`.ParameterSpace`
before propagating these uncertainties through a system of :class:`.MDODiscipline`,
by means of a :class:`.DOEScenario`.

.. seealso::

    :class:`.OTDistribution`
    :class:`.SPDistribution`
    :class:`.OTDistributionFitter`

The sub-package :mod:`~gemseo.uncertainty.sensitivity` offers an abstract level
for sensitivity analysis, as well as concrete features.
These sensitivity analyses compute indices by means of various methods:
correlation measures, Morris technique and Sobol' variance decomposition.
This sub-package is based in particular on OpenTURNS.

.. seealso::

    :class:`.CorrelationAnalysis`
    :class:`.MorrisAnalysis`
    :class:`.SobolAnalysis`
    :class:`.HSICAnalysis`

The sub-package :mod:`~gemseo.uncertainty.statistics` offers an abstract level
for statistics, as well as parametric and empirical versions.
Empirical statistics are estimated from a :class:`.Dataset`
while parametric statistics are analytical properties of a :class:`.Distribution`
fitted from a :class:`.Dataset`.

.. seealso::

    :class:`.EmpiricalStatistics`
    :class:`.ParametricStatistics`
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Collection
    from collections.abc import Iterable
    from collections.abc import Sequence
    from pathlib import Path

    from gemseo.algos.parameter_space import ParameterSpace
    from gemseo.core.discipline import MDODiscipline
    from gemseo.datasets.dataset import Dataset
    from gemseo.uncertainty.distributions.distribution import Distribution
    from gemseo.uncertainty.sensitivity.analysis import SensitivityAnalysis
    from gemseo.uncertainty.statistics.statistics import Statistics


def get_available_distributions(base_class_name: str = "Distribution") -> list[str]:
    """Get the available probability distributions.

    Args:
        base_class_name: The name of the base class of the probability distributions,
            e.g. ``"Distribution"``, ``"OTDistribution"`` or ``"SPDistribution"``.

    Returns:
        The names of the available probability distributions.
    """
    from gemseo.uncertainty.distributions.factory import DistributionFactory

    factory = DistributionFactory()
    class_names = factory.class_names
    if base_class_name == "Distribution":
        return class_names

    return [
        class_name
        for class_name in class_names
        if base_class_name
        in [cls.__name__ for cls in factory.get_class(class_name).mro()]
    ]


def create_distribution(
    variable: str,
    distribution_name: str,
    dimension: int = 1,
    **options,
) -> Distribution:
    """Create a distribution.

    Args:
        variable: The name of the random variable.
        distribution_name: The name of a class
            implementing a probability distribution,
            e.g. 'OTUniformDistribution' or 'SPDistribution'.
        dimension: The dimension of the random variable.
        **options: The distribution options.

    Examples:
        >>> from gemseo.uncertainty import create_distribution
        >>>
        >>> distribution = create_distribution(
        ...     "x", "OTNormalDistribution", dimension=2, mu=1, sigma=2
        ... )
        >>> print(distribution)
        Normal(mu=1, sigma=2)
        >>> print(distribution.mean, distribution.standard_deviation)
        [1. 1.] [2. 2.]
        >>> samples = distribution.compute_samples(10)
        >>> print(samples.shape)
        (10, 2)
    """
    from gemseo.uncertainty.distributions.factory import DistributionFactory

    factory = DistributionFactory()
    return factory.create(
        distribution_name, variable=variable, dimension=dimension, **options
    )


def get_available_sensitivity_analyses() -> list[str]:
    """Get the available sensitivity analyses."""
    from gemseo.uncertainty.sensitivity.factory import SensitivityAnalysisFactory

    factory = SensitivityAnalysisFactory()
    return factory.available_sensitivity_analyses


def create_statistics(
    dataset: Dataset,
    variable_names: Iterable[str] | None = None,
    tested_distributions: Sequence[str] | None = None,
    fitting_criterion: str = "BIC",
    selection_criterion: str = "best",
    level: float = 0.05,
    name: str | None = None,
) -> Statistics:
    """Create a statistics toolbox, either parametric or empirical.

    If parametric, the toolbox selects a distribution from candidates,
    based on a fitting criterion and on a selection strategy.

    Args:
        dataset: A dataset.
        variable_names: The variables of interest.
            If ``None``, consider all the variables from dataset.
        tested_distributions: The names of
            the tested distributions.
        fitting_criterion: The name of a goodness-of-fit criterion,
            measuring how the distribution fits the data.
            Use :meth:`.ParametricStatistics.get_criteria`
            to get the available criteria.
        selection_criterion: The name of a selection criterion
            to select a distribution from candidates.
            Either 'first' or 'best'.
        level: A test level,
            i.e. the risk of committing a Type 1 error,
            that is an incorrect rejection of a true null hypothesis,
            for criteria based on a test hypothesis.
        name: A name for the statistics toolbox instance.
            If ``None``, use the concatenation of class and dataset names.

    Returns:
        A statistics toolbox.

    Examples:
        >>> from gemseo import (
        ...     create_discipline,
        ...     create_parameter_space,
        ...     create_scenario,
        ... )
        >>> from gemseo.uncertainty import create_statistics
        >>>
        >>> expressions = {"y1": "x1+2*x2", "y2": "x1-3*x2"}
        >>> discipline = create_discipline(
        ...     "AnalyticDiscipline", expressions=expressions
        ... )
        >>>
        >>> parameter_space = create_parameter_space()
        >>> parameter_space.add_random_variable(
        ...     "x1", "OTUniformDistribution", minimum=-1, maximum=1
        ... )
        >>> parameter_space.add_random_variable(
        ...     "x2", "OTNormalDistribution", mu=0.5, sigma=2
        ... )
        >>>
        >>> scenario = create_scenario(
        ...     [discipline],
        ...     "DisciplinaryOpt",
        ...     "y1",
        ...     parameter_space,
        ...     scenario_type="DOE",
        ... )
        >>> scenario.execute({"algo": "OT_MONTE_CARLO", "n_samples": 100})
        >>>
        >>> dataset = scenario.to_dataset(opt_naming=False)
        >>>
        >>> statistics = create_statistics(dataset)
        >>> mean = statistics.compute_mean()
    """
    from gemseo.uncertainty.statistics.empirical import EmpiricalStatistics as EmpStats
    from gemseo.uncertainty.statistics.parametric import (
        ParametricStatistics as ParamStats,
    )

    if tested_distributions is None:
        statistical_analysis = EmpStats(dataset, variable_names, name)
    else:
        statistical_analysis = ParamStats(
            dataset,
            tested_distributions,
            variable_names,
            fitting_criterion,
            level,
            selection_criterion,
            name,
        )
    return statistical_analysis


def create_sensitivity_analysis(
    analysis: str,
    disciplines: Collection[MDODiscipline],
    parameter_space: ParameterSpace,
    **options,
) -> SensitivityAnalysis:
    """Create the sensitivity analysis.

    Args:
        analysis: The name of a sensitivity analysis class.
        disciplines: The disciplines.
        parameter_space: A parameter space.
        **options: The DOE algorithm options.

    Returns:
        The toolbox for these sensitivity indices.

    Examples:
        >>> from gemseo import create_discipline, create_parameter_space
        >>> from gemseo.uncertainty import create_sensitivity_analysis
        >>>
        >>> expressions = {"y1": "x1+2*x2", "y2": "x1-3*x2"}
        >>> discipline = create_discipline(
        ...     "AnalyticDiscipline", expressions=expressions
        ... )
        >>>
        >>> parameter_space = create_parameter_space()
        >>> parameter_space.add_random_variable(
        ...     "x1", "OTUniformDistribution", minimum=-1, maximum=1
        ... )
        >>> parameter_space.add_random_variable(
        ...     "x2", "OTNormalDistribution", mu=0.5, sigma=2
        ... )
        >>>
        >>> analysis = create_sensitivity_analysis(
        ...     "CorrelationIndices", [discipline], parameter_space, n_samples=1000
        ... )
        >>> indices = analysis.compute_indices()
    """
    from gemseo.uncertainty.sensitivity.factory import SensitivityAnalysisFactory

    factory = SensitivityAnalysisFactory()

    name = analysis
    if "Analysis" not in name:
        name += "Analysis"
    name = name[0].upper() + name[1:]

    return factory.create(name, disciplines, parameter_space, **options)


def load_sensitivity_analysis(file_path: str | Path) -> SensitivityAnalysis:
    """Load a sensitivity analysis from the disk.

    Args:
        file_path: The path to the file.

    Returns:
        The sensitivity analysis.
    """
    from gemseo.uncertainty.sensitivity.analysis import SensitivityAnalysis

    return SensitivityAnalysis.from_pickle(file_path)
