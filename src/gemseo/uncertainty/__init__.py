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
before propagating these uncertainties through a system of :class:`.Discipline`,
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
while parametric statistics are analytical properties of a :class:`.BaseDistribution`
fitted from a :class:`.Dataset`.

.. seealso::

    :class:`.EmpiricalStatistics`
    :class:`.ParametricStatistics`
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemseo.utils.pickle import from_pickle as from_pickle

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Sequence
    from pathlib import Path

    from gemseo.algos.parameter_space import ParameterSpace as ParameterSpace
    from gemseo.core.discipline import Discipline as Discipline
    from gemseo.datasets.dataset import Dataset
    from gemseo.datasets.io_dataset import IODataset as IODataset
    from gemseo.uncertainty.distributions.base_distribution import BaseDistribution
    from gemseo.uncertainty.sensitivity.base_sensitivity_analysis import (
        BaseSensitivityAnalysis,
    )
    from gemseo.uncertainty.statistics.base_statistics import BaseStatistics


def get_available_distributions(base_class_name: str = "BaseDistribution") -> list[str]:
    """Get the available probability distributions.

    Args:
        base_class_name: The name of the base class of the probability distributions,
            e.g. ``"BaseDistribution"``, ``"OTDistribution"`` or ``"SPDistribution"``.

    Returns:
        The names of the available probability distributions.
    """
    from gemseo.uncertainty.distributions.factory import DistributionFactory

    factory = DistributionFactory()
    class_names = factory.class_names
    if base_class_name == "BaseDistribution":
        return class_names

    return [
        class_name
        for class_name in class_names
        if base_class_name
        in [cls.__name__ for cls in factory.get_class(class_name).mro()]
    ]


def create_distribution(
    distribution_name: str,
    **options,
) -> BaseDistribution:
    """Create a distribution.

    Args:
        distribution_name: The name of a class
            implementing a probability distribution,
            e.g. 'OTUniformDistribution' or 'SPDistribution'.
        **options: The distribution options.

    Examples:
        >>> from gemseo.uncertainty import create_distribution
        >>>
        >>> distribution = create_distribution("OTNormalDistribution", mu=1, sigma=2)
        >>> print(distribution)
        Normal(mu=1, sigma=2)
        >>> print(distribution.mean, distribution.standard_deviation)
        1.0 2.0
        >>> samples = distribution.compute_samples(10)
        >>> print(samples.shape)
        (10,)
    """
    from gemseo.uncertainty.distributions.factory import DistributionFactory

    factory = DistributionFactory()
    return factory.create(distribution_name, **options)


def get_available_sensitivity_analyses() -> list[str]:
    """Get the available sensitivity analyses."""
    from gemseo.uncertainty.sensitivity.factory import SensitivityAnalysisFactory

    return SensitivityAnalysisFactory().class_names


def create_statistics(
    dataset: Dataset,
    variable_names: Iterable[str] = (),
    tested_distributions: Sequence[str] = (),
    fitting_criterion: str = "BIC",
    selection_criterion: str = "best",
    level: float = 0.05,
    name: str = "",
) -> BaseStatistics:
    """Create a statistics toolbox, either parametric or empirical.

    If parametric, the toolbox selects a distribution from candidates,
    based on a fitting criterion and on a selection strategy.

    Args:
        dataset: A dataset.
        variable_names: The variables of interest.
            If empty, consider all the variables from dataset.
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
            If empty, use the concatenation of class and dataset names.

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
        ...     "y1",
        ...     parameter_space,
        ...     formulation_name="DisciplinaryOpt",
        ...     scenario_type="DOE",
        ... )
        >>> scenario.execute(algo_name="OT_MONTE_CARLO", n_samples=100)
        >>>
        >>> dataset = scenario.to_dataset(opt_naming=False)
        >>>
        >>> statistics = create_statistics(dataset)
        >>> mean = statistics.compute_mean()
    """
    from gemseo.uncertainty.statistics.empirical_statistics import EmpiricalStatistics
    from gemseo.uncertainty.statistics.parametric_statistics import ParametricStatistics

    if tested_distributions:
        statistical_analysis = ParametricStatistics(
            dataset,
            tested_distributions,
            variable_names,
            fitting_criterion,
            level,
            selection_criterion,
            name,
        )
    else:
        statistical_analysis = EmpiricalStatistics(dataset, variable_names, name)
    return statistical_analysis


def create_sensitivity_analysis(
    analysis: str,
    samples: IODataset | str | Path = "",
) -> BaseSensitivityAnalysis:
    """Create the sensitivity analysis.

    Args:
        analysis: The name of a sensitivity analysis class.
        samples: The samples for the estimation of the sensitivity indices,
            either as an :class:`.IODataset`
            or as a pickle file path generated from
            the :class:`.IODataset.to_pickle` method.
            If empty, use :meth:`.compute_samples`.

    Returns:
        The sensitivity analysis.
    """
    from gemseo.uncertainty.sensitivity.factory import SensitivityAnalysisFactory

    factory = SensitivityAnalysisFactory()

    name = analysis
    if "Analysis" not in name:
        name += "Analysis"
    name = name[0].upper() + name[1:]

    return factory.create(name, samples=samples)


def load_sensitivity_analysis(file_path: str | Path) -> BaseSensitivityAnalysis:
    """Load a sensitivity analysis from the disk.

    Args:
        file_path: The path to the file.

    Returns:
        The sensitivity analysis.
    """
    return from_pickle(file_path)
