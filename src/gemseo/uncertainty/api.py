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
# Contributors:
#    INITIAL AUTHORS - initial API and implementation and/or initial
#                           documentation
#        :author: Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""The API for uncertainty quantification and management."""
from __future__ import annotations

from typing import Collection
from typing import Iterable
from typing import Sequence

from gemseo.algos.parameter_space import ParameterSpace
from gemseo.core.dataset import Dataset
from gemseo.core.discipline import MDODiscipline
from gemseo.uncertainty.distributions.distribution import Distribution
from gemseo.uncertainty.sensitivity.analysis import SensitivityAnalysis  # noqa: F401
from gemseo.uncertainty.statistics.statistics import Statistics


def get_available_distributions() -> list[str]:
    """Get the available distributions."""
    from gemseo.uncertainty.distributions.factory import DistributionFactory

    factory = DistributionFactory()
    return factory.available_distributions


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
        >>> from gemseo.uncertainty.api import create_distribution
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
    variables_names: Iterable[str] | None = None,
    tested_distributions: Sequence[str] | None = None,
    fitting_criterion: str = "BIC",
    selection_criterion="best",
    level: float = 0.05,
    name: str | None = None,
) -> Statistics:
    """Create a statistics toolbox, either parametric or empirical.

    If parametric, the toolbox selects a distribution from candidates,
    based on a fitting criterion and on a selection strategy.

    Args:
        dataset: A dataset.
        variables_names: The variables of interest.
            If None, consider all the variables from dataset.
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
            If None, use the concatenation of class and dataset names.

    Returns:
        A statistics toolbox.

    Examples:
        >>> from gemseo.api import (
        ...     create_discipline,
        ...     create_parameter_space,
        ...     create_scenario
        ... )
        >>> from gemseo.uncertainty.api import create_statistics
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
        ...     scenario_type="DOE"
        ... )
        >>> scenario.execute({'algo': 'OT_MONTE_CARLO', 'n_samples': 100})
        >>>
        >>> dataset = scenario.export_to_dataset(opt_naming=False)
        >>>
        >>> statistics = create_statistics(dataset)
        >>> mean = statistics.compute_mean()
    """
    from gemseo.uncertainty.statistics.empirical import EmpiricalStatistics as EmpStats
    from gemseo.uncertainty.statistics.parametric import (
        ParametricStatistics as ParamStats,
    )

    if tested_distributions is None:
        statistical_analysis = EmpStats(dataset, variables_names, name)
    else:
        statistical_analysis = ParamStats(
            dataset,
            tested_distributions,
            variables_names,
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
        >>> from gemseo.api import create_discipline, create_parameter_space
        >>> from gemseo.uncertainty.api import create_sensitivity_analysis
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


load_sensitivity_analysis = SensitivityAnalysis.load
