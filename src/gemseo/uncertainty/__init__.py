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

The package [gemseo.uncertainty][gemseo.uncertainty] provides several functionalities
to quantify and manage uncertainties.
Most of them can be used from the high-level functions provided by this module.

The sub-package  [gemseo.uncertainty.distribution][gemseo.uncertainty.distribution]
offers an abstract level
for probability distributions, as well as interfaces to the OpenTURNS and SciPy ones.
It is also possible to fit a probability distribution from data
or select the most likely one from a list of candidates.
These distributions can be used to define random variables
in a [ParameterSpace][gemseo.space.parameter.ParameterSpace]
before propagating these uncertainties through
a system of [Discipline][gemseo.core.discipline.discipline.Discipline],
by means of an
[EvaluationScenario][gemseo.scenario.evaluation.EvaluationScenario].

See Also:
    [OTDistribution][gemseo.uncertainty.distribution.openturns.distribution.OTDistribution]
    [SPDistribution][gemseo.uncertainty.distribution.scipy.distribution.SPDistribution]
    [OTDistributionFitter][gemseo.uncertainty.distribution.openturns.distribution_fitter.OTDistributionFitter]

The sub-package [gemseo.uncertainty.sensitivity][gemseo.uncertainty.sensitivity]
offers an abstract level
for sensitivity analysis, as well as concrete features.
These sensitivity analyses compute indices by means of various methods:
correlation measures, Morris technique and Sobol' variance decomposition.
This sub-package is based in particular on OpenTURNS.

See Also:
    [CorrelationAnalysis][gemseo.uncertainty.sensitivity.correlation.CorrelationAnalysis]
    [MorrisAnalysis][gemseo.uncertainty.sensitivity.morris.MorrisAnalysis]
    [SobolAnalysis][gemseo.uncertainty.sensitivity.sobol.SobolAnalysis]
    [HSICAnalysis][gemseo.uncertainty.sensitivity.hsic.HSICAnalysis]
    [FORMAnalysis][gemseo.uncertainty.sensitivity.form.FORMAnalysis]

The sub-package [gemseo.uncertainty.statistic][gemseo.uncertainty.statistic]
offers an abstract level
for statistics, as well as parametric and empirical versions.
Empirical statistics are estimated from a [Dataset][gemseo.dataset.dataset.Dataset]
while parametric statistics are analytical properties of a
[BaseDistribution][gemseo.uncertainty.distribution.core.base.BaseDistribution]
fitted from a [Dataset][gemseo.dataset.dataset.Dataset].

See Also:
    [EmpiricalStatistics][gemseo.uncertainty.statistic.empirical.EmpiricalStatistics]
    [OTParametricStatistics][gemseo.uncertainty.statistic.ot_parametric.OTParametricStatistics]
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

from gemseo.util.pickle import from_pickle as from_pickle

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Sequence

    from gemseo.dataset.dataset import Dataset
    from gemseo.dataset.io_dataset import IODataset as IODataset
    from gemseo.uncertainty.distribution.core.base import BaseDistribution
    from gemseo.uncertainty.sensitivity.core.base import BaseGenericSensitivityAnalysis
    from gemseo.uncertainty.sensitivity.core.base import (
        BaseSensitivityAnalysis as BaseSensitivityAnalysis,
    )
    from gemseo.uncertainty.statistic.core.base import BaseStatistics
    from gemseo.util.typing import StrPath


def get_available_distributions(base_class_name: str = "BaseDistribution") -> list[str]:
    """Get the available probability distributions.

    Args:
        base_class_name: The name of the base class of the probability distributions,
            e.g. `"BaseDistribution"`, `"OTDistribution"` or `"SPDistribution"`.

    Returns:
        The names of the available probability distributions.
    """
    from gemseo.uncertainty.distribution.factory import DISTRIBUTION_FACTORY

    class_names = DISTRIBUTION_FACTORY.class_names
    if base_class_name == "BaseDistribution":
        return class_names

    return [
        class_name
        for class_name in class_names
        if base_class_name
        in [cls.__name__ for cls in DISTRIBUTION_FACTORY.get_class(class_name).mro()]
    ]


def create_distribution(
    distribution_name: str,
    **options: Any,
) -> BaseDistribution:
    """Create a distribution.

    Args:
        distribution_name: The name of a class
            implementing a probability distribution,
            e.g. 'OTUniformDistribution' or 'SPDistribution'.
        **options: The distribution options.
    """
    from gemseo.uncertainty.distribution.factory import DISTRIBUTION_FACTORY

    return DISTRIBUTION_FACTORY.create(distribution_name, **options)


def get_available_sensitivity_analyses() -> list[str]:
    """Get the available sensitivity analyses."""
    from gemseo.uncertainty.sensitivity.factory import SENSITIVITY_ANALYSIS_FACTORY

    return SENSITIVITY_ANALYSIS_FACTORY.class_names


def create_statistics(
    dataset: Dataset,
    variable_names: Iterable[str] = (),
    tested_distributions: Sequence[str] = (),
    fitting_criterion: str = "",
    selection_criterion: str = "best",
    level: float = 0.05,
    name: str = "",
) -> BaseStatistics:
    """Create a toolbox to estimate statistics, either empirically or parametrically.

    If parametrically,
    the toolbox selects a distribution from candidates,
    based on a goodness-of-fit criterion and on a selection strategy.

    Args:
        dataset: A dataset.
        variable_names: The names of the variables of interest.
            If empty, consider all the variables of the dataset.
        tested_distributions: The names of the probability distributions
            to be used as candidates.
            Either SciPy class names or OpenTURNS class names.
            Do not mix SciPy and OpenTURNS class names.
        fitting_criterion: The name of a goodness-of-fit criterion,
            measuring how a distribution fits the data.
            If empty,
            use
            [OTDistributionFitter.default_fitting_criterion][gemseo.uncertainty.distribution.openturns.distribution_fitter.OTDistributionFitter.default_fitting_criterion]
            or
            [SPDistributionFitter.default_fitting_criterion][gemseo.uncertainty.distribution.scipy.distribution_fitter.SPDistributionFitter.default_fitting_criterion]
            according to the type of `tested_distributions`.
        selection_criterion: The name of a selection criterion
            to select a distribution from `tested_distributions`.
            Either `"first"`
            (select the first distribution satisfying a fitting criterion)
            or `"best"`
            (select the distribution that best satisfies a fitting criterion).
        level: A test level,
            i.e. the risk of committing a Type 1 error,
            that is an incorrect rejection of a true null hypothesis,
            for criteria based on a test hypothesis.
        name: A name for the statistics toolbox.
            If empty, concatenate the statistics class name and the dataset name.

    Returns:
        A statistics toolbox.
    """
    import openturns as ot

    from gemseo.uncertainty.statistic.empirical import EmpiricalStatistics
    from gemseo.uncertainty.statistic.ot_parametric import OTParametricStatistics
    from gemseo.uncertainty.statistic.sp_parametric import SPParametricStatistics

    if tested_distributions:
        cls = (
            OTParametricStatistics
            if hasattr(ot, tested_distributions[0])
            else SPParametricStatistics
        )
        statistical_analysis = cls(
            dataset,
            tested_distributions,
            variable_names=variable_names,
            fitting_criterion=fitting_criterion,
            level=level,
            selection_criterion=selection_criterion,
            name=name,
        )
    else:
        statistical_analysis = EmpiricalStatistics(dataset, variable_names, name)
    return statistical_analysis


def create_sensitivity_analysis(
    analysis: str,
    samples: IODataset | StrPath = "",
) -> BaseGenericSensitivityAnalysis:
    """Create the sensitivity analysis.

    Args:
        analysis: The name of a sensitivity analysis class.
        samples: The samples for the estimation of the sensitivity indices,
            either as an [IODataset][gemseo.dataset.io_dataset.IODataset]
            or as a pickle file path generated
            from the [to_pickle()][gemseo.util.pickle.to_pickle] function.
            If empty, use
            [compute_samples()][gemseo.uncertainty.sensitivity.core.base.BaseSensitivityAnalysis.compute_samples].

    Returns:
        The sensitivity analysis.
    """
    from gemseo.uncertainty.sensitivity.factory import SENSITIVITY_ANALYSIS_FACTORY

    factory = SENSITIVITY_ANALYSIS_FACTORY

    name = analysis
    if "Analysis" not in name:
        name += "Analysis"
    name = name[0].upper() + name[1:]

    return factory.create(name, samples=samples)


def load_sensitivity_analysis(file_path: StrPath) -> BaseGenericSensitivityAnalysis:
    """Load a sensitivity analysis from the disk.

    Args:
        file_path: The path to the file.

    Returns:
        The sensitivity analysis.
    """
    return from_pickle(file_path)
