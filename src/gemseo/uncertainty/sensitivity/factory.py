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
"""Module with a factory to create an instance of :class:`.SensitivityAnalysis`."""
from __future__ import annotations

from typing import Any
from typing import Collection
from typing import Iterable
from typing import Mapping

from gemseo.algos.doe.doe_lib import DOELibraryOptionType
from gemseo.algos.parameter_space import ParameterSpace
from gemseo.core.discipline import MDODiscipline
from gemseo.core.factory import Factory
from gemseo.uncertainty.sensitivity.analysis import SensitivityAnalysis


class SensitivityAnalysisFactory:
    """Factory to build instances of :class:`.SensitivityAnalysis`.

    At initialization, this factory scans the following modules
    to search for subclasses of this class:

    - the modules located in ``gemseo.uncertainty.sensitivity`` and its sub-packages,
    - the modules referenced in the ``GEMSEO_PATH,``
    - the modules referenced in the ``PYTHONPATH`` and starting with ``gemseo_``.

    Then, it can check if a class is present or return the list of available classes.

    Lastly, it can create an instance of a class.

    Examples:
        >>> from numpy import pi
        >>> from gemseo.api import create_discipline, create_parameter_space
        >>> from gemseo.uncertainty.sensitivity.factory import (
        ...     SensitivityAnalysisFactory
        ... )
        >>>
        >>> expressions = {"y": "sin(x1)+7*sin(x2)**2+0.1*x3**4*sin(x1)"}
        >>> discipline = create_discipline(
        ...     "AnalyticDiscipline", expressions=expressions
        ... )
        >>>
        >>> parameter_space = create_parameter_space()
        >>> parameter_space.add_random_variable(
        ...     "x1", "OTUniformDistribution", minimum=-pi, maximum=pi
        ... )
        >>> parameter_space.add_random_variable(
        ...     "x2", "OTUniformDistribution", minimum=-pi, maximum=pi
        ... )
        >>> parameter_space.add_random_variable(
        ...     "x3", "OTUniformDistribution", minimum=-pi, maximum=pi
        ... )
        >>>
        >>> factory = SensitivityAnalysisFactory()
        >>> analysis = factory.create(
        ...     "MorrisIndices", discipline, parameter_space, n_replicates=5
        ... )
        >>> indices = analysis.compute_indices()
    """

    def __init__(self) -> None:  # noqa: D107
        self.factory = Factory(SensitivityAnalysis, ("gemseo.uncertainty.sensitivity",))

    def create(
        self,
        sensitivity_analysis: str,
        disciplines: Collection[MDODiscipline],
        parameter_space: ParameterSpace,
        n_samples: int | None = None,
        output_names: Iterable[str] = None,
        algo: str | None = None,
        algo_options: Mapping[str, DOELibraryOptionType] | None = None,
        formulation: str = "MDF",
        **formulation_options: Any,
    ) -> SensitivityAnalysis:
        """Create the sensitivity analysis.

        Args:
            sensitivity_analysis: The name of a class
                defining a sensitivity analysis.
            disciplines: The discipline or disciplines to use for the analysis.
            parameter_space: A parameter space.
            n_samples: A number of samples.
                If ``None``, the number of samples is computed by the algorithm.
            output_names: The disciplines' outputs to be considered for the analysis.
                If ``None``, use all the outputs.
            algo: The name of the DOE algorithm.
                If ``None``, use the :attr:`.SensitivityAnalysis.DEFAULT_DRIVER`.
            algo_options: The options of the DOE algorithm.
            formulation: The name of the :class:`.MDOFormulation` to sample the
                disciplines.
            **formulation_options: The options of the :class:`.MDOFormulation`.

        Returns:
            A sensitivity analysis.
        """
        return self.factory.create(
            sensitivity_analysis,
            disciplines=disciplines,
            parameter_space=parameter_space,
            n_samples=n_samples,
            output_names=output_names,
            algo=algo,
            algo_options=algo_options,
            formulation=formulation,
            **formulation_options,
        )

    @property
    def available_sensitivity_analyses(self) -> list[str]:
        """The available classes for sensitivity analysis."""
        return self.factory.classes

    def is_available(
        self,
        sensitivity_analysis: str,
    ) -> bool:
        """Check the availability of a SensitivityAnalysis child.

        Args:
            sensitivity_analysis: The name of the sensitivity analysis.

        Returns:
            Whether the type of sensitivity analysis is available.
        """
        return self.factory.is_available(sensitivity_analysis)
