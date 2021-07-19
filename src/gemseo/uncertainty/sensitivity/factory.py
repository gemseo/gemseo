# -*- coding: utf-8 -*-
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

from __future__ import division, unicode_literals

import logging
from typing import List

from gemseo.algos.design_space import DesignSpace
from gemseo.core.discipline import MDODiscipline
from gemseo.core.factory import Factory
from gemseo.uncertainty.sensitivity.analysis import SensitivityAnalysis

LOGGER = logging.getLogger(__name__)


class SensitivityAnalysisFactory(object):
    """Factory to build instances of :class:`.SensitivityAnalysis`.

    At initialization, this factory scans the following modules
    to search for subclasses of this class:

    - the modules located in "gemseo.uncertainty.sensitivity" and its sub-packages,
    - the modules referenced in the "GEMSEO_PATH",
    - the modules referenced in the "PYTHONPATH" and starting with "gemseo_".

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
        ...     "AnalyticDiscipline", expressions_dict=expressions
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

    def __init__(self):  # type: (...) -> None  # noqa: D107
        self.factory = Factory(SensitivityAnalysis, ("gemseo.uncertainty.sensitivity",))

    def create(
        self,
        sensitivity_analysis,  # type:str
        discipline,  # type: MDODiscipline
        parameter_space,  # type: DesignSpace
        **options
    ):  # type:  (...) -> SensitivityAnalysis
        """Create the sensitivity analysis.

        Args:
            sensitivity_analysis (str): The name of a class
                defining a sensitivity analysis.
            discipline (MDODiscipline): A discipline.
            parameter_space (ParameterSpace): A parameter space.
            **options: The options of the sensitivity analysis.

        Returns:
            SensitivityAnalysis: A sensitivity analysis.
        """
        return self.factory.create(
            sensitivity_analysis,
            discipline=discipline,
            parameter_space=parameter_space,
            **options
        )

    @property
    def available_sensitivity_analyses(self):  # type: (...)-> List[str]
        """The available classes for sensitivity analysis."""
        return self.factory.classes

    def is_available(
        self,
        sensitivity_analysis,  # type: str
    ):  # type: (...) -> bool
        """Check the availability of a SensitivityAnalysis child.

        Args:
            sensitivity_analysis: The name of the sensitivity analysis.

        Returns:
            Availability of a sensitivity analysis.

            True if the type of sensitivity analysis is available.
        """
        return self.factory.is_available(sensitivity_analysis)
