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
"""The settings of the quasi-Monte Carlo sampling algorithm."""

from __future__ import annotations

from typing import ClassVar

from openturns import LowDiscrepancyExperiment as OTLowDiscrepancyExperiment
from openturns import LowDiscrepancySequenceImplementation

from gemseo.uncertainty.reliability.openturns.mc_settings import OT_MC_Settings


class BaseOTQMCSettings(OT_MC_Settings):  # noqa: N801
    """The base class for the settings of the quasi-Monte Carlo (QMC) sampling algorithms."""  # noqa: E501

    _SEQUENCE_CLASS: ClassVar[type[LowDiscrepancySequenceImplementation]]
    """The OpenTURNS class to instantiate the low-discrepancy sequence."""

    def create_experiment(self) -> OTLowDiscrepancyExperiment:  # noqa: D102
        sequence = self._SEQUENCE_CLASS()
        experiment = OTLowDiscrepancyExperiment(sequence, 1)
        experiment.setRandomize(True)
        return experiment
