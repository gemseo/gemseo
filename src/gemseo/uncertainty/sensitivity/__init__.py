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
"""Capabilities to run a sensitivity analysis.

This package contains:

- an abstract class
  [BaseSensitivityAnalysis][gemseo.uncertainty.sensitivity.base.BaseSensitivityAnalysis]]
  to define the concept of sensitivity analysis,
- a factory
  [SensitivityAnalysisFactory][gemseo.uncertainty.sensitivity.factory.SensitivityAnalysisFactory]
  to create instances of
  [BaseSensitivityAnalysis][gemseo.uncertainty.sensitivity.base.BaseSensitivityAnalysis]],
- concrete classes implementing this abstract class:
  [CorrelationAnalysis][gemseo.uncertainty.sensitivity.correlation.CorrelationAnalysis]
  (based on OpenTURNS),
  [MorrisAnalysis][gemseo.uncertainty.sensitivity.morris.MorrisAnalysis],
  [SobolAnalysis][gemseo.uncertainty.sensitivity.sobol.SobolAnalysis]
  (based on OpenTURNS)
  and [HSICAnalysis][gemseo.uncertainty.sensitivity.hsic.HSICAnalysis]
  (based on OpenTURNS).
"""

from __future__ import annotations
