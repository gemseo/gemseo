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
r"""Scenario adapters.

A scenario adapter is an :class:`.MDODiscipline` wrapping a :class:`.Scenario`.
A call to :meth:`.MDODiscipline.execute` triggers calls to :meth:`.Scenario.execute`.

For instance, let us consider an :class:`.MDOScenario` defining a gradient-based
constrained minimization of a cost function over a :class:`.DesignSpace` from several
:class:`.MDODiscipline` instances. If this optimization problem is not convex, it is
advisable to set up a multi-start strategy to repeat this minimization from different
starting points. in order to find a *good* local minimum. In this case, an
:class:`.MDOScenarioAdapter` takes a design value as input, use it as initial design
value of the minimization algorithm and outputs some variables of interest such as the
objective and constraints at the optimum. Then, this :class:`.MDOScenarioAdapter` can be
used as any :class:`.MDODiscipline` in a :class:`.DOEScenario` defining a sampling-based
version of the previous problem. In other words, this :class:`.DOEScenario` repeats the
gradient-based optimization from several starting points and returns the best local
minimum.

The scenario adapters can also be useful for bi-level optimization. Let us consider an
optimization problem with two design variables, namely :math:`x_1` and :math:`x_2`. The
wrapped :class:`.MDOScenario` solves the optimization problem with respect to the design
variables :math:`x_1` and another :class:`.MDOScenario` considers this
:class:`.MDOScenarioAdapter` to solve the optimization problem with respect to
:math:`x_2`. It is particularly relevant when the design variables have different
natures, e.g. :math:`x_1` is discrete and :math:`x_2` is continuous, and that dedicated
algorithms exist.
"""

from __future__ import annotations
