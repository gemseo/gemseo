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

A scenario adapter is a [Discipline][gemseo.core.discipline.discipline.Discipline]
wrapping an
[EvaluationScenario][gemseo.scenario.evaluation.EvaluationScenario].
A call to [Discipline.execute()][gemseo.core.discipline.discipline.Discipline.execute]
triggers calls
to
[EvaluationScenario.execute][gemseo.scenario.evaluation.EvaluationScenario.execute].

[EvaluationScenarioAdapter][gemseo.scenario.adapter.evaluation.EvaluationScenarioAdapter]
adapts any [EvaluationScenario][gemseo.scenario.evaluation.EvaluationScenario]
and outputs the last design point that it evaluated.
[MDOScenarioAdapter][gemseo.scenario.adapter.mdo.MDOScenarioAdapter] derives from it
and shall be used
when the scenario solves an
[OptimizationProblem][gemseo.optimization.problem.OptimizationProblem],
e.g. an [MDOScenario][gemseo.scenario.mdo.MDOScenario],
as it outputs the optimum,
can output the Lagrange multipliers of the optimal solution
and can be linearized by post-optimal analysis.

For instance,
let us consider an [MDOScenario][gemseo.scenario.mdo.MDOScenario]
defining a gradient-based constrained minimization of a cost function
over a [DesignSpace][gemseo.space.design.DesignSpace]
from several [Discipline][gemseo.core.discipline.discipline.Discipline] instances.
If this optimization problem is not convex,
it is advisable to set up a multi-start strategy
to repeat this minimization from different starting points
in order to find a *good* local minimum.
In this case,
an
[MDOScenarioAdapter][gemseo.scenario.adapter.mdo.MDOScenarioAdapter]
takes a design value as input,
use it as initial design value of the minimization algorithm
and outputs some variables of interest
such as the objective and constraints at the optimum.
Then,
this
[MDOScenarioAdapter][gemseo.scenario.adapter.mdo.MDOScenarioAdapter]
can be used as any [Discipline][gemseo.core.discipline.discipline.Discipline]
in a [MDOScenario][gemseo.scenario.mdo.MDOScenario] using a DOE algorithm.
In other words,
this [MDOScenario][gemseo.scenario.mdo.MDOScenario] repeats
the gradient-based optimization from several starting points
and returns the best local minimum.

The scenario adapters can also be useful for bi-level optimization.
Let us consider an optimization problem with two design variables,
namely $x_1$ and $x_2$.
The wrapped [MDOScenario][gemseo.scenario.mdo.MDOScenario] solves
the optimization problem with respect to the design variables $x_1$
and another [MDOScenario][gemseo.scenario.mdo.MDOScenario] considers
this
[MDOScenarioAdapter][gemseo.scenario.adapter.mdo.MDOScenarioAdapter]
to solve the optimization problem with respect to $x_2$.
It is particularly relevant when the design variables have different natures,
e.g. $x_1$ is discrete and $x_2$ is continuous,
and that dedicated algorithms exist.
"""

from __future__ import annotations
