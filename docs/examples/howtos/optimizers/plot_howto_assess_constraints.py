# Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com
#
# This work is licensed under a BSD 0-Clause License.
#
# Permission to use, copy, modify, and/or distribute this software
# for any purpose with or without fee is hereby granted.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL
# WARRANTIES WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL
# THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT, INDIRECT,
# OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING
# FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT,
# NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION
# WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
"""# Assess the constraints at a point

## Problem

You need to know not only whether a point satisfies the constraints
of an optimization problem,
but also which constraints it violates,
by how much,
and which constraint values are missing.

## Solution

Call
[assess()][gemseo.optimization.core.constraints.Constraints.assess]
on the `constraints` attribute of the
[OptimizationProblem][gemseo.optimization.problem.OptimizationProblem]
with the map from the constraint names to the constraint values.
It returns a
[ConstraintsAssessment][gemseo.optimization.core.constraints_assessment.ConstraintsAssessment]
giving the status and the violations of each constraint.

## Step-by-step guide
"""

from __future__ import annotations

from numpy import array
from numpy import nan

from gemseo.core.function.array_function import ArrayFunction
from gemseo.optimization import OptimizationProblem
from gemseo.space import DesignSpace

# %%
# ### Prerequisites
#
# An optimization problem with two design variables $x_0$ and $x_1$,
# the objective $f(x) = x_0^2 + x_1^2$,
# the inequality constraint $g(x) = x_0 + x_1 - 1 \leq 0$
# and the equality constraint $h(x) = x_0 - x_1 = 0$:

design_space = DesignSpace()
design_space.add_real_variable("x", size=2, lower_bound=-1.0, upper_bound=1.0)

g = ArrayFunction(lambda x: array([x[0] + x[1] - 1.0]), name="g")
h = ArrayFunction(lambda x: array([x[0] - x[1]]), name="h")

problem = OptimizationProblem(design_space)
problem.objective = ArrayFunction(lambda x: array([x @ x]), name="f")
problem.add_constraint(g, constraint_type="ineq")
problem.add_constraint(h, constraint_type="eq")

# %%
# ### 1. Assess a feasible point
#
# A point is the map from the constraint names to the constraint values,
# here computed at $x = (0.25, 0.25)$:

x = array([0.25, 0.25])
point = {"g": g.evaluate(x), "h": h.evaluate(x)}
assessment = problem.constraints.assess(point)
assessment.are_satisfied

# %%
# The status of each constraint is in `name_to_assessment`:

assessment.name_to_assessment

# %%
# ### 2. Assess a point violating the constraints
#
# At $x = (1, 0.6)$,
# both constraints are violated:

x = array([1.0, 0.6])
point = {"g": g.evaluate(x), "h": h.evaluate(x)}
assessment = problem.constraints.assess(point)
assessment.are_satisfied

# %%
# The assessment of a constraint gives its status,
# its squared violation,
# its maximum violation
# and its number of violated components:

assessment.name_to_assessment["g"]

# %%
# !!! note
#     The violation of a constraint is, component-wise,
#     `max(g - problem.tolerances.inequality, 0)` for an inequality
#     and `max(abs(h) - problem.tolerances.equality, 0)` for an equality.
#
# The squared violation sums the squares of the violations over the constraints,
# so it is in the squared unit of the constraints:

assessment.squared_violation

# %%
# The maximum violation is the largest violation over the constraints,
# so it is in the unit of the constraints:

assessment.max_violation

# %%
# ### 3. Assess a point with a NaN constraint value
#
# A constraint with a NaN (Not a Number) component is violated,
# with an infinite maximum violation:

point = {"g": array([nan]), "h": array([0.0])}
assessment = problem.constraints.assess(point)
assessment.name_to_assessment["g"]

# %%
# ### 4. Assess a point missing a constraint value
#
# A constraint whose value is absent from the point is not satisfied:

point = {"g": array([-0.5])}
assessment = problem.constraints.assess(point)
assessment.are_satisfied

# %%
# The names of the missing constraints are in `missing_names`:

assessment.missing_names

# %%
# The assessment of a missing constraint has infinite violations:

assessment.name_to_assessment["h"]

# %%
# Its number of violated components is 0,
# as no component of a missing value is evaluated.

# %%
# ## Summary
#
# [Constraints.assess()][gemseo.optimization.core.constraints.Constraints.assess]
# returns a
# [ConstraintsAssessment][gemseo.optimization.core.constraints_assessment.ConstraintsAssessment]
# telling whether all the constraints are satisfied (`are_satisfied`),
# the status and violations of each constraint (`name_to_assessment`),
# the total squared violation (`squared_violation`),
# the maximum violation (`max_violation`)
# and the missing constraints (`missing_names`).
# A NaN constraint value is violated
# and a missing one is not satisfied,
# both with infinite violations.
#
# ## One step further
#
# The tolerances used to assess the constraints are
# [problem.tolerances][gemseo.optimization.problem.OptimizationProblem.tolerances].
#
# For the plain yes/no answer,
# use
# [is_point_feasible()][gemseo.optimization.core.constraints.Constraints.is_point_feasible].
#
# For a design point of the optimization history,
# use
# [check_design_point_is_feasible()][gemseo.optimization.history.OptimizationHistory.check_design_point_is_feasible],
# which also returns its squared violation.
