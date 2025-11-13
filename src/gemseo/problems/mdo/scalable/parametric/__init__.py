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
r"""A parametric scalable problem.

The scalable MDO problem aims to minimize the objective
$x_0^Tx_0 + \sum_{i=1}^N y_i^Ty_i$
whilst satisfying the constraints $t_1-y_1\leq 0,\ldots,t_N-y_N\leq 0$.

See Also:
    [MainDiscipline][gemseo.problems.mdo.scalable.parametric.disciplines.main_discipline.MainDiscipline]

$y_1,\ldots,y_N$ are computed by $N$ strongly coupled disciplines
as $y_i=a_i-D_{i,0}x_0-D_{i,i}x_i+\sum_{j=1\atop j\neq i}^N C_{i,j}y_j$
where $a_i$, $D_{i,0}$, $D_{i,i}$ and $C_{i,j}$
are realizations of random matrices whose coefficients are independent random variables
identically distributed as the uniform distribution over $[0,1]$.

See Also:
    [ScalableDiscipline][gemseo.problems.mdo.scalable.parametric.core.disciplines.scalable_discipline.ScalableDiscipline]

The design vector $x=(x_0,x_1,\ldots,x_N)$
belongs to the design space $[0,1]^{d_0}\times[0,1]^{d_1}\ldots[0,1]^{d_N}$.

See Also:
    [ScalableDesignSpace][gemseo.problems.mdo.scalable.parametric.scalable_design_space.ScalableDesignSpace]

The implementation proposes a core that is not based on GEMSEO objects
(only NumPy and SciPy capabilities) to experiment the scalable problem outside GEMSEO
as well as GEMSEO versions of these core elements.

This problem is said to be *scalable*
because several sizing features can be chosen by the user:

- the number of scalable disciplines $N$,
- the number of shared design variables $x_0$,
- the number of local design variables $x_i$ for each scalable discipline,
- the number of coupling variables $y_i$ for each scalable discipline.

The scalable problem is particularly useful to compare different MDO formulations
with respect to the sizing configuration.

The class
[ScalableProblem][gemseo.problems.mdo.scalable.parametric.scalable_problem.ScalableProblem]
helps to define a scalable problem
from
[ScalableDisciplineSettings][gemseo.problems.mdo.scalable.parametric.core.scalable_discipline_settings.ScalableDisciplineSettings],
a number of shared design variables and a level of feasibility.
It also proposes a method
[create_scenario()][gemseo.problems.mdo.scalable.parametric.scalable_problem.ScalableProblem.create_scenario]
to create a scenario for a
[BaseMDOFormulation][gemseo.formulations.base_mdo_formulation.BaseMDOFormulation]
and a method
[create_quadratic_programming_problem()][gemseo.problems.mdo.scalable.parametric.scalable_problem.ScalableProblem.create_quadratic_programming_problem]
to rewrite the MDO problem as a quadratic
[OptimizationProblem][gemseo.algos.optimization_problem.OptimizationProblem].
Lastly,
the problem can be made uncertain by adding a centered random vector
per coupling equation:
$Y_i=a_i-D_{i,0}x_0-D_{i,i}x_i+\sum_{j=1\atop j\neq i}^N C_{i,j}y_j+U_i$.
These random vectors $U_1,\ldots,U_N$ are independent
and the covariance matrix of $U_i$ is denoted $\Sigma_i$.

!!! quote "References"
    Nathan P. Tedford and Joaquim R.R.A. Martins.
    [Benchmarking multidisciplinary design optimization algorithms.](http://dx.doi.org/10.1007/s11081-009-9082-6)
    Optimization and Engineering, 11(1):159-183, February 2010.

    Amine Aziz-Alaoui, Olivier Roustant, and Matthias De Lozzo.
    A scalable problem to benchmark robust multidisciplinary design optimization
    techniques. Optimization and Engineering, 25(2):941-958, 2024.

"""

from __future__ import annotations
