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

Based on :cite:`TedfordMartins2010`,
the scalable MDO problem proposed in :cite:`azizalaoui:hal-04002825`
aims to minimize the objective :math:`x_0^Tx_0 + \sum_{i=1}^N y_i^Ty_i`
whilst satisfying the constraints :math:`t_1-y_1\leq 0,\ldots,t_N-y_N\leq 0`.

.. seealso:: :class:`~.disciplines.main_discipline.MainDiscipline`

:math:`y_1,\ldots,y_N` are computed by :math:`N` strongly coupled disciplines
as :math:`y_i=a_i-D_{i,0}x_0-D_{i,i}x_i+\sum_{j=1\atop j\neq i}^N C_{i,j}y_j`
where :math:`a_i`, :math:`D_{i,0}`, :math:`D_{i,i}` and :math:`C_{i,j}`
are realizations of random matrices whose coefficients are independent random variables
identically distributed as the uniform distribution over :math:`[0,1]`.

.. seealso:: :class:`~.disciplines.scalable_discipline.ScalableDiscipline`

The design vector :math:`x=(x_0,x_1,\ldots,x_N)`
belongs to the design space :math:`[0,1]^{d_0}\times[0,1]^{d_1}\ldots[0,1]^{d_N}`.

.. seealso:: :class:`~.scalable_design_space.ScalableDesignSpace`

The implementation proposes a core that is not based on |g| objects
(only NumPy and SciPy capabilities) to experiment the scalable problem outside |g|
as well as |g| versions of these core elements.

This problem is said to be *scalable*
because several sizing features can be chosen by the user:

- the number of scalable disciplines :math:`N`,
- the number of shared design variables :math:`x_0`,
- the number of local design variables :math:`x_i` for each scalable discipline,
- the number of coupling variables :math:`y_i` for each scalable discipline.

The scalable problem is particularly useful to compare different MDO formulations
with respect to the sizing configuration.

The class :class:`~.scalable_problem.ScalableProblem` helps to define a scalable problem
from :class:`~.core.scalable_discipline_settings.ScalableDisciplineSettings`,
a number of shared design variables and a level of feasibility.
It also proposes a method :meth:`~.scalable_problem.ScalableProblem.create_scenario`
to create a scenario for an :class:`.MDOFormulation`
and a method :meth:`.ScalableProblem.create_quadratic_programming_problem`
to rewrite the MDO problem as a quadratic :class:`.OptimizationProblem`.
Lastly,
the problem can be made uncertain by adding a centered random vector
per coupling equation:
:math:`Y_i=a_i-D_{i,0}x_0-D_{i,i}x_i+\sum_{j=1\atop j\neq i}^N C_{i,j}y_j+U_i`.
These random vectors :math:`U_1,\ldots,U_N` are independent
and the covariance matrix of :math:`U_i` is denoted :math:`\Sigma_i`.
"""

from __future__ import annotations
