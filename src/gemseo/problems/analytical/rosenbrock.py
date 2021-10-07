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
#    INITIAL AUTHORS - API and implementation and/or documentation
#        :author: Damien Guenot
#        :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""
The Rosenbrock analytic problem
*******************************
"""
from __future__ import division, unicode_literals

import logging

from numpy import array, atleast_2d, ones, zeros
from scipy.optimize import rosen, rosen_der

from gemseo.algos.design_space import DesignSpace
from gemseo.algos.opt_problem import OptimizationProblem
from gemseo.core.discipline import MDODiscipline
from gemseo.core.mdofunctions.mdo_function import MDOFunction

LOGGER = logging.getLogger(__name__)


class Rosenbrock(OptimizationProblem):
    r"""**Rosenbrock** :class:`.OptimizationProblem`
    uses the Rosenbrock objective function

    .. math::

       f(x) = \sum_{i=2}^{n_x} 100(x_{i} - x_{i-1}^2)^2 + (1 - x_{i-1})^2

    with the default :class:`.DesignSpace`
    :math:`[-0.2,0.2]^{n_x}`.
    """

    def __init__(self, n_x=2, l_b=-2.0, u_b=2.0, scalar_var=False, initial_guess=None):
        """
        The constructor initializes the Rosenbrock
        :class:`.OptimizationProblem`
        by defining the :class:`.DesignSpace` and
        the objective function.

        :param n_x: problem dimension
        :type n_x: int
        :param l_b: lower bound (common value to all variables)
        :type l_b: float
        :param u_b: upper bound (common value to all variables)
        :type u_b: float
        :param scalar_var: if True the design space will
            contain only scalar variables
            (as many as the problem dimension); if False
            the design space will contain a
            single multidimensional variable (whose size
            equals the problem dimension)
        :type scalar_var: bool
        :param initial_guess: initial guess for optimal solution
        :type initial_guess: numpy array
        """
        design_space = DesignSpace()
        if scalar_var:
            args = ["x" + str(i) for i in range(1, n_x + 1)]
            for arg in args:
                design_space.add_variable(arg, l_b=l_b, u_b=u_b)
        else:
            args = ["x"]
            design_space.add_variable("x", size=n_x, l_b=l_b, u_b=u_b)
        if initial_guess is None:
            design_space.set_current_x(zeros(n_x))
        else:
            design_space.set_current_x(initial_guess)

        super(Rosenbrock, self).__init__(design_space)
        expr = "sum(100*(x[1:] - x[:-1]**2)**2 + (1 - x[:-1])**2"
        self.objective = MDOFunction(
            rosen,
            name="rosen",
            f_type=MDOFunction.TYPE_OBJ,
            jac=rosen_der,
            expr=expr,
            args=args,
        )

    def get_solution(self):
        """Return the theoretical optimal value.

        :returns: design variables values of optimized values,
            function value at optimum
        :rtype: numpy array
        """
        x_opt = ones(self.design_space.dimension)
        f_opt = 0.0
        return x_opt, f_opt


class RosenMF(MDODiscipline):
    r"""**RosenMF**, a multi-fidelity Rosenbrock
    :class:`.MDODiscipline`,
    returns the value:

    .. math::

       \mathrm{fidelity} * \mathrm{Rosenbrock}(x)

    where both :math:`\mathrm{fidelity}` and :math:`x`
    are provided as input data.
    """

    def __init__(self, dimension=2):
        """The constructor defines the default inputs of the :class:`.MDODiscipline`,
        namely the default design parameter values and the fidelity.

        :param dimension: problem dimension
        :type dimension: int
        """
        super(RosenMF, self).__init__(auto_detect_grammar_files=True)
        self.default_inputs = {"x": zeros(dimension), "fidelity": array([1.0])}

    def _run(self):
        fidelity = self.local_data["fidelity"]
        x_val = self.local_data["x"]
        self.local_data["rosen"] = fidelity * rosen(x_val)

    def _compute_jacobian(self, inputs=None, outputs=None):
        x_val = self.local_data["x"]
        fidelity = self.local_data["fidelity"]
        self.jac = {
            "rosen": {
                "x": atleast_2d(fidelity * rosen_der(x_val)),
                "fidelity": atleast_2d(rosen(x_val)),
            }
        }
