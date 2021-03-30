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
#       :author: Pierre-Jean Barjhoux
#       :author: Francois Gallard, integration and cleanup
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""
Implementation of the Lagrange multipliers
******************************************
"""
from __future__ import absolute_import, division, unicode_literals

from future import standard_library
from numpy import arange, array, atleast_2d, concatenate, where, zeros
from numpy.linalg import lstsq, matrix_rank, norm

from gemseo.algos.design_space import DesignSpace
from gemseo.algos.opt_problem import OptimizationProblem
from gemseo.third_party.prettytable import PrettyTable

standard_library.install_aliases()


from gemseo import LOGGER


class LagrangeMultipliers(object):
    r"""
    Class that implements the computation of Lagrange Multipliers

    Denote :math:`x^\ast` an optimal solution of the optimization problem
    below.

    .. math::
        \begin{aligned}
            & \text{Minimize}    & & f(x) \\
            & \text{relative to} & & x \\
            & \text{subject to}  & & \left\{\begin{aligned}
                                                & g(x)\le0, \\
                                                & h(x)=0, \\
                                                & \ell\le x\le u.
                                     \end{aligned}\right.
        \end{aligned}

    If the constraints are qualified at :math:`x^\ast` then the Lagrange
    multipliers of :math:`x^\ast` are the vectors :math:`\lambda_g`,
    :math:`\lambda_h`, :math:`\lambda_\ell` and :math:`\lambda_u` satisfying

    .. math::
        \left\{\begin{aligned}
            &\frac{\partial f}{\partial x}(x^\ast)
            +\lambda_g^\top\frac{\partial g}{\partial x}(x^\ast)
            +\lambda_h^\top\frac{\partial h}{\partial x}(x^\ast)
            +\sum_j\lambda_{\ell,j}+\sum_j\lambda_{u,j}
            =0,\\
            &\lambda_{g,i}\ge0\text{ if }g_i(x^\ast)=0,
            \text{ otherwise }\lambda_{g,i}=0,\\
            &\lambda_{\ell,j}\le0\text{ if }x^\ast_j=\ell_j,
            \text{ otherwise }\lambda_{\ell,j}=0,\\
            &\lambda_{u,j}\ge0\text{ if }x^\ast_j=u_j,
            \text{ otherwise }\lambda_{u,j}=0.
        \end{aligned}\right.

    """
    LOWER_BOUNDS = "lower_bounds"
    UPPER_BOUNDS = "upper_bounds"
    INEQUALITY = "inequality"
    EQUALITY = "equality"
    CSTR_LABELS = [LOWER_BOUNDS, UPPER_BOUNDS, INEQUALITY, EQUALITY]

    def __init__(self, opt_problem):
        """
        Constructor

        :param opt_problem: optimization problem on which Lagrange multipliers
            shall be computed
        """
        self._check_inputs(opt_problem)
        self.opt_problem = opt_problem
        self.active_lb_names = []
        self.active_ub_names = []
        self.active_ineq_names = []
        self.active_eq_names = []
        self.lagrange_multipliers = None
        self.__normalized = opt_problem.preprocess_options.get("normalize", False)

    @staticmethod
    def _check_inputs(opt_problem):
        """
        Checks inputs : verify that opt_problem is an
        instance of OptimizationProblem

        :param opt_problem: optimization problem on which Lagrange multipliers
            shall be computed
        """
        if not isinstance(opt_problem, OptimizationProblem):
            raise ValueError(
                "Argument of LagrangeMultiplier class"
                + " has to be an instance of OptimizationProblem"
            )
        if opt_problem.solution is None:
            raise ValueError("Optimization problem was not solved !")

    def compute(self, x_vect, ineq_tolerance=1e-6, rcond=-1):
        """
        Computes and returns the Lagrange multipliers,
        as a post-processing of the optimal point.

        This solves :
        (d ActiveConstraints)'               d Objective
        (-------------------)  . Lambda = -  -----------
        (d X                )                d X

        :param x_vect: x point on which the multipliers shall be computed
        :param ineq_tolerance: tolerance on inequality constraints
        :param rcond: float, optional
               Cut-off ratio for small singular values of the jacobian.
               see sipy.linalg.lsq
        """
        LOGGER.info("Computation of Lagrange multipliers")
        # get jacobian of all active constraints, and an
        # ordered list of their name
        jac_act, _ = self._get_jac_act(x_vect, ineq_tolerance)
        if jac_act is None:
            # There is no active constraint
            multipliers = []
            self._store_multipliers(multipliers)
            return self.lagrange_multipliers
        lhs = jac_act.T
        act_constr_nb = lhs.shape[1]
        rank = matrix_rank(lhs)
        LOGGER.info("Found %s active constraints", str(act_constr_nb))
        LOGGER.info("Rank of jacobian = %s", str(rank))
        if act_constr_nb > rank:
            LOGGER.warning("Number of active constraints > rank !")

        # get jacobian of objective
        obj_jac = self._get_obj_jac(x_vect)
        rhs = -obj_jac.T

        mul, residuals, _, sval = lstsq(lhs, rhs, rcond=rcond)
        LOGGER.info("Min singular values of jacobian = %s", str(sval.min()))
        LOGGER.info("Residuals norm = %s", str(norm(residuals)))

        # stores multipliers in a dictionary
        self._store_multipliers(mul)

        return self.lagrange_multipliers

    def _get_act_bound_jac(self, act_bounds):
        """
        Returns the jacobian of active  bounds constraints
        sign is not taken into account (matrix is made of 0 and 1)
        """
        dspace = self.opt_problem.design_space
        x_dim = dspace.dimension
        dim_act = sum(len(where(bnd)[0]) for bnd in act_bounds.values())
        if dim_act == 0:
            return None, []
        act_array = concatenate([act_bounds[var] for var in dspace.variables_names])

        bnd_jac = zeros((dim_act, x_dim))

        if self.__normalized:
            norm_factor = dspace.get_upper_bounds() - dspace.get_lower_bounds()
            act_jac = norm_factor[act_array]
        else:
            act_jac = 1.0
        bnd_jac[arange(dim_act), act_array] = act_jac
        indexed_varnames = array(dspace.get_indexed_variables_names())
        act_b_names = indexed_varnames[act_array].tolist()
        return bnd_jac, act_b_names

    def __get_act_ineq_jac(self, x_vect, ineq_tolerance=1e-6):
        """
        Returns the jacobian of active inequality
        constraints defined by user in
        the optimization problem
        """
        # retrieves the active functions and the indices :
        # a function is active if at least
        # one of its component (in case of multidimensional constraints) is
        # active
        act_func = self.opt_problem.get_active_ineq_constraints(x_vect, ineq_tolerance)

        dspace = self.opt_problem.design_space

        if self.__normalized:
            x_vect = dspace.normalize_vect(x_vect)
        jac = []
        names = []

        for func, act_set in act_func.items():
            if True in act_set:
                ineq_jac = func.jac(x_vect)
                if len(ineq_jac.shape) == 1:
                    # Make sure the Jacobian is a 2-dimensional array
                    ineq_jac = ineq_jac.reshape((1, x_vect.size))
                else:
                    ineq_jac = ineq_jac[act_set, :]
                jac.append(ineq_jac)
                if func.dim == 1:
                    names.append(func.name)
                else:
                    names += [
                        func.name + DesignSpace.SEP + str(i) for i in range(func.dim)
                    ]
        if jac:
            jac = concatenate(jac)
        else:
            jac = None
        return jac, names

    def _get_act_eq_jac(self, x_vect):
        """Returns jacobian of active equality constraints defined by user in
        the optimization problem
        """
        eq_functions = self.opt_problem.get_eq_constraints()
        # loop on equality functions
        # NB: as the solution (x_vect) is supposed to be feasible,
        # all functions (on all dimensions) are supposed to be active
        jac = []
        names = []
        dspace = self.opt_problem.design_space

        if self.__normalized:
            x_vect = dspace.normalize_vect(x_vect)

        for eq_function in eq_functions:
            eq_jac = atleast_2d(eq_function.jac(x_vect))
            jac.append(eq_jac)
            if eq_function.dim == 1:
                names.append(eq_function.name)
            else:
                names += [
                    eq_function.name + DesignSpace.SEP + str(i)
                    for i in range(eq_jac.shape[0])
                ]
        if jac:
            jac = concatenate(jac)
        else:
            jac = None
        return jac, names

    def _get_obj_jac(self, x_vect):
        """Returns objective jacobian"""
        if self.__normalized:
            x_vect = self.opt_problem.design_space.normalize_vect(x_vect)

        return self.opt_problem.objective.jac(x_vect)

    def _get_jac_act(self, x_vect, ineq_tolerance=1e-6):
        """Returns active constraints jacobian, and the
        name of each component of each function
        """
        # Bounds jacobian
        dspace = self.opt_problem.design_space
        act_lb, act_ub = dspace.get_active_bounds(x_vect, tol=ineq_tolerance)
        lb_jac_act, self.active_lb_names = self._get_act_bound_jac(act_lb)
        if lb_jac_act is not None:
            lb_jac_act *= -1
        ub_jac_act, self.active_ub_names = self._get_act_bound_jac(act_ub)

        # inequality names
        tol = ineq_tolerance
        ineq_jac, self.active_ineq_names = self.__get_act_ineq_jac(x_vect, tol)
        # equality names
        eq_jac, eq_names_act = self._get_act_eq_jac(x_vect)
        self.active_eq_names = eq_names_act

        names_list = (
            self.active_lb_names
            + self.active_ub_names
            + self.active_ineq_names
            + eq_names_act
        )
        jac_list = [lb_jac_act, ub_jac_act, ineq_jac, eq_jac]
        jac_list = [jac for jac in jac_list if jac is not None]
        if jac_list:
            jac_act_arr = concatenate(jac_list, axis=0)
        else:
            # There no active constraint
            jac_act_arr = None

        return jac_act_arr, names_list

    def _store_multipliers(self, multipliers):
        """Stores multipliers in a dictionary"""
        lag = {}

        i_min = 0
        n_act = len(self.active_lb_names)
        if n_act > 0:
            l_b_mult = multipliers[i_min : i_min + n_act]
            lag[self.LOWER_BOUNDS] = (self.active_lb_names, l_b_mult)
            i_min += n_act
            wrong_inds = where(l_b_mult < 0.0)[0]
            if wrong_inds.size > 0:
                names_neg = array(self.active_lb_names)[wrong_inds]
                LOGGER.warning(
                    "Negative Lagrange multipliers for "
                    "lower bounds on variables"
                    "%s !",
                    str(names_neg),
                )
        n_act = len(self.active_ub_names)
        if n_act > 0:
            u_b_mult = multipliers[i_min : i_min + n_act]
            lag[self.UPPER_BOUNDS] = (self.active_ub_names, u_b_mult)
            i_min += n_act
            wrong_inds = where(u_b_mult < 0.0)[0]
            if wrong_inds.size > 0:
                names_neg = array(self.active_ub_names)[wrong_inds]
                LOGGER.warning(
                    "Negative Lagrange multipliers for "
                    "upper bounds on variables"
                    "%s !",
                    str(names_neg),
                )
        n_act = len(self.active_ineq_names)
        if n_act > 0:
            ineq_mult = multipliers[i_min : i_min + n_act]
            lag[self.INEQUALITY] = (self.active_ineq_names, ineq_mult)
            i_min += n_act
            wrong_inds = where(ineq_mult < 0.0)[0]
            if wrong_inds.size > 0:
                names_neg = array(self.active_ineq_names)[wrong_inds]
                LOGGER.warning(
                    "Negative Lagrange multipliers for "
                    "inequality constraints"
                    "%s !",
                    str(names_neg),
                )
        if self.active_eq_names:
            lag[self.EQUALITY] = (
                self.active_eq_names,
                multipliers[i_min : i_min + n_act],
            )
            i_min += n_act

        self.lagrange_multipliers = lag

    def _get_pretty_table(self):
        """Displays Lagrange Multipliers"""
        table = PrettyTable(
            ["Constraint type", "Constraint name", "Lagrange Multiplier"]
        )

        for cstr_type, nam_val in self.lagrange_multipliers.items():
            for name, value in zip(nam_val[0], nam_val[1]):
                table.add_row([cstr_type, name, value])

        return table

    def log_me(self):
        """Logs a representation of the optimization problem characteristics
        logs self.__repr__ message
        """
        msg = str(self)
        for line in msg.split("\n"):
            LOGGER.info(line)

    def __str__(self, *args, **kwargs):
        """
        Textual representation of the design space
        """
        desc = "Lagrange multipliers : "
        desc += "\n" + str(self._get_pretty_table().get_string())
        return desc
