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
#        :author: Damien Guenot
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
#         Francois Gallard : refactoring for v1, May 2016
"""SNOPT optimization library wrapper."""

from __future__ import division, unicode_literals

import logging
from typing import Any, Callable, Dict, Optional, Tuple, Union

from numpy import append, array, concatenate
from numpy import float as np_float
from numpy import float64, hstack
from numpy import int as np_int
from numpy import isinf, ndarray, ones, reshape
from numpy import str as np_str
from numpy import vstack, where, zeros
from optimize.snopt7 import SNOPT_solver

from gemseo.algos.opt.opt_lib import OptimizationLibrary
from gemseo.algos.opt_result import OptimizationResult

LOGGER = logging.getLogger(__name__)
INFINITY = 1e30

OptionType = Optional[Union[str, int, float, bool, ndarray]]

SnOptPreprocessType = Tuple[
    Callable[[int, int, int, int, ndarray, int], Any],
    ndarray,
    ndarray,
    ndarray,
    int,
]


class SnOpt(OptimizationLibrary):
    """SNOPT optimization library interface.

    See OptimizationLibrary.
    """

    LIB_COMPUTE_GRAD = False
    OPTIONS_MAP = {"max_iter": "Iteration_limit"}

    MESSAGES_DICT = {
        1: "optimality conditions satisfied",
        2: "feasible point found",
        3: "requested accuracy could not be achieved",
        11: "infeasible linear constraints",
        12: "infeasible linear equalities",
        13: "nonlinear infeasibilities minimized",
        14: "infeasibilities minimized",
        21: "unbounded objective",
        22: "constraint violation limit reached",
        31: "iteration limit reached",
        32: "major iteration limit reached",
        33: "the superbasics limit is too small",
        41: "current point cannot be improved ",
        42: "singular basis",
        43: "cannot satisfy the general constraints",
        44: "ill-conditioned null-space basis",
        51: "incorrect objective derivatives",
        52: "incorrect constraint derivatives",
        61: "undefined function at the first feasible point",
        62: "undefined function at the initial point",
        63: "unable to proceed into undefined region",
        72: "terminated during constraint evaluation",
        73: "terminated during objective evaluation",
        74: "terminated from monitor routine",
        81: "work arrays must have at least 500 elements",
        82: "not enough character storage",
        83: "not enough integer storage",
        84: "not enough real storage",
        91: "invalid input argument",
        92: "basis file dimensions do not match this problem",
        141: "wrong number of basic variables",
        142: "error in basis package",
    }

    def __init__(self):
        """Constructor.

        Generate the library dict, contains the list
        of algorithms with their characteristics:
         * does it require gradient
         * does it handle equality constraints
         * does it handle inequality constraints
        """
        super(SnOpt, self).__init__()
        self.__n_ineq_constraints = 0
        self.__n_eq_constraints = 0
        self.lib_dict = {
            "SNOPTB": {
                self.INTERNAL_NAME: "SNOPTB",
                self.REQUIRE_GRAD: True,
                self.HANDLE_EQ_CONS: True,
                self.HANDLE_INEQ_CONS: True,
                self.DESCRIPTION: "Sparse Nonlinear OPTimizer (SNOPT)",
                self.WEBSITE: "https://ccom.ucsd.edu/~optimizers",
            }
        }

    def _get_options(
        self,
        ftol_rel=1e-9,  # type: float
        ftol_abs=1e-9,  # type: float
        xtol_rel=1e-9,  # type: float
        xtol_abs=1e-9,  # type: float
        max_time=0,  # type: float
        max_iter=999,  # type: int # pylint: disable=W0221
        normalize_design_space=True,  # type: bool
        **kwargs  # type: OptionType
    ):  # type: (...) -> Dict[str, Any]
        """Set the options.

        Args:
            ftol_rel: A stop criteria, the relative tolerance on the
               objective function.
               If abs(f(xk)-f(xk+1))/abs(f(xk))<= ftol_rel: stop.
            ftol_abs: A stop criteria, the absolute tolerance on the objective
               function. If abs(f(xk)-f(xk+1))<= ftol_rel: stop.
            xtol_rel: A stop criteria, the relative tolerance on the
               design variables. If norm(xk-xk+1)/norm(xk)<= xtol_rel: stop.
            xtol_abs: A stop criteria, the absolute tolerance on the
                   design variables. If norm(xk-xk+1)<= xtol_abs: stop.
            max_time: max_time: The maximum runtime in seconds,
                disabled if 0.
            max_iter: The maximum number of iterations,
                i.e. unique calls to f(x).
            normalize_design_space: If True, scales variables to [0, 1].
            **kwargs: The additional options.
        """
        nds = normalize_design_space
        return self._process_options(
            max_iter=max_iter,
            normalize_design_space=nds,
            ftol_rel=ftol_rel,
            ftol_abs=ftol_abs,
            xtol_rel=xtol_rel,
            xtol_abs=xtol_abs,
            max_time=max_time,
            **kwargs
        )

    @staticmethod
    def __eval_func(
        func,  # type: Callable[[ndarray], ndarray]
        xn_vect,  # type: ndarray
    ):  # type: (...) -> Tuple[ndarray, int]
        """Evaluate a function at the given points.

        Try to call it, if it fails, return a -1 status.

        Args
            func: The function to call.
            xn_vect : The arguments for the function evaluation.

        Returns:
            The function value at `xn_vect` and the status of the evaluation.
        """
        try:
            val = func(xn_vect)
            return val, 1
        # SNOPT can handle non-computable points
        except Exception as func_error:
            LOGGER.error("SNOPT failed to evaluate the function !")
            LOGGER.error(func_error.args[0])
        return array([float("nan")]), -1

    # cb_ means callback and avoids pylint to raise warnings
    # about unused arguments
    def cb_opt_objective_snoptb(
        self,
        mode,  # type: int
        nn_obj,  # type: int
        xn_vect,  # type: ndarray
        n_state=0,  # type: int
    ):  # type: (...) -> Tuple[int, ndarray, ndarray]
        r"""Evaluate the objective function and gradient.

        Use the snOpt conventions for mode and status
        (from web.stanford.edu/group/SOL/guides/sndoc7.pdf).

        Args:
            mode: Flag to indicate whether the obj, the gradient or both must
                be assigned during the present call of the function
                (0 :math:`\leq` mode :math:`\leq` 2).
                mode = 2, assign the obj and the known components of the gradient.
                mode = 1, assign the known components of gradient. obj is ignored.
                mode = 0, only the obj needs to be assigned; the
                gradient is ignored.
            nn_obj: The number of design variables.
            xn_vect: The normalized design vector.
            n_state: An indicator for the first and last call to the current
                function.
                n_state = 0: NTR.
                n_state = 1: first call to driver.cb_opt_objective_snoptb.
                n_state > 1, snOptB is calling subroutine for the last time and:
                n_state = 2       and the current x is optimal
                n_state  = 3, the problem appears to be infeasible
                n_state  = 4, the problem appears to be unbounded;
                n_state  = 5,  an iterations limit was reached.

        Returns:
            The solution status, the evaluation of the objective function and its
                gradient.

        """
        obj_func = self.problem.objective
        status = -1
        if mode == 0:
            obj_f, status = self.__eval_func(obj_func.func, xn_vect)
            obj_df = ones((xn_vect.shape[0],)) * 666.0

        if mode == 1:
            obj_df, status = self.__eval_func(obj_func.jac, xn_vect)
            obj_f = -666.0

        if mode == 2:
            obj_df, f_status = self.__eval_func(obj_func.jac, xn_vect)
            obj_f, df_status = self.__eval_func(obj_func.func, xn_vect)
            if f_status == 1 and df_status == 1:
                status = 1
            else:
                status = -1
        return status, obj_f, obj_df

    def __snoptb_create_c(
        self, xn_vect  # type: ndarray
    ):  # type: (...) -> Tuple[ndarray, int]
        """Return the evaluation of the constraints at the design vector.

        Args:
            xn_vect: The normalized design variables vector.

        Returns:
            The evaluation of the constraints at `xn_vect` and the status of
                the evaluation.
        """
        cstr = array([])
        for constraint in self.problem.get_eq_constraints():
            c_val, status = self.__eval_func(constraint, xn_vect)
            if status == -1:
                return c_val, -1
            cstr = hstack((cstr, c_val))

        for constraint in self.problem.get_ineq_constraints():
            c_val, status = self.__eval_func(constraint, xn_vect)
            if status == -1:
                return c_val, -1
            cstr = hstack((cstr, c_val))
        return cstr, 1

    def __snoptb_create_dc(
        self, xn_vect  # type: ndarray
    ):  # type: (...) -> Tuple[ndarray, int]
        """Evaluate the constraints gradient at the design vector xn_vect.

        Args:
            xn_vect: The normalized design variables vector.

        Returns:
            The evaluation of the constraints gradient at xn_vect and the status
                of the computation.
        """
        dcstr = array([])
        # First equality constraints then inequality
        for constraint in self.problem.get_eq_constraints():
            dc_val, status = self.__eval_func(constraint.jac, xn_vect)
            if status == -1:
                return dc_val, status
            if dcstr:
                dcstr = vstack((dcstr, dc_val))
            else:
                dcstr = dc_val
        for constraint in self.problem.get_ineq_constraints():
            dc_val, status = self.__eval_func(constraint.jac, xn_vect)
            if status == -1:
                return dc_val, status
            if dcstr.size > 0:
                dcstr = vstack((dcstr, dc_val))
            else:
                dcstr = dc_val

        if len(dcstr.shape) > 1:
            dcstr = reshape(dcstr.T, dcstr.shape[0] * dcstr.shape[1])

        return dcstr, 1

    # cb_ means callback and avoids pylint to raise warnings
    # about unused arguments
    def cb_opt_constraints_snoptb(
        self,
        mode,  # type: int
        nn_con,  # type: int
        nn_jac,  # type: int
        ne_jac,  # type: int
        xn_vect,  # type: ndarray
        n_state,  # type: int
    ):  # type: (...) -> Tuple[int, ndarray, ndarray]
        """Evaluate the constraint functions and their gradient.

        Use the snOpt conventions (from
        web.stanford.edu/group/SOL/guides/sndoc7.pdf).

        Args:
            mode: A flag that indicates whether the obj, the gradient or both must
                be assigned during the present call of function (0 ≤ mode ≤ 2).
                mode = 2, assign obj and the known components of gradient.
                mode = 1, assign the known components of gradient. obj is ignored.
                mode = 0, only obj need be assigned; gradient is ignored.
            nn_con: The number of non-linear constraints.
            nn_jac: The number of dv involved in non-linear
                constraint functions.
            ne_jac: The number of non-zero elements in the constraints gradient.
                If dcstr is 2D, then ne_jac = nn_con*nn_jac.
            xn_vect: The normalized design vector.
            n_state: An indicator for the first and last call to the current
                function
                n_state = 0: NTR.
                n_state = 1: first call to driver.cb_opt_objective_snoptb.
                n_state > 1, snOptB is calling subroutine for the last time and:
                n_state = 2       and the current x is optimal
                n_state  = 3, the problem appears to be infeasible
                n_state  = 4, the problem appears to be unbounded;
                n_state  = 5,  an iterations limit was reached.

        Returns:
            The solution status, the evaluation of the constraint function and
                its gradient.
        """
        if mode == 0:
            cstr, status = self.__snoptb_create_c(xn_vect)
            dcstr = (
                ones(
                    (
                        (self.__n_eq_constraints + self.__n_ineq_constraints)
                        * xn_vect.shape[0]
                    )
                )
                * 666.0
            )
            status = 1

        elif mode == 1:
            dcstr, status = self.__snoptb_create_dc(xn_vect)
            cstr = ones((self.__n_eq_constraints + self.__n_ineq_constraints,)) * 666.0

        elif mode == 2:
            cstr, c_status = self.__snoptb_create_c(xn_vect)
            dcstr, dc_status = self.__snoptb_create_dc(xn_vect)
            if c_status == 1 and dc_status == 1:
                status = 1
            else:
                status = -1
        return status, cstr, dcstr

    # cb_ means callback and avoids pylint to raise warnings
    # about unused arguments
    @staticmethod
    def cb_snopt_dummy_func(
        mode,  # type: int
        nn_con,  # type: int
        nn_jac,  # type: int
        ne_jac,  # type: int
        xn_vect,  # type: ndarray
        n_state,  # type: int
    ):  # type: (...) -> float
        """Return a dummy output for unconstrained problems.

        Args:
            mode: A flag that indicates whether the obj, the gradient or both must
                be assigned during the present call of function (0 ≤ mode ≤ 2).
                mode = 2, assign obj and the known components of gradient.
                mode = 1, assign the known components of gradient. obj is ignored.
                mode = 0, only obj need be assigned; gradient is ignored.
            nn_con: The number of non-linear constraints.
            nn_jac: The number of dv involved in non-linear
                constraint functions.
            ne_jac: The number of non-zero elements in the constraints gradient.
                If dcstr is 2D, then ne_jac = nn_con*nn_jac.
            xn_vect: The normalized design vector.
            n_state: An indicator for the first and last call to the current
                function
                n_state = 0: NTR.
                n_state = 1: first call to driver.cb_opt_objective_snoptb.
                n_state > 1, snOptB is calling subroutine for the last time and:
                n_state = 2       and the current x is optimal
                n_state  = 3, the problem appears to be infeasible
                n_state  = 4, the problem appears to be unbounded;
                n_state  = 5,  an iterations limit was reached.

        Returns:
            A dummy output.
        """
        return 1.0

    def __preprocess_snopt_constraints(
        self, names  # type: ndarray(dtype=np_str)
    ):  # type: (...) -> SnOptPreprocessType
        """Set the snopt parameters according to the constraints.

        Args:
            names: The names of the design variables and constraints to
                be stored in the snopt internal process.

        Returns:
            The pointer to the constraint value & gradient,
            the array of lower bound constraints,
            the array of upper bound constraints,
            the design variable names and constraint names,
            and the number of constraints.
        """
        blc = array(())
        buc = array(())
        if self.__n_eq_constraints > 0:
            blc = zeros((self.__n_eq_constraints,))
            buc = zeros((self.__n_eq_constraints,))
            ceqlist = ["c_eq" + str(i) for i in range(self.__n_eq_constraints)]
            names = append(names, array(ceqlist, dtype=str))
        if self.__n_ineq_constraints > 0:
            min_inf = ones(self.__n_ineq_constraints) * -INFINITY
            blc = append(blc, min_inf)
            buc = append(buc, zeros(self.__n_ineq_constraints))
            cieqlist = ["c_ie" + str(i) for i in range(self.__n_ineq_constraints)]
            names = append(names, array(cieqlist, dtype=str))
        # Mandatory dummy free row if unconstrained
        n_constraints = self.__n_ineq_constraints + self.__n_eq_constraints
        if n_constraints == 0:
            n_constraints = 1
            funcon = self.cb_snopt_dummy_func
            blc = append(blc, ones((1,)) * -INFINITY)
            buc = append(buc, ones((1,)) * INFINITY)
            names = append(names, array(["dummy"], dtype=np_str))
        else:
            funcon = self.cb_opt_constraints_snoptb
        return funcon, blc, buc, names, n_constraints

    def _run(
        self, **options  # type: OptionType
    ):  # type: (...) -> OptimizationResult
        """Run the algorithm, to be overloaded by subclasses.

        Args:
            **options: The options for the algorithm,
                see the associated JSON file.

        Returns:
            The optimization result.

        Raises:
            KeyError: If an unknown option or incorrect type is provided.
        """
        normalize_ds = options.pop(self.NORMALIZE_DESIGN_SPACE_OPTION, True)
        # Get the  bounds anx x0
        x_0, l_b, u_b = self.get_x0_and_bounds_vects(normalize_ds)
        self.__n_ineq_constraints = self.problem.get_ineq_cstr_total_dim()
        self.__n_eq_constraints = self.problem.get_eq_cstr_total_dim()

        snopt_problem = SNOPT_solver(name="SNOPTB")

        for (key, value) in options.items():
            try:
                snopt_problem.setOption(key.replace("_", " "), value)
            except RuntimeError:
                raise KeyError(
                    "Unknown option or incorrect type :" + str(key) + ":" + str(value)
                )
        snopt_problem.setOption("Verbose", True)
        snopt_problem.setOption("Solution print", True)

        #        n_nonlinear_constraints = n_ineq_constraints + n_eq_constraints
        #        It is assumed that all constraints provided to optimizer are not
        #        inear
        #        n_linear_constraints = 0
        #        n_nonlinear_constraints = n_ineq_constraints + n_eq_constraints

        n_design_variables = x_0.shape[0]

        # Get the normalized bounds:
        l_b = where(isinf(l_b), -INFINITY, l_b)
        u_b = where(isinf(u_b), INFINITY, u_b)

        names = array(["dv" + str(i) for i in range(n_design_variables)], dtype=str)
        i_obj = array([0], np_int)

        funcon, blc, buc, names, n_constraints = self.__preprocess_snopt_constraints(
            names
        )
        n_nl_func = n_constraints
        x0_snopt = append(x_0, zeros((n_constraints,), dtype=float64))
        lower_bounds = concatenate((l_b, blc))
        upper_bounds = concatenate((u_b, buc))
        jacobian = ones((n_nl_func, x_0.shape[0]))
        obj_add = array([0.0], np_float)
        snopt_problem.snoptb(
            # name='SNOPTB',
            m=n_nl_func,
            x0=x0_snopt,
            n=n_design_variables,
            nnCon=n_constraints,
            nnObj=n_design_variables,
            nnJac=n_design_variables,
            iObj=i_obj,
            bl=lower_bounds,
            bu=upper_bounds,
            J=jacobian,
            ObjAdd=obj_add,
            funcon=funcon,
            Names=names,
            funobj=self.cb_opt_objective_snoptb,
        )

        message = self.MESSAGES_DICT[snopt_problem.info]
        status = snopt_problem.info
        return self.get_optimum_from_database(message, status)
