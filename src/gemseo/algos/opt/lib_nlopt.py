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
"""NLOPT library wrapper"""

from __future__ import absolute_import, division, print_function, unicode_literals

from builtins import range, str, super

import nlopt
from future import standard_library
from nlopt import RoundoffLimited
from numpy import atleast_1d, atleast_2d

from gemseo.algos.opt.opt_lib import OptimizationLibrary
from gemseo.algos.stop_criteria import TerminationCriterion
from gemseo.core.function import MDOFunction

standard_library.install_aliases()


from gemseo import LOGGER


class NloptRoundOffException(Exception):
    """Nlopt roundoff error"""


class Nlopt(OptimizationLibrary):
    """NLOPT optimization library interface.

    See OptimizationLibrary.
    """

    LIB_COMPUTE_GRAD = False
    STOPVAL = "stopval"
    CTOL_ABS = "ctol_abs"
    INIT_STEP = "init_step"
    SUCCESS = "NLOPT_SUCCESS: Generic success return value"
    STOPVAL_REACHED = (
        "NLOPT_STOPVAL_REACHED: Optimization stopped  "
        "because stopval (above) was reached"
    )
    FTOL_REACHED = (
        "NLOPT_FTOL_REACHED: Optimization stopped "
        "because ftol_rel or ftol_abs (above) was reached"
    )

    XTOL_REACHED = (
        "NLOPT_XTOL_REACHED Optimization stopped "
        "because xtol_rel or xtol_abs (above) was reached"
    )

    MAXEVAL_REACHED = (
        "NLOPT_MAXEVAL_REACHED: Optimization stopped "
        "because maxeval (above) was reached"
    )

    MAXTIME_REACHED = (
        "NLOPT_MAXTIME_REACHED: Optimization stopped "
        "because maxtime (above) was reached"
    )
    FAILURE = "NLOPT_FAILURE:    Generic failure code"

    INVALID_ARGS = (
        "NLOPT_INVALID_ARGS: Invalid arguments (e.g. lower "
        "bounds are bigger than upper bounds, an unknown"
        " algorithm was specified, etcetera)."
    )
    OUT_OF_MEMORY = "OUT_OF_MEMORY: Ran out of memory"
    ROUNDOFF_LIMITED = (
        "NLOPT_ROUNDOFF_LIMITED: Halted because "
        "roundoff errors limited progress. (In this "
        "case, the optimization still typically "
        "returns a useful result.)"
    )
    FORCED_STOP = (
        "NLOPT_FORCED_STOP: Halted because of a forced "
        "termination: the user called nlopt_force_stop"
        "(opt) on the optimization’s nlopt_opt"
        " object opt from the user’s objective "
        "function or constraints."
    )

    NLOPT_MESSAGES = {
        1: SUCCESS,
        2: STOPVAL_REACHED,
        3: FTOL_REACHED,
        4: XTOL_REACHED,
        5: MAXEVAL_REACHED,
        6: MAXTIME_REACHED,
        -1: FAILURE,
        -2: INVALID_ARGS,
        -3: OUT_OF_MEMORY,
        -4: ROUNDOFF_LIMITED,
        -5: FORCED_STOP,
    }

    def __init__(self):
        """
        Constructor
        """
        super(Nlopt, self).__init__()

        nlopt_doc = "https://nlopt.readthedocs.io/en/latest/NLopt_Algorithms/"
        self.lib_dict = {
            "NLOPT_MMA": {
                self.INTERNAL_NAME: nlopt.LD_MMA,
                self.REQUIRE_GRAD: True,
                self.HANDLE_INEQ_CONS: True,
                self.HANDLE_EQ_CONS: False,
                self.DESCRIPTION: "Method of Moving Asymptotes (MMA)"
                "implemented in the NLOPT library",
                self.WEBSITE: "{}#mma-method-of-moving-asymptotes-and-ccsa".format(
                    nlopt_doc
                ),
            },
            "NLOPT_COBYLA": {
                self.INTERNAL_NAME: nlopt.LN_COBYLA,
                self.REQUIRE_GRAD: False,
                self.HANDLE_EQ_CONS: True,
                self.HANDLE_INEQ_CONS: True,
                self.DESCRIPTION: "Constrained Optimization BY Linear "
                "Approximations (COBYLA) implemented "
                "in the NLOPT library",
                self.WEBSITE: "{}".format(nlopt_doc)
                + "#cobyla-constrained-optimization-by-linear-"
                "approximations",
            },
            "NLOPT_SLSQP": {
                self.INTERNAL_NAME: nlopt.LD_SLSQP,
                self.REQUIRE_GRAD: True,
                self.HANDLE_EQ_CONS: True,
                self.HANDLE_INEQ_CONS: True,
                self.DESCRIPTION: "Sequential Least-Squares Quadratic "
                "Programming (SLSQP) implemented in "
                "the NLOPT library",
                self.WEBSITE: nlopt_doc + "#slsqp",
            },
            "NLOPT_BOBYQA": {
                self.INTERNAL_NAME: nlopt.LN_BOBYQA,
                self.REQUIRE_GRAD: False,
                self.HANDLE_EQ_CONS: False,
                self.HANDLE_INEQ_CONS: False,
                self.DESCRIPTION: "Bound Optimization BY Quadratic "
                "Approximation (BOBYQA) implemented "
                "in the NLOPT library",
                self.WEBSITE: nlopt_doc + "#bobyqa",
            },
            "NLOPT_BFGS": {
                self.INTERNAL_NAME: nlopt.LD_LBFGS,
                self.REQUIRE_GRAD: True,
                self.HANDLE_EQ_CONS: False,
                self.HANDLE_INEQ_CONS: False,
                self.DESCRIPTION: "Broyden-Fletcher-Goldfarb-Shanno method "
                "(BFGS) implemented in the NLOPT library",
                self.WEBSITE: nlopt_doc + "#low-storage-bfgs",
            },
            # Does not work on Rastrigin => banned
            #             'NLOPT_ESCH': { Does not work on Rastrigin
            #                 self.INTERNAL_NAME: nlopt.GN_ESCH,
            #                 self.REQUIRE_GRAD: False,
            #                 self.HANDLE_EQ_CONS: False,
            #                 self.HANDLE_INEQ_CONS: False},
            "NLOPT_NEWUOA": {
                self.INTERNAL_NAME: nlopt.LN_NEWUOA_BOUND,
                self.REQUIRE_GRAD: False,
                self.HANDLE_EQ_CONS: False,
                self.HANDLE_INEQ_CONS: False,
                self.DESCRIPTION: "NEWUOA + bound constraints implemented "
                "in the NLOPT library",
                self.WEBSITE: nlopt_doc + "#newuoa-bound-constraints",
            },
            # Does not work on Rastrigin => banned
            #             'NLOPT_ISRES': {
            #                 self.INTERNAL_NAME: nlopt.GN_ISRES,
            #                 self.REQUIRE_GRAD: False,
            #                 self.HANDLE_EQ_CONS: True,
            #                 self.HANDLE_INEQ_CONS: True}
        }
        for key in self.lib_dict:
            self.lib_dict[key][self.LIB] = self.__class__.__name__

    def _get_options(
        self,
        ftol_abs=1e-14,  # pylint: disable=W0221
        xtol_abs=1e-14,
        max_iter=999,
        ftol_rel=1e-8,
        xtol_rel=1e-8,
        max_time=0.0,
        ctol_abs=1e-6,
        stopval=None,
        normalize_design_space=True,
        eq_tolerance=1e-2,
        ineq_tolerance=1e-4,
        init_step=0.25,
        **kwargs
    ):
        r"""Sets the options

        :param max_iter: maximum number of iterations
        :type max_iter: int
        :param ftol_abs: Objective function tolerance
        :type ftol_abs: float
        :param xtol_abs: Design parameter tolerance
        :type xtol_abs: float
        :param ftol_rel: Relative objective function tolerance
        :type ftol_rel: float
        :param xtol_rel: Relative design parameter tolerance
        :type xtol_rel: float
        :param max_time: Maximum time
        :type max_time: float
        :param ctol_abs: Absolute tolerance for constraints
        :type ctol_abs: float
        :param normalize_design_space: If True, scales variables in [0, 1]
        :type normalize_design_space: bool
        :param stopval: Stop when an objective value of at least stopval
            is found:
            stop minimizing when an objective value :math:`\leq` stopval is
            found,
            or stop maximizing a value :math:`\geq` stopval is found.
        :type stopval: float
        :param eq_tolerance: equality tolerance
        :type eq_tolerance: float
        :param ineq_tolerance: inequality tolerance
        :type ineq_tolerance: float
        :param kwargs: additional options
        :type kwargs: kwargs
        :param init_step: initial step size for derivavtive free algorithms
            increasing init_step will make the initial DOE in COBYLA
            wider steps in the design variables. By defaults, each variable
            is set to x0 + a perturbation that worths 0.25*(ub_i-x0_i) for i
            in xrange(len(x0))
        :type init_step: float
        """
        nds = normalize_design_space
        popts = self._process_options(
            ftol_abs=ftol_abs,
            xtol_abs=xtol_abs,
            max_iter=max_iter,
            ftol_rel=ftol_rel,
            xtol_rel=xtol_rel,
            max_time=max_time,
            ctol_abs=ctol_abs,
            normalize_design_space=nds,
            stopval=stopval,
            eq_tolerance=eq_tolerance,
            ineq_tolerance=ineq_tolerance,
            init_step=init_step,
            **kwargs
        )

        return popts

    def __opt_objective_grad_nlopt(self, xn_vect, grad):
        """
        Objective function + gradient function of the optimizer for nlopt
        :param xn_vect: normalized design vector
        :param grad: gradient array
        :returns: objective Gradients array
        :rtype: float
        """
        obj_func = self.problem.objective
        if grad.size > 0:
            grad[:] = obj_func.jac(xn_vect)
        return obj_func.func(xn_vect).real

    def __make_constraint(self, func, jac, index_cstr):
        """
        Builds nlopt-like constraints: no vector functions allowed
        The database will avoid multiple evaluations
        :param func: function pointer
        :param jac: jacobian pointer
        :param index_cstr: index of the constraint
        """

        def cstr_fun_grad(xn_vect, grad):
            """Function which is given as a pointer
            to optimizer for constraints and constraints
            gradient if required

            :param xn_vect: normalized design vector
            :param grad: gradient array

            """
            if self.lib_dict[self.algo_name][self.REQUIRE_GRAD]:
                if grad.size > 0:
                    cstr_jac = jac(xn_vect)
                    grad[:] = atleast_2d(cstr_jac)[
                        index_cstr,
                    ]
            return atleast_1d(func(xn_vect).real)[index_cstr]

        return cstr_fun_grad

    def __add_constraints(self, nlopt_problem, ctol=0.0):
        """
        Function that add all constraints (UDF+formulation) to
        optimization algorithm
        :param nlopt_problem: optimization problem
        :rtype: unconstrainted nlopt problem
        :returns: updated nlnlopt_problem
        """
        for constraint in self.problem.constraints:
            f_type = constraint.f_type
            func = constraint.func
            jac = constraint.jac
            dim = constraint.dim
            for idim in range(dim):
                nl_fun = self.__make_constraint(func, jac, idim)
                if f_type == MDOFunction.TYPE_INEQ:
                    nlopt_problem.add_inequality_constraint(nl_fun, ctol)
                elif f_type == MDOFunction.TYPE_EQ:
                    nlopt_problem.add_equality_constraint(nl_fun, ctol)

    def __set_prob_options(self, nlopt_problem, **opt_options):
        """
        Settings of options for nlopt algorithms
        :param nlopt_problem: optimization problem from nlopt
        :param opt_options: optimization options of nlopt
        """

        nlopt_problem.set_xtol_abs(opt_options[self.X_TOL_ABS])
        nlopt_problem.set_xtol_rel(opt_options[self.X_TOL_REL])
        nlopt_problem.set_ftol_rel(opt_options[self.F_TOL_REL])
        nlopt_problem.set_ftol_abs(opt_options[self.F_TOL_ABS])
        nlopt_problem.set_maxeval(opt_options[self.MAX_ITER])
        nlopt_problem.set_maxtime(opt_options[self.MAX_TIME])
        nlopt_problem.set_initial_step(opt_options[self.INIT_STEP])
        if self.STOPVAL in opt_options:
            stopval = opt_options[self.STOPVAL]
            if stopval is not None:
                nlopt_problem.set_stopval(stopval)

        return nlopt_problem

    def _run(self, **options):
        """Runs the algorithm, to be overloaded by subclasses

        :param options: the options dict for the algorithm,
            see associated JSON file

        """
        normalize_ds = options.pop(self.NORMALIZE_DESIGN_SPACE_OPTION, True)
        # Get the  bounds anx x0
        x_0, l_b, u_b = self.get_x0_and_bounds_vects(normalize_ds)

        nlopt_problem = nlopt.opt(self.internal_algo_name, x_0.shape[0])

        # Set the normalized bounds:
        nlopt_problem.set_lower_bounds(l_b.real)
        nlopt_problem.set_upper_bounds(u_b.real)

        nlopt_problem.set_min_objective(self.__opt_objective_grad_nlopt)
        if self.CTOL_ABS in options:
            ctol = options[self.CTOL_ABS]
        self.__add_constraints(nlopt_problem, ctol)
        nlopt_problem = self.__set_prob_options(nlopt_problem, **options)
        try:
            nlopt_problem.optimize(x_0.real)
        except (RoundoffLimited, RuntimeError) as err:
            LOGGER.error(
                "NLOPT run failed : %s, %s",
                str(err.args[0]),
                str(err.__class__.__name__),
            )
            raise TerminationCriterion()
        # status = nlopt_problem.last_optimize_result()
        message = self.NLOPT_MESSAGES[nlopt_problem.last_optimize_result()]
        status = nlopt_problem.last_optimize_result()
        return self.get_optimum_from_database(message, status)
