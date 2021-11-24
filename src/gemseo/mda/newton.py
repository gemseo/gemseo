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
#        :author: Charlie Vanaret, Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""A set of Newton algorithm variants for solving MDAs.

Root finding methods include:

- `Newton-Raphson <https://en.wikipedia.org/wiki/Newton%27s_method>`__
- `quasi-Newton methods <https://en.wikipedia.org/wiki/Quasi-Newton_method>`__

Each of these methods is implemented by a class in this module.
Both inherit from a common abstract cache.
"""
from __future__ import division, unicode_literals

import logging
from copy import deepcopy
from typing import Any, Dict, List, Mapping, Optional, Sequence, Union

from numpy import ndarray
from numpy.linalg import norm
from scipy.optimize import root

from gemseo.core.coupling_structure import MDOCouplingStructure
from gemseo.core.discipline import MDODiscipline
from gemseo.mda.mda import MDA
from gemseo.utils.data_conversion import DataConversion

# from gemseo.core.parallel_execution import DisciplinesParallelExecution
LOGGER = logging.getLogger(__name__)


class MDARoot(MDA):
    """Abstract class implementing MDAs based on (Quasi-)Newton methods."""

    _ATTR_TO_SERIALIZE = MDA._ATTR_TO_SERIALIZE + ("strong_couplings",)

    def __init__(
        self,
        disciplines,  # type: Sequence[MDODiscipline]
        max_mda_iter=10,  # type: int
        name=None,  # type: Optional[str]
        grammar_type=MDODiscipline.JSON_GRAMMAR_TYPE,  # type: str
        tolerance=1e-6,  # type: float
        linear_solver_tolerance=1e-12,  # type: float
        warm_start=False,  # type: bool
        use_lu_fact=False,  # type: bool
        coupling_structure=None,  # type: Optional[MDOCouplingStructure]
        log_convergence=False,  # type: bool
        linear_solver="DEFAULT",  # type: str
        linear_solver_options=None,  # type: Mapping[str,Any]
    ):  # type: (...) -> None
        self.tolerance = 1e-6
        self.max_mda_iter = 10
        super(MDARoot, self).__init__(
            disciplines,
            max_mda_iter=max_mda_iter,
            name=name,
            grammar_type=grammar_type,
            tolerance=tolerance,
            linear_solver_tolerance=linear_solver_tolerance,
            warm_start=warm_start,
            use_lu_fact=use_lu_fact,
            coupling_structure=coupling_structure,
            log_convergence=log_convergence,
            linear_solver=linear_solver,
            linear_solver_options=linear_solver_options,
        )
        self._set_default_inputs()
        self._compute_input_couplings()
        # parallel execution
        # ==================================================================
        # if 1 < n_processes:
        #     self.parallel_execution = DisciplinesParallelExecution(
        #         self.disciplines, n_processes)
        # else:
        #    self.parallel_execution = None
        # ==================================================================

    def _initialize_grammars(self):  # type: (...) -> None
        for disciplines in self.disciplines:
            self.input_grammar.update_from(disciplines.input_grammar)
            self.output_grammar.update_from(disciplines.output_grammar)

    def execute_all_disciplines(
        self,
        input_local_data,  # type: Mapping[str,ndarray]
    ):  # type: (...) -> None
        """Execute all self.disciplines.

        Args:
            input_local_data: The input data of the disciplines.
        """
        # Set status of sub disciplines
        # if self.parallel_execution is not None:
        #     self.disciplines = self.parallel_execution
        # .execute(input_local_data)
        # else:
        for discipline in self.disciplines:
            discipline.reset_statuses_for_run()
            discipline.execute(deepcopy(input_local_data))

        outputs = [discipline.get_output_data() for discipline in self.disciplines]
        for data in outputs:
            self.local_data.update(data)


class MDANewtonRaphson(MDARoot):
    r"""Newton solver for MDA.

    The `Newton-Raphson method
    <https://en.wikipedia.org/wiki/Newton%27s_method>`__ is parameterized by a
    relaxation factor :math:`\alpha \in (0, 1]` to limit the length of the
    steps taken along the Newton direction.  The new iterate is given by:

    .. math::

       x_{k+1} = x_k - \alpha f'(x_k)^{-1} f(x_k)
    """

    _ATTR_TO_SERIALIZE = MDARoot._ATTR_TO_SERIALIZE + (
        "assembly",
        "relax_factor",
        "linear_solver",
        "linear_solver_options",
        "matrix_type",
    )

    def __init__(
        self,
        disciplines,  # type: Sequence[MDODiscipline]
        max_mda_iter=10,  # type: int
        relax_factor=0.99,  # type: float
        name=None,  # type: Optional[str]
        grammar_type=MDODiscipline.JSON_GRAMMAR_TYPE,  # type: str
        linear_solver="DEFAULT",  # type: str
        tolerance=1e-6,  # type: float
        linear_solver_tolerance=1e-12,  # type: float
        warm_start=False,  # type: bool
        use_lu_fact=False,  # type: bool
        coupling_structure=None,  # type: Optional[MDOCouplingStructure]
        log_convergence=False,  # type:bool
        linear_solver_options=None,  # type: Mapping[str,Any]
    ):
        """
        Args:
            relax_factor: The relaxation factor in the Newton step.
        """
        super(MDANewtonRaphson, self).__init__(
            disciplines,
            max_mda_iter=max_mda_iter,
            name=name,
            grammar_type=grammar_type,
            tolerance=tolerance,
            linear_solver_tolerance=linear_solver_tolerance,
            warm_start=warm_start,
            use_lu_fact=use_lu_fact,
            linear_solver=linear_solver,
            linear_solver_options=linear_solver_options,
            coupling_structure=coupling_structure,
            log_convergence=log_convergence,
        )
        self.relax_factor = self.__check_relax_factor(relax_factor)
        self.linear_solver = linear_solver

    @staticmethod
    def __check_relax_factor(
        relax_factor,  # type: float
    ):  # type:(...) -> float
        """Check that the relaxation factor in the Newton step is in (0, 1].

        Args:
            relax_factor: The relaxation factor.
        """
        if relax_factor <= 0.0 or relax_factor > 1:
            raise ValueError(
                "Newton relaxation factor should belong to (0, 1] "
                "(current value: {}).".format(relax_factor)
            )
        return relax_factor

    def _newton_step(self):  # type: (...) -> None
        """Execute the full Newton step.

        Compute the increment :math:`-[dR/dW]^{-1}.R` and run the disciplines.
        """
        newton_dict = self.assembly.compute_newton_step(
            self.local_data,
            self.strong_couplings,
            self.relax_factor,
            self.linear_solver,
            matrix_type=self.matrix_type,
            **self.linear_solver_options
        )
        # update current solution with Newton step
        exec_data = deepcopy(self.local_data)
        for c_var, c_step in newton_dict.items():
            exec_data[c_var] += c_step
        self.reset_disciplines_statuses()
        self.execute_all_disciplines(exec_data)

    def _run(self):  # type: (...) -> None
        if self.warm_start:
            self._couplings_warm_start()
        # execute the disciplines
        current_couplings = self._current_input_couplings()
        self.reset_disciplines_statuses()
        self.execute_all_disciplines(self.local_data)
        new_couplings = self._current_input_couplings()

        # store initial residual
        current_iter = 1
        self._compute_residual(
            current_couplings,
            new_couplings,
            current_iter,
            first=True,
            log_normed_residual=self.log_convergence,
        )
        current_couplings = new_couplings

        while not self._termination(current_iter):
            self._newton_step()
            new_couplings = self._current_input_couplings()

            # store current residual
            current_iter += 1
            self._compute_residual(
                current_couplings,
                new_couplings,
                current_iter,
                log_normed_residual=self.log_convergence,
            )
            current_couplings = new_couplings


class MDAQuasiNewton(MDARoot):
    """Quasi-Newton solver for MDA.

    `Quasi-Newton methods
    <https://en.wikipedia.org/wiki/Quasi-Newton_method>`__
    include numerous variants
    (
    `Broyden
    <https://en.wikipedia.org/wiki/Broyden%27s_method>`__,
    `Levenberg-Marquardt
    <https://en.wikipedia.org/wiki/Levenberg%E2%80%93Marquardt_algorithm>`__,
    ...).
    The name of the variant should be provided
    as a parameter :code:`method` of the class.

    The new iterate is given by:

    .. math::

       x_{k+1} = x_k - \\rho_k B_k f(x_k)

    where :math:`\\rho_k` is a coefficient chosen
    in order to minimize the convergence and :math:`B_k` is an approximation
    of the inverse of the Jacobian :math:`Df(x_k)^{-1}`.
    """

    # quasi-Newton methods
    HYBRID = "hybr"
    LEVENBERG_MARQUARDT = "lm"
    BROYDEN1 = "broyden1"
    BROYDEN2 = "broyden2"
    ANDERSON = "anderson"
    LINEAR_MIXING = "linearmixing"
    DIAG_BROYDEN = "diagbroyden"
    EXCITING_MIXING = "excitingmixing"
    KRYLOV = "krylov"
    DF_SANE = "df-sane"

    QUASI_NEWTON_METHODS = [
        HYBRID,
        LEVENBERG_MARQUARDT,
        BROYDEN1,
        BROYDEN2,
        ANDERSON,
        LINEAR_MIXING,
        DIAG_BROYDEN,
        EXCITING_MIXING,
        KRYLOV,
        DF_SANE,
    ]

    _ATTR_TO_SERIALIZE = MDARoot._ATTR_TO_SERIALIZE + (
        "method",
        "use_gradient",
        "assembly",
        "normed_residual",
    )

    def __init__(
        self,
        disciplines,  # type: Sequence[MDODiscipline]
        max_mda_iter=10,  # type: int
        name=None,  # type: Optional[str]
        grammar_type=MDODiscipline.JSON_GRAMMAR_TYPE,  # type: str
        method=HYBRID,  # type: str
        use_gradient=False,  # type: bool
        tolerance=1e-6,  # type: float
        linear_solver_tolerance=1e-12,  # type: float
        warm_start=False,  # type: bool
        use_lu_fact=False,  # type: bool
        coupling_structure=None,  # type: Optional[MDOCouplingStructure]
        linear_solver="DEFAULT",  # type: str
        linear_solver_options=None,  # type: Mapping[str,Any]
    ):
        """
        Args:
            method: The name of the method in scipy root finding,
                among :attr:`QUASI_NEWTON_METHODS`.
            use_gradient: Whether to use the analytic gradient of the discipline.

        Raises:
            ValueError: If the method is not a valid quasi-Newton method.
        """
        super(MDAQuasiNewton, self).__init__(
            disciplines,
            max_mda_iter=max_mda_iter,
            name=name,
            grammar_type=grammar_type,
            tolerance=tolerance,
            linear_solver_tolerance=linear_solver_tolerance,
            warm_start=warm_start,
            use_lu_fact=use_lu_fact,
            linear_solver=linear_solver,
            linear_solver_options=linear_solver_options,
            coupling_structure=coupling_structure,
        )
        if method not in self.QUASI_NEWTON_METHODS:
            msg = "Method '{}' is not a valid quasi-Newton method.".format(method)
            raise ValueError(msg)
        self.method = method
        self.use_gradient = use_gradient
        self.local_residual_history = []
        self.last_outputs = None  # used for computing the residual history

    def _solver_options(self):  # type: (...) -> Dict[str,Union[float,int]]
        """Determine options for the solver, based on the resolution method."""
        options = {}
        if self.method in [
            self.BROYDEN1,
            self.BROYDEN2,
            self.ANDERSON,
            self.LINEAR_MIXING,
            self.DIAG_BROYDEN,
            self.EXCITING_MIXING,
            self.KRYLOV,
        ]:
            options["ftol"] = self.tolerance
            options["maxiter"] = self.max_mda_iter
        elif self.method in [self.LEVENBERG_MARQUARDT]:
            options["xtol"] = self.tolerance
            options["maxiter"] = self.max_mda_iter
        elif self.method in [self.DF_SANE]:
            options["fatol"] = self.tolerance
            options["maxfev"] = self.max_mda_iter
        elif self.method in [self.HYBRID]:
            options["xtol"] = self.tolerance
            options["maxfev"] = self.max_mda_iter
        return options

    def _methods_with_callback(self):  # type: (...) -> List[str]
        """Determine whether resolution method accepts a callback function."""
        return [self.BROYDEN1, self.BROYDEN2]

    def _run(self):  # type: (...) -> Dict[str,ndarray]
        if self.warm_start:
            self._couplings_warm_start()
        self.reset_disciplines_statuses()
        self.execute_all_disciplines(deepcopy(self.local_data))

        couplings = self.strong_couplings

        if not couplings:
            msg = (
                "MDAQuasiNewton found no strong couplings. Executed all"
                "disciplines once."
            )
            LOGGER.warning(msg)
            return self.local_data

        options = self._solver_options()
        self.current_iter = 0

        def fun(
            x_vect,  # type: ndarray
        ):  # type: (...) -> ndarray
            """Evaluate all residuals, possibly in parallel.

            Args:
                x_vect: The value of the design variables.
            """
            self.current_iter += 1
            # transform input vector into a dict
            input_values = DataConversion.update_dict_from_array(
                self.local_data, couplings, x_vect
            )
            # compute all residuals
            self.reset_disciplines_statuses()
            self.execute_all_disciplines(input_values)
            residuals = self.assembly.residuals(input_values, couplings).real
            #             if residuals.size == 1:  # Weak couplings already treated
            #                 return residuals[0]
            return residuals

        jac = None
        if self.use_gradient:

            for discipline in self.disciplines:
                # Tells the discipline what to linearize
                outs = discipline.get_output_data_names()
                to_linearize = set(outs) & set(couplings)
                discipline.add_differentiated_outputs(list(to_linearize))
                inpts = discipline.get_input_data_names()
                to_linearize = set(inpts) & set(couplings)
                discipline.add_differentiated_inputs(list(to_linearize))
            # linearize the residuals

            def jacobian(
                x_vect,  # type: ndarray
            ):  # type: (...) -> ndarray
                """Linearize all residuals.

                Args:
                    x_vect: The value of the design variables.
                """
                # transform input vector into a dict
                input_values = DataConversion.update_dict_from_array(
                    self.local_data, couplings, x_vect
                )
                # linearize all residuals
                self.reset_disciplines_statuses()
                for discipline in self.disciplines:
                    discipline.linearize(input_values)
                # assemble the system
                n_couplings = 0
                for coupling in couplings:
                    discipline = self.coupling_structure.find_discipline(coupling)
                    size = list(discipline.jac[coupling].values())[0].shape[0]
                    n_couplings += size
                self.assembly.compute_sizes(couplings, couplings, couplings)
                dresiduals = self.assembly.dres_dvar(
                    couplings, couplings, n_couplings, n_couplings
                ).todense()
                return dresiduals

            jac = jacobian

        # initial solution
        y_0 = DataConversion.dict_to_array(self.local_data, couplings).real
        # callback function to retrieve the residual at iteration k
        norm_0 = norm(y_0.real)
        if self.reset_history_each_run:
            self.residual_history = []

        # callback function to store residuals
        self.last_outputs = y_0
        if self.method in self._methods_with_callback():

            def callback(
                y_k,  # type: ndarray
                _,
            ):  # type: (...) -> None
                """Store the current residual in the history.

                Args:
                    y_k: The coupling variables.
                    _: ignored
                """
                delta = norm((y_k - self.last_outputs).real) / norm_0
                self.residual_history.append((delta, 0))  # iter number?
                self.last_outputs = y_k

        else:
            callback = None

        # solve the system
        y_opt = root(
            fun, x0=y_0, method=self.method, jac=jac, callback=callback, options=options
        )
        self._warn_convergence_criteria(self.current_iter)

        # transform optimal vector into a dict
        self.local_data = DataConversion.update_dict_from_array(
            self.local_data, couplings, y_opt.x
        )
        return self.local_data
