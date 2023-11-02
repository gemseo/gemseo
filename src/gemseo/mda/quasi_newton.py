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
"""A set of quasi Newton algorithm variants for solving MDAs.

`quasi-Newton methods <https://en.wikipedia.org/wiki/Quasi-Newton_method>`__
"""
from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from typing import ClassVar

from numpy import array
from scipy.optimize import root

from gemseo.core.discipline import MDODiscipline
from gemseo.mda.root import MDARoot
from gemseo.utils.data_conversion import split_array_to_dict_of_arrays
from gemseo.utils.data_conversion import update_dict_of_arrays_from_array

if TYPE_CHECKING:
    from typing import Any
    from typing import Mapping
    from typing import Sequence

    from numpy.typing import NDArray

    from gemseo.core.coupling_structure import MDOCouplingStructure
    from gemseo.core.discipline_data import DisciplineData

LOGGER = logging.getLogger(__name__)


class MDAQuasiNewton(MDARoot):
    r"""Quasi-Newton solver for MDA.

    `Quasi-Newton methods <https://en.wikipedia.org/wiki/Quasi-Newton_method>`__
    include numerous variants (
    `Broyden <https://en.wikipedia.org/wiki/Broyden%27s_method>`__,
    `Levenberg-Marquardt <https://en.wikipedia.org/wiki/Levenberg-Marquardt_algorithm>`
    __, ...).

    The name of the variant should be provided via the :code:`method` parameter of the
    class.

    The new iterate is given by:

    .. math::

       x_{k+1} = x_k - \\rho_k B_k f(x_k)

    where :math:`\\rho_k` is a coefficient chosen in order to minimize the convergence
    and :math:`B_k` is an approximation of :math:`Df(x_k)^{-1}`, the inverse of the
    Jacobian of :math:`f` at :math:`x_k`.
    """

    # Available quasi-Newton methods
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

    # TODO: API: use enums.
    QUASI_NEWTON_METHODS: ClassVar[list[str]] = [
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

    def __init__(
        self,
        disciplines: Sequence[MDODiscipline],
        max_mda_iter: int = 10,
        name: str | None = None,
        grammar_type: MDODiscipline.GrammarType = MDODiscipline.GrammarType.JSON,
        method: str = HYBRID,
        use_gradient: bool = False,
        tolerance: float = 1e-6,
        linear_solver_tolerance: float = 1e-12,
        warm_start: bool = False,
        use_lu_fact: bool = False,
        coupling_structure: MDOCouplingStructure | None = None,
        linear_solver: str = "DEFAULT",
        linear_solver_options: Mapping[str, Any] | None = None,
    ) -> None:
        """
        Args:
            method: The name of the method in scipy root finding, among
                :attr:`.QUASI_NEWTON_METHODS`.
            use_gradient: Whether to use the analytic gradient of the discipline.

        Raises:
            ValueError: If the method is not a valid quasi-Newton method.
        """  # noqa:D205 D212 D415
        self.method = method
        super().__init__(
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
            raise ValueError(f"Method '{method}' is not a valid quasi-Newton method.")

        self.use_gradient = use_gradient
        self.local_residual_history = []
        self.last_outputs = None  # used for computing the residual history

    # TODO: API: prepend verb.
    def _solver_options(self) -> dict[str, float | int]:
        """Determine options for the solver, based on the resolution method."""
        options = {}
        if self.method in {
            self.BROYDEN1,
            self.BROYDEN2,
            self.ANDERSON,
            self.LINEAR_MIXING,
            self.DIAG_BROYDEN,
            self.EXCITING_MIXING,
            self.KRYLOV,
        }:
            options["ftol"] = self.tolerance
            options["maxiter"] = self.max_mda_iter
        elif self.method == self.LEVENBERG_MARQUARDT:
            options["xtol"] = self.tolerance
            options["maxiter"] = self.max_mda_iter
        elif self.method == self.DF_SANE:
            options["fatol"] = self.tolerance
            options["maxfev"] = self.max_mda_iter
        elif self.method == self.HYBRID:
            options["xtol"] = self.tolerance
            options["maxfev"] = self.max_mda_iter
        return options

    # TODO: API: prepend verb.
    def _methods_with_callback(self) -> list[str]:
        """Determine whether resolution method accepts a callback function."""
        return [self.BROYDEN1, self.BROYDEN2]

    def _run(self) -> DisciplineData:
        if self.warm_start:
            self._couplings_warm_start()

        self.reset_disciplines_statuses()
        self.execute_all_disciplines(self.local_data)
        self._compute_coupling_sizes()

        if not self.strong_couplings:
            msg = (
                "MDAQuasiNewton found no strong couplings. Executed all"
                "disciplines once."
            )
            LOGGER.warning(msg)
            self.local_data[self.RESIDUALS_NORM] = array([0.0])
            return self.local_data

        options = self._solver_options()
        self.current_iter = 0

        def fun(
            x_vect: NDArray,
        ) -> NDArray:
            """Evaluate all residuals, possibly in parallel.

            Args:
                x_vect: The value of the design variables.
            """
            self.current_iter += 1
            # transform input vector into a dict
            input_values = update_dict_of_arrays_from_array(
                self.local_data, self.strong_couplings, x_vect
            )
            # compute all residuals
            self.reset_disciplines_statuses()
            self.execute_all_disciplines(input_values)
            return self.assembly.residuals(input_values, self.strong_couplings).real

        jac = None
        if self.use_gradient:
            self.assembly.set_newton_differentiated_ios(self.strong_couplings)

            def jacobian(
                x_vect: NDArray,
            ) -> NDArray:
                """Linearize all residuals.

                Args:
                    x_vect: The value of the design variables.
                """
                # transform input vector into a dict
                self.local_data.update(
                    split_array_to_dict_of_arrays(
                        x_vect, self._coupling_sizes, self.strong_couplings
                    )
                )
                # linearize all residuals
                self.reset_disciplines_statuses()
                for discipline in self.disciplines:
                    discipline.linearize(self.local_data)
                # assemble the system
                self.assembly.compute_sizes(
                    self.strong_couplings, self.strong_couplings, self.strong_couplings
                )
                dresiduals = self.assembly.assemble_jacobian(
                    self.strong_couplings, self.strong_couplings, is_residual=True
                )
                return dresiduals.toarray().real

            jac = jacobian

        # initial solution
        self.current_couplings = self._current_working_couplings().real
        # callback function to retrieve the residual at iteration k
        if self.reset_history_each_run:
            self.residual_history = []

        # callback function to store residuals
        if self.method in self._methods_with_callback():

            def callback(
                new_couplings: NDArray,
                _,
            ) -> None:
                """Store the current residual in the history.

                Args:
                    new_couplings: The new coupling variables.
                    _: ignored
                """
                self._compute_residual(self.current_couplings, new_couplings)
                self.current_couplings = new_couplings

        else:
            callback = None

        # solve the system
        y_opt = root(
            fun,
            x0=self.current_couplings,
            method=self.method,
            jac=jac,
            callback=callback,
            tol=self.tolerance,
            options=options,
        )
        self._warn_convergence_criteria()

        # transform optimal vector into a dict
        self.local_data.update(
            split_array_to_dict_of_arrays(
                y_opt.x, self._coupling_sizes, self.strong_couplings
            )
        )

        if self.method in self._methods_with_callback():
            self.local_data[self.RESIDUALS_NORM] = array([self.normed_residual])

        return self.local_data

    def _initialize_grammars(self) -> None:
        for disciplines in self.disciplines:
            self.input_grammar.update(disciplines.input_grammar)
            self.output_grammar.update(disciplines.output_grammar)

        if self.method in self._methods_with_callback():
            self._add_residuals_norm_to_output_grammar()
