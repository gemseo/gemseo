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
from typing import Callable
from typing import ClassVar

from numpy import array
from numpy import ndarray
from scipy.optimize import root

from gemseo.core.discipline import MDODiscipline
from gemseo.mda.root import MDARoot

if TYPE_CHECKING:
    from collections.abc import Mapping
    from collections.abc import Sequence
    from typing import Any

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

    __current_couplings: ndarray
    """The current values of the coupling variables."""

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

        self.method = method

        if self.method not in self._methods_with_callback():
            del self.output_grammar[self.RESIDUALS_NORM]

        self.use_gradient = use_gradient

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
        """Determine whether resolution method accepts a callback function.

        Returns:
            The names of the methods with callback.
        """
        return [self.BROYDEN1, self.BROYDEN2]

    def __get_jacobian_computer(self) -> Callable[[ndarray], ndarray] | None:
        """Return the function to compute the jacobian.

        Returns:
            The callable to compute the jacobian.
        """
        if not self.use_gradient:
            return None

        self.assembly.set_newton_differentiated_ios(self._resolved_variable_names)

        def compute_jacobian(
            x_vect: ndarray,
        ) -> ndarray:
            """Linearize all residuals.

            Args:
                x_vect: The value of the design variables.

            Returns:
                The linearized residuals.
            """
            self._update_local_data(x_vect)

            self.reset_disciplines_statuses()
            for discipline in self.disciplines:
                discipline.linearize(self._local_data)

            self.assembly.compute_sizes(
                self._resolved_variable_names,
                self._resolved_variable_names,
                self._resolved_variable_names,
            )

            return (
                self.assembly.assemble_jacobian(
                    self._resolved_variable_names,
                    self._resolved_variable_names,
                    is_residual=True,
                )
                .toarray()
                .real
            )

        return compute_jacobian

    def __get_residual_history_callback(self) -> Callable[[ndarray, Any], None] | None:
        """Return the callback used to store the residual history."""
        if self.method not in self._methods_with_callback():
            return None

        def callback(
            new_couplings: ndarray,
            _,
        ) -> None:
            """Store the current residual in the history.

            Args:
                new_couplings: The new coupling variables.
                _: ignored
            """
            self._compute_residual()
            self.__current_couplings = new_couplings

        return callback

    def __compute_residuals(
        self,
        x_vect: ndarray,
    ) -> ndarray:
        """Evaluate all residuals, possibly in parallel.

        Args:
            x_vect: The value of the design variables.

        Returns:
            The residuals.
        """
        self.current_iter += 1
        # Work on a temporary copy so _update_local_data can be called.
        local_data_copy = self._local_data.copy()
        self._update_local_data(x_vect)
        input_data = self._local_data
        self._local_data = local_data_copy
        self.reset_disciplines_statuses()
        self.execute_all_disciplines(input_data)
        self._update_residuals(input_data)
        return self.assembly.residuals(input_data, self._resolved_variable_names).real

    def _run(self) -> DisciplineData:
        super()._run()

        self.reset_disciplines_statuses()
        self.execute_all_disciplines(self._local_data)

        if not self.strong_couplings:
            msg = (
                "MDAQuasiNewton found no strong couplings. Executed all"
                "disciplines once."
            )
            LOGGER.warning(msg)
            self._local_data[self.RESIDUALS_NORM] = array([0.0])
            return self._local_data

        self.current_iter = 0

        if self.reset_history_each_run:
            self.residual_history = []

        # initial solution
        self.__current_couplings = self.get_current_resolved_variables_vector().real

        # solve the system
        y_opt = root(
            self.__compute_residuals,
            x0=self.__current_couplings,
            method=self.method,
            jac=self.__get_jacobian_computer(),
            callback=self.__get_residual_history_callback(),
            tol=self.tolerance,
            options=self._solver_options(),
        )

        self._warn_convergence_criteria()

        self._update_local_data(y_opt.x)

        if self.method in self._methods_with_callback():
            self._local_data[self.RESIDUALS_NORM] = array([self.normed_residual])

        return self._local_data
