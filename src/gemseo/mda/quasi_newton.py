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
from strenum import StrEnum

from gemseo.mda.base_mda_root import BaseMDARoot
from gemseo.utils.constants import READ_ONLY_EMPTY_DICT

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

    from gemseo.core.coupling_structure import CouplingStructure
    from gemseo.core.discipline import Discipline
    from gemseo.core.discipline.discipline_data import DisciplineData
    from gemseo.typing import StrKeyMapping

LOGGER = logging.getLogger(__name__)


class MDAQuasiNewton(BaseMDARoot):
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

    class QuasiNewtonMethod(StrEnum):
        """A quasi-Newton method."""

        ANDERSON = "anderson"
        BROYDEN1 = "broyden1"
        BROYDEN2 = "broyden2"
        DF_SANE = "df-sane"
        DIAG_BROYDEN = "diagbroyden"
        EXCITING_MIXING = "excitingmixing"
        HYBRID = "hybr"
        KRYLOV = "krylov"
        LEVENBERG_MARQUARDT = "lm"
        LINEAR_MIXING = "linearmixing"

    _METHODS_SUPPORTING_CALLBACKS: ClassVar[
        tuple[QuasiNewtonMethod, QuasiNewtonMethod]
    ] = (QuasiNewtonMethod.BROYDEN1, QuasiNewtonMethod.BROYDEN2)
    """The methods supporting callback functions."""

    __current_couplings: ndarray
    """The current values of the coupling variables."""

    def __init__(
        self,
        disciplines: Sequence[Discipline],
        max_mda_iter: int = 10,
        name: str = "",
        method: QuasiNewtonMethod = QuasiNewtonMethod.HYBRID,
        use_gradient: bool = False,
        tolerance: float = 1e-6,
        linear_solver_tolerance: float = 1e-12,
        warm_start: bool = False,
        use_lu_fact: bool = False,
        coupling_structure: CouplingStructure | None = None,
        linear_solver: str = "DEFAULT",
        linear_solver_options: StrKeyMapping = READ_ONLY_EMPTY_DICT,
        execute_before_linearizing: bool = False,
    ) -> None:
        """
        Args:
            method: The name of the method in scipy root finding.
            use_gradient: Whether to use the analytic gradient of the discipline.

        Raises:
            ValueError: If the method is not a valid quasi-Newton method.
        """  # noqa:D205 D212 D415
        super().__init__(
            disciplines,
            max_mda_iter=max_mda_iter,
            name=name,
            tolerance=tolerance,
            linear_solver_tolerance=linear_solver_tolerance,
            warm_start=warm_start,
            use_lu_fact=use_lu_fact,
            linear_solver=linear_solver,
            linear_solver_options=linear_solver_options,
            coupling_structure=coupling_structure,
            execute_before_linearizing=execute_before_linearizing,
        )
        self.method = method

        if self.method not in self._METHODS_SUPPORTING_CALLBACKS:
            del self.output_grammar[self.NORMALIZED_RESIDUAL_NORM]

        self.use_gradient = use_gradient

    def _get_options(self) -> dict[str, float | int]:
        """Determine options for the solver, based on the resolution method."""
        options = {}
        if self.method in {
            self.QuasiNewtonMethod.BROYDEN1,
            self.QuasiNewtonMethod.BROYDEN2,
            self.QuasiNewtonMethod.ANDERSON,
            self.QuasiNewtonMethod.LINEAR_MIXING,
            self.QuasiNewtonMethod.DIAG_BROYDEN,
            self.QuasiNewtonMethod.EXCITING_MIXING,
            self.QuasiNewtonMethod.KRYLOV,
        }:
            options["ftol"] = self.tolerance
            options["maxiter"] = self.max_mda_iter
        elif self.method == self.QuasiNewtonMethod.LEVENBERG_MARQUARDT:
            options["xtol"] = self.tolerance
            options["maxiter"] = self.max_mda_iter
        elif self.method == self.QuasiNewtonMethod.DF_SANE:
            options["fatol"] = self.tolerance
            options["maxfev"] = self.max_mda_iter
        elif self.method == self.QuasiNewtonMethod.HYBRID:
            options["xtol"] = self.tolerance
            options["maxfev"] = self.max_mda_iter
        return options

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
            self._update_local_data_from_array(x_vect)

            for discipline in self.disciplines:
                discipline.linearize(self.io.data)

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
        if self.method not in self._METHODS_SUPPORTING_CALLBACKS:
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
            self._compute_normalized_residual_norm()
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
        local_data_copy = self.io.data.copy()
        self._update_local_data_from_array(x_vect)
        local_data_before_execution = self.io.data
        self.io.data = local_data_copy
        self._execute_disciplines_and_update_local_data(local_data_before_execution)
        self._compute_residuals(local_data_before_execution)
        return self.assembly.residuals(
            local_data_before_execution, self._resolved_variable_names
        ).real

    def _run(self) -> DisciplineData:
        super()._run()

        self._execute_disciplines_and_update_local_data()

        if not self.strong_couplings:
            msg = (
                "MDAQuasiNewton found no strong couplings. Executed all"
                "disciplines once."
            )
            LOGGER.warning(msg)
            self.io.data[self.NORMALIZED_RESIDUAL_NORM] = array([0.0])
            return self.io.data

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
            options=self._get_options(),
        )

        self._warn_convergence_criteria()

        self._update_local_data_from_array(y_opt.x)

        if self.method in self._METHODS_SUPPORTING_CALLBACKS:
            self.io.data[self.NORMALIZED_RESIDUAL_NORM] = array([self.normed_residual])

        return self.io.data
