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

from gemseo.mda.base_parallel_mda_solver import BaseParallelMDASolver
from gemseo.mda.quasi_newton_settings import MDAQuasiNewton_Settings
from gemseo.mda.quasi_newton_settings import QuasiNewtonMethod

if TYPE_CHECKING:
    from collections.abc import Sequence
    from typing import Any

    from gemseo.core.discipline import Discipline

LOGGER = logging.getLogger(__name__)


class MDAQuasiNewton(BaseParallelMDASolver):
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

    Settings: ClassVar[type[MDAQuasiNewton_Settings]] = MDAQuasiNewton_Settings
    """The pydantic model for the settings."""

    _METHODS_SUPPORTING_CALLBACKS: ClassVar[tuple[QuasiNewtonMethod, ...]] = (
        QuasiNewtonMethod.BROYDEN1,
        QuasiNewtonMethod.BROYDEN2,
    )
    """The methods supporting callback functions."""

    disciplines: tuple[Discipline, ...]
    """The disciplines."""

    settings: MDAQuasiNewton_Settings
    """The settings of the MDA"""

    def __init__(
        self,
        disciplines: Sequence[Discipline],
        settings_model: MDAQuasiNewton_Settings | None = None,
        **settings: Any,
    ) -> None:
        """
        Raises:
            ValueError: If the method is not a valid quasi-Newton method.
        """  # noqa:D205 D212 D415
        super().__init__(disciplines, settings_model=settings_model, **settings)
        self._set_resolved_variables(self.coupling_structure.strong_couplings)

        if self.settings.method not in self._METHODS_SUPPORTING_CALLBACKS:
            del self.io.output_grammar[self.NORMALIZED_RESIDUAL_NORM]

    def __get_options(self) -> dict[str, float | int]:
        """Get the options adapted to the resolution method."""
        options = {}
        if self.settings.method in {
            QuasiNewtonMethod.BROYDEN1,
            QuasiNewtonMethod.BROYDEN2,
            QuasiNewtonMethod.ANDERSON,
            QuasiNewtonMethod.LINEAR_MIXING,
            QuasiNewtonMethod.DIAG_BROYDEN,
            QuasiNewtonMethod.EXCITING_MIXING,
            QuasiNewtonMethod.KRYLOV,
        }:
            options["ftol"] = self.settings.tolerance
            options["maxiter"] = self.settings.max_mda_iter
        elif self.settings.method == QuasiNewtonMethod.LEVENBERG_MARQUARDT:
            options["xtol"] = self.settings.tolerance
            options["maxiter"] = self.settings.max_mda_iter
        elif self.settings.method == QuasiNewtonMethod.DF_SANE:
            options["fatol"] = self.settings.tolerance
            options["maxfev"] = self.settings.max_mda_iter
        else:  # Necessarily the HYBRID quasi-Newton method
            options["xtol"] = self.settings.tolerance
            options["maxfev"] = self.settings.max_mda_iter
        return options

    def __get_jacobian_computer(self) -> Callable[[ndarray], ndarray] | None:
        """Return the function to compute the jacobian.

        Returns:
            The callable to compute the jacobian.
        """
        if not self.settings.use_gradient:
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

            for discipline in self._disciplines:
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
        if self.settings.method not in self._METHODS_SUPPORTING_CALLBACKS:
            return None

        def callback(iterate: ndarray, residual: ndarray) -> None:
            """Store the current residual in the history.

            Args:
                iterate: The current iterate.
                residual: The associated residual.
            """
            self._compute_normalized_residual_norm()

        return callback

    def __compute_residuals(self, x_vect: ndarray) -> ndarray:
        """Evaluate all residuals, possibly in parallel.

        Args:
            x_vect: The value of the design variables.

        Returns:
            The residuals.
        """
        self._update_local_data_from_array(x_vect)

        local_data_before_execution = self.io.data.copy()
        self._execute_disciplines_and_update_local_data()
        self._compute_residuals(local_data_before_execution)

        return self.get_current_resolved_residual_vector()

    def _execute(self) -> None:
        super()._execute()

        self._execute_disciplines_and_update_local_data()

        if not self.coupling_structure.strong_couplings:
            msg = (
                "MDAQuasiNewton found no strong couplings. Executed all"
                "disciplines once."
            )
            LOGGER.warning(msg)
            self.io.data[self.NORMALIZED_RESIDUAL_NORM] = array([0.0])
            return self.io.data

        self._current_iter = 0

        if self.reset_history_each_run:
            self.residual_history = []

        # solve the system
        y_opt = root(
            self.__compute_residuals,
            x0=self.get_current_resolved_variables_vector().real,
            method=self.settings.method,
            jac=self.__get_jacobian_computer(),
            callback=self.__get_residual_history_callback(),
            tol=self.settings.tolerance,
            options=self.__get_options(),
        )

        self._check_stopping_criteria()

        self._update_local_data_from_array(y_opt.x)

        if self.settings.method in self._METHODS_SUPPORTING_CALLBACKS:
            self.io.update_output_data({
                self.NORMALIZED_RESIDUAL_NORM: array([self.normed_residual]),
            })
        return None
