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
"""The Newton-Raphson algorithm for solving MDAs.

`Newton-Raphson <https://en.wikipedia.org/wiki/Newton%27s_method>`__
"""
from __future__ import annotations

import logging
from typing import Any
from typing import Mapping
from typing import Sequence

from numpy import concatenate
from numpy import ndarray

from gemseo.core.coupling_structure import MDOCouplingStructure
from gemseo.core.discipline import MDODiscipline
from gemseo.core.discipline_data import DisciplineData
from gemseo.mda.root import MDARoot
from gemseo.utils.data_conversion import update_dict_of_arrays_from_array

LOGGER = logging.getLogger(__name__)


class MDANewtonRaphson(MDARoot):
    r"""Newton solver for MDA.

    The `Newton-Raphson method
    <https://en.wikipedia.org/wiki/Newton%27s_method>`__ is parameterized by a
    relaxation factor :math:`\alpha \in (0, 1]` to limit the length of the
    steps taken along the Newton direction.  The new iterate is given by:

    .. math::

       x_{k+1} = x_k - \alpha f'(x_k)^{-1} f(x_k)
    """

    __newton_linear_solver_name: str
    """The name of the linear solver for the Newton method: can be "DEFAULT", "GMRES" or
    "BICGSTAB"."""

    __newton_linear_solver_options: Mapping[str, Any] | None
    """The options of the Newton linear solver."""

    _newton_coupling_names: set(str)
    """The coupling data names used to form the residuals."""

    def __init__(
        self,
        disciplines: Sequence[MDODiscipline],
        max_mda_iter: int = 10,
        relax_factor: float = 0.99,
        name: str | None = None,
        grammar_type: MDODiscipline.GrammarType = MDODiscipline.GrammarType.JSON,
        linear_solver: str = "DEFAULT",
        tolerance: float = 1e-6,
        linear_solver_tolerance: float = 1e-12,
        warm_start: bool = False,
        use_lu_fact: bool = False,
        coupling_structure: MDOCouplingStructure | None = None,
        log_convergence: bool = False,
        linear_solver_options: Mapping[str, Any] = None,
        newton_linear_solver_name: str = "DEFAULT",
        newton_linear_solver_options: Mapping[str, Any] | None = None,
        parallel: bool = False,
        use_threading: bool = True,
        n_processes: int = MDARoot.N_CPUS,
    ) -> None:
        """
        Args:
            relax_factor: The relaxation factor in the Newton step.
            newton_linear_solver: The name of the linear solver for the Newton method.
            newton_linear_solver_options: The options for the Newton linear solver.

        Raises:
            ValueError: When there are no coupling variables, or when there are
                weakly coupled disciplines.
                In these cases, use MDAChain.
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
            log_convergence=log_convergence,
            parallel=parallel,
            use_threading=use_threading,
            n_processes=n_processes,
        )
        self.__relax_factor = 0.0
        self.relax_factor = relax_factor
        self.linear_solver = linear_solver

        # We use all couplings to form the Newton matrix otherwise the effect of the
        # weak couplings are not taken into account in the coupling updates.
        self._newton_coupling_names = set(self.all_couplings)

        self.__newton_linear_solver_name = newton_linear_solver_name
        self.__newton_linear_solver_options = newton_linear_solver_options or {}

        if not self._newton_coupling_names:
            raise ValueError(
                "There is no couplings to compute. Please consider using MDAChain."
            )

        strongly_coupled = self.coupling_structure.get_strongly_coupled_disciplines()
        if len(strongly_coupled) < len(self.disciplines):
            raise ValueError(
                "The MDANewtonRaphson has weakly coupled disciplines, "
                "which is not supported. Please consider using a MDAChain with "
                "the option inner_mda_name='MDANewtonRaphson'."
            )

    @property
    def relax_factor(self) -> float:
        """The relaxation factor in the Newton step."""
        return self.__relax_factor

    @relax_factor.setter
    def relax_factor(self, value: float) -> None:
        self.__relax_factor = value
        self.__check_relax_factor()

    def __check_relax_factor(self) -> None:
        """Check that the relaxation factor in the Newton step is in (0, 1].

        Raises:
            ValueError: When the relaxation factor is not in (0, 1].
        """
        if not (0.0 < self.relax_factor <= 1.0):
            raise ValueError(
                "Newton relaxation factor should belong to (0, 1] "
                f"(current value: {self.relax_factor})."
            )

    def _get_coupling_values(self, data: Mapping[str, ndarray]) -> ndarray:
        """Concatenates the values of the coupling variables.

        Args:
            data: The data to take the values of the couplings from.

        Returns:
            The concatenated coupling values.
        """
        return concatenate([data[name] for name in self._newton_coupling_names])

    # TODO: API: prepend with verb.
    def _newton_step(self, input_data: dict[str:Any], residuals: ndarray) -> ndarray:
        """Compute the full Newton step without relaxation.

        The Newton step is :math:`-[dR/dY]^{-1}.R`, where R is the coupling residuals
        and Y is the coupling vector.

        Args:
            input_data: The input data for the disciplines.
            residuals: The residuals vector.

        Returns:
            The Newton step.
        """
        newton_step, is_converged = self.assembly.compute_newton_step(
            input_data,
            self._newton_coupling_names,
            self.__newton_linear_solver_name,
            matrix_type=self.matrix_type,
            residuals=residuals,
            **self.__newton_linear_solver_options,
        )
        if not is_converged:
            LOGGER.warning(
                "The linear solver %s failed "
                "to converge during the Newton step computation.",
                self.__newton_linear_solver_name,
            )
        return newton_step

    def _get_disciplines_output_data(self) -> DisciplineData:
        """Retreive the disciplines output data.

        Returns:
            A concatenated mapping containing all disciplines output data.
        """
        disciplines_data = self.disciplines[0].get_output_data()
        for discipline in self.disciplines[1:]:
            disciplines_data.update(discipline.get_output_data())

        return disciplines_data

    def _run(self) -> None:
        if self.warm_start:
            self._couplings_warm_start()

        input_data = self.local_data.copy()  # Dont alter self inputs
        current_couplings = self._get_coupling_values(input_data)
        # The residuals for the Newton step must be computed on all couplings
        # otherwise cases with several cycles of strong couplings that are
        # weakly coupled together will fail.
        first_iteration = True
        while True:
            self.execute_all_disciplines(input_data, update_local_data=False)
            # self.linearize_all_disciplines(input_data)
            # Dont update local data otherwise the linearization is not performed
            # at the same values as execution and this is expansive.
            disc_output_data = self._get_disciplines_output_data()
            new_couplings = self._get_coupling_values(disc_output_data)
            self._compute_residual(
                current_couplings,
                new_couplings,
                log_normed_residual=self.log_convergence,
            )
            if self._stop_criterion_is_reached:
                break
            if first_iteration:
                self.assembly.set_newton_differentiated_ios(self._newton_coupling_names)
                first_iteration = False
            # Linearize the disciplines with the same input_data as execution
            # to ensure minimal computation time and apply a pure Newton step
            self.linearize_all_disciplines(input_data, execute=False)
            residuals = (new_couplings - current_couplings).real
            current_couplings += self.relax_factor * self._newton_step(
                input_data, residuals
            )

            update_dict_of_arrays_from_array(
                input_data, self._newton_coupling_names, current_couplings, copy=False
            )

        self._update_local_data_from_disciplines()
