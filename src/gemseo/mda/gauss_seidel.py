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
#        :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""A Gauss Seidel algorithm for solving MDAs."""
from __future__ import annotations

from typing import Any
from typing import Mapping
from typing import Sequence

from gemseo.core.chain import MDOChain
from gemseo.core.coupling_structure import MDOCouplingStructure
from gemseo.core.discipline import MDODiscipline
from gemseo.mda.mda import MDA


class MDAGaussSeidel(MDA):
    """An MDA analysis based on the Gauss-Seidel algorithm.

    This algorithm is an iterative technique to solve the linear system:

    .. math::

       Ax = b

    by decomposing the matrix :math:`A`
    into the sum of a lower triangular matrix :math:`L_*`
    and a strictly upper triangular matrix :math:`U`.

    The new iterate is given by:

    .. math::

       x_{k+1} = L_*^{-1}(b-Ux_k)
    """

    _ATTR_TO_SERIALIZE = MDA._ATTR_TO_SERIALIZE + (
        "strong_couplings",
        "over_relax_factor",
        "normed_residual",
    )

    def __init__(
        self,
        disciplines: Sequence[MDODiscipline],
        name: str | None = None,
        max_mda_iter: int = 10,
        grammar_type: str = MDODiscipline.JSON_GRAMMAR_TYPE,
        tolerance: float = 1e-6,
        linear_solver_tolerance: float = 1e-12,
        warm_start: bool = False,
        use_lu_fact: bool = False,
        over_relax_factor: float = 1.0,
        coupling_structure: MDOCouplingStructure | None = None,
        log_convergence: bool = False,
        linear_solver: str = "DEFAULT",
        linear_solver_options: Mapping[str, Any] = None,
    ) -> None:
        """
        Args:
            over_relax_factor: The relaxation coefficient,
                used to make the method more robust,
                if ``0<over_relax_factor<1`` or faster if ``1<over_relax_factor<=2``.
                If ``over_relax_factor =1.``, it is deactivated.
        """
        self.chain = MDOChain(disciplines, grammar_type=grammar_type)
        super().__init__(
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
        assert over_relax_factor > 0.0
        assert over_relax_factor <= 2.0
        self.over_relax_factor = over_relax_factor
        self._set_default_inputs()
        self._compute_input_couplings()

    def _initialize_grammars(self):
        self.input_grammar.update(self.chain.input_grammar)
        self.output_grammar.update(self.chain.output_grammar)
        self._add_residuals_norm_to_output_grammar()

    def _run(self):
        # Run the disciplines in a sequential way
        # until the difference between outputs is under tolerance.
        if self.warm_start:
            self._couplings_warm_start()
        current_couplings = 0.0

        relax = self.over_relax_factor
        use_relax = relax != 1.0

        while not self._stop_criterion_is_reached or self._current_iter == 0:
            for discipline in self.disciplines:
                discipline.execute(self.local_data)
                outs = discipline.get_output_data()
                if use_relax:
                    # First time this output is computed, update directly local data
                    self.local_data.update(
                        {k: v for k, v in outs.items() if k not in self.local_data}
                    )
                    # The couplings already exist in the local data,
                    # so the over relaxation can be applied
                    self.local_data.update(
                        {
                            k: relax * v + (1.0 - relax) * self.local_data[k]
                            for k, v in outs.items()
                            if k in self.local_data
                        }
                    )
                else:
                    self.local_data.update(outs)

            new_couplings = self._current_strong_couplings()
            self._compute_residual(
                current_couplings,
                new_couplings,
                log_normed_residual=self.log_convergence,
            )
            current_couplings = new_couplings

        for discipline in self.disciplines:  # Update all outputs without relax
            self.local_data.update(discipline.get_output_data())
