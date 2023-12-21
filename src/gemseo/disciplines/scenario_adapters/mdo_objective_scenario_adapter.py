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
#                        documentation
#        :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
#        :author: Pierre-Jean Barjhoux, Benoit Pauwels - MDOScenarioAdapter
#                                                        Jacobian computation
"""A scenario adapter overwriting the local data with the optimal objective."""

from __future__ import annotations

from typing import TYPE_CHECKING

from numpy import atleast_1d

from gemseo.algos.post_optimal_analysis import PostOptimalAnalysis
from gemseo.disciplines.scenario_adapters.mdo_scenario_adapter import MDOScenarioAdapter

if TYPE_CHECKING:
    from collections.abc import Sequence


class MDOObjectiveScenarioAdapter(MDOScenarioAdapter):
    """A scenario adapter overwriting the local data with the optimal objective."""

    def _retrieve_top_level_outputs(self) -> None:
        formulation = self.scenario.formulation
        opt_problem = formulation.opt_problem
        top_level_disciplines = formulation.get_top_level_disc()

        # Get the optimal outputs
        optimum = opt_problem.design_space.get_current_value(as_dict=True)
        f_opt = opt_problem.get_optimum()[0]
        if not opt_problem.minimize_objective:
            f_opt = -f_opt
        if not opt_problem.is_mono_objective:
            raise ValueError("The objective function must be single-valued.")

        # Overwrite the adapter local data
        objective = opt_problem.objective.output_names[0]
        if objective in self._output_names:
            self.local_data[objective] = atleast_1d(f_opt)

        for output in self._output_names:
            if output != objective:
                for discipline in top_level_disciplines:
                    if discipline.is_output_existing(output) and output not in optimum:
                        self.local_data[output] = discipline.local_data[output]

                value = optimum.get(output)
                if value is not None:
                    self.local_data[output] = value

    def _compute_jacobian(
        self,
        inputs: Sequence[str] | None = None,
        outputs: Sequence[str] | None = None,
    ) -> None:
        MDOScenarioAdapter._compute_jacobian(self, inputs, outputs)
        # The gradient of the objective function cannot be computed by the
        # disciplines, but the gradients of the constraints can.
        # The objective function is assumed independent of non-optimization
        # variables.
        obj_name = self.scenario.formulation.opt_problem.objective.output_names[0]
        mult_cstr_jac_key = PostOptimalAnalysis.MULT_DOT_CONSTR_JAC
        self.jac[obj_name] = dict(self.jac[mult_cstr_jac_key])
