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
#    INITIAL AUTHORS - initial API and implementation and/or
#                        initial documentation
#        :author: Francois Gallard, Charlie Vanaret
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""The Individual Discipline Feasible (IDF) formulation."""

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from typing import ClassVar

from numpy import abs as np_abs
from numpy import concatenate
from numpy import zeros

from gemseo.core.chains.parallel_chain import MDOParallelChain
from gemseo.core.coupling_structure import CouplingStructure
from gemseo.core.discipline import Discipline
from gemseo.core.mdo_functions.consistency_constraint import ConsistencyConstraint
from gemseo.core.mdo_functions.taylor_polynomials import compute_linear_approximation
from gemseo.formulations.base_mdo_formulation import BaseMDOFormulation
from gemseo.formulations.idf_settings import IDF_Settings
from gemseo.mda.mda_chain import MDAChain
from gemseo.utils.string_tools import pretty_repr

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Sequence
    from typing import Any

    from gemseo.algos.design_space import DesignSpace
    from gemseo.core.discipline import Discipline
    from gemseo.typing import RealArray

LOGGER = logging.getLogger(__name__)


class IDF(BaseMDOFormulation):
    r"""The Individual Discipline Feasible (IDF) formulation.

    The IDF formulation expresses an MDO problem as

    .. math::

       \begin{aligned}
       & \underset{x,z,y^t}{\text{min}} & & f(x, z, y^t) \\
       & \text{subject to}     & & g(x, z, y^t) \le 0 \\
       &                       & & h(x, z, y^t) = 0 \\
       &                       & & y_i(x_i, z, y^t_{j \neq i}) - y_i^t = 0,
                                   \quad \forall i \in \{1,\ldots, N\}
       \end{aligned}

    where

    - :math:`N` is the number of disciplines,
    - :math:`f` is the objective function,
    - :math:`g` are the inequality constraint functions,
    - :math:`h` are the equality constraint functions,
    - :math:`z` are the global design variables,
    - :math:`x=(x_1,x_2,\ldots,x_N)` are the local design variables,
    - :math:`x_i` are the design variables specific to the :math:`i`-th discipline,
    - :math:`y=(y_1,y_2,\ldots,y_N)` are the coupling variables
      outputted by the disciplines,
    - :math:`y_i` are the coupling variables outputted by the :math:`i`-th discipline,
    - :math:`y^t=(y_1^t,y_2^t,\ldots,y_N^t)` are the *target* coupling variables
      used by the disciplines in input,
    - :math:`y_i^t` are the target coupling variables
      used by the :math:`i`-th discipline,

    Note that:

    1. the search space includes
       both the design variables and the target coupling variables,
    2. the original constraints are supplemented
       by equality constraints called *consistency constraints*.
    3. the disciplinary analysis is made at each optimization iteration
       while the multidisciplinary analysis, i.e. :math:`y=y^t`, is made at the optimum.
    4. the use of the target coupling variables in input of the disciplines
       instead of the coupling variables decouples the multidisciplinary process
       and makes it possible to evaluate the disciplines in parallel.
    """  # noqa: E501

    Settings: ClassVar[type[IDF_Settings]] = IDF_Settings

    _settings: IDF_Settings

    __coupling_structure: CouplingStructure
    """The coupling structure of the disciplines."""

    _parallel_exec: MDOParallelChain | None
    """The process to execute the disciplines in parallel, if parallelization."""

    def __init__(  # noqa: D107
        self,
        disciplines: Sequence[Discipline],
        objective_name: str,
        design_space: DesignSpace,
        settings_model: IDF_Settings | None = None,
        **settings: Any,
    ) -> None:
        super().__init__(
            disciplines,
            objective_name,
            design_space,
            settings_model=settings_model,
            **settings,
        )
        if (n_processes := self._settings.n_processes) > 1:
            LOGGER.info(
                "IDF formulation: running in parallel on %s processes.",
                n_processes,
            )
            self._parallel_exec = MDOParallelChain(
                self.disciplines,
                use_threading=self._settings.use_threading,
                n_processes=n_processes,
            )
        else:
            self._parallel_exec = None

        self.__coupling_structure = CouplingStructure(disciplines)
        self._update_top_level_disciplines()
        self._build_constraints()
        self._build_objective_from_disc(objective_name)
        if self._settings.start_at_equilibrium:
            self._compute_equilibrium()

    @property
    def all_couplings(self) -> list[str]:
        """The inputs of disciplines that are also outputs of other disciplines."""
        return self.__coupling_structure.all_couplings

    @property
    def coupling_structure(self) -> CouplingStructure:
        """The coupling structure of the disciplines."""
        return self.__coupling_structure

    @property
    def normalize_constraints(self) -> bool:
        """Whether the outputs of the coupling consistency constraints are scaled."""
        return self._settings.normalize_constraints

    def _compute_equilibrium(self) -> None:
        """Perform an MDA to bring the target coupling variables at equilibrium."""
        design_space = self.optimization_problem.design_space

        # Perform the MDA.
        mda_chain = MDAChain(
            self.disciplines,
            settings_model=self._settings.mda_chain_settings_for_start_at_equilibrium,
        )
        initial = design_space.get_current_value(as_dict=True)
        final = mda_chain.execute(initial)

        # Set the target coupling variables to their equilibrium values.
        msg = "IDF formulation: the initial MDA sets %s to %s (before: %s)."
        for name in self.__coupling_structure.all_couplings:
            LOGGER.info(msg, name, current := final[name], initial[name])
            design_space.set_current_variable(name, current)

    def _update_top_level_disciplines(self) -> None:
        """Update the default input values of the top-level disciplines.

        Raises:
            ValueError: When a coupling variable is not defined in the design space.
        """
        strong_couplings = set(self.__coupling_structure.all_couplings)
        variable_names = self.optimization_problem.design_space
        if not strong_couplings.issubset(variable_names):
            missing_variables = strong_couplings.difference(variable_names)
            msg = (
                "IDF formulation: "
                f"the variables {pretty_repr(missing_variables, use_and=True)} "
                f"must be added to the design space."
            )
            raise ValueError(msg)
        self._set_default_input_values_from_design_space()

    def get_top_level_disciplines(  # noqa:D102
        self, include_sub_formulations: bool = False
    ) -> tuple[Discipline, ...]:
        if self._parallel_exec is None:
            return self.disciplines

        return (self._parallel_exec,)

    def _get_normalization_factor(
        self,
        output_couplings: Iterable[str],
    ) -> RealArray:
        """Compute the normalization factor for all output couplings.

        A normalization factor is the difference between the lower and upper bounds.

        Args:
            output_couplings: The names of the coupling variables.

        Returns:
            The concatenation of the normalization factors for all output couplings.
        """
        get_upper_bound = self.optimization_problem.design_space.get_upper_bound
        get_lower_bound = self.optimization_problem.design_space.get_lower_bound
        return concatenate([
            np_abs(get_upper_bound(name) - get_lower_bound(name))
            for name in output_couplings
        ])

    def _build_constraints(self) -> None:
        """Create the consistency constraints and add them to the optimization problem.

        The consistency constraints are equality constraints of the form "y = y_copy".
        """
        get_output_couplings = self.__coupling_structure.get_output_couplings
        for discipline in self.disciplines:
            if couplings := get_output_couplings(discipline, strong=False):
                constraint = ConsistencyConstraint(couplings, self)
                discipline_adapter = constraint.coupling_function.discipline_adapter
                if discipline_adapter.is_linear:
                    constraint = compute_linear_approximation(
                        constraint,
                        zeros(discipline_adapter.input_dimension),
                        f_type=constraint.ConstraintType.EQ,
                    )
                self.optimization_problem.add_constraint(constraint)
