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
#
# Copyright 2024 Capgemini
# Contributors:
#    INITIAL AUTHORS - API and implementation and/or documentation
#        :author: Charlie Vanaret
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
#        :author: Jean-Christophe Giret
"""An advanced MDA splitting algorithm based on graphs."""

from __future__ import annotations

import logging
from itertools import repeat
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar

from numpy import array

from gemseo.core._process_flow.execution_sequences.sequential import (
    SequentialExecSequence,
)
from gemseo.core.chains.chain import MDOChain
from gemseo.core.chains.initialization_chain import MDOInitializationChain
from gemseo.core.chains.parallel_chain import MDOParallelChain
from gemseo.mda.base_mda import BaseMDA
from gemseo.mda.base_mda import BaseProcessFlow
from gemseo.mda.base_mda import _BaseMDAProcessFlow
from gemseo.mda.base_mda_settings import BaseMDASettings
from gemseo.mda.factory import MDAFactory
from gemseo.mda.mda_chain_settings import MDAChain_Settings
from gemseo.utils.constants import READ_ONLY_EMPTY_DICT

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Mapping
    from collections.abc import Sequence

    from gemseo.core.discipline.discipline import Discipline
    from gemseo.core.discipline.discipline_data import DisciplineData
    from gemseo.mda.base_mda_solver import BaseMDASolver
    from gemseo.typing import RealArray
    from gemseo.typing import StrKeyMapping
    from gemseo.utils.matplotlib_figure import FigSizeType

LOGGER = logging.getLogger(__name__)


class _ProcessFlow(_BaseMDAProcessFlow):
    """The process data and execution flow."""

    _node: MDAChain

    def get_data_flow(  # noqa:D102
        self,
    ) -> list[tuple[Discipline, Discipline, list[str]]]:
        return self._node.mdo_chain.get_process_flow().get_data_flow()

    def get_execution_flow(self) -> SequentialExecSequence:  # noqa:D102
        exec_s = SequentialExecSequence()
        exec_s.extend(self._node.mdo_chain.get_process_flow().get_execution_flow())
        return exec_s

    def get_disciplines_in_data_flow(self) -> list[Discipline]:  # noqa: D102
        return self._node.mdo_chain.get_process_flow().get_disciplines_in_data_flow()  # noqa: E501


class MDAChain(BaseMDA):
    """A chain of MDAs.

    The execution sequence is provided by the :class:`.DependencyGraph`.
    """

    Settings: ClassVar[type[MDAChain_Settings]] = MDAChain_Settings
    """The pydantic model for the settings."""

    _process_flow_class: ClassVar[type[BaseProcessFlow]] = _ProcessFlow

    inner_mdas: list[BaseMDASolver]
    """The ordered MDAs."""

    mdo_chain: MDOChain
    """The chain of MDAs."""

    settings: MDAChain_Settings
    """The settings of the MDA"""

    __inner_mda_class: BaseMDASolver
    """The inner MDA class."""

    def __init__(  # noqa: D107
        self,
        disciplines: Sequence[Discipline],
        settings_model: MDAChain_Settings | None = None,
        **settings: Any,
    ) -> None:
        self.mdo_chain = None
        super().__init__(disciplines, settings_model=settings_model, **settings)

        if (
            not self.coupling_structure.all_couplings
            and not self.settings.chain_linearize
        ):
            LOGGER.warning("No coupling in MDA, switching chain_linearize to True.")
            self.settings.chain_linearize = True

        self.inner_mdas = []
        self.__inner_mda_class = MDAFactory().get_class(self.settings.inner_mda_name)
        self.mdo_chain = self._create_mdo_chain()

        self._initialize_grammars()
        self._check_consistency()
        self._compute_input_coupling_names()

        # cascade the tolerance
        for mda in self.inner_mdas:
            mda.settings.tolerance = self.settings.tolerance

    @BaseMDA.scaling.setter
    def scaling(self, scaling: BaseMDA.ResidualScaling) -> None:  # noqa: D102
        self._scaling = scaling
        for mda in self.inner_mdas:
            mda.scaling = scaling

    def set_bounds(  # noqa: D102
        self,
        variable_names_to_bounds: Mapping[
            str, tuple[RealArray | None, RealArray | None]
        ],
    ) -> None:
        for inner_mda in self.inner_mdas:
            inner_mda.set_bounds(variable_names_to_bounds)

    def _create_mdo_chain(self) -> MDOChain:
        """Create an MDO chain from the execution sequence of the disciplines."""
        if not self.settings.sub_coupling_structures:
            sub_coupling_structures = repeat(None)
        else:
            sub_coupling_structures = self.settings.sub_coupling_structures

        self.__sub_coupling_structures_iterator = iter(sub_coupling_structures)

        chained_disciplines = []
        for parallel_tasks in self.coupling_structure.sequence:
            process = self.__create_process_from_disciplines(parallel_tasks)
            chained_disciplines.append(process)

        return MDOChain(chained_disciplines, name="MDA chain")

    def __create_process_from_disciplines(
        self,
        parallel_tasks: list[tuple[Discipline, ...]],
    ) -> Discipline:
        """Create a process from disciplines.

        This method creates a process that will be appended to the main inner
        :class:`.MDOChain` of the :class:`.MDAChain`. Depending on the number and type
        of disciplines, as well as the options provided by the user, the process may be
        a sole discipline, a :class:`.BaseMDA`, a :class:`MDOChain`, or a
        :class:`MDOParallelChain`.

        Args:
            parallel_tasks: The parallel tasks to be processed.

        Returns:
            A process.
        """
        parallel_disciplines = self.__compute_parallel_disciplines(parallel_tasks)

        if len(parallel_disciplines) == 1:
            return parallel_disciplines[0]

        if self.settings.mdachain_parallelize_tasks:
            return MDOParallelChain(
                parallel_disciplines,
                **self.settings.mdachain_parallel_settings,
            )
        return MDOChain(parallel_disciplines)

    def __compute_parallel_disciplines(
        self,
        parallel_tasks: list[tuple[Discipline, ...]],
    ) -> Sequence[Discipline | BaseMDA]:
        """Compute the parallel disciplines.

        This method computes the parallel disciplines,
        if any.
        If there is any coupled disciplines in a parallel task,
        a :class:`.BaseMDA` is created,
        based on the :class:`.BaseMDA` options provided.

        Args:
            parallel_tasks: The parallel tasks.

        Returns:
            The parallel disciplines.
        """
        parallel_disciplines = []
        for coupled_disciplines in parallel_tasks:
            if self.__requires_mda(coupled_disciplines):
                ordered_disciplines = [
                    discipline_
                    for discipline_ in self._disciplines
                    if discipline_ in coupled_disciplines
                ]

                settings_model = self.__create_inner_mda_settings()
                settings_model.coupling_structure = next(
                    self.__sub_coupling_structures_iterator
                )

                discipline = self.__inner_mda_class(
                    disciplines=ordered_disciplines,
                    settings_model=settings_model,
                )

                self.inner_mdas.append(discipline)
            else:
                discipline = coupled_disciplines[0]

            parallel_disciplines.append(discipline)

        self.settings._sub_mdas = self.inner_mdas
        return parallel_disciplines

    def __create_inner_mda_settings(self) -> BaseMDASettings:
        """Create the inner MDA settings model."""
        inner_settings = dict(self.settings.inner_mda_settings) | {
            name: setting
            for name, setting in self.settings
            if name in BaseMDASettings.model_fields
        }
        return self.__inner_mda_class.Settings(**inner_settings)

    def __requires_mda(self, disciplines: tuple[Discipline, ...]) -> bool:
        """Whether the disciplines require to be embed in an MDA.

        Args:
            disciplines: The disciplines to check.
        """
        return len(disciplines) > 1 or (
            len(disciplines) == 1
            and self.coupling_structure.is_self_coupled(disciplines[0])
            and not isinstance(disciplines[0], BaseMDA)
        )

    def _initialize_grammars(self) -> None:
        """Define all inputs and outputs of the chain."""
        if self.mdo_chain is None:  # First call by super class must be ignored.
            return
        self.io.input_grammar = self.mdo_chain.io.input_grammar.copy()
        self.io.output_grammar = self.mdo_chain.io.output_grammar.copy()

    def _check_consistency(self) -> None:
        """Check if there is no more than 1 equation per variable.

        For instance if a strong coupling is not also a self coupling.
        """
        if self.mdo_chain is None:  # First call by super class must be ignored.
            return
        super()._check_consistency()

    def execute(  # noqa:D102
        self,
        input_data: StrKeyMapping = READ_ONLY_EMPTY_DICT,
    ) -> DisciplineData:
        # The initialization is needed for MDA loops.
        if (
            self.settings.initialize_defaults
            and len(self._disciplines) > 1
            and len(self.coupling_structure.strong_couplings) > 0
        ):
            init_chain = MDOInitializationChain(
                self._disciplines,
                available_data_names=input_data,
            )

            self.io.input_grammar.defaults.update({
                key: value
                for key, value in init_chain.execute(input_data).items()
                if key in self.io.input_grammar
            })
            self.settings.initialize_defaults = False
        return super().execute(input_data=input_data)

    def _execute(self) -> None:
        super()._execute()

        self.io.data = self.mdo_chain.execute(self.io.data)

        res_sum = 0.0
        for mda in self.inner_mdas:
            res_local = mda.io.data.get(self.NORMALIZED_RESIDUAL_NORM)
            if res_local is not None:
                res_sum += res_local[-1] ** 2

        self.io.update_output_data({
            self.NORMALIZED_RESIDUAL_NORM: array([res_sum**0.5])
        })

    def _compute_jacobian(
        self,
        input_names: Sequence[str] = (),
        output_names: Sequence[str] = (),
    ) -> None:
        if self.settings.chain_linearize:
            self.mdo_chain.add_differentiated_inputs(input_names)
            self.mdo_chain.add_differentiated_outputs(output_names)
            # the Jacobian of the MDA chain is the Jacobian of the MDO chain
            self.mdo_chain.linearize(self.io.get_input_data())
            self.jac = self.mdo_chain.jac
        else:
            super()._compute_jacobian(input_names, output_names)

    def add_differentiated_inputs(  # noqa:D102
        self,
        input_names: Iterable[str] = (),
    ) -> None:
        BaseMDA.add_differentiated_inputs(self, input_names)
        if self.settings.chain_linearize:
            self.mdo_chain.add_differentiated_inputs(input_names)

    def add_differentiated_outputs(  # noqa: D102
        self,
        output_names: Iterable[str] = (),
    ) -> None:
        BaseMDA.add_differentiated_outputs(self, output_names)
        if self.settings.chain_linearize:
            self.mdo_chain.add_differentiated_outputs(output_names)

    @property
    def normed_residual(self) -> float:
        """The normed_residuals, computed from the sub-MDAs residuals."""
        return sum(mda.normed_residual**2 for mda in self.inner_mdas) ** 0.5

    @normed_residual.setter
    def normed_residual(
        self,
        normed_residual: float,
    ) -> None:
        """Set the normed_residual.

        Has no effect, since the normed residuals are defined by inner-MDAs residuals
        (see associated property).

        Here for compatibility with mother class.
        """

    def plot_residual_history(  # noqa: D102
        self,
        show: bool = False,
        save: bool = True,
        n_iterations: int | None = None,
        logscale: tuple[int, int] = (),
        filename: Path | str = "",
        fig_size: FigSizeType = (50.0, 10.0),
    ) -> None:
        if filename:
            file_path = Path(filename)
        for mda in self.inner_mdas:
            if filename:
                path = file_path.parent / f"{mda.__class__.__name__}_{file_path.name}"
            else:
                path = filename
            mda.plot_residual_history(
                show, save, n_iterations, logscale, path, fig_size
            )
