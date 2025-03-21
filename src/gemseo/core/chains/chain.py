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
"""Chains of disciplines."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import ClassVar

from strenum import LowercaseStrEnum
from strenum import StrEnum

from gemseo.core._process_flow.base_process_flow import BaseProcessFlow
from gemseo.core.coupling_structure import CouplingStructure
from gemseo.core.dependency_graph import DependencyGraph
from gemseo.core.derivatives.chain_rule import traverse_add_diff_io
from gemseo.core.derivatives.jacobian_operator import JacobianOperator
from gemseo.core.discipline import Discipline
from gemseo.core.process_discipline import ProcessDiscipline
from gemseo.utils.compatibility.scipy import array_classes
from gemseo.utils.derivatives.approximation_modes import ApproximationMode
from gemseo.utils.enumeration import merge_enums

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Sequence

    from numpy import ndarray


class _ProcessFlow(BaseProcessFlow):
    """The process flow."""

    def get_data_flow(  # noqa: D102
        self,
    ) -> list[tuple[Discipline, Discipline, list[str]]]:
        disciplines = self.get_disciplines_in_data_flow()
        graph = DependencyGraph(disciplines)
        disciplines_couplings = graph.get_disciplines_couplings()

        # Add discipline inner couplings (ex. MDA case)
        for discipline in disciplines:
            disciplines_couplings.extend(discipline.get_process_flow().get_data_flow())

        return disciplines_couplings


class MDOChain(ProcessDiscipline):
    """Chain of disciplines that is based on a predefined order of execution."""

    _process_flow_class: ClassVar[type[BaseProcessFlow]] = _ProcessFlow

    class _DerivationMode(LowercaseStrEnum):
        """The derivation modes."""

        REVERSE = "reverse"
        """The reverse Jacobian accumulation, chain rule from outputs to inputs."""

        AUTO = "auto"
        """Automatic switch between direct, reverse or adjoint depending on data
        sizes."""

    LinearizationMode = merge_enums(
        "LinearizationMode",
        StrEnum,
        ApproximationMode,
        _DerivationMode,
    )

    def __init__(
        self,
        disciplines: Sequence[Discipline],
        name: str = "",
    ) -> None:
        """
        Args:
            disciplines: The disciplines.
            name: The name of the discipline.
                If ``None``, use the class name.
        """  # noqa: D205, D212, D415
        super().__init__(disciplines, name=name)
        self._coupling_structure = None
        self._last_diff_inouts = None
        self._initialize_grammars()

    def _initialize_grammars(self) -> None:
        """Define the input and output grammars from the disciplines' ones."""
        self.io.input_grammar.clear()
        self.io.output_grammar.clear()
        for discipline in self._disciplines:
            self.io.input_grammar.update(
                discipline.io.input_grammar,
                excluded_names=self.io.output_grammar,
            )
            self.io.output_grammar.update(discipline.io.output_grammar)

    def _execute(self) -> None:
        for discipline in self._disciplines:
            self.io.data.update(discipline.execute(self.io.data))

    def reverse_chain_rule(
        self,
        chain_outputs: Iterable[str],
        discipline: Discipline,
    ) -> None:
        """Chain the derivatives with a new discipline in the chain in reverse mode.

        Perform chain ruling:
        (notation: D is total derivative, d is partial derivative)

        D out    d out      dinpt_1    d output      dinpt_2
        -----  = -------- . ------- + -------- . --------
        D new_in  d inpt_1  d new_in   d inpt_2   d new_in


        D out    d out        d out      dinpt_2
        -----  = -------- + -------- . --------
        D z      d z         d inpt_2     d z


        D out    d out      [dinpt_1   d out      d inpt_1    dinpt_2 ]
        -----  = -------- . [------- + -------- . --------  . --------]
        D z      d inpt_1   [d z       d inpt_1   d inpt_2     d z    ]

        Args:
            discipline: The new discipline to compose in the chain.
            chain_outputs: The outputs to lineariza.
        """
        # TODO : only linearize wrt needed inputs/inputs
        # use coupling_structure graph path for that
        last_cached = discipline.io.get_input_data()
        # The graph traversal algorithm avoid to compute unnecessary Jacobians
        discipline.linearize(last_cached, execute=False, compute_all_jacobians=False)

        for output_name in chain_outputs:
            if output_name in self.jac:
                # This output has already been taken from previous disciplines
                # Derivatives must be composed using the chain rule

                # Make a copy of the keys because the dict is changed in the
                # loop
                common_inputs = sorted(
                    set(self.jac[output_name].keys()).intersection(discipline.jac)
                )
                for input_name in common_inputs:
                    # Store reference to the current Jacobian
                    curr_jac = self.jac[output_name][input_name]
                    for new_in, new_jac in discipline.jac[input_name].items():
                        # Chain rule the derivatives
                        # TODO: sum BEFORE dot
                        if isinstance(new_jac, JacobianOperator):
                            # NumPy array @ JacobianOperator is not supported, thus
                            # imposing to explicitly use the __rmatmul__ method.
                            loc_dot = new_jac.__rmatmul__(curr_jac)
                        else:
                            loc_dot = curr_jac @ new_jac

                        # when input_name==new_in, we are in the case of an
                        # input being also an output
                        # in this case we must only compose the derivatives
                        if new_in in self.jac[output_name] and input_name != new_in:
                            # The output is already linearized wrt this
                            # input_name. We are in the case:
                            # d o     d o    d o     di_2
                            # ----  = ---- + ----- . -----
                            # d z     d z    d i_2    d z
                            if isinstance(loc_dot, JacobianOperator):
                                self.jac[output_name][new_in] = (
                                    loc_dot + self.jac[output_name][new_in]
                                )
                            else:
                                self.jac[output_name][new_in] += loc_dot
                        else:
                            # The output is not yet linearized wrt this
                            # input_name.  We are in the case:
                            #  d o      d o     di_1   d o     di_2
                            # -----  = ------ . ---- + ----  . ----
                            #  d x      d i_1   d x    d i_2    d x
                            self.jac[output_name][new_in] = loc_dot

            elif output_name in discipline.jac:
                # Output of the chain not yet filled in jac,
                # Take the jacobian dict of the current discipline to
                # Initialize. Make a copy !
                self.jac[output_name] = MDOChain.copy_jacs(discipline.jac[output_name])

    def _compute_diff_in_outs(
        self,
        input_names: Iterable[str],
        output_names: Iterable[str],
    ) -> None:
        if self._coupling_structure is None:
            self._coupling_structure = CouplingStructure(self._disciplines)

        diff_ios = (set(input_names), set(output_names))
        if self._last_diff_inouts != diff_ios:
            traverse_add_diff_io(
                self._coupling_structure.graph.graph, input_names, output_names
            )
            self._last_diff_inouts = diff_ios

    def _compute_jacobian(
        self,
        input_names: Iterable[str] = (),
        output_names: Iterable[str] = (),
    ) -> None:
        self._compute_diff_in_outs(input_names, output_names)

        # Initializes self jac with copy of last discipline (reverse mode)
        last_discipline = self._disciplines[-1]
        # TODO : only linearize wrt needed inputs/inputs
        # use coupling_structure graph path for that
        last_cached = last_discipline.io.get_input_data()

        # The graph traversal algorithm avoid to compute unnecessary Jacobians
        last_discipline.linearize(last_cached, execute=False)
        self.jac = self.copy_jacs(last_discipline.jac)

        # reverse mode of remaining disciplines
        remaining_disciplines = self._disciplines[:-1]
        for discipline in remaining_disciplines[::-1]:
            self.reverse_chain_rule(output_names, discipline)

        # Remove differentiations that should not be there,
        # because inputs are not inputs of the chain
        for output_jacobian in self.jac.values():
            # Copy keys because the dict in changed in the loop
            input_names_before_loop = list(output_jacobian.keys())
            for input_name in input_names_before_loop:
                if input_name not in input_names:
                    del output_jacobian[input_name]

        # Add differentiations that should be there,
        # because inputs of the chain but not of all disciplines.
        self._init_jacobian(
            input_names,
            output_names,
            fill_missing_keys=True,
            init_type=Discipline.InitJacobianType.SPARSE,
        )

    @staticmethod
    def copy_jacs(
        jacobian: dict[str, dict[str, ndarray]],
    ) -> dict[str, dict[str, ndarray]]:
        """Deepcopy a Jacobian dictionary.

        Args:
            jacobian: The Jacobian dictionary,
                which is a nested dictionary as ``{'out': {'in': derivatives}}``.

        Returns:
            The deepcopy of the Jacobian dictionary.
        """
        jacobian_copy = {}
        for output_name, output_jacobian in jacobian.items():
            if isinstance(output_jacobian, dict):
                output_jacobian_copy = {}
                jacobian_copy[output_name] = output_jacobian_copy
                for input_name, derivatives in output_jacobian.items():
                    output_jacobian_copy[input_name] = derivatives.copy()
            elif isinstance(output_jacobian, (array_classes, JacobianOperator)):
                jacobian_copy[output_name] = output_jacobian.copy()

        return jacobian_copy
