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
#        :author: Benoit Pauwels - Stacked data management
#               (e.g. iteration index)
#        :author: Gilberto Ruiz Jimenez
"""The MDOFunction subclass to create a function from a Discipline."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Callable
from typing import ClassVar

from gemseo.core.mdo_functions.discipline_adapter_generator import (
    DisciplineAdapterGenerator,
)
from gemseo.core.mdo_functions.mdo_function import MDOFunction

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Sequence

    from gemseo.core.discipline import Discipline
    from gemseo.core.mdo_functions.discipline_adapter import DisciplineAdapter
    from gemseo.formulations.base_formulation import BaseFormulation
    from gemseo.typing import BooleanArray
    from gemseo.typing import NumberArray


class FunctionFromDiscipline(MDOFunction):
    """A function computing some outputs of a discipline for some of its inputs.

    This function is called from an input vector defined from a larger number of input
    variables.
    """

    __all_input_names: Iterable[str]
    """The names of the discipline inputs for the function.

    Use all the design variables of ``formulation.design_space`` if empty.
    """

    __all_differentiated_input_names: Iterable[str]
    """The names of the inputs to differentiate the function.

    If ``formulation.differentiated_input_names_substitute`` is empty,
    use ``__all_input_names``.
    """

    __input_names: Sequence[str]
    """The names of the discipline inputs for the discipline adapter."""

    __differentiated_input_names: Sequence[str]
    """The names of the inputs to differentiate the discipline adapter.

    If ``formulation.differentiated_input_names_substitute`` is empty,
    use ``__input_names``.
    """

    __input_mask: BooleanArray | None
    """The input components to unmask, set at the first evaluation of the function."""

    __unmask_x_swap_order: Callable[
        [Iterable[str], NumberArray, Iterable[str] | None, NumberArray | None],
        NumberArray,
    ]
    """Unmask a vector from a subset of names, with respect to a set of names."""

    __get_x_mask_x_swap_order: Callable[
        [Iterable[str], Iterable[str] | None], NumberArray
    ]
    """Mask a vector from a subset of names, with respect to a set of names."""

    __discipline_adapter: DisciplineAdapter
    """The discipline adapter."""

    generator_class: ClassVar[type[DisciplineAdapterGenerator]] = (
        DisciplineAdapterGenerator
    )
    """The class used to generator the :class:`DisciplineAdapter`."""

    def __init__(
        self,
        output_names: Iterable[str],
        formulation: BaseFormulation,
        discipline: Discipline | None = None,
        top_level_disc: bool = True,
        input_names: Sequence[str] = (),
        all_input_names: Iterable[str] = (),
        is_differentiable: bool = True,
    ) -> None:
        """
        Args:
            output_names: The discipline output names
                defining the function output vector.
            formulation: The MDO formulation to which the function will be attached.
            discipline: The discipline computing these outputs.
                If ``None``,
                the discipline is detected from the inner disciplines.
            top_level_disc: Whether the inner disciplines
                are the top level disciplines of the formulation;
                otherwise,
                the disciplines used to instantiate the formulation are considered.
            input_names: The discipline input names defining the input vector
                of the discipline adapter.
                If empty,
                use the names of the discipline inputs that are also design variables.
            all_input_names: The discipline input names
                defining the function input vector.
                If empty,
                use all the design variables.
            is_differentiable: Whether the function is differentiable.
        """  # noqa: D205, D212, D415
        self.__all_input_names = all_input_names
        self.__input_names = input_names
        self.__input_mask = None
        self.__unmask_x_swap_order = formulation.unmask_x_swap_order
        self.__get_x_mask_x_swap_order = formulation.get_x_mask_x_swap_order
        discipline_adapter_generator = self.__get_discipline_adapter_generator(
            formulation, output_names, discipline, top_level_disc
        )
        if not input_names:
            discipline = discipline_adapter_generator.discipline
            self.__input_names = formulation.get_x_names_of_disc(discipline)

        self.__discipline_adapter = discipline_adapter_generator.get_function(
            self.__input_names,
            output_names,
            is_differentiable=is_differentiable,
            differentiated_input_names_substitute=formulation.differentiated_input_names_substitute,  # noqa: E501
        )
        self.__differentiated_input_names = (
            self.__discipline_adapter.differentiated_input_names_substitute
        )
        if formulation.differentiated_input_names_substitute:
            self.__all_differentiated_input_names = self.__differentiated_input_names
        else:
            self.__all_differentiated_input_names = all_input_names
        super().__init__(
            self._func_to_wrap,
            jac=self._jac_to_wrap,
            name=self.__discipline_adapter.name,
            input_names=self.__input_names,
            expr=self.__discipline_adapter.expr,
            dim=self.__discipline_adapter.dim,
            output_names=self.__discipline_adapter.output_names,
        )

    @property
    def discipline_adapter(self) -> DisciplineAdapter:
        """The discipline adapter."""
        return self.__discipline_adapter

    def _func_to_wrap(self, x_vect: NumberArray) -> NumberArray:
        """Compute the outputs.

        Args:
            x_vect: The input vector

        Returns:
            The value of the outputs.
        """
        return self.__discipline_adapter.evaluate(x_vect[self._input_mask])

    def _jac_to_wrap(self, x_vect: NumberArray) -> NumberArray:
        """Compute the gradient of the outputs.

        Args:
            x_vect: The input vector.

        Returns:
            The value of the gradient of the outputs.
        """
        # TODO: The support of sparse Jacobians requires modifications here.
        jac = self.__unmask_x_swap_order(
            self.__differentiated_input_names,
            self.__discipline_adapter.jac(x_vect[self._input_mask]),
            self.__all_differentiated_input_names,
        )
        jac.astype(x_vect.dtype)
        return jac

    @property
    def _input_mask(self) -> BooleanArray:
        """The components of interest in the full input vector."""
        if self.__input_mask is None:
            self.__input_mask = self.__get_x_mask_x_swap_order(
                self.__input_names, self.__all_input_names
            )

        return self.__input_mask

    @classmethod
    def __get_discipline_adapter_generator(
        cls,
        formulation: BaseFormulation,
        output_names: Iterable[str],
        discipline: Discipline | None,
        use_top_level_disciplines: bool,
    ) -> DisciplineAdapterGenerator:
        """Create the generator of discipline adapters.

        Args:
            formulation: The formulation to which the function will be attached.
            output_names: The discipline outputs defining the function output vector.
            discipline: The discipline computing these outputs.
                If ``None``,
                the discipline is detected from the inner disciplines.
            use_top_level_disciplines: Whether the inner disciplines
                are the top level disciplines of the formulation;
                otherwise,
                the disciplines used to instantiate the formulation are considered.

        Returns:
            The generator of discipline adapters.

        Raises:
            ValueError: If no discipline is found.
        """
        if discipline is not None:
            return cls.generator_class(discipline, formulation.variable_sizes)

        for discipline in (
            formulation.get_top_level_disciplines()
            if use_top_level_disciplines
            else formulation.disciplines
        ):
            if discipline.io.output_grammar.has_names(output_names):
                return cls.generator_class(discipline, formulation.variable_sizes)

        msg = (
            f"No discipline known by formulation {formulation.__class__.__name__}"
            f" has all outputs named {output_names}."
        )
        raise ValueError(msg)
