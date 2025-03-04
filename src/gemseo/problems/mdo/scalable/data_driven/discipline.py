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
#                  initial documentation
#        :author:  Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Scalable discipline.

The :mod:`~gemseo.problems.mdo.scalable.data_driven.discipline`
implements the concept of scalable discipline.
This is a particular discipline
built from an input-output training dataset associated with a function
and generalizing its behavior to a new user-defined problem dimension,
that is to say new user-defined input and output dimensions.

Alone or in interaction with other objects of the same type,
a scalable discipline can be used to compare the efficiency of an algorithm
applying to disciplines with respect to the problem dimension,
e.g. optimization algorithm, surrogate model, MDO formulation, MDA, ...

The :class:`.ScalableDiscipline` class implements this concept.
It inherits from the :class:`.Discipline` class
in such a way that it can easily be used in a :class:`.Scenario`.
It is composed of a :class:`.ScalableModel`.

The user only needs to provide:

- the name of a class overloading :class:`.ScalableModel`,
- a dataset as an :class:`.Dataset`
- variables sizes as a dictionary
  whose keys are the names of inputs and outputs
  and values are their new sizes.
  If a variable is missing, its original size is considered.

The :class:`.ScalableModel` parameters can also be filled in,
otherwise the model uses default values.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

from gemseo.core.discipline import Discipline
from gemseo.problems.mdo.scalable.data_driven.factory import ScalableModelFactory
from gemseo.utils.constants import READ_ONLY_EMPTY_DICT
from gemseo.utils.data_conversion import split_array_to_dict_of_arrays

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Mapping

    from gemseo.datasets.io_dataset import IODataset
    from gemseo.typing import StrKeyMapping


class DataDrivenScalableDiscipline(Discipline):
    """A scalable discipline."""

    _ATTR_NOT_TO_SERIALIZE = Discipline._ATTR_NOT_TO_SERIALIZE.union(["scalable_model"])

    def __init__(
        self,
        name: str,
        data: IODataset,
        sizes: Mapping[str, int] = READ_ONLY_EMPTY_DICT,
        **parameters: Any,
    ) -> None:
        """
        Args:
            name: The name of the class of the scalable model.
            data: The training dataset.
            sizes: The sizes of the input and output variables.
                If empty, use the original sizes.
            **parameters: The parameters for the model.
        """  # noqa: D205 D212
        self.scalable_model = ScalableModelFactory().create(
            name, data=data, sizes=sizes, **parameters
        )
        super().__init__(self.scalable_model.name)
        self._initialize_grammars(data)
        self.io.input_grammar.defaults = self.scalable_model.default_input_data
        self.add_differentiated_inputs(self.io.input_grammar)
        self.add_differentiated_outputs(self.io.output_grammar)

    def _initialize_grammars(self, data: IODataset) -> None:
        """Initialize input and output grammars from data names.

        Args:
            data: The training dataset.
        """
        self.io.input_grammar.update_from_names(
            data.get_variable_names(data.INPUT_GROUP)
        )
        self.io.output_grammar.update_from_names(
            data.get_variable_names(data.OUTPUT_GROUP)
        )

    def _run(self, input_data: StrKeyMapping) -> StrKeyMapping | None:
        return self.scalable_model.scalable_function(input_data)

    def _compute_jacobian(
        self,
        input_names: Iterable[str] = (),
        output_names: Iterable[str] = (),
    ) -> None:
        """Compute the Jacobian of outputs wrt inputs and store the values.

        Args:
            input_names: The name of the input variables.
            output_names: The names of the output functions.
        """
        self._init_jacobian(
            input_names, output_names, Discipline.InitJacobianType.EMPTY
        )
        jac = self.scalable_model.scalable_derivatives(self.io.data)
        input_names = self.scalable_model.input_names
        jac = {
            fname: split_array_to_dict_of_arrays(
                jac[fname], self.scalable_model.sizes, input_names
            )
            for fname in self.io.output_grammar
        }
        self.jac = jac
