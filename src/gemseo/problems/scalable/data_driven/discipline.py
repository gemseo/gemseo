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

The :mod:`~gemseo.problems.scalable.data_driven.discipline`
implements the concept of scalable discipline.
This is a particular discipline
built from an input-output learning dataset associated with a function
and generalizing its behavior to a new user-defined problem dimension,
that is to say new user-defined input and output dimensions.

Alone or in interaction with other objects of the same type,
a scalable discipline can be used to compare the efficiency of an algorithm
applying to disciplines with respect to the problem dimension,
e.g. optimization algorithm, surrogate model, MDO formulation, MDA, ...

The :class:`.ScalableDiscipline` class implements this concept.
It inherits from the :class:`.MDODiscipline` class
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
from typing import Iterable
from typing import Mapping

from gemseo.core.discipline import MDODiscipline
from gemseo.problems.scalable.data_driven.factory import ScalableModelFactory
from gemseo.utils.data_conversion import split_array_to_dict_of_arrays

if TYPE_CHECKING:
    from gemseo.datasets.io_dataset import IODataset


class ScalableDiscipline(MDODiscipline):
    """A scalable discipline."""

    _ATTR_NOT_TO_SERIALIZE = MDODiscipline._ATTR_NOT_TO_SERIALIZE.union([
        "scalable_model"
    ])

    def __init__(
        self,
        name: str | None,
        data: IODataset,
        sizes: Mapping[str, int] | None = None,
        **parameters: Any,
    ) -> None:
        """
        Args:
            name: The name of the class of the scalable model.
            data: The learning dataset.
            sizes: The sizes of the input and output variables.
                If ``None``, use the original sizes.
            **parameters: The parameters for the model.
        """  # noqa: D205 D212
        self.scalable_model = ScalableModelFactory().create(
            name, data=data, sizes=sizes, **parameters
        )
        super().__init__(self.scalable_model.name)
        self.initialize_grammars(data)
        self.default_inputs = self.scalable_model.default_inputs
        self.re_exec_policy = self.ReExecutionPolicy.DONE
        self.add_differentiated_inputs(self.get_input_data_names())
        self.add_differentiated_outputs(self.get_output_data_names())

    def initialize_grammars(self, data: IODataset) -> None:
        """Initialize input and output grammars from data names.

        Args:
            data: The learning dataset.
        """
        self.input_grammar.update_from_names(data.get_variable_names(data.INPUT_GROUP))
        self.output_grammar.update_from_names(
            data.get_variable_names(data.OUTPUT_GROUP)
        )

    def _run(self) -> None:
        self.local_data.update(self.scalable_model.scalable_function(self.local_data))

    def _compute_jacobian(
        self,
        inputs: Iterable[str] | None = None,
        outputs: Iterable[str] | None = None,
    ) -> None:
        """Compute the Jacobian of outputs wrt inputs and store the values.

        Args:
            inputs: The name of the input variables.
            outputs: The names of the output functions.
        """
        self._init_jacobian(inputs, outputs, MDODiscipline.InitJacobianType.EMPTY)
        jac = self.scalable_model.scalable_derivatives(self.local_data)
        input_names = self.scalable_model.input_names
        jac = {
            fname: split_array_to_dict_of_arrays(
                jac[fname], self.scalable_model.sizes, input_names
            )
            for fname in self.get_output_data_names()
        }
        self.jac = jac
