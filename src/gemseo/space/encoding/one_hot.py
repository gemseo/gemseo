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

"""The one-hot encoding of the discrete and catalog variables of a design space."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import ClassVar

from numpy import arange
from numpy import asarray
from numpy import float64
from numpy import int64
from numpy import isnan
from numpy import minimum
from numpy import ones
from numpy import result_type
from numpy import unique
from numpy import zeros
from scipy.sparse import csr_array

from gemseo.core.function.array_function import ArrayFunction
from gemseo.core.function.linear_function import LinearFunction
from gemseo.enum import VariableType
from gemseo.space.design import DesignSpace
from gemseo.space.encoding.base import BaseSpaceEncoding
from gemseo.space.variable.factory import deterministic_variable_factory

if TYPE_CHECKING:
    from collections.abc import Collection
    from collections.abc import Iterable
    from collections.abc import Mapping

    from numpy.typing import DTypeLike

    from gemseo.util.typing import IntegerArray
    from gemseo.util.typing import NumberArray
    from gemseo.util.typing import RealArray


@dataclass(frozen=True, eq=False)
class _EncodedVariable:
    """A scalar variable encoded by a one-hot vector."""

    index: int
    """The index of the variable in the original space."""

    weights_name: str
    """The name of the weights of the variable in the encoded space."""

    weight_slice: slice
    """The slice of the weights of the variable."""

    choices: RealArray
    """The choices of the variable, sorted in ascending order.

    For a catalog variable, the positions of its alternatives.
    """

    @property
    def n_weights(self) -> int:
        """The number of weights, namely of choices or alternatives."""
        return len(self.choices)

    def encode(self, positions: IntegerArray) -> IntegerArray:
        """Encode choices of the variable by one-hot vectors.

        Args:
            positions: The positions of the choices, with any shape.

        Returns:
            The vertices of the hypercube of the weights that stand for the choices,
            with the shape of the positions followed by the number of weights.
        """
        return (arange(self.n_weights) == positions[..., None]).astype(int64)

    def decode(self, weights: NumberArray) -> RealArray:
        """Decode weights to the values of the variable.

        Args:
            weights: The weights of the variable, in the encoded space,
                with any leading shape followed by the number of weights.

        Returns:
            The values, in the original coordinates,
            of the choices of largest weight,
            the first one in case of a tie.
        """
        return self.choices[weights.argmax(axis=-1)]


class SpaceOneHotEncoding(BaseSpaceEncoding):
    r"""The one-hot encoding of the discrete and catalog variables of a design space.

    In the encoded space,
    every encoded variable is replaced by a variable
    named after it with the suffix `_weights`,
    e.g. `x_weights` for `x`,
    with one weight per choice of a discrete variable
    or per alternative of a catalog variable.
    The weights are integers, $0$ or $1$.
    The other variables are carried over as they are,
    the categorical ones included.
    The current value of an encoded variable is the vertex of the hypercube
    that stands for its current value,
    e.g. `[0, 1, 0]` for the second of three choices.

    A value of the original space is encoded
    only when the value of each encoded variable is one of its choices,
    the choices of a catalog variable being the positions of its alternatives.

    A value of the encoded space is decoded as the choice of largest weight,
    so that the decoded value is always a choice,
    even from weights that are all zero or several of which are one;
    a tie goes to the first of the tied choices,
    namely the lowest one,
    so that the order of the choices matters in that case only;
    weights with a NaN component are rejected.
    The components carried over are returned as they are, unchecked.

    [create_constraints][gemseo.space.encoding.one_hot.SpaceOneHotEncoding.create_constraints]
    returns, for each encoded variable `x` with weights $w$,
    the constraint $\sum_i w_i = 1$, named `x_one_hot`,
    which the integer weights satisfy only at the one-hot codes.

    The name of the weights of an encoded variable
    cannot be the name of a variable carried over.
    """  # noqa: E501

    _weights_suffix: ClassVar[str] = "_weights"
    """The suffix naming the weights of an encoded variable in the encoded space.

    The variable `x_weights` holds the weights of the encoded variable `x`.
    """

    __name_to_encoded_variable: dict[str, _EncodedVariable]
    """The encoded variables bound to their names, in the order of the space."""

    __carried_indices: IntegerArray
    """The indices, in the original space, of the components carried over."""

    __carried_encoded_indices: IntegerArray
    """The indices, in the encoded space, of the components carried over."""

    __weight_indices: IntegerArray
    """The indices, in the encoded space, of the weights."""

    __encoded_dimension: int
    """The dimension of the encoded space."""

    def __init__(
        self,
        space: DesignSpace,
        encode_discrete: bool = True,
        encode_catalog: bool = True,
    ) -> None:
        """
        Args:
            encode_discrete: Whether the discrete variables are encoded.
            encode_catalog: Whether the catalog variables are encoded.
        """  # noqa: D205, D212
        super().__init__(space)
        name_to_indices = space.variables.name_to_indices
        name_to_encoded_variable = {}
        carried_names = set()
        carried_indices = []
        carried_encoded_indices = []
        weight_indices = []
        encoded_start = 0
        for name, variable in space.variables.items():
            indices = name_to_indices[name]
            if variable.type == VariableType.DISCRETE and encode_discrete:
                choices = asarray(variable.choices, dtype=float64)
            elif variable.type == VariableType.CATALOG and encode_catalog:
                # The coordinate of a catalog variable is the position of an
                # alternative, so these positions are its choices.
                choices = arange(len(variable.labels), dtype=float64)
            else:
                size = len(indices)
                carried_names.add(name)
                carried_indices.extend(indices)
                carried_encoded_indices.extend(
                    range(encoded_start, encoded_start + size)
                )
                encoded_start += size
                continue

            n_weights = len(choices)
            weight_indices.extend(range(encoded_start, encoded_start + n_weights))
            name_to_encoded_variable[name] = _EncodedVariable(
                indices[0],
                f"{name}{self._weights_suffix}",
                slice(encoded_start, encoded_start + n_weights),
                choices,
            )
            encoded_start += n_weights

        self.__check_weights_names(name_to_encoded_variable.values(), carried_names)
        self.__name_to_encoded_variable = name_to_encoded_variable
        self.__weight_indices = asarray(weight_indices, dtype=int64)
        self.__carried_indices = asarray(carried_indices, dtype=int64)
        self.__carried_encoded_indices = asarray(carried_encoded_indices, dtype=int64)
        self.__encoded_dimension = encoded_start
        # The encoded space is built now,
        # from the state of the space the maps above are built from.
        _ = self.encoded_space

    @classmethod
    def __check_weights_names(
        cls,
        encoded_variables: Iterable[_EncodedVariable],
        carried_names: Collection[str],
    ) -> None:
        """Check that the names of the weights are not names of carried variables.

        Args:
            encoded_variables: The encoded variables.
            carried_names: The names of the variables carried over.

        Raises:
            ValueError: When the name of the weights of an encoded variable
                is the name of a variable carried over.
        """
        taken_names = [
            encoded_variable.weights_name
            for encoded_variable in encoded_variables
            if encoded_variable.weights_name in carried_names
        ]
        if taken_names:
            msg = (
                f"{cls.__name__} names the weights of an encoded variable x "
                "as x_weights, a name that no other variable of the design space "
                "may take; the design space has variables named "
                f"{', '.join(taken_names)}."
            )
            raise ValueError(msg)

    @property
    def encoded_names(self) -> frozenset[str]:
        """The names of the variables this encoding replaces by weights."""
        return frozenset(self.__name_to_encoded_variable)

    @staticmethod
    def __scatter_components(
        array: NumberArray,
        source_indices: IntegerArray,
        destination_indices: IntegerArray,
        dimension: int,
        dtype: DTypeLike,
    ) -> NumberArray:
        """Copy components of an array to the components of a larger zero array.

        Args:
            array: The array to read, with any leading shape.
            source_indices: The indices of the components to read.
            destination_indices: The indices where to write them.
            dimension: The size of the last axis of the output.
            dtype: The data type of the output.

        Returns:
            The array with the leading shape of the input
            and a last axis of the given size,
            zero except at the destination indices.
        """
        output = zeros((*array.shape[:-1], dimension), dtype=dtype)
        output[..., destination_indices] = array[..., source_indices]
        return output

    def _create_encoded_space(self) -> DesignSpace:  # noqa: D102
        space = self._original_space
        name_to_encoded_variable = self.__name_to_encoded_variable
        if not name_to_encoded_variable:
            # Nothing to encode: the encoded space is the original one.
            return space

        # The current values are encoded as `encode` encodes a value.
        name_to_value = space._current_value
        name_to_weights = self.__encode({
            name: asarray(value)[0]
            for name in name_to_encoded_variable
            if (value := name_to_value.get(name)) is not None
        })
        # The encoded space is a plain design space,
        # since the constructor of a subclass may take other arguments
        # and register variables of its own.
        # The normalization of the integer variables is a state of the space,
        # not of a variable, so it is copied.
        encoded_space = DesignSpace(space.name)
        encoded_space.enable_integer_variables_normalization = (
            space.enable_integer_variables_normalization
        )
        # The variables are registered one by one,
        # since the names and the sizes of the encoded ones change,
        # which rules out `DesignSpace._copy_with_variables`.
        for name in space.variables:
            encoded_variable = name_to_encoded_variable.get(name)
            if encoded_variable is None:
                encoded_space._add_variable_from(space, name)
                continue

            encoded_space._register_variable(
                encoded_variable.weights_name,
                deterministic_variable_factory.create(
                    VariableType.INTEGER,
                    size=encoded_variable.n_weights,
                    lower_bound=0,
                    upper_bound=1,
                ),
                name_to_weights.get(name),
            )

        return encoded_space

    def encode(self, value: NumberArray) -> NumberArray:  # noqa: D102
        if not self.__name_to_encoded_variable:
            return value

        encoded_value = self.__scatter_components(
            value,
            self.__carried_indices,
            self.__carried_encoded_indices,
            self.__encoded_dimension,
            result_type(value.dtype, float64),
        )
        name_to_encoded_variable = self.__name_to_encoded_variable
        name_to_weights = self.__encode({
            name: value[..., encoded_variable.index]
            for name, encoded_variable in name_to_encoded_variable.items()
        })
        for name, weights in name_to_weights.items():
            encoded_value[..., name_to_encoded_variable[name].weight_slice] = weights

        return encoded_value

    def __encode(
        self, name_to_component: Mapping[str, NumberArray]
    ) -> dict[str, IntegerArray]:
        """Encode values of encoded variables by one-hot vectors.

        Args:
            name_to_component: The values of encoded variables,
                in the original coordinates, with any shape,
                bound to the names of the variables.

        Returns:
            The vertices of the hypercube of the weights that stand for the values,
            with their shape followed by the number of weights,
            bound to the names of the variables.

        Raises:
            ValueError: When a value is not a choice.
        """
        name_to_weights = {}
        errors = []
        for name, component in name_to_component.items():
            encoded_variable = self.__name_to_encoded_variable[name]
            choices = encoded_variable.choices
            # A value that is a choice is found by `searchsorted`;
            # any other value, NaN included, is caught by the check below.
            positions = minimum(
                choices.searchsorted(component), encoded_variable.n_weights - 1
            )
            outside = component != choices[positions]
            if outside.any():
                errors.append(
                    f"{name}: {unique(component[outside]).tolist()} not in "
                    f"{choices.tolist()}"
                )
            else:
                name_to_weights[name] = encoded_variable.encode(positions)

        if errors:
            msg = (
                f"{type(self).__name__} cannot encode values that are not choices, "
                "the choices of a catalog variable being the positions "
                f"of its alternatives: {'; '.join(errors)}."
            )
            raise ValueError(msg)

        return name_to_weights

    def decode(self, value: NumberArray) -> NumberArray:  # noqa: D102
        # The carried components are not checked,
        # so that the integer ones can be handed back unrounded.
        if not self.__name_to_encoded_variable:
            return value

        original_value = self.__scatter_components(
            value,
            self.__carried_encoded_indices,
            self.__carried_indices,
            self._original_space.dimension,
            result_type(value.dtype, float64),
        )
        name_to_encoded_variable = self.__name_to_encoded_variable
        if isnan(value[..., self.__weight_indices]).any():
            errors = []
            for name, encoded_variable in name_to_encoded_variable.items():
                weights = value[..., encoded_variable.weight_slice]
                if isnan(weights).any():
                    errors.append(f"{name}: {weights.tolist()}")

            msg = (
                f"{type(self).__name__} cannot decode weights with a NaN component, "
                f"weights with no NaN component being expected: {'; '.join(errors)}."
            )
            raise ValueError(msg)

        for encoded_variable in name_to_encoded_variable.values():
            original_value[..., encoded_variable.index] = encoded_variable.decode(
                value[..., encoded_variable.weight_slice]
            )

        return original_value

    def create_constraints(self) -> tuple[LinearFunction, ...]:  # noqa: D102
        dimension = self.__encoded_dimension
        # The constraints are printed with the names of the encoded components,
        # e.g. `d_weights[0]`, rather than generic ones.
        input_names = self.encoded_space.get_indexed_variable_names()
        constraints: list[LinearFunction] = []
        for name, encoded_variable in self.__name_to_encoded_variable.items():
            weight_slice = encoded_variable.weight_slice
            weight_indices = range(weight_slice.start, weight_slice.stop)
            n_weights = len(weight_indices)
            constraints.append(
                LinearFunction(
                    csr_array(
                        (ones(n_weights), (zeros(n_weights, int64), weight_indices)),
                        shape=(1, dimension),
                    ),
                    f"{name}_one_hot",
                    f_type=ArrayFunction.FunctionType.EQ,
                    input_names=input_names,
                    value_at_zero=-1.0,
                )
            )

        return tuple(constraints)
