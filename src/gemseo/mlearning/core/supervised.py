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
#                         documentation
#        :author: Matthias De Lozzo
#        :author: Syver Doving Agdestein
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
r"""This module contains the base class for the supervised machine learning algorithms.

Supervised machine learning is a task of learning relationships
between input and output variables based on an input-output dataset.
One usually distinguishes between two types of supervised machine learning algorithms,
based on the nature of the outputs.
For a continuous output variable,
a *regression* is performed,
while for a discrete output variable,
a *classification* is performed.

Given a set of input variables
:math:`x \\in \\mathbb{R}^{n_{\\text{samples}}\\times n_{\\text{inputs}}}` and
a set of output variables
:math:`y\\in \\mathbb{K}^{n_{\\text{samples}}\\times n_{\\text{outputs}}}`,
where :math:`n_{\\text{inputs}}` is the dimension of the input variable,
:math:`n_{\\text{outputs}}` is the dimension of the output variable,
:math:`n_{\\text{samples}}` is the number of training samples and
:math:`\\mathbb{K}` is either :math:`\\mathbb{R}` or :math:`\\mathbb{N}`
for regression and classification tasks respectively,
a supervised learning algorithm seeks to find a function
:math:`f: \\mathbb{R}^{n_{\\text{inputs}}} \\to
\\mathbb{K}^{n_{\\text{outputs}}}` such that :math:`y=f(x)`.

In addition,
we often want to impose some additional constraints on the function :math:`f`,
mainly to ensure that it has a generalization capacity beyond the training data,
i.e. it is able to correctly predict output values of new input values.
This is called regularization.
Assuming :math:`f` is parametrized by a set of parameters :math:`\\theta`,
and denoting :math:`f_\\theta` the parametrized function,
one typically seeks to minimize a function of the form

.. math::

    \\mu(y, f_\\theta(x)) + \\Omega(\\theta),

where :math:`\\mu` is a distance-like measure,
typically a mean squared error,
a cross entropy in the case of a regression,
or a probability to be maximized in the case of a classification,
and :math:`\\Omega` is a regularization term that limits the parameters
from over-fitting, typically some norm of its argument.

The :mod:`~gemseo.mlearning.core.supervised` module implements this concept
through the :class:`.MLSupervisedAlgo` class based on an :class:`.IODataset`.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Iterable
from collections.abc import Mapping
from collections.abc import Sequence
from types import MappingProxyType
from typing import TYPE_CHECKING
from typing import ClassVar
from typing import NoReturn
from typing import Union

from numpy import array
from numpy import hstack
from numpy import ndarray

from gemseo.datasets.io_dataset import IODataset
from gemseo.mlearning.core.ml_algo import DataType
from gemseo.mlearning.core.ml_algo import DefaultTransformerType
from gemseo.mlearning.core.ml_algo import MLAlgo
from gemseo.mlearning.core.ml_algo import MLAlgoParameterType
from gemseo.mlearning.core.ml_algo import SavedObjectType as MLAlgoSaveObjectType
from gemseo.mlearning.core.ml_algo import TransformerType
from gemseo.mlearning.data_formatters.supervised_data_formatters import (
    SupervisedDataFormatters,
)
from gemseo.mlearning.transformers.dimension_reduction.dimension_reduction import (
    DimensionReduction,
)
from gemseo.mlearning.transformers.scaler.min_max_scaler import MinMaxScaler
from gemseo.utils.data_conversion import split_array_to_dict_of_arrays

if TYPE_CHECKING:
    from gemseo.mlearning.transformers.transformer import Transformer

SavedObjectType = Union[MLAlgoSaveObjectType, Sequence[str], dict[str, ndarray]]


class MLSupervisedAlgo(MLAlgo):
    """Supervised machine learning algorithm.

    Inheriting classes shall overload the :meth:`!MLSupervisedAlgo._fit` and
    :meth:`!MLSupervisedAlgo._predict` methods.
    """

    input_names: list[str]
    """The names of the input variables."""

    input_space_center: dict[str, ndarray]
    """The center of the input space."""

    output_names: list[str]
    """The names of the output variables."""

    SHORT_ALGO_NAME: ClassVar[str] = "MLSupervisedAlgo"
    DEFAULT_TRANSFORMER: DefaultTransformerType = MappingProxyType({
        IODataset.INPUT_GROUP: MinMaxScaler()
    })

    DataFormatters = SupervisedDataFormatters

    def __init__(
        self,
        data: IODataset,
        transformer: TransformerType = MLAlgo.IDENTITY,
        input_names: Iterable[str] | None = None,
        output_names: Iterable[str] | None = None,
        **parameters: MLAlgoParameterType,
    ) -> None:
        """
        Args:
            input_names: The names of the input variables.
                If ``None``, consider all the input variables of the learning dataset.
            output_names: The names of the output variables.
                If ``None``, consider all the output variables of the learning dataset.
        """  # noqa: D205 D212
        super().__init__(data, transformer=transformer, **parameters)
        self.input_names = input_names or data.get_variable_names(data.INPUT_GROUP)
        self.output_names = output_names or data.get_variable_names(data.OUTPUT_GROUP)
        self.__groups_to_names = {
            data.INPUT_GROUP: self.input_names,
            data.OUTPUT_GROUP: self.output_names,
        }
        self.input_space_center = array([])
        self.__input_dimension = 0
        self.__output_dimension = 0
        self.__reduced_dimensions = (0, 0)
        self._transformed_variable_sizes = {}
        self._transformed_input_sizes = {}
        self._transformed_output_sizes = {}
        self._input_variables_to_transform = [
            key for key in self.transformer if key in self.input_names
        ]
        self._transform_input_group = self.learning_set.INPUT_GROUP in self.transformer
        self._output_variables_to_transform = [
            key for key in self.transformer if key in self.output_names
        ]
        self._transform_output_group = (
            self.learning_set.OUTPUT_GROUP in self.transformer
        )

    @property
    def _reduced_dimensions(self) -> tuple[int, int]:
        """The input and output reduced dimensions."""
        if self.__reduced_dimensions == (0, 0):
            self.__reduced_dimensions = self.__compute_reduced_dimensions()

        return self.__reduced_dimensions

    @property
    def input_dimension(self) -> int:
        """The input space dimension."""
        if not self.__input_dimension and self.learning_set is not None:
            variable_to_size = self.learning_set.variable_names_to_n_components
            self.__input_dimension = sum(
                variable_to_size[name] for name in self.input_names
            )

        return self.__input_dimension

    @property
    def output_dimension(self) -> int:
        """The output space dimension."""
        if not self.__output_dimension and self.learning_set is not None:
            variable_to_size = self.learning_set.variable_names_to_n_components
            self.__output_dimension = sum(
                variable_to_size[name] for name in self.output_names
            )

        return self.__output_dimension

    def _transform_data(self, data: ndarray, name: str, inverse: bool) -> ndarray:
        """
        Args:
            data: The original data array.
            name: The name of the variable or group to transform.
            inverse: Whether to use the inverse transformation.

        Returns:
            The transformed data.
        """  # noqa: D205 D212
        if inverse:
            function = self.transformer[name].inverse_transform
        else:
            function = self.transformer[name].transform
        return function(data)

    def _transform_data_from_variable_names(
        self,
        data: ndarray,
        names: Iterable[str],
        names_to_sizes: Mapping[str, int],
        names_to_transform: Sequence[str],
        inverse: bool,
    ) -> ndarray:
        """Transform a data array.

        Args:
            data: The original data array.
            names: The variables representing the columns of the array.
            names_to_sizes: The sizes of the variables.
            names_to_transform: The names of the variables to transform.
            inverse: Whether to use the inverse transformation.

        Returns:
            The transformed data array.
        """
        data = split_array_to_dict_of_arrays(data, names_to_sizes, names)
        transformed_data = []
        for name in names:
            if name in names_to_transform:
                transformed_data.append(self._transform_data(data[name], name, inverse))
            else:
                transformed_data.append(data[name])

        return hstack(transformed_data)

    def _learn(
        self,
        indices: Sequence[int] | None,
        fit_transformers: bool,
    ) -> None:
        dataset = self.learning_set
        if indices is None:
            indices = Ellipsis

        input_data = dataset.get_view(
            group_names=dataset.INPUT_GROUP, variable_names=self.input_names
        ).to_numpy()[indices]
        output_data = dataset.get_view(
            group_names=dataset.OUTPUT_GROUP, variable_names=self.output_names
        ).to_numpy()[indices]
        self.input_space_center = split_array_to_dict_of_arrays(
            input_data.mean(0),
            self.learning_set.variable_names_to_n_components,
            self.input_names,
        )

        if fit_transformers:
            if self._transform_input_group or self._input_variables_to_transform:
                input_data = self.__fit_transformer(
                    indices,
                    True,
                    self._input_variables_to_transform,
                )

            if self._transform_output_group or self._output_variables_to_transform:
                output_data = self.__fit_transformer(
                    indices,
                    False,
                    self._output_variables_to_transform,
                )

        self._fit(input_data, output_data)
        self.__compute_transformed_variable_sizes()

    def __fit_transformer(
        self,
        indices: Ellipsis | Sequence[int],
        input_group: bool,
        names: Sequence[str],
    ) -> ndarray:
        """Fit a transformer.

        Args:
            indices: The indices of the learning samples.
            input_group: Whether to consider the input group.
                Otherwise, consider the output one.
            names: The variable names having dedicated transformers.

        Returns:
            The transformed data.
        """
        if names:
            return self.__fit_transformer_from_names(input_group, names, indices)
        return self.__fit_transformer_from_group(input_group, indices)

    def __fit_transformer_from_names(
        self, input_group: bool, names: Iterable[str], indices: Ellipsis | Sequence[int]
    ) -> ndarray:
        """Fit a transformer from variable names.

        Args:
            input_group: Whether to consider the input group.
                Otherwise, consider the output one.
            names: The variable names having dedicated transformers.
            indices: The indices of the learning samples.

        Returns:
            The transformed data.

        Raises:
            NotImplementedError: When an output transformer needs to be fitted
                from both input and output data.
        """
        dataset = self.learning_set
        transformed_data = []
        for name in self.__groups_to_names[self.__get_group_name(input_group)]:
            if name not in names:
                transformed_data.append(
                    dataset.get_view(variable_names=name).to_numpy()
                )
                continue

            transformed_data.append(
                self.__fit_and_transform_data(
                    [name], self.transformer[name], indices, input_group
                )
            )

        return hstack(transformed_data)

    def __get_group_name(self, input_group: bool) -> str:
        """Return the name of the group.

        Args:
            input_group: Whether to consider the input group.
                Otherwise, consider the output one.

        Returns:
            The name of the group.
        """
        if input_group:
            return self.learning_set.INPUT_GROUP
        return self.learning_set.OUTPUT_GROUP

    def __fit_transformer_from_group(
        self, input_group: bool, indices: Ellipsis | Sequence[int]
    ) -> ndarray:
        """Fit a transformer from a group name.

        Args:
            input_group: Whether to consider the input group.
                Otherwise, consider the output one.
            indices: The indices of the learning samples.

        Returns:
            The transformed data.
        """
        group = self.__get_group_name(input_group)
        return self.__fit_and_transform_data(
            self.__groups_to_names[group], self.transformer[group], indices, input_group
        )

    def __fit_and_transform_data(
        self,
        names: Iterable[str],
        transformer: Transformer,
        indices: Ellipsis | Sequence[int],
        input_group: bool,
    ) -> ndarray:
        """Fit and transform data.

        Args:
            names: The names of the variables to be transformed.
            transformer: The transformer to be applied.
            indices: The indices of the learning samples.
            input_group: Whether to consider the input group.
                Otherwise, consider the output one.

        Returns:
            The transformed data.

        Raises:
            NotImplementedError: When the output transformer needs to be fitted
                from both input and output data.
        """
        data = self.learning_set.get_view(variable_names=(names)).to_numpy()[indices]
        if not transformer.CROSSED:
            return transformer.fit_transform(data)

        if not input_group:
            raise NotImplementedError(
                "The transformer {} cannot be applied to the outputs "
                "to build a supervised machine learning algorithm.".format(
                    transformer.__class__.__name__
                )
            )

        return transformer.fit_transform(
            data,
            self.learning_set.get_view(variable_names=self.output_names).to_numpy()[
                indices
            ],
        )

    @abstractmethod
    def _fit(
        self,
        input_data: ndarray,
        output_data: ndarray,
    ) -> NoReturn:
        """Fit input-output relationship from the learning data.

        Args:
            input_data: The input data with the shape (n_samples, n_inputs).
            output_data: The output data with shape (n_samples, n_outputs).
        """

    @DataFormatters.format_input_output
    def predict(
        self,
        input_data: DataType,
    ) -> DataType:
        """Predict output data from input data.

        The user can specify these input data either as a NumPy array,
        e.g. ``array([1., 2., 3.])``
        or as a dictionary,
        e.g.  ``{'a': array([1.]), 'b': array([2., 3.])}``.

        If the numpy arrays are of dimension 2,
        their i-th rows represent the input data of the i-th sample;
        while if the numpy arrays are of dimension 1,
        there is a single sample.

        The type of the output data and the dimension of the output arrays
        will be consistent
        with the type of the input data and the size of the input arrays.

        Args:
            input_data: The input data.

        Returns:
            The predicted output data.
        """
        return self._predict(input_data)

    @abstractmethod
    def _predict(
        self,
        input_data: ndarray,
    ) -> NoReturn:
        """Predict output data from input data.

        Args:
            input_data: The input data with shape (n_samples, n_inputs).

        Returns:
            output_data: The output data with shape (n_samples, n_outputs).
        """

    def __compute_reduced_dimensions(self) -> tuple[int, int]:
        """Return the reduced input and output dimensions after transformations.

        Returns:
            The reduced input and output dimensions.
        """
        input_dimension = 0
        output_dimension = 0
        input_names = [*self.input_names, IODataset.INPUT_GROUP]
        output_names = [*self.output_names, IODataset.OUTPUT_GROUP]

        for key in self.transformer:
            transformer = self.transformer.get(key)
            if key in input_names:
                if isinstance(transformer, DimensionReduction):
                    input_dimension += transformer.n_components
                else:
                    input_dimension += (
                        self.learning_set.variable_names_to_n_components.get(
                            key, self.input_dimension
                        )
                    )

            if key in output_names:
                if isinstance(transformer, DimensionReduction):
                    output_dimension += transformer.n_components
                else:
                    output_dimension += (
                        self.learning_set.variable_names_to_n_components.get(
                            key, self.output_dimension
                        )
                    )

        input_dimension = input_dimension or self.input_dimension
        output_dimension = output_dimension or self.output_dimension
        return input_dimension, output_dimension

    def __compute_transformed_variable_sizes(self) -> None:
        """Compute the sizes of the transformed variables."""
        if self._transformed_variable_sizes:
            return

        for name in self.input_names + self.output_names:
            transformer = self.transformer.get(name)
            if transformer is None or not isinstance(transformer, DimensionReduction):
                self._transformed_variable_sizes[
                    name
                ] = self.learning_set.variable_names_to_n_components[name]
            else:
                self._transformed_variable_sizes[name] = transformer.n_components

        self._transformed_input_sizes = {
            name: size
            for name, size in self._transformed_variable_sizes.items()
            if name in self.input_names
        }
        self._transformed_output_sizes = {
            name: size
            for name, size in self._transformed_variable_sizes.items()
            if name in self.output_names
        }

    @property
    def input_data(self) -> ndarray:
        """The input data matrix."""
        return self.learning_set.get_view(
            group_names=self.learning_set.INPUT_GROUP,
            variable_names=self.input_names,
            indices=self._learning_samples_indices,
        ).to_numpy()

    @property
    def output_data(self) -> ndarray:
        """The output data matrix."""
        return self.learning_set.get_view(
            group_names=self.learning_set.OUTPUT_GROUP,
            variable_names=self.output_names,
            indices=self._learning_samples_indices,
        ).to_numpy()

    def _get_objects_to_save(self) -> dict[str, SavedObjectType]:
        objects = super()._get_objects_to_save()
        objects["input_names"] = self.input_names
        objects["output_names"] = self.output_names
        objects["input_space_center"] = self.input_space_center
        objects["_transformed_input_sizes"] = self._transformed_input_sizes
        objects["_transformed_output_sizes"] = self._transformed_output_sizes
        objects["_transform_input_group"] = self._transform_input_group
        objects["_transform_output_group"] = self._transform_output_group
        objects["_input_variables_to_transform"] = self._input_variables_to_transform
        objects["_output_variables_to_transform"] = self._output_variables_to_transform
        return objects
