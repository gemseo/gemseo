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
:math:`x \in \mathbb{R}^{n_{\text{samples}}\times n_{\text{inputs}}}` and
a set of output variables
:math:`y \in \mathbb{K}^{n_{\text{samples}}\times n_{\text{outputs}}}`,
where :math:`n_{\text{inputs}}` is the dimension of the input variable,
:math:`n_{\text{outputs}}` is the dimension of the output variable,
:math:`n_{\text{samples}}` is the number of training samples and
:math:`\mathbb{K}` is either :math:`\mathbb{R}` or :math:`\mathbb{N}`
for regression and classification tasks respectively,
a supervised learning algorithm seeks to find a function
:math:`f: \mathbb{R}^{n_{\text{inputs}}} \to
\mathbb{K}^{n_{\text{outputs}}}` such that :math:`y=f(x)`.

In addition,
we often want to impose some additional constraints on the function :math:`f`,
mainly to ensure that it has a generalization capacity beyond the training data,
i.e. it is able to correctly predict output values of new input values.
This is called regularization.
Assuming :math:`f` is parametrized by a set of parameters :math:`\theta`,
and denoting :math:`f_\theta` the parametrized function,
one typically seeks to minimize a function of the form

.. math::

    \mu(y, f_\theta(x)) + \Omega(\theta),

where :math:`\mu` is a distance-like measure,
typically a mean squared error,
a cross entropy in the case of a regression,
or a probability to be maximized in the case of a classification,
and :math:`\Omega` is a regularization term that limits the parameters
from over-fitting, typically some norm of its argument.

The :mod:`~gemseo.mlearning.core.supervised` module implements this concept
through the :class:`.BaseMLSupervisedAlgo` class based on an :class:`.IODataset`.
"""

from __future__ import annotations

from abc import abstractmethod
from collections.abc import Iterable
from collections.abc import Mapping
from collections.abc import Sequence
from types import MappingProxyType
from typing import TYPE_CHECKING
from typing import ClassVar
from typing import Union

from numpy import concatenate
from numpy import hstack
from numpy import ndarray

from gemseo.algos.design_space import DesignSpace
from gemseo.datasets.io_dataset import IODataset
from gemseo.mlearning.core.algos.ml_algo import BaseMLAlgo
from gemseo.mlearning.core.algos.ml_algo import DataType
from gemseo.mlearning.core.algos.ml_algo import DefaultTransformerType
from gemseo.mlearning.core.algos.ml_algo import SavedObjectType as MLAlgoSaveObjectType
from gemseo.mlearning.core.algos.supervised_settings import BaseMLSupervisedAlgoSettings
from gemseo.mlearning.data_formatters.supervised_data_formatters import (
    SupervisedDataFormatters,
)
from gemseo.mlearning.transformers.dimension_reduction.base_dimension_reduction import (
    BaseDimensionReduction,
)
from gemseo.mlearning.transformers.scaler.min_max_scaler import MinMaxScaler
from gemseo.utils.data_conversion import split_array_to_dict_of_arrays

if TYPE_CHECKING:
    from gemseo.mlearning.transformers.base_transformer import BaseTransformer
    from gemseo.typing import RealArray

SavedObjectType = Union[MLAlgoSaveObjectType, Sequence[str], dict[str, ndarray]]


class BaseMLSupervisedAlgo(BaseMLAlgo):
    """Supervised machine learning algorithm.

    Inheriting classes shall overload the :meth:`!BaseMLSupervisedAlgo._fit` and
    :meth:`!BaseMLSupervisedAlgo._predict` methods.
    """

    input_names: list[str]
    """The names of the input variables."""

    input_space_center: dict[str, RealArray]
    """The center of the input space."""

    output_names: list[str]
    """The names of the output variables."""

    validity_domain: DesignSpace
    """The validity domain.

    This is the hypercube defined by the lower and upper bounds of the input variables
    computed from the training dataset.
    """

    __input_dimension: int
    """The dimension of the input space."""

    __groups_to_names: dict[str, str]
    """The variable names associated with group names."""

    __output_dimension: int
    """The dimension of the output space."""

    __reduced_input_dimension: int
    """The dimension of the transformed input space."""

    __reduced_output_dimension: int
    """The dimension of the transformed output space."""

    _input_variables_to_transform: list[str]
    """The names of the input variables to transform."""

    _output_variables_to_transform: list[str]
    """The names of the output variables to transform."""

    _transform_input_group: bool
    """Whether to transform the variables of the input group."""

    _transform_output_group: bool
    """Whether to transform the variables of the output group."""

    _transformed_input_sizes: dict[str, int]
    """The sizes of the transformed input variables."""

    _transformed_output_sizes: dict[str, int]
    """The sizes of the transformed output variables."""

    _transformed_variable_sizes: dict[str, int]
    """The sizes of the variables in the transformed spaces."""

    SHORT_ALGO_NAME: ClassVar[str] = "BaseMLSupervisedAlgo"
    DEFAULT_TRANSFORMER: DefaultTransformerType = MappingProxyType({
        IODataset.INPUT_GROUP: MinMaxScaler()
    })

    DataFormatters = SupervisedDataFormatters

    Settings: ClassVar[type[BaseMLSupervisedAlgoSettings]] = (
        BaseMLSupervisedAlgoSettings
    )

    def _post_init(self):
        super()._post_init()
        data = self.learning_set
        self.input_names = list(self._settings.input_names) or data.get_variable_names(
            data.INPUT_GROUP
        )
        self.output_names = list(
            self._settings.output_names
        ) or data.get_variable_names(data.OUTPUT_GROUP)
        self.__groups_to_names = {
            data.INPUT_GROUP: self.input_names,
            data.OUTPUT_GROUP: self.output_names,
        }
        self.input_space_center = {}
        self.__input_dimension = 0
        self.__output_dimension = 0
        self.__reduced_input_dimension = 0
        self.__reduced_output_dimension = 0
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
        self.validity_domain = DesignSpace()
        for input_name in self.input_names:
            data = self.learning_set.get_view(
                variable_names=input_name, group_names=self.learning_set.INPUT_GROUP
            ).to_numpy()
            self.validity_domain.add_variable(
                input_name,
                size=data.shape[1],
                lower_bound=data.min(axis=0),
                upper_bound=data.max(axis=0),
            )

    @property
    def _reduced_input_dimension(self) -> int:
        """The reduced input dimension."""
        if not self.__reduced_input_dimension:
            self.__set_reduced_dimensions()

        return self.__reduced_input_dimension

    @property
    def _reduced_output_dimension(self) -> int:
        """The reduced output dimension."""
        if not self.__reduced_output_dimension:
            self.__set_reduced_dimensions()

        return self.__reduced_output_dimension

    @property
    def input_dimension(self) -> int:
        """The input space dimension."""
        data = self.learning_set
        if not self.__input_dimension and data is not None:
            input_data = data.input_dataset
            self.__input_dimension = sum(
                input_data.get_view(variable_names=name).shape[1]
                for name in self.input_names
            )

        return self.__input_dimension

    @property
    def output_dimension(self) -> int:
        """The output space dimension."""
        data = self.learning_set
        if not self.__output_dimension and data is not None:
            output_data = data.output_dataset
            self.__output_dimension = sum(
                output_data.get_view(variable_names=name).shape[1]
                for name in self.output_names
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

        return concatenate(transformed_data, axis=-1)

    def _learn(
        self,
        indices: Sequence[int],
        fit_transformers: bool,
    ) -> None:
        dataset = self.learning_set
        if not indices:
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

        if self._transform_input_group or self._input_variables_to_transform:
            input_data = self.__transform_data_from_group_or_names(
                indices, True, self._input_variables_to_transform, fit_transformers
            )

        if self._transform_output_group or self._output_variables_to_transform:
            output_data = self.__transform_data_from_group_or_names(
                indices, False, self._output_variables_to_transform, fit_transformers
            )

        self._fit(input_data, output_data)
        self.__compute_transformed_variable_sizes()

    def __transform_data_from_group_or_names(
        self,
        indices: Ellipsis | Sequence[int],
        input_group: bool,
        names: Sequence[str],
        fit: bool,
    ) -> ndarray:
        """Transform data from variable names or a group name.

        Args:
            indices: The indices of the learning samples.
            input_group: Whether to consider the input group.
                Otherwise, consider the output one.
            names: The variable names having dedicated transformers.
            fit: Whether to fit the transformers before applying transformation.

        Returns:
            The transformed data.
        """
        if names:
            return self.__transform_data_from_names(input_group, names, indices, fit)

        return self.__transform_data_from_group(input_group, indices, fit)

    def __transform_data_from_names(
        self,
        input_group: bool,
        names: Iterable[str],
        indices: Ellipsis | Sequence[int],
        fit: bool,
    ) -> ndarray:
        """Transform data from variable names.

        Args:
            input_group: Whether to consider the input group.
                Otherwise, consider the output one.
            names: The variable names having dedicated transformers.
            indices: The indices of the learning samples.
            fit: Whether to fit the transformers before applying transformation.

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
                self.__transform_data(
                    [name], self.transformer[name], indices, input_group, fit
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

    def __transform_data_from_group(
        self, input_group: bool, indices: Ellipsis | Sequence[int], fit: bool
    ) -> ndarray:
        """Transform data from a group name.

        Args:
            input_group: Whether to consider the input group.
                Otherwise, consider the output one.
            indices: The indices of the learning samples.
            fit: Whether to fit the transformers before applying transformation.

        Returns:
            The transformed data.
        """
        group = self.__get_group_name(input_group)
        return self.__transform_data(
            self.__groups_to_names[group],
            self.transformer[group],
            indices,
            input_group,
            fit,
        )

    def __transform_data(
        self,
        names: Iterable[str],
        transformer: BaseTransformer,
        indices: Ellipsis | Sequence[int],
        input_group: bool,
        fit: bool,
    ) -> ndarray:
        """Transform data.

        Args:
            names: The names of the variables to be transformed.
            transformer: The transformer to be applied.
            indices: The indices of the learning samples.
            input_group: Whether to consider the input group.
                Otherwise, consider the output one.
            fit: Whether to fit the transformers before applying transformation.

        Returns:
            The transformed data.

        Raises:
            NotImplementedError: When the output transformer needs to be fitted
                from both input and output data.
        """
        data = self.learning_set.get_view(variable_names=(names)).to_numpy()[indices]
        if not transformer.CROSSED:
            if fit:
                return transformer.fit_transform(data)

            return transformer.transform(data)

        if not input_group:
            msg = (
                f"The transformer {transformer.__class__.__name__} "
                "cannot be applied to the outputs "
                "to build a supervised machine learning algorithm."
            )
            raise NotImplementedError(msg)

        if fit:
            return transformer.fit_transform(
                data,
                self.learning_set.get_view(variable_names=self.output_names).to_numpy()[
                    indices
                ],
            )

        return transformer.transform(data)

    @abstractmethod
    def _fit(
        self,
        input_data: RealArray,
        output_data: ndarray,
    ) -> None:
        """Fit input-output relationship from the learning data.

        Args:
            input_data: The input data with the shape (n_samples, n_inputs).
            output_data: The output data with shape (n_samples, n_outputs).
        """

    @DataFormatters.format_input_output()
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
        input_data: RealArray,
    ) -> ndarray:
        """Predict output data from input data.

        Args:
            input_data: The input data with shape (n_samples, n_inputs).

        Returns:
            output_data: The output data with shape (n_samples, n_outputs).
        """

    def __set_reduced_dimensions(self) -> None:
        """Set the input and output dimensions after transformations."""
        input_dimension = 0
        output_dimension = 0
        input_names = (IODataset.INPUT_GROUP, *self.input_names)
        output_names = (IODataset.OUTPUT_GROUP, *self.output_names)

        in_data = self.learning_set.input_dataset
        out_data = self.learning_set.output_dataset
        for name in input_names:
            transformer = self.transformer.get(name)
            if isinstance(transformer, BaseDimensionReduction):
                input_dimension += transformer.n_components
            elif name != IODataset.INPUT_GROUP:
                input_dimension += in_data.get_view(variable_names=name).shape[1]

            if name == IODataset.INPUT_GROUP and input_dimension:
                break

        for name in output_names:
            transformer = self.transformer.get(name)
            if isinstance(transformer, BaseDimensionReduction):
                output_dimension += transformer.n_components
            elif name != IODataset.OUTPUT_GROUP:
                output_dimension += out_data.get_view(variable_names=name).shape[1]

            if name == IODataset.OUTPUT_GROUP and output_dimension:
                break

        self.__reduced_input_dimension = input_dimension or self.input_dimension
        self.__reduced_output_dimension = output_dimension or self.output_dimension

    def __compute_transformed_variable_sizes(self) -> None:
        """Compute the sizes of the transformed variables."""
        if self._transformed_variable_sizes:
            return

        for name in self.input_names + self.output_names:
            transformer = self.transformer.get(name)
            if transformer is None or not isinstance(
                transformer, BaseDimensionReduction
            ):
                self._transformed_variable_sizes[name] = (
                    self.learning_set.variable_names_to_n_components[name]
                )
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
    def input_data(self) -> RealArray:
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
