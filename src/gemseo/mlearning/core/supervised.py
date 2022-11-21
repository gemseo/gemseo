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
"""This module contains the base class for the supervised machine learning algorithms.

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
through the :class:`.MLSupervisedAlgo` class based on a :class:`.Dataset`.
"""
from __future__ import annotations

from abc import abstractmethod
from types import MappingProxyType
from typing import Callable
from typing import ClassVar
from typing import Dict
from typing import Iterable
from typing import Mapping
from typing import NoReturn
from typing import Sequence
from typing import Union

from numpy import array
from numpy import atleast_2d
from numpy import hstack
from numpy import ndarray

from gemseo.core.dataset import Dataset
from gemseo.mlearning.core.ml_algo import DataType
from gemseo.mlearning.core.ml_algo import DefaultTransformerType
from gemseo.mlearning.core.ml_algo import MLAlgo
from gemseo.mlearning.core.ml_algo import MLAlgoParameterType
from gemseo.mlearning.core.ml_algo import SavedObjectType as MLAlgoSaveObjectType
from gemseo.mlearning.core.ml_algo import TransformerType
from gemseo.mlearning.transform.dimension_reduction.dimension_reduction import (
    DimensionReduction,
)
from gemseo.mlearning.transform.scaler.min_max_scaler import MinMaxScaler
from gemseo.mlearning.transform.transformer import Transformer
from gemseo.utils.data_conversion import concatenate_dict_of_arrays_to_array
from gemseo.utils.data_conversion import split_array_to_dict_of_arrays

SavedObjectType = Union[MLAlgoSaveObjectType, Sequence[str], Dict[str, ndarray]]


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
    DEFAULT_TRANSFORMER: DefaultTransformerType = MappingProxyType(
        {Dataset.INPUT_GROUP: MinMaxScaler()}
    )

    def __init__(
        self,
        data: Dataset,
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
        """
        super().__init__(data, transformer=transformer, **parameters)
        self.input_names = input_names or data.get_names(data.INPUT_GROUP)
        self.output_names = output_names or data.get_names(data.OUTPUT_GROUP)
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
            key for key in self.transformer.keys() if key in self.input_names
        ]
        self._transform_input_group = self.learning_set.INPUT_GROUP in self.transformer
        self._output_variables_to_transform = [
            key for key in self.transformer.keys() if key in self.output_names
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
            self.__input_dimension = sum(
                self.learning_set.sizes[name] for name in self.input_names
            )

        return self.__input_dimension

    @property
    def output_dimension(self) -> int:
        """The output space dimension."""
        if not self.__output_dimension and self.learning_set is not None:
            self.__output_dimension = sum(
                self.learning_set.sizes[name] for name in self.output_names
            )

        return self.__output_dimension

    class DataFormatters(MLAlgo.DataFormatters):
        """Decorators for supervised algorithms."""

        @classmethod
        def format_dict(
            cls,
            predict: Callable[[ndarray], ndarray],
        ) -> Callable[[DataType], DataType]:
            """Make an array-based function be called with a dictionary of NumPy arrays.

            Args:
                predict: The function to be called;
                    it takes a NumPy array in input and returns a NumPy array.

            Returns:
                A function making the function 'predict' work with
                either a NumPy data array
                or a dictionary of NumPy data arrays indexed by variables names.
                The evaluation will have the same type as the input data.
            """

            def wrapper(
                self,
                input_data: DataType,
                *args,
                **kwargs,
            ) -> DataType:
                """Evaluate 'predict' with either array or dictionary-based input data.

                Firstly,
                the pre-processing stage converts the input data to a NumPy data array,
                if these data are expressed as a dictionary of NumPy data arrays.

                Then,
                the processing evaluates the function 'predict'
                from this NumPy input data array.

                Lastly,
                the post-processing transforms the output data
                to a dictionary of output NumPy data array
                if the input data were passed as a dictionary of NumPy data arrays.

                Args:
                    input_data: The input data.
                    *args: The positional arguments of the function 'predict'.
                    **kwargs: The keyword arguments of the function 'predict'.

                Returns:
                    The output data with the same type as the input one.
                """
                as_dict = isinstance(input_data, dict)
                if as_dict:
                    input_data = concatenate_dict_of_arrays_to_array(
                        input_data, self.input_names
                    )

                output_data = predict(self, input_data, *args, **kwargs)
                if as_dict:
                    return split_array_to_dict_of_arrays(
                        output_data,
                        self.learning_set.sizes,
                        self.output_names,
                    )

                return output_data

            return wrapper

        @classmethod
        def format_samples(
            cls,
            predict: Callable[[ndarray], ndarray],
        ) -> Callable[[ndarray], ndarray]:
            """Make a 2D NumPy array-based function work with 1D NumPy array.

            Args:
                predict: The function to be called;
                    it takes a 2D NumPy array in input
                    and returns a 2D NumPy array.
                    The first dimension represents the samples
                    while the second one represents the components of the variables.

            Returns:
                A function making the function 'predict' work with
                either a 1D NumPy array or a 2D NumPy array.
                The evaluation will have the same dimension as the input data.
            """

            def wrapper(
                self,
                input_data: DataType,
                *args,
                **kwargs,
            ) -> DataType:
                """Evaluate 'predict' with either a 1D or 2D NumPy data array.

                Firstly,
                the pre-processing stage converts the input data
                to a 2D NumPy data array.

                Then,
                the processing evaluates the function 'predict'
                from this 2D NumPy data array.

                Lastly,
                the post-processing converts the output data to a 1D NumPy data array
                if the dimension of the input data is equal to 1.

                Args:
                    input_data: The input data.
                    *args: The positional arguments of the function 'predict'.
                    **kwargs: The keyword arguments of the function 'predict'.

                Returns:
                    The output data with the same dimension as the input one.
                """
                single_sample = input_data.ndim == 1
                output_data = predict(self, atleast_2d(input_data), *args, **kwargs)
                if single_sample:
                    output_data = output_data[0]

                return output_data

            return wrapper

        @classmethod
        def format_transform(
            cls,
            transform_inputs: bool = True,
            transform_outputs: bool = True,
        ) -> Callable[[ndarray], ndarray]:
            """Force a function to transform its input and/or output variables.

            Args:
                transform_inputs: Whether to transform the input variables.
                transform_outputs: Whether to transform the output variables.

            Returns:
                A function evaluating a function of interest,
                after transforming its input data
                and/or before transforming its output data.
            """

            def format_transform_(
                predict: Callable[[ndarray], ndarray],
            ) -> Callable[[ndarray], ndarray]:
                """Apply transformation to inputs and inverse transformation to outputs.

                Args:
                    predict: The function of interest to be called.

                Returns:
                    A function evaluating the function 'predict',
                    after transforming its input data
                    and/or before transforming its output data.
                """

                def wrapper(
                    self,
                    input_data: ndarray,
                    *args,
                    **kwargs,
                ) -> ndarray:
                    """Evaluate 'predict' after or before data transformation.

                    Firstly,
                    the pre-processing stage transforms the input data if required.

                    Then,
                    the processing evaluates the function 'predict'.

                    Lastly,
                    the post-processing stage transforms the output data if required.

                    Args:
                        input_data: The input data.
                        *args: The positional arguments of the function.
                        **kwargs: The keyword arguments of the function.

                    Returns:
                        Either the raw output data of 'predict'
                        or a transformed version according to the requirements.
                    """
                    if transform_inputs:
                        if self._transform_input_group:
                            input_data = self._transform_data(
                                input_data, self.learning_set.INPUT_GROUP, False
                            )

                        if self._input_variables_to_transform:
                            input_data = self._transform_data_from_variable_names(
                                input_data,
                                self.input_names,
                                self.learning_set.sizes,
                                self._input_variables_to_transform,
                                False,
                            )

                    output_data = predict(self, input_data, *args, **kwargs)

                    if not transform_outputs or (
                        not self._transform_output_group
                        and not self._output_variables_to_transform
                    ):
                        return output_data

                    if self._transform_output_group:
                        output_data = self._transform_data(
                            output_data, self.learning_set.OUTPUT_GROUP, True
                        )

                    return self._transform_data_from_variable_names(
                        output_data,
                        self.output_names,
                        self._transformed_output_sizes,
                        self._output_variables_to_transform,
                        True,
                    )

                return wrapper

            return format_transform_

        @classmethod
        def format_input_output(
            cls,
            predict: Callable[[ndarray], ndarray],
        ) -> Callable[[DataType], DataType]:
            """Make a function robust to type, array shape and data transformation.

            Args:
                predict: The function of interest to be called.

            Returns:
                A function calling the function of interest 'predict',
                while guaranteeing consistency in terms of data type and array shape,
                and applying input and/or output data transformation if required.
            """

            @cls.format_dict
            @cls.format_samples
            @cls.format_transform()
            def wrapper(self, input_data, *args, **kwargs):
                return predict(self, input_data, *args, **kwargs)

            return wrapper

    def _transform_data(self, data: ndarray, name: str, inverse: bool) -> ndarray:
        """
        Args:
            data: The original data array.
            name: The name of the variable or group to transform.
            inverse: Whether to use the inverse transformation.

        Returns:
            The transformed data.
        """
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

        input_data = dataset.get_data_by_names(self.input_names, False)[indices]
        output_data = dataset.get_data_by_names(self.output_names, False)[indices]
        self.input_space_center = split_array_to_dict_of_arrays(
            input_data.mean(0), self.learning_set.sizes, self.input_names
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
        else:
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
                transformed_data.append(dataset.get_data_by_names([name], False))
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
        else:
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
        data = self.learning_set.get_data_by_names(names, False)[indices]
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
            data, self.learning_set.get_data_by_names(self.output_names, False)[indices]
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
        e.g. :code:`array([1., 2., 3.])`
        or as a dictionary,
        e.g.  :code:`{'a': array([1.]), 'b': array([2., 3.])}`.

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
        input_names = self.input_names + [Dataset.INPUT_GROUP]
        output_names = self.output_names + [Dataset.OUTPUT_GROUP]

        for key in self.transformer:
            transformer = self.transformer.get(key)
            if key in input_names:
                if isinstance(transformer, DimensionReduction):
                    input_dimension += transformer.n_components
                else:
                    input_dimension += self.learning_set.sizes.get(
                        key, self.input_dimension
                    )

            if key in output_names:
                if isinstance(transformer, DimensionReduction):
                    output_dimension += transformer.n_components
                else:
                    output_dimension += self.learning_set.sizes.get(
                        key, self.output_dimension
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
                self._transformed_variable_sizes[name] = self.learning_set.sizes[name]
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
        data = self.learning_set.get_data_by_names(self.input_names, False)
        return data[self.learning_samples_indices]

    @property
    def output_data(self) -> ndarray:
        """The output data matrix."""
        data = self.learning_set.get_data_by_names(self.output_names, False)
        return data[self.learning_samples_indices]

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
