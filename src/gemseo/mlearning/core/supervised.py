# -*- coding: utf-8 -*-
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
from __future__ import division
from __future__ import unicode_literals

from typing import Callable
from typing import Dict
from typing import Iterable
from typing import Mapping
from typing import NoReturn
from typing import Optional
from typing import Sequence
from typing import Tuple
from typing import Union

from numpy import array
from numpy import atleast_2d
from numpy import ndarray

from gemseo.core.dataset import Dataset
from gemseo.mlearning.core.ml_algo import DataType
from gemseo.mlearning.core.ml_algo import MLAlgo
from gemseo.mlearning.core.ml_algo import MLAlgoParameterType
from gemseo.mlearning.core.ml_algo import SavedObjectType as MLAlgoSaveObjectType
from gemseo.mlearning.core.ml_algo import TransformerType
from gemseo.mlearning.transform.dimension_reduction.dimension_reduction import (
    DimensionReduction,
)
from gemseo.mlearning.transform.scaler.min_max_scaler import MinMaxScaler
from gemseo.utils.data_conversion import concatenate_dict_of_arrays_to_array
from gemseo.utils.data_conversion import split_array_to_dict_of_arrays

SavedObjectType = Union[MLAlgoSaveObjectType, Sequence[str], Dict[str, ndarray]]


class MLSupervisedAlgo(MLAlgo):
    """Supervised machine learning algorithm.

    Inheriting classes shall overload the :meth:`!MLSupervisedAlgo._fit` and
    :meth:`!MLSupervisedAlgo._predict` methods.

    Attributes:
        input_names (List[str]): The names of the input variables.
        input_space_center (Dict[str, ndarray]): The center of the input space.
        output_names (List[str]): The names of the output variables.
    """

    ABBR = "MLSupervisedAlgo"
    DEFAULT_TRANSFORMER = {Dataset.INPUT_GROUP: MinMaxScaler()}

    def __init__(
        self,
        data,  # type: Dataset
        transformer=DEFAULT_TRANSFORMER,  # type: Optional[Mapping[str,TransformerType]]
        input_names=None,  # type: Optional[Iterable[str]]
        output_names=None,  # type: Optional[Iterable[str]]
        **parameters,  # type: MLAlgoParameterType
    ):  # type: (...) -> None
        """
        Args:
            input_names: The names of the input variables.
                If None, consider all input variables mentioned in the learning dataset.
            output_names: The names of the output variables.
                If None, consider all input variables mentioned in the learning dataset.
        """
        super(MLSupervisedAlgo, self).__init__(
            data, transformer=transformer, **parameters
        )
        self.input_names = input_names or data.get_names(data.INPUT_GROUP)
        self.input_space_center = array([])
        self.__input_dimension = 0
        self.output_names = output_names or data.get_names(data.OUTPUT_GROUP)
        self.__output_dimension = 0
        self.__reduced_dimensions = (0, 0)

    @property
    def _reduced_dimensions(self):  # type: (...) -> Tuple[int, int]
        """The input and output reduced dimensions."""
        if self.__reduced_dimensions == (0, 0):
            self.__reduced_dimensions = self.__compute_reduced_dimensions()

        return self.__reduced_dimensions

    @property
    def input_dimension(self):  # type: (...) -> int
        """The input space dimension."""
        if not self.__input_dimension and self.learning_set is not None:
            self.__input_dimension = sum(
                [self.learning_set.sizes[name] for name in self.input_names]
            )

        return self.__input_dimension

    @property
    def output_dimension(self):  # type: (...) -> int
        """The output space dimension."""
        if not self.__output_dimension and self.learning_set is not None:
            self.__output_dimension = sum(
                [self.learning_set.sizes[name] for name in self.output_names]
            )

        return self.__output_dimension

    class DataFormatters(MLAlgo.DataFormatters):
        """Decorators for supervised algorithms."""

        @classmethod
        def format_dict(
            cls,
            predict,  # type: Callable[[ndarray],ndarray]
        ):  # type: (...) -> Callable[[DataType],DataType]
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
                input_data,  # type: DataType
                *args,
                **kwargs,
            ):  # type: (...) -> DataType
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
            predict,  # type: Callable[[ndarray],ndarray]
        ):  # type: (...) -> Callable[[ndarray],ndarray]
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
                input_data,  # type: DataType
                *args,
                **kwargs,
            ):  # type: (...) -> DataType
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
            transform_inputs=True,  # type: bool
            transform_outputs=True,  # type: bool
        ):  # type: (...) -> Callable[[ndarray],ndarray]
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
                predict,  # type: Callable[[ndarray],ndarray]
            ):  # type: (...) -> Callable[[ndarray],ndarray]
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
                    input_data,  # type: DataType
                    *args,
                    **kwargs,
                ):  # type: (...) -> DataType
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
                    inputs = self.learning_set.INPUT_GROUP
                    if transform_inputs and inputs in self.transformer:
                        input_data = self.transformer[inputs].transform(input_data)

                    output_data = predict(self, input_data, *args, **kwargs)
                    outputs = self.learning_set.OUTPUT_GROUP
                    if transform_outputs and outputs in self.transformer:
                        return self.transformer[outputs].inverse_transform(output_data)

                    return output_data

                return wrapper

            return format_transform_

        @classmethod
        def format_input_output(
            cls,
            predict,  # type: Callable[[ndarray],ndarray]
        ):  # type: (...) -> Callable[[DataType],DataType]
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

    def _learn(
        self,
        indices,  # type: Optional[Sequence[int]]
    ):  # type: (...) -> None
        """
        Raises:
            NotImplementedError: If an output transformer modifies
                both the input and the output variables, e.g. :class:`.PLS`.
        """
        input_group = self.learning_set.INPUT_GROUP
        output_group = self.learning_set.OUTPUT_GROUP
        input_data = self.learning_set.get_data_by_names(self.input_names, False)
        output_data = self.learning_set.get_data_by_names(self.output_names, False)

        if indices is not None:
            input_data = input_data[indices]
            output_data = output_data[indices]

        self.input_space_center = split_array_to_dict_of_arrays(
            input_data.mean(0), self.learning_set.sizes, self.input_names
        )

        if input_group in self.transformer:
            transformer = self.transformer[input_group]
            if transformer.CROSSED:
                input_data = transformer.fit_transform(input_data, output_data)
            else:
                input_data = transformer.fit_transform(input_data)

        if output_group in self.transformer:
            transformer = self.transformer[output_group]
            if self.transformer[output_group].CROSSED:
                raise NotImplementedError(
                    "The transformer of type {} cannot be applied to the outputs "
                    "to build a supervised machine learning algorithm".format(
                        self.transformer[output_group].__class__.__name__
                    )
                )
            else:
                output_data = transformer.fit_transform(output_data)

        self._fit(input_data, output_data)

    def _fit(
        self,
        input_data,  # type: ndarray
        output_data,  # type: ndarray
    ):  # type: (...) -> NoReturn
        """Fit input-output relationship from the learning data.

        Args:
            input_data: The input data with the shape (n_samples, n_inputs).
            output_data: The output data with shape (n_samples, n_outputs).
        """
        raise NotImplementedError

    @DataFormatters.format_input_output
    def predict(
        self,
        input_data,  # type: DataType
    ):  # type: (...) -> DataType
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

    def _predict(
        self,
        input_data,  # type: ndarray
    ):  # type: (...) -> NoReturn
        """Predict output data from input data.

        Args:
            input_data: The input data with shape (n_samples, n_inputs).

        Returns:
            output_data: The output data with shape (n_samples, n_outputs).
        """
        raise NotImplementedError

    def __compute_reduced_dimensions(self):  # type: (...) -> Tuple[int,int]
        """Return the reduced input and output dimensions after transformations.

        Returns:
            The reduced input and output dimensions.
        """
        transformer = self.transformer.get(Dataset.INPUT_GROUP)
        if isinstance(transformer, DimensionReduction):
            input_dimension = transformer.n_components
        else:
            input_dimension = self.input_dimension

        transformer = self.transformer.get(Dataset.OUTPUT_GROUP)
        if isinstance(transformer, DimensionReduction):
            output_dimension = transformer.n_components
        else:
            output_dimension = self.output_dimension

        return input_dimension, output_dimension

    @property
    def input_data(self):  # type: (...) -> ndarray
        """The input data matrix."""
        return self.learning_set.get_data_by_names(self.input_names, False)

    @property
    def output_data(self):  # type: (...) -> ndarray
        """The output data matrix."""
        return self.learning_set.get_data_by_names(self.output_names, False)

    def _get_objects_to_save(self):  # type: (...) -> Dict[str,SavedObjectType]
        objects = super(MLSupervisedAlgo, self)._get_objects_to_save()
        objects["input_names"] = self.input_names
        objects["output_names"] = self.output_names
        objects["input_space_center"] = self.input_space_center
        return objects
