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
"""Data formatters for supervised machine learning algorithms."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from numpy import atleast_2d

if TYPE_CHECKING:
    from typing import Any
    from typing import Callable

    from numpy import ndarray

    from gemseo.mlearning import MLSupervisedAlgo
    from gemseo.mlearning.core.ml_algo import DataType
from gemseo.mlearning.data_formatters.base_data_formatters import BaseDataFormatters
from gemseo.utils.data_conversion import concatenate_dict_of_arrays_to_array
from gemseo.utils.data_conversion import split_array_to_dict_of_arrays


class SupervisedDataFormatters(BaseDataFormatters):
    """Data formatters for supervised machine learning algorithms."""

    @classmethod
    def format_dict(
        cls,
        func: Callable[[MLSupervisedAlgo, ndarray, Any, ...], ndarray],
    ) -> Callable[[MLSupervisedAlgo, DataType, Any, ...], DataType]:
        """Make an array-based function be called with a dictionary of NumPy arrays.

        Args:
            func: The function to be called;
                it takes a NumPy array in input and returns a NumPy array.

        Returns:
            A function making the function ``func`` work with
            either a NumPy data array
            or a dictionary of NumPy data arrays indexed by variables names.
            The evaluation will have the same type as the input data.
        """

        def wrapper(
            algo: MLSupervisedAlgo,
            input_data: DataType,
            *args: Any,
            **kwargs: Any,
        ) -> DataType:
            """Evaluate ``func`` with either array or dictionary-based input data.

            Firstly,
            the pre-processing stage converts the input data to a NumPy data array,
            if these data are expressed as a dictionary of NumPy data arrays.

            Then,
            the processing evaluates the function ``func``
            from this NumPy input data array.

            Lastly,
            the post-processing transforms the output data
            to a dictionary of output NumPy data array
            if the input data were passed as a dictionary of NumPy data arrays.

            Args:
                algo: The supervised learning algorithm.
                input_data: The input data.
                *args: The positional arguments of the function ``func``.
                **kwargs: The keyword arguments of the function ``func``.

            Returns:
                The output data with the same type as the input one.
            """
            as_dict = isinstance(input_data, Mapping)
            if as_dict:
                input_data = concatenate_dict_of_arrays_to_array(
                    input_data, algo.input_names
                )

            output_data = func(algo, input_data, *args, **kwargs)
            if as_dict:
                return split_array_to_dict_of_arrays(
                    output_data,
                    algo.learning_set.variable_names_to_n_components,
                    algo.output_names,
                )

            return output_data

        return wrapper

    @classmethod
    def format_samples(
        cls,
        func: Callable[[MLSupervisedAlgo, ndarray, Any, ...], ndarray],
    ) -> Callable[[MLSupervisedAlgo, ndarray, Any, ...], ndarray]:
        """Make a 2D NumPy array-based function work with 1D NumPy array.

        Args:
            func: The function to be called;
                it takes a 2D NumPy array in input
                and returns a 2D NumPy array.
                The first dimension represents the samples
                while the second one represents the components of the variables.

        Returns:
            A function making the function ``func`` work with
            either a 1D NumPy array or a 2D NumPy array.
            The evaluation will have the same dimension as the input data.
        """

        def wrapper(
            algo: MLSupervisedAlgo,
            input_data: DataType,
            *args: Any,
            **kwargs: Any,
        ) -> DataType:
            """Evaluate ``func`` with either a 1D or 2D NumPy data array.

            Firstly,
            the pre-processing stage converts the input data
            to a 2D NumPy data array.

            Then,
            the processing evaluates the function ``func``
            from this 2D NumPy data array.

            Lastly,
            the post-processing converts the output data to a 1D NumPy data array
            if the dimension of the input data is equal to 1.

            Args:
                algo: The supervised learning algorithm.
                input_data: The input data.
                *args: The positional arguments of the function ``func``.
                **kwargs: The keyword arguments of the function ``func``.

            Returns:
                The output data with the same dimension as the input one.
            """
            single_sample = input_data.ndim == 1
            output_data = func(algo, atleast_2d(input_data), *args, **kwargs)
            if single_sample:
                output_data = output_data[0]

            return output_data

        return wrapper

    @classmethod
    def format_transform(
        cls,
        transform_inputs: bool = True,
        transform_outputs: bool = True,
    ) -> Callable[
        [Callable[[MLSupervisedAlgo, ndarray, Any, ...], ndarray]],
        Callable[[MLSupervisedAlgo, ndarray, Any, ...], ndarray],
    ]:
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
            func: Callable[[MLSupervisedAlgo, ndarray, Any, ...], ndarray],
        ) -> Callable[[MLSupervisedAlgo, ndarray, Any, ...], ndarray]:
            """Apply transformation to inputs and inverse transformation to outputs.

            Args:
                func: The function of interest to be called.

            Returns:
                A function evaluating the function ``func``,
                after transforming its input data
                and/or before transforming its output data.
            """

            def wrapper(
                algo: MLSupervisedAlgo,
                input_data: ndarray,
                *args: Any,
                **kwargs: Any,
            ) -> ndarray:
                """Evaluate ``func`` after or before data transformation.

                Firstly,
                the pre-processing stage transforms the input data if required.

                Then,
                the processing evaluates the function ``func``.

                Lastly,
                the post-processing stage transforms the output data if required.

                Args:
                    algo: The supervised learning algorithm.
                    input_data: The input data.
                    *args: The positional arguments of the function.
                    **kwargs: The keyword arguments of the function.

                Returns:
                    Either the raw output data of ``func``
                    or a transformed version according to the requirements.
                """
                if transform_inputs:
                    if algo._transform_input_group:
                        input_data = algo._transform_data(
                            input_data, algo.learning_set.INPUT_GROUP, False
                        )

                    if algo._input_variables_to_transform:
                        input_data = algo._transform_data_from_variable_names(
                            input_data,
                            algo.input_names,
                            algo.learning_set.variable_names_to_n_components,
                            algo._input_variables_to_transform,
                            False,
                        )

                output_data = func(algo, input_data, *args, **kwargs)

                if not transform_outputs or (
                    not algo._transform_output_group
                    and not algo._output_variables_to_transform
                ):
                    return output_data

                if algo._transform_output_group:
                    output_data = algo._transform_data(
                        output_data, algo.learning_set.OUTPUT_GROUP, True
                    )

                return algo._transform_data_from_variable_names(
                    output_data,
                    algo.output_names,
                    algo._transformed_output_sizes,
                    algo._output_variables_to_transform,
                    True,
                )

            return wrapper

        return format_transform_

    @classmethod
    def format_input_output(
        cls,
        func: Callable[[MLSupervisedAlgo, ndarray, Any, ...], ndarray],
    ) -> Callable[[MLSupervisedAlgo, DataType, Any, ...], DataType]:
        """Make a function robust to type, array shape and data transformation.

        Args:
            func: The function of interest to be called.

        Returns:
            A function calling the function of interest ``func``,
            while guaranteeing consistency in terms of data type and array shape,
            and applying input and/or output data transformation if required.
        """

        @cls.format_dict
        @cls.format_samples
        @cls.format_transform()
        def wrapper(
            algo: MLSupervisedAlgo, input_data: DataType, *args: Any, **kwargs: Any
        ) -> DataType:
            """Compute output data from input data.

            Args:
                algo: The supervised learning algorithm.
                input_data: The input data.
                *args: The positional arguments of the function.
                **kwargs: The keyword arguments of the function.

            Returns:
                The output data.
            """
            return func(algo, input_data, *args, **kwargs)

        return wrapper
