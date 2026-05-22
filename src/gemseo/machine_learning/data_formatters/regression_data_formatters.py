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
"""Data formatters for regression models."""

from __future__ import annotations

from collections.abc import Mapping
from functools import wraps
from typing import TYPE_CHECKING

from numpy import eye

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

    from gemseo.machine_learning.core.models.ml_model import DataType
    from gemseo.machine_learning.regression.models.base_regressor import BaseRegressor
    from gemseo.typing import RealArray

from gemseo.machine_learning.data_formatters.supervised_data_formatters import (
    SupervisedDataFormatters,
)
from gemseo.utils.data_conversion import concatenate_dict_of_arrays_to_array
from gemseo.utils.data_conversion import split_array_to_dict_of_arrays


class RegressionDataFormatters(SupervisedDataFormatters):
    """Data formatters for regression models."""

    @classmethod
    def format_dict_jacobian(
        cls,
        func: Callable[[BaseRegressor, RealArray, Any, ...], RealArray],
    ) -> Callable[[BaseRegressor, DataType, Any, ...], DataType]:
        """Make an array-based function callable with a dictionary of NumPy arrays.

        Args:
            func: The function to be called;
                it takes a NumPy array in input and returns a NumPy array.

        Returns:
            The wrapped `func` function, callable with
            either a NumPy data array
            or a dictionary of numpy data arrays indexed by variables names.
            The return value will have the same type as the input data.
        """

        @wraps(func)
        def wrapper(
            model: BaseRegressor, input_data: DataType, *args: Any, **kwargs: Any
        ) -> DataType:
            """Evaluate `func` with either array or dictionary-based data.

            Firstly,
            the pre-processing stage converts the input data to a NumPy data array,
            if these data are expressed as a dictionary of NumPy data arrays.

            Then,
            the processing evaluates the function `func`
            from this NumPy input data array.

            Lastly,
            the post-processing transforms the output data
            to a dictionary of output NumPy data array
            if the input data were passed as a dictionary of NumPy data arrays.

            Args:
                model: The regression model.
                input_data: The input data.
                *args: The positional arguments of the function `func`.
                **kwargs: The keyword arguments of the function `func`.

            Returns:
                The output data with the same type as the input one.
            """
            as_dict = isinstance(input_data, Mapping)
            if as_dict:
                input_data = concatenate_dict_of_arrays_to_array(
                    input_data, model.input_names
                )
            single_sample = len(input_data.shape) == 1
            jacobians = func(model, input_data, *args, **kwargs)
            if as_dict:
                varsizes = model.learning_set.variable_name_to_n_components
                if single_sample:
                    jacobians = split_array_to_dict_of_arrays(
                        jacobians, varsizes, model.output_names, model.input_names
                    )
                else:
                    jacobians = split_array_to_dict_of_arrays(
                        jacobians, varsizes, model.output_names, model.input_names
                    )
            return jacobians

        return wrapper

    @classmethod
    def transform_jacobian(
        cls,
        func: Callable[[BaseRegressor, RealArray, Any, ...], RealArray],
    ) -> Callable[[BaseRegressor, RealArray, Any, ...], RealArray]:
        """Apply transformation to inputs and inverse transformation to outputs.

        Args:
            func: The function of interest to be called.

        Returns:
            A function evaluating the function `func`,
            after transforming its input data
            and/or before transforming its output data.
        """

        @wraps(func)
        def wrapper(
            model: BaseRegressor, input_data: RealArray, *args: Any, **kwargs: Any
        ) -> RealArray:
            """Evaluate `func` after or before data transformation.

            Firstly,
            the pre-processing stage transforms the input data if required.

            Then,
            the processing evaluates the function `func`.

            Lastly,
            the post-processing stage transforms the output data if required.

            Args:
                model: The regression model.
                input_data: The input data.
                *args: The positional arguments of the function.
                **kwargs: The keyword arguments of the function.

            Returns:
                Either the raw output data of `func`
                or a transformed version according to the requirements.

            Raises:
                NotImplementedError: When the transformer is applied to a variable
                    rather than to a group of variables.
            """
            if (
                model._input_variables_to_transform
                or model._output_variables_to_transform
            ):
                # TODO: implement this case
                msg = (
                    "The Jacobian of regression models cannot be computed "
                    "when the transformed quantities are variables; "
                    "please transform the whole group 'inputs' or 'outputs' "
                    "or do not use data transformation."
                )
                raise NotImplementedError(msg)

            inputs = model.learning_set.INPUT_GROUP
            if inputs in model.transformer:
                jac = model.transformer[inputs].compute_jacobian(input_data)
                input_data = model.transformer[inputs].transform(input_data)
            else:
                jac = eye(input_data.shape[1])

            jac = func(model, input_data, *args, **kwargs) @ jac
            output_data = model.predict_raw(input_data)

            outputs = model.learning_set.OUTPUT_GROUP
            if outputs in model.transformer:
                jac = (
                    model.transformer[outputs].compute_jacobian_inverse(output_data)
                    @ jac
                )
            return jac

        return wrapper
