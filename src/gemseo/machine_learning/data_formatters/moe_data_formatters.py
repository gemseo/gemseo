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
"""Data formatters for mixture of experts."""

from __future__ import annotations

import functools
from collections.abc import Mapping
from typing import TYPE_CHECKING

from gemseo.machine_learning.data_formatters.regression_data_formatters import (
    RegressionDataFormatters,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Any

    from numpy import ndarray

    from gemseo.machine_learning.core.models.ml_model import DataType
    from gemseo.machine_learning.regression.models.moe import MOERegressor
    from gemseo.typing import RealArray
from gemseo.utils.data_conversion import concatenate_dict_of_arrays_to_array


class MOEDataFormatters(RegressionDataFormatters):
    """Data formatters for mixture of experts."""

    @classmethod
    def format_predict_class_dict(
        cls,
        func: Callable[[MOERegressor, RealArray, Any, ...], ndarray],
    ) -> Callable[[MOERegressor, DataType, Any, ...], DataType]:
        """Make an array-based function be called with a dictionary of NumPy arrays.

        Args:
            func: The function to be called;
                it takes a NumPy array in input and returns a NumPy array.

        Returns:
            A function making a function work with
            either a NumPy data array
            or a dictionary of NumPy data arrays indexed by variables names.
            The evaluation will have the same type as the input data.
        """

        @functools.wraps(func)
        def wrapper(
            model: MOERegressor,
            input_data: DataType,
            *args: Any,
            **kwargs: Any,
        ) -> DataType:
            """Evaluate `func` with either array or dictionary-based input data.

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
                model: The mixture of experts.
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
            output_data = func(model, input_data, *args, **kwargs)
            if as_dict:
                output_data = {model.LABELS: output_data}
            return output_data

        return wrapper
