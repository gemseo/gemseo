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

# -*-mode: python; py-indent-offset: 4; tab-width: 8; coding:utf-8 -*-
# Copyright (c) 2018 IRT-AESE.
# All rights reserved.
#
# Contributors:
#    INITIAL AUTHORS - API and implementation and/or documentation
#        :author: Arthur Piat
#        :author: François Gallard: initial author of the scilab version
#                                   of MatlabDataProcessorWrapper
#        :author: Nicolas Roussouly: GEMSEO integration

#
#    OTHER AUTHORS   - MACROSCOPIC CHANGES

"""Definition of Matlab data processor.

Overview
--------

The class and functions in this module enables to
manipulate data from and toward the Matlab workspace.
It also enables to read and write Matlab data file (.mat).
"""

from typing import Mapping, Union

import matlab
import scipy.io
from numpy import array, iscomplexobj, ndarray

from gemseo.core.data_processor import DataProcessor
from gemseo.utils.py23_compat import Path


class MatlabDataProcessor(DataProcessor):
    """A Matlab data processor.

    Convert GEMSEO format to Matlab format.

    Examples:
        >>> # Build a new instance
        >>> proc = MatlabDataProcessor()
        >>> # initial python data
        >>> d = {"x": array([2]), "y": array([2j], dtype="complex")}
        >>> # process data to matlab format
        >>> res = proc.pre_process_data(d)
        >>> print(res)
        >>>
        >>> # initial data in matlab format
        >>> d = {"y": double([2, 3]), "x": double([2j], is_complex=True)}
        >>> # process to Python format
        >>> res = proc.post_process_data(d)
        >>> print(res)
    """

    def pre_process_data(
        self, data  # type: Mapping[str, ndarray]
    ):  # type: (...) -> Mapping[str, matlab.double]
        """Transform data from GEMSEO to Matlab.

        The function takes a dict of ndarray and return
        a dict of matlab.double.

        Args:
            data: The input data.

        Returns:
            The data with matlab array types.
        """
        processed_data = {}

        for name, values in data.items():
            if isinstance(values, matlab.double):
                processed_data[name] = values
            else:
                processed_data[name] = array2double(values)

        return processed_data

    def post_process_data(
        self, data  # type: Mapping[str, matlab.double]
    ):  # type: (...) -> Mapping[str, ndarray]
        """Transform the output data from Matlab to GEMSEO.

        Args:
            data: The data with matlab arrays.

        Returns:
            The data with numpy arrays.
        """
        processed_data = {}

        for name, values in data.items():
            if isinstance(values, ndarray):
                processed_data[name] = values.copy()
            else:
                processed_data[name] = double2array(values)

        return processed_data


def load_matlab_file(
    file_path,  # type: Union[str, Path]
):  # type: (...) -> Mapping[str, matlab.double]
    """Read .mat file and convert it to usable format for Matlab.

    Args:
        file_path: The path to a .mat file.

    Returns:
        The dict of matlab.double.
    """
    row_data = scipy.io.loadmat(str(file_path))
    clean_data = {}
    for parameter in row_data:
        if parameter not in ["__header__", "__globals__", "__version__"]:
            clean_data[parameter] = array2double(row_data[parameter])

    return clean_data


def save_matlab_file(
    dict_to_save,  # type: dict
    file_path="output_dict",  # type: Union[str, Path]
    *args,  # type: bool
    **kwargs  # type: bool
):  # type: (...) -> None
    """Create a .mat file from dict of ndarray.

    Args:
        dict_to_save: The dict of ndarray to be saved.
        file_path: The path where to sabe the file.
        *args: The list of scipy.io.savemat options.
        **kwargs: The dict of scipy.io.savemat options.

    Raises:
        ValueError: If the saved dictionary is nor composed
            of ndarray only.
    """
    saved_dict = dict_to_save.copy()

    for key, value in saved_dict.items():
        if isinstance(value, matlab.double):
            saved_dict[key] = double2array(value)
        elif not isinstance(value, ndarray):
            msg = "The saved dict must be composed of ndarray only"
            raise ValueError(msg)

    scipy.io.savemat(file_name=str(file_path), mdict=saved_dict, *args, **kwargs)


def array2double(
    data_array,  # type: ndarray
):  # type (..) -> matlab.double
    """Turn a ndarray into a matlab.double.

    Args:
        data_array: The numpy array to be converted.

    Returns:
        The matlab.double value.
    """
    is_cmplx = iscomplexobj(data_array)
    if len(data_array.shape) == 1:
        return matlab.double(data_array.tolist(), is_complex=is_cmplx)[0]
    else:
        return matlab.double(data_array.tolist(), is_complex=is_cmplx)


def double2array(
    matlab_double,  # type: matlab.double
):  # type: (...) -> ndarray
    """Turn a matlab double into ndarray.

    Args:
        matlab_double: The matlab.double values.

    Returns:
        The array of values.
    """
    if iscomplexobj(matlab_double):
        d_type = "complex"
    else:
        d_type = None

    # note here that we can treat string as well
    # -> we put string into an array as float
    #    (otherwise the array has no shape)
    if isinstance(matlab_double, float) or isinstance(matlab_double, str):
        output = array([matlab_double], dtype=d_type)
    else:
        output = array(matlab_double, dtype=d_type)

    if output.shape[0] == 1 and len(output.shape) > 1:
        output = output[0]

    return output


def convert_array_from_matlab(
    data,  # type: Mapping[str, matlab.double]
):  # type: (...) -> Mapping[str, ndarray]
    """Convert dict of matlab.output to dict of ndarray.

    Args:
        data: The dict of matlab.double.

    Returns:
        The dict of ndarray.
    """
    output_values = {}
    for matlab_key in data:
        current_value = data[matlab_key]
        output_values[matlab_key] = double2array(current_value)
    return output_values


def convert_array_to_matlab(
    data,  # type: Mapping[str, ndarray]
):  # type: (...) -> Mapping[str, matlab.double]
    """Convert gems dict of ndarray to dict of matlab.double.

    Args:
        data: The dict of ndarray.

    Returns:
        The dict of matlab.double.
    """
    output = {}
    for keys in data:
        current_data = data[keys]
        if not (len(current_data) == 1):
            output[keys] = array2double(current_data)
        else:
            if iscomplexobj(current_data):
                output[keys] = complex(current_data)
            else:
                output[keys] = float(current_data)

    return output
