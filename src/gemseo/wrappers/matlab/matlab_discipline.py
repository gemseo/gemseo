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
# Contributors:
#    INITIAL AUTHORS - API and implementation and/or documentation
#        :author: François Gallard: initial Scilab version
#        :author: Arthur Piat: conversion Scilab to Matlab and complementary features
#        :author: Nicolas Roussouly: GEMSEO integration
"""Definition of the Matlab discipline.

Overview
--------

This module contains the :class:`.MatlabDiscipline`
which enables to automatically create a wrapper of any Matlab function.
This class can be used in order to interface any Matlab code
and to use it inside a MDO process.
"""
from __future__ import annotations

import logging
import os
import re
from os.path import exists
from os.path import join
from pathlib import Path
from typing import Any
from typing import Mapping
from typing import Sequence

import matlab.engine
import numpy as np

from gemseo.core.discipline import MDODiscipline
from gemseo.wrappers.matlab.engine import get_matlab_engine
from gemseo.wrappers.matlab.engine import MatlabEngine
from gemseo.wrappers.matlab.matlab_data_processor import convert_array_from_matlab
from gemseo.wrappers.matlab.matlab_data_processor import double2array
from gemseo.wrappers.matlab.matlab_data_processor import load_matlab_file
from gemseo.wrappers.matlab.matlab_data_processor import MatlabDataProcessor
from gemseo.wrappers.matlab.matlab_data_processor import save_matlab_file
from gemseo.wrappers.matlab.matlab_parser import MatlabParser

LOGGER = logging.getLogger(__name__)


class MatlabDiscipline(MDODiscipline):
    """Base wrapper for matlab discipline.

    Generates a discipline of given matlab function and wrap it to be executed with
    GEMSEO. Can be used on encrypted, MATLAB build-in and user made function.

    Examples:
        >>> # build the discipline from the MATLAB function "function.m"
        >>> disc = MatlabDiscipline("function.m")
        >>> # Execute the discipline
        >>> disc.execute({"x" : array([2.]), "y" : array([1.])})
        >>>
        >>> # build discipline with initial data from MATLAB file
        >>> disc = MatlabDiscipline("function.m", matlab_data_file="data.mat")
        >>> # execute discipline from default values
        >>> disc.execute()
        >>>
        >>> # build discipline from MATLAB file located in matlab_files directory
        >>> disc = MatlabDiscipline("function.m", search_file="matlab_files")
        >>>
        >>> # build discipline with jacobian returned by the matlab function
        >>> disc = MatlabDiscipline("function.m", is_jac_returned_by_func=True)
        >>> disc.execute({"x" : array([2.]), "y" : array([1.])})
        >>> # print jacboian values
        >>> print(disc.jac)

    Note:
        If ``is_jac_returned_by_func`` is True, jacobian matrices must be returned
        by the matlab function itself. In such case, function outputs must contain
        standard output as well as new outputs for jacobian terms. These new
        outputs must follow naming convention described in function
        :meth:`.MatlabDiscipline._get_jac_name`. They can be returned
        in any order.
    """

    JAC_PREFIX = "jac_"

    def __init__(
        self,
        matlab_fct: str | Path,
        input_names: Sequence[str] | None = None,
        output_names: Sequence[str] | None = None,
        add_subfold_path: bool = False,
        search_file: str | None = None,
        matlab_engine_name: str = "matlab",
        matlab_data_file: str | Path | None = None,
        name: str | None = None,
        clean_cache_each_n: int | None = None,
        input_grammar_file: str | None = None,
        output_grammar_file: str | None = None,
        auto_detect_grammar_files: bool = False,
        check_opt_data: bool = True,
        cache_type: str = MDODiscipline.SIMPLE_CACHE,
        grammar_type: str = MDODiscipline.JSON_GRAMMAR_TYPE,
        cache_file_path: str | None = None,
        is_jac_returned_by_func: bool = False,
    ) -> None:
        """
        Args:
            matlab_fct: The path of the Matlab file or Name of the function.
            input_names: The list of input variables.
            output_names: The list of output variables.
            add_subfold_path: If True, add all sub-folder to matlab engine path.
            search_file: The root directory to launch the research of matlab file.
            matlab_engine_name: The name of the singleton used for this discipline.
            matlab_data_file: The .mat file or path containing default values of data.
            name: The name of discipline.
            clean_cache_each_n: Iteration interval at which matlab workspace is cleaned.
            input_grammar_file: The file for input grammar description,
                if None, name + "_input.json" is used.
            output_grammar_file: The file for output grammar description.
            auto_detect_grammar_files: If no input and output grammar files
                are provided,
                auto_detect_grammar_files uses a naming convention
                to associate a grammar file to a discipline:
                searches in the "comp_dir" directory containing the
                discipline source file for files basenames
                self.name _input.json and self.name _output.json.
            check_opt_data: If true, check input and output data of
                discipline, pass if False.
            cache_type: The type of cache policy, SIMPLE_CACHE
                or HDF5_CACHE.
            grammar_type: The type of grammar to use for IO declaration
                either JSON_GRAMMAR_TYPE or SIMPLE_GRAMMAR_TYPE.
            cache_file_path: The file to store the data,
                mandatory when HDF caching is used.
            is_jac_returned_by_func: If True, the jacobian matrices should be returned
                of matlab function with standard outputs.
                Default is False.
                If True, the conventional name 'jac_dout_din' is used as jacobian
                term of any output 'out' with respect to input 'in'.
        """  # noqa: D205, D212, D415
        super().__init__(
            name=name,
            input_grammar_file=input_grammar_file,
            output_grammar_file=output_grammar_file,
            auto_detect_grammar_files=auto_detect_grammar_files,
            cache_type=cache_type,
            grammar_type=grammar_type,
            cache_file_path=cache_file_path,
        )
        self.__fct_name = None

        matlab_fct = str(matlab_fct)
        if input_names is None or output_names is None:
            parser = MatlabParser()

            if search_file is not None:
                path = self.search_file(matlab_fct, search_file)
                parser.parse(path)
                if matlab_data_file is not None and not exists(str(matlab_data_file)):
                    matlab_data_file = self.search_file(
                        str(matlab_data_file), search_file, ".mat"
                    )
            else:
                parser.parse(matlab_fct)

            input_data = parser.inputs
            output_data = parser.outputs
            function_path = (parser.directory / parser.function_name).with_suffix(".m")
        else:
            function_path = matlab_fct

        if input_names is not None:
            input_data = input_names
        if output_names is not None:
            output_data = output_names

        self.__engine = get_matlab_engine(matlab_engine_name)
        self.__inputs = input_data
        self.__outputs = output_data
        # init size with -1 -> means that size is currently unknown
        self.__is_size_known = False
        self.__inputs_size = {var: -1 for var in self.__inputs}
        self.__outputs_size = {var: -1 for var in self.__outputs}

        # self.outputs can be filtered here
        self.__is_jac_returned_by_func = is_jac_returned_by_func
        self.__jac_output_names = []
        self.__jac_output_indices = []
        if self.__is_jac_returned_by_func:
            self.__filter_jacobian_in_outputs()

        self.__check_function(function_path, add_subfold_path)
        self.__check_opt_data = check_opt_data
        self.cleaning_interval = clean_cache_each_n
        self.__init_default_data(
            matlab_data_file,
            input_grammar_file,
            output_grammar_file,
            auto_detect_grammar_files,
        )
        self.data_processor = MatlabDataProcessor()

        if self.__is_jac_returned_by_func:
            self.__reorder_and_check_jacobian_consistency()

    @property
    def engine(self) -> MatlabEngine:
        """The matlab engine of the discipline.

        The engine is associated to the ``matlab_engine_name`` provided at the instance
        construction.
        """
        return self.__engine

    @property
    def function_name(self) -> str:
        """Return the name of the function."""
        return self.__fct_name

    @staticmethod
    def search_file(
        file_name: str,
        root_dir: str,
        extension: str = ".m",
    ) -> str:
        """Locate recursively a file in the given root directory.

        Args:
            file_name: The name of the file to be located.
            root_dir: The root directory to launch the research.
            extension: The extension of the file in case not given by user.

        Returns:
            The path of the given file.

        Raises:
            IOError:
                * If two files are found in same directory;
                * If no file is found.
        """
        found_file = False
        re_matfile = re.compile(r"\S+\.\S*")
        grps = re_matfile.search(file_name)
        if grps is None:
            file_name += extension

        file_path = ""
        for subdir, _, files in os.walk(str(root_dir)):
            for file_loc in files:
                if file_loc == file_name:
                    if found_file:
                        msg = "At least two files {} were in directory {}".format(
                            file_name, root_dir
                        )
                        msg += f"\n File one: {file_path};"
                        msg += f"\n File two: {join(subdir, file_loc)}."
                        raise OSError(msg)
                    found_file = True
                    file_path = join(subdir, file_loc)
                    dir_name = subdir

        if not found_file:
            raise OSError(f"No file: {file_name}, found in directory: {root_dir}.")

        LOGGER.info("File: %s found in directory: %s.", file_name, dir_name)
        return file_path

    def __check_function(
        self,
        matlab_fct: str | Path,
        add_subfold_path: bool,
    ) -> None:
        """Check the availability of the prescribed MATLAB function.

        The function manages encrypted, build-in and user made function and
        unify their use.

        Args:
            matlab_fct: A name for the matlab function to be wrapped.
            add_subfold_path: If true, add all sub-folders of the function to
                matlab search path.

        Raises:
            NameError: If the function (or file) does not exist.
        """
        path = Path(matlab_fct)
        if path.exists():
            # Test if the file exists in the system
            self.__engine.add_path(path.parent, add_subfolder=add_subfold_path)
            self.__fct_name = path.stem
        elif self.__engine.exist(matlab_fct)[0]:
            # If file does not exist, try to find an existing build-in function in
            # engine
            self.__fct_name = matlab_fct
        else:
            # If no file and build-in function exist, raise error
            msg = f'No existing file or function "{matlab_fct}".'
            raise NameError(msg)

    def __init_default_data(
        self,
        matlab_data_file: str,
        input_grammar_file: str,
        output_grammar_file: str,
        auto_detect_grammar_files: bool,
    ) -> None:
        """Initialize default data of the discipline.

        Args:
            matlab_data_file: The path to the .mat containing default values of data
            input_grammar_file: The file for input grammar description,
                if None, name + "_input.json" is used.
            output_grammar_file: The file for output grammar description.
            auto_detect_grammar_files: If True, no input and output grammar files.
        """
        if matlab_data_file is not None:
            saved_values = convert_array_from_matlab(
                load_matlab_file(str(matlab_data_file).replace(".mat", ""))
            )

        if input_grammar_file is None and not auto_detect_grammar_files:
            # Here, we temporary init inputs data with an array of
            # size 1 but that could not be the right size...
            # The right size can be known from either matlab_data_file or evaluating
            # the matlab function
            input_data = dict.fromkeys(self.__inputs, np.array([0.1]))
            if matlab_data_file is not None:
                input_data = self.__update_data(input_data.copy(), saved_values)

        if output_grammar_file is None and not auto_detect_grammar_files:
            # same remark as above about the size
            output_data = dict.fromkeys(self.__outputs, np.array([0.1]))
            if matlab_data_file is not None:
                output_data = self.__update_data(output_data.copy(), saved_values)

        self.input_grammar.update_from_data(input_data)

        self.output_grammar.update_from_data(output_data)

        # If none input matlab data is prescribed, we cannot know
        # the size of inputs and outputs. Thus, we must evaluate
        # the function in order to know the sizes
        if matlab_data_file is not None:
            self.__is_size_known = True
            for input_name, input_value in input_data.items():
                self.__inputs_size[input_name] = len(input_value)

            for output_name, output_value in output_data.items():
                self.__outputs_size[output_name] = len(output_value)

        default_values = input_data.copy()
        default_values.update(output_data)
        self.default_inputs = default_values

    def __filter_jacobian_in_outputs(self) -> None:
        """Filter jacobians in outputs names.

        This function is applied when _is_jac_returned_by_func is True.
        In such case, the function extracts the jacobian component from the list of
        output names returned by the matlab function. It thus fills
        _jac_output_names attributes as well as _jac_output_indices which
        corresponds to indices of jacobian component in the list of outputs returned
        by the matlab function.

        After applying this function, _outputs attribute no longer contains jacobian
        output names but only standard outputs.

        In order to filter jacobian component, this function just checks that
        jacobian names are prefixed by 'jac_'.
        """
        output_names = list(self.__outputs)

        # select jacobian output and remove them from self.outputs
        for i, out_name in enumerate(self.__outputs):
            if out_name[0:4] == self.JAC_PREFIX:
                self.__jac_output_names.append(out_name)
                self.__jac_output_indices.append(i)
                output_names.remove(out_name)

        # here self.outputs only contains output responses (no jacobian)
        self.__outputs = output_names

    def __reorder_and_check_jacobian_consistency(self) -> None:
        """This function checks jacobian output consistency.

        This function is used when _is_jac_returned_by_func is True.

        The function is called after calling jacobian filtering
        :meth:`.MatlabDiscipline._filter_jacobian_in_outputs`. It enables to:
        * check that all outputs have a jacobian matrix with respect to all inputs;
        * reorder the list of jacobian names (and indices) following the order
          from iterating over outputs then inputs lists;

        In order to check that all jacobian components exist, the function
        uses the conventional naming described in
        :meth:`.MatlabDiscipline._get_jac_name`.

        Raises:
            ValueError:
                * If the number of jacobian outputs is wrong;
                * If a specific jacobian output has the wrong name.
        """
        conventional_jac_names = self.__get_conventional_jac_names()
        new_indices = [-1] * len(conventional_jac_names)

        if len(conventional_jac_names) != len(self.__jac_output_names):
            raise ValueError(
                "The number of jacobian outputs does "
                "not correspond to what it should be. "
                "Make sure that all outputs have a jacobian "
                "matrix with respect to inputs."
            )

        not_found = []
        for i, name in enumerate(conventional_jac_names):
            try:
                idx = self.__jac_output_names.index(name)
                new_indices[i] = self.__jac_output_indices[idx]
            except ValueError:
                not_found.append(name)

        if not_found:
            raise ValueError(
                "Jacobian terms {} are not found in the "
                "list of conventional names. It is reminded that "
                "jacobian terms' name should be "
                "such as 'jac_dout_din'".format(not_found)
            )

        self.__jac_output_names = conventional_jac_names
        self.__jac_output_indices = new_indices

    def __get_conventional_jac_names(self) -> list[str]:
        """Return the list of jacobian names following the conventional naming.

        The conventional naming is described in :meth:`.MatlabDiscipline._get_jac_name`.
        """
        return [
            self.__get_jac_name(out_var, in_var)
            for out_var in self.__outputs
            for in_var in self.__inputs
        ]

    def __get_jac_name(
        self,
        out_var: str,
        in_var: str,
    ) -> str:
        """Return the name of jacobian given input and ouput variables.

        The conventional naming of jacobian component is the following:
        if outputs have any names ``out_1``, ``out_2``...
        and inputs are ``in_1``, ``in_2``... Therefore, names of jacobian
        components returned
        by the matlab function must be: ``jac_dout_1_din_1``,
        ``jac_dout_1_din_2``,
        ``jac_dout_2_din_1``, ``jac_dout_2_din_2``... which means
        that the names must be prefixed by ``jac_``, and followed by
        ``doutput`` and ``dinput`` seperated by ``_``.

        Args:
            out_var: The output variable name.
            in_var: The input variable name.

        Returns:
            The jacobian matrix name of output with respect to input.
        """
        return str(
            "{prefix}d{outv}_d{inv}".format(
                prefix=self.JAC_PREFIX, outv=out_var, inv=in_var
            )
        )

    def check_input_data(
        self,
        input_data: Mapping[str, np.ndarray],
        raise_exception: bool = True,
    ) -> None:  # noqa: D102
        if self.__check_opt_data:
            super().check_input_data(input_data, raise_exception=raise_exception)

    def check_output_data(self, raise_exception: bool = True) -> None:
        if self.__check_opt_data:
            super().check_output_data(raise_exception=raise_exception)

    def _run(self) -> None:
        """Run the Matlab discipline.

        If jacobian values are returned by the matlab function, they are filtered and
        used in order to fill :attr:`.MatlabDiscipline.jac`.

        Raises:
            ValueError:
                * If the execution of the matlab function fails.
                * If the size of the jacobian output matrix is wrong.
        """
        # import pudb;pudb.set_trace()
        input_vals = self.get_input_data()
        list_of_values = [input_vals.get(k) for k in self.__inputs if k in input_vals]

        try:
            out_vals = self.__engine.execute_function(
                self.__fct_name,
                *list_of_values,
                nargout=len(self.__outputs) + len(self.__jac_output_names),
            )

        except matlab.engine.MatlabExecutionError:
            LOGGER.error("Discipline: %s execution failed", self.name)
            raise

        # filter output values if jacobian is returned
        jac_vals = []

        if self.__is_jac_returned_by_func:
            out_vals = np.array(out_vals, dtype=object)
            jac_vals = [out_vals[idx] for idx in self.__jac_output_indices]
            out_vals = np.delete(out_vals, self.__jac_output_indices)
            # --> now out_vals only contains output responses (no jacobian)

        if self.cleaning_interval is not None:
            if self._n_calls.value % self.cleaning_interval == 0:
                self.__engine.execute_function("clear", "all", nargout=0)
                LOGGER.info(
                    "MATLAB cache cleaned: Discipline called %s times",
                    self._n_calls.value,
                )

        out_names = self.__outputs

        if len(out_names) == 1:
            self.store_local_data(**{out_names[0]: double2array(out_vals)})
        else:
            for out_n, out_v in zip(out_names, out_vals):
                self.store_local_data(**{out_n: double2array(out_v)})

        if not self.__is_size_known:
            for i, var in enumerate(self.__inputs):
                self.__inputs_size[var] = len(list_of_values[i])
            for var in self.__outputs:
                self.__outputs_size[var] = len(self.local_data[var])
            self.__is_size_known = True

        if self.__is_jac_returned_by_func:
            # fill jac dict
            self._init_jacobian()
            cpt = 0
            for out_name in self.__outputs:
                self.jac[out_name] = {}
                for in_name in self.__inputs:
                    self.jac[out_name][in_name] = np.atleast_2d(jac_vals[cpt])

                    if self.jac[out_name][in_name].shape != (
                        self.__outputs_size[out_name],
                        self.__inputs_size[in_name],
                    ):
                        raise ValueError(
                            "Jacobian term 'jac_d{}_d{}' "
                            "has the wrong size {} whereas it should "
                            "be {}.".format(
                                out_name,
                                in_name,
                                self.jac[out_name][in_name].shape,
                                (
                                    self.__outputs_size[out_name],
                                    self.__inputs_size[in_name],
                                ),
                            )
                        )

                    cpt += 1

            self._is_linearized = True

    @staticmethod
    def __update_data(
        data: Mapping[str, Any],
        other_data: Mapping[str, Any],
    ) -> Mapping[str]:
        """Update the values of a data mapping without adding new data names.

        Args:
            data: The data to be updated.
            other_data: The data to update ``data``.

        Returns:
            The updated data.
        """
        for key, value in other_data.items():
            if key in data.keys():
                data[key] = value

        return data

    def save_data_to_matlab(self, file_path: str | Path) -> None:
        """Save local data to matlab .mat format.

        Args:
            file_path: The path where to save the file.
        """
        file_path = Path(file_path)
        save_matlab_file(self.local_data, file_path=file_path)
        msg = "Local data of discipline {} exported to {}.mat successfully.".format(
            self.name, file_path.name
        )
        LOGGER.info(msg)

    @property
    def cleaning_interval(self) -> int:
        """Get and/or set the flushing interval for matlab disciplines."""
        return self.__cleaning_interval

    @cleaning_interval.setter
    def cleaning_interval(self, cleaning_interval):
        if cleaning_interval is not None:
            is_integer = cleaning_interval % 1 == 0
            if not is_integer:
                raise ValueError(
                    "The parameter 'cleaning_interval' argument must be an integer."
                )
        self.__cleaning_interval = cleaning_interval
