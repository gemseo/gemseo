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
#        :author: François Gallard : initial Scilab version
#        :author: Arthur Piat : conversion Scilab to Matlab
#        :author: Nicolas Roussouly: GEMSEO integration
"""Definition of the Matlab parser.

Overview
--------

This module contains the :class:`.MatlabParser`
which enables to parse Matlab files in order to automatically
detect inputs and outputs.
This class is basically used through the :class:`.MatlabDiscipline` class
in order to build a discipline based on the Matlab function.
"""
from __future__ import annotations

import logging
import re
from pathlib import Path

LOGGER = logging.getLogger(__name__)


class MatlabParser:
    """Parse Matlab file to identify inputs and outputs.

    Examples:
        >>> # Parse the matlab function "fucntion.m"
        >>> parser = MatlabParser("function.m")
        >>> print(parser.inputs)
        >>> print(parser.outputs)
    """

    RE_FILE_FMT = re.compile(r".*(\\.*)*\.m\b")
    RE_ENCRYPTED_FCT = re.compile(r".*(\\.*)*\.p\b")
    RE_OUTPUTS = re.compile(r"(\[(.*?)])|(function( *?)(.*?)=)")
    RE_FUNCTION = re.compile(r"=(.*?)\(")
    RE_ARGS = re.compile(r"\((.*?)\)")

    def __init__(self, full_path: str | None = None) -> None:

        """
        Args:
            full_path: The path to the matlab file.
                If ``None``, the user shall parse the file explicitly.
        """  # noqa: D205, D212, D415
        self.__inputs = None
        self.__outputs = None
        self.__fct_name = None
        self.__fct_dir = None

        if full_path is not None:
            self.parse(full_path)

    @property
    def function_name(self) -> str:
        """Return the name of the function."""
        return self.__fct_name

    @property
    def directory(self) -> Path:
        """Return the directory of the function."""
        return self.__fct_dir

    @property
    def inputs(self):
        """Return the inputs."""
        return self.__inputs

    @property
    def outputs(self):
        """Return the outputs."""
        return self.__outputs

    def __check_path(self, file_path: str | Path) -> None:
        """Check the format of the file.

        Args:
            file_path: The path to the matlab the file.

        Raises:
            IOError: If the matlab file does not exist.
            ValueError:
                * If the matlab file is encrypted;
                * If the matlab function is neither a script nor a function.
        """
        if not file_path.exists():
            raise OSError(
                "The function directory for Matlab "
                "sources {} does not exists.".format(str(file_path))
            )

        file_path = str(file_path)

        re_encrypted_file_groups = self.RE_ENCRYPTED_FCT.search(file_path)

        if re_encrypted_file_groups is not None:
            raise ValueError(
                "The given file {} is encrypted "
                "and cannot be parsed.".format(file_path)
            )

        re_file_groups = self.RE_FILE_FMT.search(file_path)

        if re_file_groups is None:
            raise ValueError(
                "The given file {} should "
                "either be a matlab function or script.".format(file_path)
            )

    def __parse_function_inputs_outputs(
        self,
        line: str,
        function_name: str,
    ) -> None:
        """Parse inputs and outputs.

        Args:
            line: The line containing the declaration of the function.
            function_name: The name of the function according to file.m.

        Raises:
            NameError:
                * If function has no name;
                * If function and file name are different.
            ValueError:
                * If function has no output;
                * If function has no input.
        """
        # TODO: refactor this function because some exceptions are not useful

        re_func_groups = self.RE_FUNCTION.search(line)

        if re_func_groups is None:
            raise NameError("Matlab function has no name.")

        fname = re_func_groups.group(0).strip()[1:-1].strip()

        if fname != function_name:
            raise NameError(
                "Function name {} does not match with file name {}.".format(
                    function_name, fname
                )
            )

        LOGGER.debug("Detected function: %s", fname)
        self.__fct_name = fname

        re_output_groups = self.RE_OUTPUTS.search(line)

        if re_output_groups is None:
            raise ValueError(f"Function {fname} has no output")

        arg_str = re_output_groups.group(0).strip()
        arg_str = arg_str.replace("[", "").replace("]", "")
        arg_str = arg_str.replace("function", "").replace(" ", "")
        arg_str = arg_str.replace("=", "")
        outs = arg_str.split(",")
        self.__outputs = [out_str.strip() for out_str in outs]
        LOGGER.debug("Outputs are: %s", outs)

        re_args_groups = self.RE_ARGS.search(line)
        if re_args_groups is None:
            raise ValueError(f"Function {fname} has no argument.")

        arg_str = re_args_groups.group(0).strip()[1:-1].strip()
        args = arg_str.split(",")
        self.__inputs = [args_str.strip() for args_str in args]
        LOGGER.debug("And arguments are: %s", args)

    def parse(self, path: str) -> None:
        """Parse a .m file in order to get inputs and outputs.

        Args:
            path: The path to the matlab file.

        Raises:
            ValueError: Raised if the file is not a matlab function.
        """
        path = Path(path).absolute()
        self.__check_path(path)
        self.__fct_dir = path.parent
        fct_name = path.stem

        is_parsed = False

        with path.open(errors="ignore") as file_handle:
            for line in file_handle.readlines():
                if line.strip().startswith("function"):
                    self.__parse_function_inputs_outputs(line, fct_name)
                    is_parsed = True
                    break

        if not is_parsed:
            raise ValueError(f"The given file {path} is not a matlab function.")
