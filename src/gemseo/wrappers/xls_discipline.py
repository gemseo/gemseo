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
#    INITIAL AUTHORS - initial API and implementation and/or
#                      initial documentation
#        :author:  Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES

"""Excel based discipline."""

from __future__ import division, unicode_literals

import atexit
import os
import shutil
import tempfile
from typing import Any, List, Mapping, Optional
from uuid import uuid4

from numpy import array

from gemseo.core.discipline import MDODiscipline
from gemseo.utils.py23_compat import Path

cwd = Path.cwd()
try:
    import xlwings
except ImportError:
    # error will be reported if the discipline is used
    os.chdir(str(cwd))
    xlwings = None


class XLSDiscipline(MDODiscipline):
    """Wraps an excel workbook into a discipline.

    .. warning::

        As this wrapper relies on the `xlswings library <https://www.xlwings.org>`__
        to handle macros and interprocess communication, it
        is only working under Windows and MacOS.
    """

    _ATTR_TO_SERIALIZE = MDODiscipline._ATTR_TO_SERIALIZE + (
        "_xls_file_path",
        "input_names",
        "output_names",
        "macro_name",
        "_copy_xls_at_setstate",
    )

    def __init__(
        self,
        xls_file_path,  # type: str
        macro_name="execute",  # type: Optional[str]
        copy_xls_at_setstate=False,  # type: bool
    ):  # type: (...) -> None
        """Initialize xls file path and macro.

        Inputs must be specified in the "Inputs" sheet, in the following
        format (A and B are the first two columns).

        +---+---+
        | A | B |
        +===+===+
        | a | 1 |
        +---+---+
        | b | 2 |
        +---+---+

        Where a is the name of the first input, and 1 is its default value
        b is the name of the second one, and 2 its default value.

        There must be no empty lines between the inputs.


           The number of rows is arbitrary but they must be contiguous and
           start at line 1.

           The same applies for the "Outputs" sheet
           (A and B are the first two columns).


        +---+---+
        | A | B |
        +===+===+
        | c | 3 |
        +---+---+

        Where c is the only output. They may be multiple.

           If the file is a .xlsm, a macro named "execute" *must* exist
           and will be called by the _run method before retrieving
           the outputs. The macro has no arguments, it takes its inputs in the
           Inputs sheet, and write the outputs to the "Outputs" sheet.

           Alternatively, the user may provide a macro name to the constructor,
           or None if no macro shall be executed.

        Args:
            xls_file_path: The path to the excel file. If the file is a `.xlsm`,
                a macro named "execute" must exist and will be called by the _run
                method before retrieving the outputs.
            macro_name: The name of the macro to be executed for a `.xlsm` file.
                If None is provided, do not run a macro.
            copy_xls_at_setstate: If True, create a copy of the original Excel file
                for each of the pickled parallel processes. This option is required
                to be set to True for parallelization in Windows platforms.

        Raises:
            ImportError: If `xlwings` cannot be imported.
        """
        if xlwings is None:
            raise ImportError("cannot import xlwings")
        super(XLSDiscipline, self).__init__()
        self._xls_file_path = Path(xls_file_path)
        self._xls_app = None
        self.macro_name = macro_name
        self.input_names = None
        self.output_names = None
        self._book = None
        self._copy_xls_at_setstate = copy_xls_at_setstate

        self.__init_workbook()
        self._init_grammars()
        self._init_defaults()
        self.re_exec_policy = self.RE_EXECUTE_DONE_POLICY

    def __init_workbook(self):  # type: (...) -> None
        """Initialize a workbook.

        Raises:
            ValueError: If there is no sheet in the Excel file
                named "Inputs" or if there is no sheet named
                "Outputs".
        """

        try:
            self._xls_app = xlwings.App(visible=False)
            self._xls_app.interactive = False
        # wide except because I cannot tell what is the exception raised by xlwings
        except:  # noqa: E722,B001
            raise RuntimeError("xlwings requires Microsoft Excel")

        # Close the app when exiting
        atexit.register(self._xls_app.quit)

        self._book = self._xls_app.books.open(str(self._xls_file_path))
        sh_names = [sheet.name for sheet in self._book.sheets]

        if "Inputs" not in sh_names:
            raise ValueError(
                "Workbook must contain a sheet named 'Inputs' "
                "that define the inputs of the discipline"
            )
        if "Outputs" not in sh_names:
            raise ValueError(
                "Workbook must contain a sheet named 'Outputs' "
                "that define the outputs of the discipline"
            )

    def __setstate__(
        self, state  # type: Mapping[str, Any]
    ):  # type: (...) -> None
        super(XLSDiscipline, self).__setstate__(state)
        self._book = None
        self._xls_app = None
        if self._copy_xls_at_setstate:
            temp_dir = Path(tempfile.gettempdir())
            temp_path = temp_dir / self._xls_file_path.name.replace(
                ".xls", str(uuid4()) + ".xls"
            )
            shutil.copy2(str(self._xls_file_path), str(temp_path))
            self._xls_file_path = temp_path
        self.__init_workbook()

    def get_attributes_to_serialize(self):  # type: (...) -> List[str]
        """Overload pickle method to define which attributes are to be serialized.

        Returns:
             The attributes to serialize.
        """
        base = super(XLSDiscipline, self).get_attributes_to_serialize()
        base += [
            "_xls_file_path",
            "input_names",
            "output_names",
            "macro_name",
            "_copy_xls_at_setstate",
        ]
        return base

    def __read_sheet_col(
        self,
        sheet_name,  # type: str
        column=0,  # type: int
    ):  # type: (...) -> List[List[str], List[str], List[float], List[float]]
        """Read a specific column of the sheet.

        Args:
            sheet_name: The name of the sheet to be read.
            column: The number of the column to be read.

        Returns:
            The column values.
        """
        sht = self._book.sheets[sheet_name]
        i = 0
        value = sht[i, column].value
        values = []
        while value is not None:
            values.append(value)
            i += 1
            value = sht[i, column].value
        return values

    def _init_grammars(self):  # type: (...) -> None
        """Initialize grammars by parsing the Inputs and Outputs sheets."""
        self.input_names = self.__read_sheet_col("Inputs", 0)
        self.output_names = self.__read_sheet_col("Outputs", 0)
        self.input_grammar.initialize_from_data_names(self.input_names)
        self.output_grammar.initialize_from_data_names(self.output_names)

    def _init_defaults(self):  # type: (...) -> None
        """Initialize the default input values.

        Raises:
            ValueError: If the "Inputs" sheet does not have the same number of
                entries in the name column and the value column.
        """
        inputs = self.__read_sheet_col("Inputs", 1)
        if len(inputs) != len(self.input_names):
            msg = (
                "Inconsistent Inputs sheet, names (first columns) and"
                " values column (second) must be of the same length"
            )
            raise ValueError(msg)

        self.default_inputs = {k: array([v]) for k, v in zip(self.input_names, inputs)}

    def __write_inputs(
        self, input_data  # type: Mapping[str, float]
    ):  # type: (...) -> None
        """Write the inputs values to the Inputs sheet."""
        sht = self._book.sheets["Inputs"]
        for i, key in enumerate(self.input_names):
            sht[i, 1].value = input_data[key][0]

    def _run(self):  # type: (...) -> None
        """Run the discipline.

        Eventually calls the execute macro.

        Raises:
            RuntimeError: If the macro fails to be executed.
            ValueError: If the "Outputs" sheet does not have the same number of
                entries in the name column and the value column.
        """
        self.__write_inputs(self.local_data)
        if self._xls_file_path.match("*.xlsm") and self.macro_name is not None:
            try:
                self._xls_app.api.Application.Run(self.macro_name)
            except Exception as err:
                macro_name = self.macro_name
                msg = "Failed to run '{}' macro: {}.".format(macro_name, err)
                raise RuntimeError(msg)
        out_vals = self.__read_sheet_col("Outputs", 1)
        if len(out_vals) != len(self.output_names):
            msg = (
                "Inconsistent Outputs sheet, names (first columns) and "
                "values column (second) must be of the same length."
            )
            raise ValueError(msg)

        outputs = {k: array([v]) for k, v in zip(self.output_names, out_vals)}
        self.store_local_data(**outputs)
