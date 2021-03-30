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

from __future__ import absolute_import, division, unicode_literals

import atexit

from future import standard_library
from numpy import array

try:
    import xlwings
except ImportError:
    # error will be reported if the discipline is used
    xlwings = None

from gemseo.core.discipline import MDODiscipline

standard_library.install_aliases()


class XLSDiscipline(MDODiscipline):
    """Wraps an excel workbook into a discipline.

    .. warning::

        As this wrapper relies on the `xlswings library <https://www.xlwings.org>`__
        to handle macros and interprocess communication, it
        is only working under Windows and MacOS.
    """

    def __init__(self, xls_file_path, macro_name="execute"):
        """Initialize xls file path and macro.

        Inputs must be specified in the "Inputs" sheet, in the following
        format (A and B are the first two columns)

        +---+---+
        | A | B |
        +===+===+
        | a | 1 |
        +---+---+
        | b | 2 |
        +---+---+

        Where a is the name of the first input, and 1 is its default value
        b is the name of the second one, and 2 its default value.

        There must be no empty lines between the inputs


           The number of rows is arbitrary but they must be contiguous and
           start at line 1

           And same for the "Outputs" sheet (A and B are the first two columns)


        +---+---+
        | A | B |
        +===+===+
        | c | 3 |
        +---+---+

        Where c is the only output. There may be multiple.

           if the file is a .xlsm, a macro named "execute" *must* exist
           and will be called by the _run method before retrieving
           the outputs. The macro has no arguments, it takes its inputs in the
           Inputs sheet, and write the outputs to the "Outputs" sheet

           Alternatively, the user may provide a macro name to the constructor,
           or None if no macro shall be executed


        Parameters
        ----------
        xls_file_path : str
            path to the excel file
            if the file is a .xlsm, a macro named "execute" must exist
            and will be called by the _run method before retrieving
            the outputs
        macro_name : str
            name of the macro to be executed for a .xlsm file
            if None is provided, do not run a macro
        """
        if xlwings is None:
            raise ImportError("cannot import xlwings")

        self._book = None

        try:
            self._xls_app = xlwings.App(visible=False)
        # wide except because I cannot tell what is the exception raised by xlwings
        except:  # noqa: E722,B001
            raise RuntimeError("xlwings requires Microsoft Excel")

        super(XLSDiscipline, self).__init__()
        self._xls_file_path = xls_file_path
        self.macro_name = macro_name

        # Close the app when exiting
        atexit.register(self._xls_app.quit)

        self._book = xlwings.Book(xls_file_path)
        sh_names = [sheet.name for sheet in self._book.sheets]
        if "Inputs" not in sh_names:
            raise ValueError(
                "Workbook must contain a sheet named 'Inputs' "
                + "that define the inputs of the discipline"
            )
        if "Outputs" not in sh_names:
            raise ValueError(
                "Workbook must contain a sheet named 'Outputs' "
                + "that define the outputs of the discipline"
            )
        self.input_names = None
        self.output_names = None

        self._init_grammars()
        self._init_defaults()
        self.re_exec_policy = self.RE_EXECUTE_DONE_POLICY

    def close(self):
        """Close the workbook."""
        if self._book is not None:
            self._book.close()
            self._book = None

    def __del__(self):
        # ensures that the gc automatically release the resources held by the book
        self.close()

    def __read_sheet_col(self, sheet_name, column=0):
        """Read a specific column of the sheet."""
        sht = self._book.sheets[sheet_name]
        i = 0
        value = sht[i, column].value
        values = []
        while value is not None:
            values.append(value)
            i += 1
            value = sht[i, column].value

        return values

    def _init_grammars(self):
        """Initialize grammars by parsing the Inputs and Outputs sheets."""
        self.input_names = self.__read_sheet_col("Inputs", 0)
        self.output_names = self.__read_sheet_col("Outputs", 0)
        self.input_grammar.initialize_from_data_names(self.input_names)
        self.output_grammar.initialize_from_data_names(self.output_names)

    def _init_defaults(self):
        """Initialize the default input values."""
        inputs = self.__read_sheet_col("Inputs", 1)
        if len(inputs) != len(self.input_names):
            msg = (
                "Inconsistent Inputs sheet, names (first columns) and"
                " values column (second) must be of the same length"
            )
            raise ValueError(msg)

        self.default_inputs = dict(zip(self.input_names, inputs))

    def __write_inputs(self, input_data):
        """Write the inputs values to the Inputs sheet."""
        sht = self._book.sheets["Inputs"]
        for i, key in enumerate(self.input_names):
            sht[i, 1].value = input_data[key][0]

    def _run(self):
        """Run the discipline.

        Eventually calls the execute macro
        """
        self.__write_inputs(self.local_data)
        if self._xls_file_path.endswith(".xlsm") and self.macro_name is not None:
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
                " values column (second) must be of the same length."
            )
            raise ValueError(msg)

        outputs = {k: array([v]) for k, v in zip(self.output_names, out_vals)}
        self.store_local_data(**outputs)
