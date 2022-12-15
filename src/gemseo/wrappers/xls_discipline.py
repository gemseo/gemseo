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
#        :author:  Francois Gallard, Gilberto Ruiz
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Excel based discipline."""
from __future__ import annotations

import atexit
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any
from typing import Mapping
from uuid import uuid4

from numpy import array

from gemseo.core.discipline import MDODiscipline

if sys.platform == "win32":
    import pythoncom

cwd = Path.cwd()
try:
    import xlwings
except ImportError:
    # error will be reported if the discipline is used
    os.chdir(str(cwd))
    xlwings = None


class XLSDiscipline(MDODiscipline):
    """Wraps an Excel workbook into a discipline.

    .. warning::

        As this wrapper relies on the `xlwings library <https://www.xlwings.org>`__
        to handle macros and interprocess communication, it
        is only working under Windows and macOS.
    """

    _ATTR_TO_SERIALIZE = MDODiscipline._ATTR_TO_SERIALIZE + (
        "_xls_file_path",
        "input_names",
        "output_names",
        "macro_name",
        "_copy_xls_at_setstate",
        "_recreate_book_at_run",
    )

    def __init__(
        self,
        xls_file_path: Path | str,
        macro_name: str | None = "execute",
        copy_xls_at_setstate: bool = False,
        recreate_book_at_run: bool = False,
    ) -> None:
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


           The number of rows is arbitrary, but they must be contiguous and
           start at line 1.

           The same applies for the "Outputs" sheet
           (A and B are the first two columns).


        +---+---+
        | A | B |
        +===+===+
        | c | 3 |
        +---+---+

        Where c is the only output. They may be multiple.

           If the file is a XLSM, a macro named "execute" *must* exist
           and will be called by the _run method before retrieving
           the outputs. The macro has no arguments, it takes its inputs in the
           Inputs sheet, and write the outputs to the "Outputs" sheet.

           Alternatively, the user may provide a macro name to the constructor,
           or ``None`` if no macro shall be executed.

        Args:
            xls_file_path: The path to the Excel file. If the file is a XLSM,
                a macro named "execute" must exist and will be called by the
                :meth:`~.XLSDiscipline._run` method before retrieving the outputs.
            macro_name: The name of the macro to be executed for a XLSM file.
                If ``None`` is provided, do not run a macro.
            copy_xls_at_setstate: If ``True``, create a copy of the original Excel file
                for each of the pickled parallel processes. This option is required
                to be set to ``True`` for parallelization in Windows platforms.
            recreate_book_at_run: Whether to rebuild the xls objects at each ``_run``
                call.

        Raises:
            ImportError: If ``xlwings`` cannot be imported.
        """
        if xlwings is None:
            raise ImportError("cannot import xlwings")
        super().__init__()
        self._xls_file_path = Path(xls_file_path)
        self._xls_app = None
        self.macro_name = macro_name
        self.input_names = None
        self.output_names = None
        self._book = None
        self._copy_xls_at_setstate = copy_xls_at_setstate
        self._recreate_book_at_run = recreate_book_at_run

        # A workbook init creates an instance of `_xls_app`.
        # It opens an Excel process and calls CoInitialize().
        # This must be done prior to initializing grammars and defaults.
        # In serial mode, the initial book is always called with quit_xls_at_exit=True.
        # In multiprocessing or multithreading,
        # the book is closed once grammars and default values have been initialized.
        quit_xls_at_exit = not (recreate_book_at_run or copy_xls_at_setstate)
        self.__create_book(quit_xls_at_exit=quit_xls_at_exit)
        self._init_grammars()
        self._init_defaults()
        self.re_exec_policy = self.RE_EXECUTE_DONE_POLICY
        if recreate_book_at_run or copy_xls_at_setstate:
            self.__reset_xls_objects()

    def __reset_xls_objects(self) -> None:
        """Close the xls app and set `_xls_app` and `_book` to `None`."""
        if self._xls_app is not None:
            self._xls_app.kill()
            self._book = None
            self._xls_app = None

    def __create_book(self, quit_xls_at_exit: bool = True) -> None:
        """Create a book with xlwings.

        Args:
            quit_xls_at_exit: Whether to force excel to quit
                when the python process exits.

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

        # In multiprocessing or sequential execution, excel closes in each process.
        # Each process keeps its own _xls_app instance from init to end.
        # It is therefore possible to register the quit() call at exit.
        if quit_xls_at_exit:
            atexit.register(self.__reset_xls_objects)

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

    def __del__(self):
        self.__reset_xls_objects()

    def __setstate__(self, state: Mapping[str, Any]) -> None:
        super().__setstate__(state)
        # If the book is recreated at _run, there is no need to create one for each
        # process.
        if self._copy_xls_at_setstate and not self._recreate_book_at_run:
            temp_dir = Path(tempfile.gettempdir())
            temp_path = temp_dir / self._xls_file_path.name.replace(
                ".xls", str(uuid4()) + ".xls"
            )
            shutil.copy2(str(self._xls_file_path), str(temp_path))
            self._xls_file_path = temp_path
            self.__create_book()

    def __read_sheet_col(
        self,
        sheet_name: str,
        column: int = 0,
    ) -> list[list[str], list[str], list[float], list[float]]:
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

    def _init_grammars(self) -> None:
        """Initialize grammars by parsing the Inputs and Outputs sheets."""
        self.input_names = self.__read_sheet_col("Inputs")
        self.output_names = self.__read_sheet_col("Outputs")
        self.input_grammar.update(self.input_names)
        self.output_grammar.update(self.output_names)

    def _init_defaults(self) -> None:
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

    def __write_inputs(self, input_data: Mapping[str, float]) -> None:
        """Write the inputs values to the Inputs sheet."""
        sht = self._book.sheets["Inputs"]
        for i, key in enumerate(self.input_names):
            sht[i, 1].value = input_data[key][0]

    def _run(self) -> None:
        """Run the discipline.

        Eventually calls the execute macro.

        Raises:
            RuntimeError: If the macro fails to be executed.
            ValueError: If the "Outputs" sheet does not have the same number of
                entries in the name column and the value column.
        """
        # If threading, the run method is called from different threads.
        # But it is not possible to pass xlwings objects between threads.
        # Since CoInitialize was implicitly called at init but has not been called
        # inside each thread, a call is made here.
        # We then initialize the workbook again to run the computation inside each
        # thread.
        # In this case, the Excel process is closed at the end of this _run method.
        if self._recreate_book_at_run:
            pythoncom.CoInitialize()
            self.__create_book(quit_xls_at_exit=False)

        self.__write_inputs(self.local_data)

        if self._xls_file_path.match("*.xlsm") and self.macro_name is not None:
            try:
                self._xls_app.api.Application.Run(self.macro_name)
            except Exception as err:
                macro_name = self.macro_name
                msg = f"Failed to run '{macro_name}' macro: {err}."
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

        # When using threads, each computation is made with a unique `_xls_app`.
        # If we do not quit at this point, we loose the reference and
        # the process ends up hung.
        # Therefore, we close everything once we have stored all we need.
        # For this same reason, overloading __del__ is not an option.
        if self._recreate_book_at_run:
            self.__reset_xls_objects()
