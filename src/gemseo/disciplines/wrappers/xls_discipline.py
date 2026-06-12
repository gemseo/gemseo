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

import contextlib
import os
import random
import shutil
import tempfile
import threading
import time
import weakref
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Final
from uuid import uuid4

from numpy import array

from gemseo.core.discipline import Discipline
from gemseo.utils.platform import PLATFORM_IS_WINDOWS

if TYPE_CHECKING:
    from collections.abc import Mapping

    from numpy import ndarray

    from gemseo.typing import StrKeyMapping

# A failed `import xlwings` has historically been observed to leave the process
# in a different working directory. Capture the cwd before the import and
# restore it on failure so importing this module is a no-op for the caller.
_cwd_before_xlwings_import = Path.cwd()
try:
    import xlwings
except ImportError:
    # error will be reported if the discipline is used
    os.chdir(str(_cwd_before_xlwings_import))
    xlwings = None
del _cwd_before_xlwings_import

# pythoncom is only available on Windows; xlwings itself also runs on macOS.
if xlwings is not None and PLATFORM_IS_WINDOWS:
    import pythoncom
else:
    pythoncom = None


_APP_CREATION_MAX_ATTEMPTS: Final[int] = 3
_APP_CREATION_RETRY_BACKOFF: Final[float] = 1.0
"""Seconds to wait between retries; scaled by the attempt index."""

_APP_CREATION_RETRY_JITTER: Final[float] = 0.5
"""Upper bound of the random delay added to each retry backoff.

De-synchronizes the retries of parallel worker processes that collided on
their first launch attempt and would otherwise collide again.
"""

_APP_CREATION_LOCK: Final[threading.Lock] = threading.Lock()
"""Serializes in-process `xlwings.App` launches.

Concurrent `CoCreateInstanceEx` calls from multiple threads regularly fail
with `Server execution failed` (HRESULT 0x80080005); serializing them
prevents that contention instead of recovering from it.
"""


def _create_xls_app() -> object:
    """Launch a hidden xlwings `App` configured for manual calculation.

    In-process launches are serialized by `_APP_CREATION_LOCK` so threads
    never race `CoCreateInstanceEx`. Launches from parallel worker processes
    (parallel DOE) can still collide with `Server execution failed`
    (HRESULT 0x80080005); a short retry loop with linear backoff and random
    jitter absorbs that contention. The app whose configuration fails
    mid-launch is killed before retrying so the Excel process is not leaked.

    Returns:
        A hidden, non-interactive xlwings `App` in manual calculation mode.

    Raises:
        RuntimeError: If every attempt fails. Chained from the last error
            raised by xlwings.
    """
    last_err: BaseException | None = None
    for attempt in range(_APP_CREATION_MAX_ATTEMPTS):
        app: object | None = None
        try:
            with _APP_CREATION_LOCK:
                app = xlwings.App(visible=False)
            app.interactive = False
            # Manual mode: recalculate cells only when explicitly requested.
            app.calculation = "manual"
        except BaseException as err:  # noqa: BLE001, PERF203
            last_err = err
            _kill_xls_app_silently(app)
            if not isinstance(err, Exception):
                raise
            if attempt < _APP_CREATION_MAX_ATTEMPTS - 1:
                time.sleep(
                    _APP_CREATION_RETRY_BACKOFF * (attempt + 1)
                    + random.uniform(0.0, _APP_CREATION_RETRY_JITTER)  # noqa: S311
                )
            continue
        return app
    msg = "xlwings requires Microsoft Excel"
    raise RuntimeError(msg) from last_err


def _kill_xls_app_silently(app: object) -> None:
    """Kill an xlwings `App`, swallowing every error.

    Used as a `weakref.finalize` callback for `XLSDiscipline` and from its
    cleanup paths. By the time it runs the app may already be dead (e.g.,
    killed by an earlier reset); raising would surface as an unraisable
    warning at interpreter shutdown or mask the original exception.

    Args:
        app: The xlwings `App` instance to kill.
    """
    if app is None:
        return
    with contextlib.suppress(Exception):
        app.kill()


def _remove_silently(path: Path) -> None:
    """Delete `path` if it still exists, swallowing OS errors.

    Used as a `weakref.finalize` callback, where raising would surface as an
    unraisable warning at interpreter shutdown.

    Args:
        path: The file to remove.
    """
    with contextlib.suppress(OSError):
        path.unlink(missing_ok=True)


def _copy_xls_to_temp(src_path: Path, owner: object) -> Path:
    """Copy `src_path` to a unique temp location tied to `owner`'s lifetime.

    A `weakref.finalize` is registered against `owner` so the temp copy is
    removed when the owner is garbage collected or at interpreter exit.

    Args:
        src_path: The workbook to copy.
        owner: The object whose lifetime governs the temp copy.

    Returns:
        The path of the temp copy.
    """
    temp_path = (
        Path(tempfile.gettempdir()) / f"{src_path.stem}_{uuid4()}{src_path.suffix}"
    )
    shutil.copy2(str(src_path), str(temp_path))
    weakref.finalize(owner, _remove_silently, temp_path)
    return temp_path


class XLSDiscipline(Discipline):
    """Wraps an Excel workbook into a discipline.

    Warning:
        As this wrapper relies on the [xlwings library](https://www.xlwings.org)
        to handle macros and interprocess communication, it
        is only working under Windows and macOS.
    """

    _ATTR_NOT_TO_SERIALIZE = Discipline._ATTR_NOT_TO_SERIALIZE.union([
        "_xls_app",
        "_book",
    ])

    def __init__(
        self,
        xls_file_path: Path | str,
        name: str = "",
        macro_name: str = "execute",
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
           or `None` if no macro shall be executed.

        Args:
            xls_file_path: The path to the Excel file. If the file is a XLSM,
                a macro named "execute" must exist and will be called by the
                `XLSDiscipline._run()` method before retrieving the outputs.
            macro_name: The name of the macro to be executed for a XLSM file.
                If empty, do not run a macro.
            copy_xls_at_setstate: If `True`, create a copy of the original Excel file
                for each of the pickled parallel processes. This option is required
                to be set to `True` for parallelization in Windows platforms.
            recreate_book_at_run: Whether to rebuild the xls objects at each `_run`
                call.

        Raises:
            ImportError: If `xlwings` cannot be imported.
        """
        if xlwings is None:
            msg = "cannot import xlwings"
            raise ImportError(msg)
        super().__init__(name)
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
        try:
            self.__create_book(quit_xls_at_exit=quit_xls_at_exit)
            self._init_grammars()
            self._init_defaults()
        except BaseException:
            # Without quit_xls_at_exit no finalizer was registered;
            # kill the app here so the Excel process is not leaked.
            self.__reset_xls_objects()
            raise
        if recreate_book_at_run or copy_xls_at_setstate:
            self.__reset_xls_objects()

    def __reset_xls_objects(self) -> None:
        """Close the xls app and set `_xls_app` and `_book` to `None`.

        The kill is silent: this method runs from cleanup paths (e.g. the
        `finally` clause of `_run`) where a COM error raised by an already
        dead app would mask the original exception. Uses `getattr` so that
        the call is safe on an instance whose `__init__` failed before
        binding `_xls_app` (e.g. when `xlwings` is missing).
        """
        if getattr(self, "_xls_app", None) is not None:
            _kill_xls_app_silently(self._xls_app)
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
        self._xls_app = _create_xls_app()

        # In multiprocessing or sequential execution, Excel closes in each process.
        # Each process keeps its own _xls_app instance from init to end.
        # Use weakref.finalize rather than atexit so the discipline can still be
        # garbage collected during the run; the App captured here is killed when
        # the discipline dies or, failing that, at interpreter exit.
        if quit_xls_at_exit:
            weakref.finalize(self, _kill_xls_app_silently, self._xls_app)

        self._book = self._xls_app.books.open(str(self._xls_file_path))
        sh_names = [sheet.name for sheet in self._book.sheets]

        if "Inputs" not in sh_names:
            msg = (
                "Workbook must contain a sheet named 'Inputs' "
                "that define the inputs of the discipline"
            )
            raise ValueError(msg)

        if "Outputs" not in sh_names:
            msg = (
                "Workbook must contain a sheet named 'Outputs' "
                "that define the outputs of the discipline"
            )
            raise ValueError(msg)

    def __setstate__(self, state: StrKeyMapping) -> None:
        super().__setstate__(state)
        # If the book is recreated at _run, there is no need to create one for each
        # process.
        if self._copy_xls_at_setstate and not self._recreate_book_at_run:
            self._xls_file_path = _copy_xls_to_temp(self._xls_file_path, self)
            self.__create_book()

    def __read_sheet_col(
        self,
        sheet_name: str,
        column: int = 0,
    ) -> list[str | float | None]:
        """Read a specific column of the sheet.

        Args:
            sheet_name: The name of the sheet to be read.
            column: The number of the column to be read.

        Returns:
            The column values.
        """
        sht = self._book.sheets[sheet_name]

        return sht[0, column].expand("down").options(ndim=1).value

    def __build_data_dict(
        self,
        names: list[str],
        values: list[float | None],
    ) -> tuple[dict[str, ndarray], list[int]]:
        """Build the data dictionary while listing Excel rows where errors are found.

        A `None` value is interpreted as an Excel error in that cell.

        Args:
            names: The variable names.
            values: The values read from the Excel sheet.

        Returns:
            The data dictionary and the list of Excel rows with None values.
        """
        data_dict = {}
        error_rows = []
        for row, (k, v) in enumerate(
            zip(names, values, strict=False),
            start=1,
        ):
            data_dict[k] = array([v])
            if v is None:
                error_rows.append(row)

        return data_dict, error_rows

    def _init_grammars(self) -> None:
        """Initialize grammars by parsing the Inputs and Outputs sheets."""
        self.input_names = self.__read_sheet_col("Inputs")
        self.output_names = self.__read_sheet_col("Outputs")
        self.io.input_grammar.update_from_names(self.input_names)
        self.io.output_grammar.update_from_names(self.output_names)

    def _init_defaults(self) -> None:
        """Initialize the default input values.

        Raises:
            ValueError: If the "Inputs" sheet does not have the same number of
                entries in the name column and the value column, or if the value column
                contains Excel errors.
        """
        inputs = self.__read_sheet_col("Inputs", 1)
        if len(inputs) != len(self.input_names):
            msg = (
                "Inconsistent Inputs sheet, names (first columns) and"
                " values column (second) must be of the same length"
            )
            raise ValueError(msg)

        input_defaults, error_rows = self.__build_data_dict(self.input_names, inputs)
        if error_rows:
            msg = (
                "Inputs sheet contains Excel errors in the second column at rows "
                f"{error_rows}"
            )
            raise ValueError(msg)
        self.io.input_grammar.defaults = input_defaults

    def __write_inputs(self, input_data: Mapping[str, ndarray]) -> None:
        """Write the input values to the Inputs sheet.

        Each value is written to the row matching the position of its name in
        the ``Inputs`` sheet (column A), not to the iteration order of
        ``input_data``.

        Args:
            input_data: The input values, keyed by name.
        """
        sht = self._book.sheets["Inputs"]
        for i, name in enumerate(self.input_names):
            input_value = input_data.get(name)
            if input_value is not None:
                sht[i, 1].value = input_value[0]

    def _run(self, input_data: StrKeyMapping) -> StrKeyMapping | None:
        """Run the discipline.

        Eventually calls the execute macro.

        Args:
            input_data: The input values, keyed by name.

        Returns:
            The output values read from the Outputs sheet, keyed by name.

        Raises:
            RuntimeError: If the macro fails to be executed.
            ValueError: If the "Outputs" sheet does not have the same number of
                entries in the name column and the value column, or if the value column
                contains Excel errors.
        """
        # If threading, the run method is called from different threads.
        # But it is not possible to pass xlwings objects between threads.
        # Since CoInitialize was implicitly called at init but has not been called
        # inside each thread, a call is made here.
        # We then initialize the workbook again to run the computation inside each
        # thread.
        # In this case, the Excel process is closed at the end of this _run method.
        if self._recreate_book_at_run and pythoncom is not None:
            pythoncom.CoInitialize()

        try:
            if self._recreate_book_at_run:
                # Inside the try block so that a workbook opening failure
                # also resets the Excel objects in the finally clause.
                self.__create_book(quit_xls_at_exit=False)

            self.__write_inputs(input_data)

            # Explicitly call calculate() because the workbook is in manual mode.
            self._xls_app.calculate()

            if self._xls_file_path.match("*.xlsm") and self.macro_name:
                try:
                    self._xls_app.api.Application.Run(self.macro_name)
                except Exception as err:
                    msg = f"Failed to run '{self.macro_name}' macro: {err}."
                    raise RuntimeError(msg) from err

            out_vals = self.__read_sheet_col("Outputs", 1)

            if len(out_vals) != len(self.output_names):
                msg = (
                    "Inconsistent Outputs sheet, names (first columns) and "
                    "values column (second) must be of the same length."
                )
                raise ValueError(msg)

            output_data, error_rows = self.__build_data_dict(
                self.output_names, out_vals
            )
            if error_rows:
                msg = (
                    "Outputs sheet contains Excel errors in the second column at "
                    f"rows {error_rows}"
                )
                raise ValueError(msg)
        finally:
            # When using threads, each computation is made with a unique `_xls_app`.
            # If we do not quit here, we lose the reference and the process ends up
            # hung. The reset must also happen on failure so that a raised exception
            # does not leak the Excel process.
            if self._recreate_book_at_run:
                self.__reset_xls_objects()
                if pythoncom is not None:
                    # Balance the CoInitialize() call above, after the COM
                    # proxies have been released by __reset_xls_objects(),
                    # so the apartment reference count does not grow on
                    # long-lived worker threads.
                    pythoncom.CoUninitialize()

        return output_data
