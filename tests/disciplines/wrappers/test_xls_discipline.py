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
# INITIAL AUTHORS - initial API and implementation and/or
#                   initial documentation
#        :author:  Francois Gallard, Gilberto Ruiz
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import annotations

import gc
import shutil
import weakref
from pathlib import Path

import pytest
from numpy import array
from numpy import exp
from numpy import isclose
from numpy import ones

from gemseo import create_design_space
from gemseo import create_discipline
from gemseo import create_scenario
from gemseo.algos.doe.diagonal_doe.settings.diagonal_doe_settings import (
    DiagonalDOE_Settings,
)
from gemseo.core.parallel_execution.discipline_execution import DiscParallelExecution
from gemseo.disciplines.wrappers import xls_discipline
from gemseo.disciplines.wrappers.xls_discipline import XLSDiscipline
from gemseo.mda.chain_settings import MDAChain_Settings
from gemseo.mda.jacobi_settings import MDAJacobi_Settings
from gemseo.utils.testing.helpers import assert_exception

DIR_PATH = Path(__file__).parent
FILE_PATH_PATTERN = str(DIR_PATH / "test_excel_fail{}.xlsx")
INPUT_DATA = {"a": array([20.25]), "b": array([3.25])}


class _FakeBooks:
    """Stand-in for `App.books` whose `open` always fails."""

    def open(self, path: str) -> None:
        """Simulate a workbook that cannot be opened.

        Args:
            path: The workbook path (ignored).

        Raises:
            OSError: Always.
        """
        msg = "simulated workbook open failure"
        raise OSError(msg)


class _FakeApp:
    """Stand-in for an xlwings `App` recording whether it was killed."""

    def __init__(self, fail_with: type[BaseException] | None = None) -> None:
        """
        Args:
            fail_with: Exception type to raise when setting `interactive`,
                simulating a failure right after the app is created.
                ``None`` means no failure.
        """
        self.killed = False
        self.calculation = ""
        self.books = _FakeBooks()
        self._fail_with = fail_with

    @property
    def interactive(self) -> bool:
        """Whether the app is interactive."""
        return False

    @interactive.setter
    def interactive(self, value: bool) -> None:
        if self._fail_with is not None:
            msg = "simulated failure"
            raise self._fail_with(msg)

    def kill(self) -> None:
        """Record that the app was killed."""
        self.killed = True


def test_missing_xlwings(monkeypatch, snapshot) -> None:
    """Check the error when xlwings cannot be imported."""
    monkeypatch.setattr(xls_discipline, "xlwings", None)
    with assert_exception(ImportError, snapshot):
        XLSDiscipline("dummy_file_path")


def test_excel_unavailable(skip_if_xlwings_is_usable, monkeypatch, snapshot) -> None:
    """Check the error when xlwings is importable but Excel cannot be launched.

    Args:
        skip_if_xlwings_is_usable: Fixture to skip the test when xlwings is usable.
    """
    with assert_exception(RuntimeError, snapshot):
        monkeypatch.setattr(xls_discipline, "_APP_CREATION_MAX_ATTEMPTS", 1)
        monkeypatch.setattr(xls_discipline, "_APP_CREATION_RETRY_BACKOFF", 0.0)
        monkeypatch.setattr(xls_discipline, "_APP_CREATION_RETRY_JITTER", 0.0)
        XLSDiscipline("dummy_file_path")


def test_app_killed_when_configuration_fails(monkeypatch, snapshot) -> None:
    """The partially configured Excel app is killed inside `_create_xls_app`.

    Args:
        monkeypatch: pytest monkeypatch fixture.
        snapshot: syrupy snapshot fixture.
    """
    apps: list[_FakeApp] = []

    def make_app(*args, **kwargs):
        app = _FakeApp(fail_with=RuntimeError)
        apps.append(app)
        return app

    fake_xlwings = type("FakeXlwings", (), {"App": staticmethod(make_app)})
    monkeypatch.setattr(xls_discipline, "xlwings", fake_xlwings)
    monkeypatch.setattr(xls_discipline, "_APP_CREATION_MAX_ATTEMPTS", 1)
    monkeypatch.setattr(xls_discipline, "_APP_CREATION_RETRY_BACKOFF", 0.0)
    monkeypatch.setattr(xls_discipline, "_APP_CREATION_RETRY_JITTER", 0.0)
    with assert_exception(RuntimeError, snapshot):
        XLSDiscipline("dummy_file_path")
    assert apps
    assert all(app.killed for app in apps)


def test_app_killed_when_init_fails_without_finalizer(monkeypatch, snapshot) -> None:
    """The Excel app is killed when `__init__` fails after the app is created.

    With `copy_xls_at_setstate=True` no finalizer is registered at book
    creation, so a failure while opening the workbook must not leak the app.

    Args:
        monkeypatch: pytest monkeypatch fixture.
        snapshot: syrupy snapshot fixture.
    """
    app = _FakeApp()
    monkeypatch.setattr(xls_discipline, "xlwings", object())
    monkeypatch.setattr(xls_discipline, "_create_xls_app", lambda: app)
    with assert_exception(OSError, snapshot):
        XLSDiscipline("dummy_file_path", copy_xls_at_setstate=True)
    assert app.killed


@pytest.mark.parametrize("exception_type", [KeyboardInterrupt, SystemExit])
def test_create_xls_app_does_not_retry_base_exception(
    monkeypatch, exception_type, snapshot
) -> None:
    """KeyboardInterrupt / SystemExit raised by xlwings.App are re-raised immediately.

    Args:
        monkeypatch: pytest monkeypatch fixture.
        exception_type: The BaseException subclass to simulate.
        snapshot: syrupy snapshot fixture.
    """
    attempts = {"value": 0}

    def raising_app(*args, **kwargs):
        attempts["value"] += 1
        raise exception_type()

    fake_xlwings = type("FakeXlwings", (), {"App": staticmethod(raising_app)})
    monkeypatch.setattr(xls_discipline, "xlwings", fake_xlwings)
    monkeypatch.setattr(xls_discipline, "_APP_CREATION_MAX_ATTEMPTS", 3)

    with assert_exception(exception_type, snapshot):
        xls_discipline._create_xls_app()

    assert attempts["value"] == 1


@pytest.mark.parametrize("exception_type", [KeyboardInterrupt, SystemExit])
def test_create_xls_app_kills_app_on_base_exception(
    monkeypatch, exception_type, snapshot
) -> None:
    """A partially configured app is killed before a BaseException re-raises.

    Args:
        monkeypatch: pytest monkeypatch fixture.
        exception_type: The BaseException subclass to simulate.
        snapshot: syrupy snapshot fixture.
    """
    apps: list[_FakeApp] = []

    def make_app(*args, **kwargs):
        app = _FakeApp(fail_with=exception_type)
        apps.append(app)
        return app

    fake_xlwings = type("FakeXlwings", (), {"App": staticmethod(make_app)})
    monkeypatch.setattr(xls_discipline, "xlwings", fake_xlwings)
    monkeypatch.setattr(xls_discipline, "_APP_CREATION_MAX_ATTEMPTS", 3)

    with assert_exception(exception_type, snapshot):
        xls_discipline._create_xls_app()

    assert len(apps) == 1
    assert apps[0].killed


def test_basic(skip_if_xlwings_is_not_usable) -> None:
    """Simple test, the output is the sum of the inputs.

    Args:
        skip_if_xlwings_is_not_usable: Fixture to skip the test when xlwings is not
            usable.
    """
    xlsd = XLSDiscipline(DIR_PATH / "test_excel.xlsx")
    xlsd.execute(INPUT_DATA)
    assert xlsd.io.data["c"] == 23.5


def test_restricted_grammar(skip_if_xlwings_is_not_usable) -> None:
    """Test that a subset of the input grammar can be used.

    Args:
        skip_if_xlwings_is_not_usable: Fixture to skip the test when xlwings is not
            usable.
    """
    xlsd = XLSDiscipline(DIR_PATH / "test_excel.xlsx")
    xlsd.io.input_grammar.clear()
    xlsd.io.input_grammar.update_from_names("a")
    xlsd.execute(INPUT_DATA)
    assert xlsd.io.data["c"] == 23.25


@pytest.mark.parametrize("file_id", range(1, 4))
def test_error_init(skip_if_xlwings_is_not_usable, file_id, snapshot) -> None:
    """Test that errors are raised for files without the proper format.

    Args:
        skip_if_xlwings_is_not_usable: Fixture to skip the test when xlwings is not
            usable.
        file_id: The id of the test file.
        snapshot: syrupy snapshot fixture.
    """
    with assert_exception(ValueError, snapshot):
        XLSDiscipline(FILE_PATH_PATTERN.format(file_id))


def test_error_execute(skip_if_xlwings_is_not_usable, snapshot) -> None:
    """Check that an exception is raised for incomplete data.

    Args:
        skip_if_xlwings_is_not_usable: Fixture to skip the test when xlwings is not
            usable.
    """
    disc = XLSDiscipline(FILE_PATH_PATTERN.format(4))
    with assert_exception(ValueError, snapshot):
        disc.execute(INPUT_DATA)


def test_excel_error_defaults(skip_if_xlwings_is_not_usable, snapshot) -> None:
    """Check that an exception is raised if default inputs contain excel errors.

    Args:
        skip_if_xlwings_is_not_usable: Fixture to skip the test when xlwings is not
            usable.
    """
    with assert_exception(ValueError, snapshot):
        XLSDiscipline(FILE_PATH_PATTERN.format(5))


def test_excel_error_execute(skip_if_xlwings_is_not_usable, snapshot) -> None:
    """Check that an exception is raised if output data contain excel errors.

    Args:
        skip_if_xlwings_is_not_usable: Fixture to skip the test when xlwings is not
            usable.
    """
    disc = XLSDiscipline(FILE_PATH_PATTERN.format(6))
    with assert_exception(ValueError, snapshot):
        disc.execute(INPUT_DATA)


def test_copy_xls_to_temp_preserves_extension(tmp_path) -> None:
    """`_copy_xls_to_temp` must keep the suffix when the stem contains `.xls`.

    Args:
        tmp_path: pytest fixture providing a temporary directory.
    """
    src = tmp_path / "model.xls.backup.xlsx"
    src.write_bytes(b"fake content")

    class Owner:
        pass

    temp_path = xls_discipline._copy_xls_to_temp(src, Owner())
    assert temp_path.suffix == ".xlsx"
    assert temp_path.name.startswith("model.xls.backup")


def test_copy_xls_to_temp_cleans_up_on_gc(tmp_path) -> None:
    """The temp file copied by `_copy_xls_to_temp` is removed when its owner is GC'd.

    Args:
        tmp_path: pytest fixture providing a temporary directory.
    """
    src = tmp_path / "src.xlsx"
    src.write_bytes(b"fake content")

    class Owner:
        pass

    owner = Owner()
    temp_path = xls_discipline._copy_xls_to_temp(src, owner)
    assert temp_path.exists()
    assert temp_path != src

    del owner
    gc.collect()

    assert not temp_path.exists()


def test_app_creation_retries_on_transient_failure(
    skip_if_xlwings_is_not_usable, monkeypatch
) -> None:
    """Transient COM failures while launching Excel are retried.

    Args:
        skip_if_xlwings_is_not_usable: Fixture to skip the test when xlwings is not
            usable.
        monkeypatch: pytest monkeypatch fixture.
    """
    real_app = xls_discipline.xlwings.App
    attempts = {"value": 0}

    def flaky_app(*args, **kwargs):
        attempts["value"] += 1
        if attempts["value"] < 3:
            msg = "simulated transient COM failure"
            raise RuntimeError(msg)
        return real_app(*args, **kwargs)

    monkeypatch.setattr(xls_discipline.xlwings, "App", flaky_app)
    monkeypatch.setattr(xls_discipline, "_APP_CREATION_RETRY_BACKOFF", 0.0)
    monkeypatch.setattr(xls_discipline, "_APP_CREATION_RETRY_JITTER", 0.0)

    disc = XLSDiscipline(DIR_PATH / "test_excel.xlsx")
    assert attempts["value"] == 3
    assert disc._xls_app is not None


def test_discipline_can_be_garbage_collected(skip_if_xlwings_is_not_usable) -> None:
    """The discipline must not be kept alive by an interpreter-exit hook.

    Args:
        skip_if_xlwings_is_not_usable: Fixture to skip the test when xlwings is not
            usable.
    """
    disc = XLSDiscipline(DIR_PATH / "test_excel.xlsx")
    ref = weakref.ref(disc)
    del disc
    gc.collect()

    assert ref() is None


def test_run_cleans_up_on_failure(skip_if_xlwings_is_not_usable, snapshot) -> None:
    """The Excel app is reset when `_run` raises with `recreate_book_at_run=True`.

    Args:
        skip_if_xlwings_is_not_usable: Fixture to skip the test when xlwings is not
            usable.
        snapshot: syrupy snapshot fixture.
    """
    disc = XLSDiscipline(FILE_PATH_PATTERN.format(6), recreate_book_at_run=True)
    assert disc._xls_app is None

    with assert_exception(ValueError, snapshot):
        disc.execute(INPUT_DATA)

    assert disc._xls_app is None


def test_run_resets_app_when_book_creation_fails(
    skip_if_xlwings_is_not_usable, tmp_path
) -> None:
    """The Excel app is reset when `_run` fails to open the workbook.

    With `recreate_book_at_run=True` the workbook is opened at each run; a
    failure there must not leak the Excel process.

    Args:
        skip_if_xlwings_is_not_usable: Fixture to skip the test when xlwings is not
            usable.
        tmp_path: pytest fixture providing a temporary directory.
    """
    xls_path = tmp_path / "test_excel.xlsx"
    shutil.copy2(DIR_PATH / "test_excel.xlsx", xls_path)
    disc = XLSDiscipline(xls_path, recreate_book_at_run=True)
    assert disc._xls_app is None

    xls_path.unlink()
    with pytest.raises(FileNotFoundError):
        disc.execute(INPUT_DATA)

    assert disc._xls_app is None


def test_multiprocessing(skip_if_xlwings_is_not_usable) -> None:
    """Test the parallel execution xls disciplines.

    Args:
        skip_if_xlwings_is_not_usable: Fixture to skip the test when xlwings is not
            usable.
    """
    xlsd = XLSDiscipline(DIR_PATH / "test_excel.xlsx", copy_xls_at_setstate=True)
    xlsd_2 = XLSDiscipline(DIR_PATH / "test_excel.xlsx", copy_xls_at_setstate=True)

    parallel_execution = DiscParallelExecution([xlsd, xlsd_2], n_processes=2)
    parallel_execution.execute([
        {"a": array([2.0]), "b": array([1.0])},
        {"a": array([5.0]), "b": array([3.0])},
    ])
    assert xlsd.io.get_output_data() == {"c": array([3.0])}
    assert xlsd_2.io.get_output_data() == {"c": array([8.0])}


def test_multithreading(skip_if_xlwings_is_not_usable) -> None:
    """Test the execution of an XLSDiscipline with threading.

    Args:
        skip_if_xlwings_is_not_usable: Fixture to skip the test when xlwings is not
            usable.
    """
    xlsd = XLSDiscipline(
        DIR_PATH / "test_excel.xlsx",
        copy_xls_at_setstate=True,
        recreate_book_at_run=True,
    )
    xlsd_2 = XLSDiscipline(DIR_PATH / "test_excel.xlsx", recreate_book_at_run=True)

    parallel_execution = DiscParallelExecution(
        [xlsd, xlsd_2], use_threading=True, n_processes=2
    )
    parallel_execution.execute([
        {"a": array([2.0]), "b": array([1.0])},
        {"a": array([5.0]), "b": array([3.0])},
    ])

    assert xlsd.io.get_output_data() == {"c": array([3.0])}
    assert xlsd_2.io.get_output_data() == {"c": array([8.0])}


def f_sellar_system(
    x_1: float = 1.0, x_shared_2: float = 3.0, y_1: float = 1.0, y_2: float = 1.0
):
    """Objective function for the sellar problem."""
    obj = x_1**2 + x_shared_2 + y_1**2 + exp(-y_2)
    c_1 = 3.16 - y_1**2
    c_2 = y_2 - 24.0
    return obj, c_1, c_2


def f_sellar_1(
    x_1: float = 1.0,
    y_2: float = 1.0,
    x_shared_1: float = 1.0,
    x_shared_2: float = 3.0,
):
    """Function for discipline sellar 1."""
    y_1 = (x_shared_1**2 + x_shared_2 + x_1 - 0.2 * y_2) ** 0.5
    return y_1  # noqa: RET504


def test_doe_multiproc_multithread(skip_if_xlwings_is_not_usable) -> None:
    """Test the execution of a parallel DOE with multithreading at the MDA level.

    At the DOE level, the parallelization uses multiprocessing to compute the samples.
    At the MDA level of each sample, an MDAJacobi uses multithreading for faster
    convergence. Both parallelization techniques shall work together.

    Args:
        skip_if_xlwings_is_not_usable: Fixture to skip the test when xlwings is not
            usable.
    """
    sellar_1 = create_discipline("AutoPyDiscipline", py_func=f_sellar_1)
    sellar_2_xls = XLSDiscipline(
        DIR_PATH / "sellar_2.xlsx",
        copy_xls_at_setstate=True,
        recreate_book_at_run=True,
    )
    sellar_system = create_discipline("AutoPyDiscipline", py_func=f_sellar_system)
    disciplines = [sellar_1, sellar_2_xls, sellar_system]

    design_space = create_design_space()
    design_space.add_variable("x_1", lower_bound=0.0, upper_bound=10.0, value=ones(1))
    design_space.add_variable(
        "x_shared_1", lower_bound=-10.0, upper_bound=10.0, value=array([4])
    )
    design_space.add_variable(
        "x_shared_2", lower_bound=0.0, upper_bound=10.0, value=array([3])
    )

    scenario = create_scenario(
        disciplines,
        "obj",
        design_space,
        formulation_name="MDF",
        main_mda_settings=MDAChain_Settings(
            tolerance=1e-14, inner_mda_settings=MDAJacobi_Settings()
        ),
    )
    scenario.add_constraint("c_1", constraint_type=scenario.ConstraintType.INEQ)
    scenario.add_constraint("c_2", constraint_type=scenario.ConstraintType.INEQ)
    scenario.execute(DiagonalDOE_Settings(n_samples=2, n_processes=2))
    assert isclose(scenario.optimization_result.f_opt, 101.0, 1e-8, 1e-8)


def test_input_order(skip_if_xlwings_is_not_usable) -> None:
    """Test that input values are written correctly regardless of input data order.

    The order or the input_data dictionary generally corresponds to the order of the
    input names in the grammar, which is inferred from the row order of the Excel file.
    If the user modifies the grammar, this order could be altered.

    This Excel sheet computes: c = 100*z + 10*a + m
    The order in the sheet is z, a, m

    Args:
        skip_if_xlwings_is_not_usable: Fixture to skip the test when xlwings is not
            usable.
    """
    xlsd = XLSDiscipline(DIR_PATH / "test_excel_input_order.xlsx")
    xlsd.io.input_grammar.clear()
    # Grammar in different order than the Inputs sheet (z, a, m)
    xlsd.io.input_grammar.update_from_names(["a", "m", "z"])
    input_data = {
        "m": array([2.0]),
        "z": array([3.0]),
        "a": array([1.0]),
    }
    xlsd.execute(input_data)
    assert xlsd.io.data["c"] == array([312.0])


#         def test_macro(self):
#             xlsd = create_discipline("XLSDiscipline",
#                                      xls_file_path=join(DIRNAME,
#                                                         "test_excel.xlsm"))
#             input_data = {"a": array([2.25]), "b": array([30.25])}
#             xlsd.execute(input_data)
#             assert xlsd.io.data["d"] == 10 * 2.25 + 20 * 30.25
#             xlsd.close()


#         def test_fail_macro(self): Causes crash...
#             infile = join(DIRNAME, "test_excel_fail.xlsm")
#             xlsd = create_discipline("XLSDiscipline", xls_file_path=infile)
#             input_data = {"a": array([2.25]), "b": array([30.25])}
#             self.assertRaises(RuntimeError, xlsd.execute, input_data)
