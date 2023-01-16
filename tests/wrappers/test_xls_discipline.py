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

from pathlib import Path

import pytest
from gemseo.api import create_design_space
from gemseo.api import create_discipline
from gemseo.api import create_scenario
from gemseo.core.parallel_execution import DiscParallelExecution
from gemseo.wrappers.xls_discipline import XLSDiscipline
from numpy import array
from numpy import exp
from numpy import ones

DIR_PATH = Path(__file__).parent
FILE_PATH_PATTERN = str(DIR_PATH / "test_excel_fail{}.xlsx")
INPUT_DATA = {"a": array([20.25]), "b": array([3.25])}


def test_missing_xlwings(skip_if_xlwings_is_usable):
    """Check error when excel is not available.

    Args:
        skip_if_xlwings_is_usable: Fixture to skip the test when xlwings is usable.
    """
    msg = "xlwings requires Microsoft Excel"
    with pytest.raises(RuntimeError, match=msg):
        XLSDiscipline("dummy_file_path")


def test_basic(skip_if_xlwings_is_not_usable):
    """Simple test, the output is the sum of the inputs.

    Args:
        skip_if_xlwings_is_not_usable: Fixture to skip the test when xlwings is not
            usable.
    """
    xlsd = XLSDiscipline(DIR_PATH / "test_excel.xlsx")
    xlsd.execute(INPUT_DATA)
    assert xlsd.local_data["c"] == 23.5


@pytest.mark.parametrize("file_id", range(1, 4))
def test_error_init(skip_if_xlwings_is_not_usable, file_id):
    """Test that errors are raised for files without the proper format.

    Args:
        skip_if_xlwings_is_not_usable: Fixture to skip the test when xlwings is not
            usable.
        file_id: The id of the test file.
    """
    with pytest.raises(ValueError):
        XLSDiscipline(FILE_PATH_PATTERN.format(file_id))


def test_error_execute(skip_if_xlwings_is_not_usable):
    """Check that an exception is raised for incomplete data.

    Args:
        skip_if_xlwings_is_not_usable: Fixture to skip the test when xlwings is not
            usable.
    """
    disc = XLSDiscipline(FILE_PATH_PATTERN.format(4))
    with pytest.raises(
        ValueError,
        match=r"Inconsistent Outputs sheet, names \(first columns\) and "
        r"values column \(second\) must be of the same length.",
    ):
        disc.execute(INPUT_DATA)


def test_multiprocessing(skip_if_xlwings_is_not_usable):
    """Test the parallel execution xls disciplines.

    Args:
        skip_if_xlwings_is_not_usable: Fixture to skip the test when xlwings is not
            usable.
    """
    xlsd = XLSDiscipline(DIR_PATH / "test_excel.xlsx", copy_xls_at_setstate=True)
    xlsd_2 = XLSDiscipline(DIR_PATH / "test_excel.xlsx", copy_xls_at_setstate=True)

    parallel_execution = DiscParallelExecution([xlsd, xlsd_2], n_processes=2)
    parallel_execution.execute(
        [{"a": array([2.0]), "b": array([1.0])}, {"a": array([5.0]), "b": array([3.0])}]
    )
    assert xlsd.get_output_data() == {"c": array([3.0])}
    assert xlsd_2.get_output_data() == {"c": array([8.0])}


def test_multithreading(skip_if_xlwings_is_not_usable):
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
    parallel_execution.execute(
        [{"a": array([2.0]), "b": array([1.0])}, {"a": array([5.0]), "b": array([3.0])}]
    )

    assert xlsd.get_output_data() == {"c": array([3.0])}
    assert xlsd_2.get_output_data() == {"c": array([8.0])}


def f_sellar_system(
    x_local: float = 1.0, x_shared_2: float = 3.0, y_1: float = 1.0, y_2: float = 1.0
) -> tuple[float, float, float]:
    """Objective function for the sellar problem."""
    obj = x_local**2 + x_shared_2 + y_1**2 + exp(-y_2)
    c_1 = 3.16 - y_1**2
    c_2 = y_2 - 24.0
    return obj, c_1, c_2


def f_sellar_1(
    x_local: float = 1.0,
    y_2: float = 1.0,
    x_shared_1: float = 1.0,
    x_shared_2: float = 3.0,
) -> float:
    """Function for discipline sellar 1."""
    y_1 = (x_shared_1**2 + x_shared_2 + x_local - 0.2 * y_2) ** 0.5
    return y_1


def test_doe_multiproc_multithread(skip_if_xlwings_is_not_usable):
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
    design_space.add_variable("x_local", l_b=0.0, u_b=10.0, value=ones(1))
    design_space.add_variable("x_shared_1", l_b=-10.0, u_b=10.0, value=array([4]))
    design_space.add_variable("x_shared_2", l_b=0.0, u_b=10.0, value=array([3]))

    scenario = create_scenario(
        disciplines,
        formulation="MDF",
        main_mda_name="MDAChain",
        inner_mda_name="MDAJacobi",
        objective_name="obj",
        design_space=design_space,
        scenario_type="DOE",
    )
    scenario.add_constraint("c_1", "ineq")
    scenario.add_constraint("c_2", "ineq")
    doe_input = {
        "algo": "DiagonalDOE",
        "n_samples": 2,
        "algo_options": {"n_processes": 2},
    }
    scenario.execute(doe_input)
    assert scenario.optimization_result.f_opt == 101.0


#         def test_macro(self):
#             xlsd = create_discipline("XLSDiscipline",
#                                      xls_file_path=join(DIRNAME,
#                                                         "test_excel.xlsm"))
#             input_data = {"a": array([2.25]), "b": array([30.25])}
#             xlsd.execute(input_data)
#             assert xlsd.local_data["d"] == 10 * 2.25 + 20 * 30.25
#             xlsd.close()


#         def test_fail_macro(self): Causes crash...
#             infile = join(DIRNAME, "test_excel_fail.xlsm")
#             xlsd = create_discipline("XLSDiscipline", xls_file_path=infile)
#             input_data = {"a": array([2.25]), "b": array([30.25])}
#             self.assertRaises(RuntimeError, xlsd.execute, input_data)
