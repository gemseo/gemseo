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
# INITIAL AUTHORS - initial API and implementation and/or
#                   initial documentation
#        :author:  Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES

import pytest
from numpy import array

from gemseo.core.parallel_execution import DiscParallelExecution
from gemseo.utils.py23_compat import Path
from gemseo.wrappers.xls_discipline import XLSDiscipline

DIR_PATH = Path(__file__).parent
FILE_PATH_PATTERN = str(DIR_PATH / "test_excel_fail{}.xlsx")
INPUT_DATA = {"a": array([20.25]), "b": array([3.25])}


@pytest.fixture(scope="module")
def import_or_skip_xlwings():
    """Fixture to skip a test when xlwings cannot be imported."""
    return pytest.importorskip("xlwings", reason="xlwings is not available")


@pytest.fixture(scope="module")
def skip_if_xlwings_is_not_usable(import_or_skip_xlwings):
    """Fixture to skip a test when xlwings has no usable excel.

    Args:
        import_or_skip_xlwings: Fixture to skip a test when
            xlwings cannot be imported.
    """
    xlwings = import_or_skip_xlwings

    try:
        xlwings.App(visible=False)
    # wide except because I cannot tell what is the exception raised by xlwings
    except:  # noqa: E722,B001
        pytest.skip("test requires excel available")


@pytest.fixture(scope="module")
def skip_if_xlwings_is_usable(import_or_skip_xlwings):
    """Fixture to skip a test when xlwings has usable excel.

    Args:
        import_or_skip_xlwings: Fixture to skip a test when
            xlwings cannot be imported.
    """
    xlwings = import_or_skip_xlwings

    try:
        xlwings.App(visible=False)
    # wide except because I cannot tell what is the exception raised by xlwings
    except:  # noqa: E722,B001
        pass
    else:
        pytest.skip("test requires no excel available")


def test_missing_xlwings(skip_if_xlwings_is_usable):
    """Check error when excel is not available."""
    msg = "xlwings requires Microsoft Excel"
    with pytest.raises(RuntimeError, match=msg):
        XLSDiscipline("dummy_file_path")


def test_basic(skip_if_xlwings_is_not_usable):
    """Simple test, the output is the sum of the inputs.

    Args:
        skip_if_xlwings_is_not_usable: Fixture to skip a test
            when xlwings has no usable excel.
    """
    xlsd = XLSDiscipline(str(DIR_PATH / "test_excel.xlsx"))
    xlsd.execute(INPUT_DATA)
    assert xlsd.local_data["c"] == 23.5


@pytest.mark.parametrize("file_id", range(1, 4))
def test_error_init(skip_if_xlwings_is_not_usable, file_id):
    """Test that errors are raised for files without the proper format.

    Args:
        skip_if_xlwings_is_not_usable: Fixture to skip a test
            when xlwings has no usable excel.
        file_id: The id of the test file.
    """
    with pytest.raises(ValueError):
        XLSDiscipline(FILE_PATH_PATTERN.format(file_id))


def test_error_execute(skip_if_xlwings_is_not_usable):
    """Check that an exception is raised for incomplete data.

    Args:
        skip_if_xlwings_is_not_usable: Fixture to skip a test
            when xlwings has no usable excel.
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
        skip_if_xlwings_is_not_usable: Fixture to skip a test
            when xlwings has no usable excel.
    """
    xlsd = XLSDiscipline(str(DIR_PATH / "test_excel.xlsx"), copy_xls_at_setstate=True)
    xlsd_2 = XLSDiscipline(str(DIR_PATH / "test_excel.xlsx"), copy_xls_at_setstate=True)

    parallel_execution = DiscParallelExecution(
        [xlsd, xlsd_2], use_threading=False, n_processes=2
    )
    parallel_execution.execute(
        [{"a": array([2.0]), "b": array([1.0])}, {"a": array([5.0]), "b": array([3.0])}]
    )

    assert xlsd.get_output_data() == {"c": array([3.0])}
    assert xlsd_2.get_output_data() == {"c": array([8.0])}


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
