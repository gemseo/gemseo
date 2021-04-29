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
    """Fixture to skip a test when xlwings has no usable excel."""
    xlwings = import_or_skip_xlwings

    try:
        xlwings.App(visible=False)
    # wide except because I cannot tell what is the exception raised by xlwings
    except:  # noqa: E722,B001
        pytest.skip("test requires excel available")


@pytest.fixture(scope="module")
def skip_if_xlwings_is_usable(import_or_skip_xlwings):
    """Fixture to skip a test when xlwings has usable excel."""
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
    xlsd = XLSDiscipline(str(DIR_PATH / "test_excel.xlsx"))
    xlsd.execute(INPUT_DATA)
    assert xlsd.local_data["c"] == 23.5
    xlsd.close()


@pytest.mark.parametrize("file_id", range(1, 4))
def test_error_init(skip_if_xlwings_is_not_usable, file_id):
    with pytest.raises(ValueError):
        XLSDiscipline(FILE_PATH_PATTERN.format(file_id))


def test_error_execute(skip_if_xlwings_is_not_usable):
    disc = XLSDiscipline(FILE_PATH_PATTERN.format(4))
    with pytest.raises(ValueError):
        disc.execute(INPUT_DATA)


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
