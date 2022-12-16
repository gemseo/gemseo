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
from __future__ import annotations

import pytest
from numpy import array

# skip if matlab API is not found
matlab = pytest.importorskip("matlab")

from gemseo.wrappers.matlab.matlab_data_processor import load_matlab_file  # noqa: E402
from gemseo.wrappers.matlab.matlab_discipline import MatlabDiscipline  # noqa: E402

from .matlab_files import MATLAB_FILES_DIR_PATH  # noqa: E402

MATLAB_SIMPLE_FUNC = MATLAB_FILES_DIR_PATH / "dummy_test.m"
MATLAB_COMPLEX_FUNC = MATLAB_FILES_DIR_PATH / "dummy_complex_fct.m"
MATLAB_SIMPLE_FUNC_MULTIDIM = MATLAB_FILES_DIR_PATH / "dummy_test_multidim.m"
MATLAB_SIMPLE_FUNC_MULTIDIM_JAC = MATLAB_FILES_DIR_PATH / "dummy_test_multidim_jac.m"
FCT_MULTIDIM_DATASET = MATLAB_FILES_DIR_PATH / "dummy_file_multidim_fct.mat"


def test_engine_property():
    """Check the engine property."""
    mat = MatlabDiscipline(MATLAB_SIMPLE_FUNC)
    assert not mat.engine.is_closed


def test_inputs_from_matlab():
    """Test input variables read from matlab file."""
    mat2 = MatlabDiscipline(MATLAB_COMPLEX_FUNC)
    assert mat2._MatlabDiscipline__inputs == ["a", "b", "c", "d", "e", "f"]


def test_inputs_from_param():
    """Test input variables given as input param."""
    mat = MatlabDiscipline(
        MATLAB_COMPLEX_FUNC, input_names=["v1", "v2", "v3", "v4", "v5", "v6"]
    )
    assert mat._MatlabDiscipline__inputs == ["v1", "v2", "v3", "v4", "v5", "v6"]


def test_outputs():
    """Test output variables."""
    mat2 = MatlabDiscipline(MATLAB_COMPLEX_FUNC)
    assert mat2._MatlabDiscipline__outputs == ["x", "y", "z"]


def test_inputs_and_outputs_size_unknown():
    """Test that size of input and output variables are unknown when initializing without
    matlab data."""
    mat1 = MatlabDiscipline(MATLAB_SIMPLE_FUNC)

    assert mat1._MatlabDiscipline__inputs_size["x"] == -1
    assert mat1._MatlabDiscipline__outputs_size["y"] == -1
    assert mat1._MatlabDiscipline__is_size_known is False


def test_inputs_and_outputs_size_known_eval():
    """Test that size of input and output variables are known after evaluating matlab
    function."""
    mat2 = MatlabDiscipline(MATLAB_SIMPLE_FUNC_MULTIDIM)
    mat2.execute({"x": array([1, 1]), "y": array([1])})

    assert mat2._MatlabDiscipline__inputs_size["x"] == 2
    assert mat2._MatlabDiscipline__inputs_size["y"] == 1
    assert mat2._MatlabDiscipline__outputs_size["z1"] == 2
    assert mat2._MatlabDiscipline__outputs_size["z2"] == 1
    assert mat2._MatlabDiscipline__is_size_known is True


def test_inputs_and_outputs_size_known_init():
    """Test the size of input and output variables are known when initializing with
    matlab data."""
    mat2 = MatlabDiscipline(
        MATLAB_SIMPLE_FUNC_MULTIDIM, matlab_data_file=FCT_MULTIDIM_DATASET
    )

    assert mat2._MatlabDiscipline__inputs_size["x"] == 2
    assert mat2._MatlabDiscipline__inputs_size["y"] == 1
    assert mat2._MatlabDiscipline__outputs_size["z1"] == 2
    assert mat2._MatlabDiscipline__outputs_size["z2"] == 1
    assert mat2._MatlabDiscipline__is_size_known is True


def test_jac_output_names_error_wrong_name():
    """Test that jacobians output raise an error if names are wrong."""
    with pytest.raises(ValueError) as excp:
        MatlabDiscipline(
            MATLAB_FILES_DIR_PATH / "dummy_test_jac_wrong.m",
            is_jac_returned_by_func=True,
        )
    assert (
        str(excp.value) == "Jacobian terms ['jac_dy_dx'] are not found "
        "in the list of conventional names. "
        "It is reminded that jacobian terms' name "
        "should be such as 'jac_dout_din'"
    )


def test_jac_output_names_error_missing_term():
    """Test that jacobians output raise an error if a term is missing."""
    with pytest.raises(
        ValueError,
        match="The number of jacobian outputs "
        "does not correspond to what it "
        "should be. Make sure that all "
        "outputs have a jacobian matrix "
        "with respect to inputs.",
    ):
        MatlabDiscipline(
            MATLAB_FILES_DIR_PATH / "dummy_test_multidim_jac_wrong.m",
            is_jac_returned_by_func=True,
        )


def test_jac_output_names():
    """Test that jacobians output name are detected when returning by the main
    function."""
    mat = MatlabDiscipline(
        MATLAB_SIMPLE_FUNC_MULTIDIM_JAC, is_jac_returned_by_func=True
    )

    assert mat._MatlabDiscipline__jac_output_names == [
        "jac_dz1_dx",
        "jac_dz1_dy",
        "jac_dz2_dx",
        "jac_dz2_dy",
    ]


def test_jac_output_indices():
    """Test that jacobians output have the right indices."""
    mat = MatlabDiscipline(
        MATLAB_SIMPLE_FUNC_MULTIDIM_JAC, is_jac_returned_by_func=True
    )

    assert mat._MatlabDiscipline__jac_output_indices == [2, 3, 4, 5]


def test_init_default_data():
    """Test that data are correctly initialized."""
    mat = MatlabDiscipline(
        matlab_fct=MATLAB_COMPLEX_FUNC,
        matlab_data_file=MATLAB_FILES_DIR_PATH / "dummy_complex_fct_database.mat",
    )
    assert array(mat.default_inputs["a"]) == pytest.approx(
        array([[[1, 1], [1, 1]], [[1, 1], [1, 1]]])
    )
    assert array(mat.default_inputs["b"]) == pytest.approx(
        array([[[1, 1], [1, 1]], [[1, 1], [1, 1]]])
    )
    assert mat.default_inputs["c"] == pytest.approx(2)
    assert array(mat.default_inputs["d"]) == pytest.approx(array([1, 2]))
    assert mat.default_inputs["e"] == pytest.approx(1)
    assert mat.default_inputs["f"] == pytest.approx(1)


def test_search_file_error_not_found():
    """Test that an error is raised if file is not found."""
    with pytest.raises(
        IOError, match="No file: dummy_test.m, " "found in directory: non_existing."
    ):
        MatlabDiscipline("dummy_test.m", search_file="non_existing")


def test_search_file_error_two_files_found():
    """Test that an error is raised if two files are found."""
    with pytest.raises(IOError) as excp:
        MatlabDiscipline("dummy_test.m", search_file=MATLAB_FILES_DIR_PATH)
    assert (
        str(excp.value) == "At least two files dummy_test.m were "
        "in directory {}\n File one: {};"
        "\n File two: {}.".format(
            MATLAB_FILES_DIR_PATH,
            MATLAB_SIMPLE_FUNC,
            MATLAB_FILES_DIR_PATH / "matlab_files_bis_test" / "dummy_test.m",
        )
    )


def test_search_file():
    """Test that file is found."""
    mat = MatlabDiscipline("dummy_test_multidim.m", search_file=MATLAB_FILES_DIR_PATH)
    assert mat._MatlabDiscipline__inputs == ["x", "y"]


def test_check_existing_function():
    """Test an existing user-made function."""
    mat = MatlabDiscipline(MATLAB_SIMPLE_FUNC)
    assert mat.function_name == "dummy_test"


def test_check_function_builtin():
    """Test a built-in function."""
    mat = MatlabDiscipline("cos", input_names=["x"], output_names=["y"])
    assert mat.function_name == "cos"


def test_run_builtin():
    """Test that built-in matlab function is correctly called and returned right
    values."""
    mat = MatlabDiscipline("cos", input_names=["x"], output_names=["out"])
    mat.execute({"x": array([0])})
    assert mat.local_data["out"] == pytest.approx(1)


def test_run_user():
    """Test that user matlab function is correctly called and returned right values."""
    mat = MatlabDiscipline(MATLAB_SIMPLE_FUNC)
    mat.execute({"x": array([2])})
    assert mat.local_data["y"] == pytest.approx(4)


def test_run_user_new_names():
    """Test that user matlab function is correctly called and returned right values when
    new names are prescribed for inputs and outputs."""
    mat = MatlabDiscipline(
        MATLAB_SIMPLE_FUNC, input_names=["in1"], output_names=["out"]
    )
    mat.execute({"in1": array([3])})
    assert mat.local_data["out"] == pytest.approx(9)


def test_run_user_multidim():
    """Test that user matlab function is correctly called and returned right values."""
    mat = MatlabDiscipline(
        MATLAB_SIMPLE_FUNC_MULTIDIM, matlab_data_file=FCT_MULTIDIM_DATASET
    )
    mat.execute({"x": array([1, 2]), "y": array([3])})
    assert array(mat.local_data["z1"]) == pytest.approx(array([1, 5]))
    assert mat.local_data["z2"] == pytest.approx(11)


def test_run_user_multidim_no_extension_data():
    """Test that an input matlab data file can be prescribed without extension."""
    mat = MatlabDiscipline(
        "dummy_test_multidim.m",
        matlab_data_file="dummy_file_multidim_fct",
        search_file=MATLAB_FILES_DIR_PATH,
    )
    mat.execute({"x": array([1, 2]), "y": array([3])})
    assert array(mat.local_data["z1"]) == pytest.approx(array([1, 5]))
    assert mat.local_data["z2"] == pytest.approx(11)


def test_run_user_multidim_jac():
    """Test that user matlab function is correctly called and returned right values when
    jacobian is defined."""
    mat = MatlabDiscipline(
        MATLAB_SIMPLE_FUNC_MULTIDIM_JAC,
        matlab_data_file=FCT_MULTIDIM_DATASET,
        is_jac_returned_by_func=True,
    )
    mat.execute({"x": array([1, 2]), "y": array([3])})
    assert array(mat.jac["z1"]["x"]) == pytest.approx(array([[2, 3], [0, 4]]))
    assert array(mat.jac["z1"]["y"]) == pytest.approx(array([[0], [54]]))
    assert array(mat.jac["z2"]["x"]) == pytest.approx(array([[4, 0]]))
    assert array(mat.jac["z2"]["y"]) == pytest.approx(array([[6]]))
    assert len(mat.local_data) == 4


def test_run_user_multidim_jac_wrong_size():
    """Test that user matlab function is correctly called and returned right values when
    jacobian is defined."""
    mat = MatlabDiscipline(
        MATLAB_FILES_DIR_PATH / "dummy_test_multidim_jac_wrong_size.m",
        matlab_data_file=FCT_MULTIDIM_DATASET,
        is_jac_returned_by_func=True,
    )
    with pytest.raises(ValueError) as excp:
        mat.execute({"x": array([1, 2]), "y": array([3])})

    assert str(excp.value) == (
        "Jacobian term 'jac_dz1_dx' has the wrong size "
        "(1, 4) whereas it should be (2, 2)."
    )


def test_check_cleaning_interval():
    """Test check that interval cleaning raise an error if not an integer."""
    with pytest.raises(
        ValueError,
        match="The parameter 'cleaning_interval' argument must be an integer.",
    ):
        MatlabDiscipline(MATLAB_SIMPLE_FUNC, clean_cache_each_n=2.3)


def test_save_data(tmp_wd):
    """Test that discipline data are correctly exported into a matlab file."""
    mat = MatlabDiscipline(MATLAB_SIMPLE_FUNC)
    mat.execute({"x": array([2])})
    output_file = "output_file.mat"
    mat.save_data_to_matlab(output_file)
    written_data = load_matlab_file(output_file)
    assert array(written_data["x"]) == pytest.approx(2)
    assert array(written_data["y"]) == pytest.approx(4)
