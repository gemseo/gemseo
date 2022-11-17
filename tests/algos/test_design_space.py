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
#    INITIAL AUTHORS - initial API and implementation and/or initial
#                           documentation
#        :author: Charlie Vanaret
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import annotations

import logging
import re
from pathlib import Path

import numpy as np
import pytest
from gemseo.algos.design_space import DesignSpace
from gemseo.algos.design_space import DesignVariable
from gemseo.algos.design_space import DesignVariableType
from gemseo.algos.opt_problem import OptimizationProblem
from gemseo.algos.opt_result import OptimizationResult
from gemseo.core.mdofunctions.mdo_function import MDOFunction
from gemseo.problems.sobieski.core.problem import SobieskiProblem
from numpy import array
from numpy import array_equal
from numpy import float64
from numpy import inf
from numpy import int32
from numpy import ndarray
from numpy import ones
from numpy import zeros
from numpy.linalg import norm
from numpy.testing import assert_equal

CURRENT_DIR = Path(__file__).parent
TEST_INFILE = CURRENT_DIR / "design_space.txt"
FAIL_HDF = CURRENT_DIR / "fail.hdf5"


@pytest.fixture
def design_space():
    """The main design space to be used by the test function.

    Feel free to add new variables.
    """
    ds = DesignSpace()
    ds.add_variable("x1", l_b=0.0, u_b=2.0)
    ds.add_variable("x2", l_b=-2.0, u_b=2.0)
    ds.add_variable("x3", var_type=DesignVariableType.INTEGER, l_b=0, u_b=2)
    ds.add_variable("x4", var_type="float", l_b=-1.0, u_b=0.0, value=-0.5)
    ds.add_variable("x5", size=3, var_type="float", l_b=-1.0, u_b=0.0, value=-0.5)
    ds.add_variable("x6", u_b=2.0)
    ds.add_variable("x7", l_b=0.0)
    ds.add_variable("x8", var_type=DesignVariableType.INTEGER, l_b=1, u_b=1)
    ds.add_variable("x9", size=3, l_b=-1.0, u_b=2.0)
    ds.add_variable("x10", size=3)
    ds.add_variable("x11", size=2)
    ds.add_variable("x12")
    ds.add_variable("x13", value=array([0.5]))
    ds.add_variable("x14", var_type=DesignVariableType.INTEGER, value=array([2.0]))
    ds.add_variable("x15")
    ds.add_variable(
        "x16", size=2, var_type=[DesignVariableType.FLOAT] * 2, value=array([1.0, 2.0])
    )
    ds.add_variable(
        "x17", size=2, var_type=[DesignVariableType.INTEGER] * 2, value=array([1, 2])
    )
    ds.add_variable("x18", l_b=-1.0, u_b=2.0)
    ds.add_variable("x19", l_b=1.0, u_b=3.0)
    ds.add_variable("x20", var_type=b"float")
    ds.add_variable("x21", value=0.5)
    ds.add_variable("x22", size=2)
    ds.add_variable("x23", l_b=0.0, u_b=1.0, value=array([1]), var_type="float")
    return ds


def test_add_variable_when_already_exists(design_space):
    """Check that adding an existing variable raises an error."""
    design_space.add_variable("varname")
    with pytest.raises(ValueError, match="Variable 'varname' already exists."):
        design_space.add_variable(name="varname")


@pytest.mark.parametrize("size", [-1, 0, 0.4])
def test_add_variable_with_wrong_size(design_space, size):
    """Check that adding a variable with a wrong size raises an error."""
    with pytest.raises(
        ValueError, match="The size of 'varname' should be a positive integer."
    ):
        design_space.add_variable(name="varname", size=size)


def test_add_variable_with_inconsistent_types_list(design_space):
    """Check that adding a variable with wrong number of types raises an error."""
    with pytest.raises(
        ValueError,
        match="The list of types for variable 'varname' should be of size 3.",
    ):
        design_space.add_variable(
            name="varname", size=3, var_type=[DesignVariableType.FLOAT] * 2
        )


def test_add_variable_with_unkown_type(design_space):
    """Check that adding a variable with unknown type raises an error."""
    with pytest.raises(ValueError, match='The type "a" of varname is not known.'):
        design_space.add_variable(name="varname", var_type="a")


def test_add_variable_with_unkown_type_from_list(design_space):
    """Check that adding a variable with unknown type raises an error."""
    with pytest.raises(ValueError, match='The type "a" of varname is not known.'):
        design_space.add_variable(
            name="varname",
            size=2,
            var_type=[DesignVariableType.FLOAT, "a"],
        )


def test_add_variable_with_unnumerizable_value(design_space):
    """Check that adding a variable with unnumerizable value raises an error."""
    expected = re.escape(
        "Value <built-in function len> of variable 'varname' is not numerizable."
    )
    with pytest.raises(ValueError, match=expected):
        design_space.add_variable(name="varname", value=len)


@pytest.mark.parametrize("arg", ["l_b", "u_b", "value"])
def test_add_variable_with_nan_value(design_space, arg):
    """Check that adding a variable with nan value raises an error."""
    with pytest.raises(ValueError, match="Value nan of variable 'varname' is NaN."):
        design_space.add_variable(name="varname", **{arg: float("nan")})


@pytest.mark.parametrize("arg,side", [("l_b", "lower"), ("u_b", "upper")])
def test_add_variable_with_inconsistent_bound_size(design_space, arg, side):
    """Check that using bounds with inconsistent size raises an."""
    with pytest.raises(
        ValueError, match=f"The {side} bounds of 'varname' should be of size 3."
    ):
        design_space.add_variable(name="varname", size=3, **{arg: [0.0, 0.0]})


def test_add_variable_with_upper_bounds_lower_than_lower_ones(design_space):
    """Check that using upper bounds lower than lower ones raises an error."""
    with pytest.raises(
        ValueError,
        match=re.escape(
            "The bounds of variable 'varname'[1] are not valid: [1.]!<[0.]."
        ),
    ):
        design_space.add_variable(
            name="varname", size=3, l_b=[0, 1.0, 0], u_b=[1, 0.0, 1]
        )


@pytest.mark.parametrize("arg", ["l_b", "u_b", "value"])
def test_add_variable_with_2d_object(design_space, arg):
    """Check that using a 2d iterable object raises an error."""
    expected = (
        "Value [[1.]] of variable 'varname' has dimension greater than 1 "
        "while a float or a 1d iterable object (array, list, tuple, ...) "
        "while a scalar was expected."
    )
    with pytest.raises(ValueError, match=re.escape(expected)):
        design_space.add_variable("varname", **{arg: [[1.0]]})


@pytest.mark.parametrize(
    "size,var_type,l_b,u_b, value",
    [
        (1, DesignVariableType.FLOAT, 1.0, 2.0, 3.0),
        (1, DesignVariableType.FLOAT, 1.0, 2.0, 0.0),
    ],
)
def test_add_variable_with_value_out_of_bounds(
    design_space, size, var_type, l_b, u_b, value
):
    """Check that setting a value out of bounds raises an error."""
    expected = (
        "The current value of variable 'varname' ({}) is not "
        "between the lower bound {} and the upper bound {}.".format(value, l_b, u_b)
    )
    with pytest.raises(ValueError, match=re.escape(expected)):
        design_space.add_variable(
            name="varname", size=size, var_type=var_type, l_b=l_b, u_b=u_b, value=value
        )


def test_creation_4():
    design_space = DesignSpace()
    design_space.add_variable("varname")
    with pytest.raises(
        KeyError,
        match=re.escape("There is no current value for the design variables: varname"),
    ):
        design_space.get_current_value(normalize=True)


def test_add_variable_value(design_space):
    design_space.add_variable(
        "varname",
        size=3,
        var_type=DesignVariableType.FLOAT,
        l_b=0.0,
        u_b=1.0,
        value=[None, None, None],
    )


@pytest.mark.parametrize(
    "current_x",
    [
        {"x1": array([1.0]), "x2": array([0.0])},
        OptimizationResult(x_opt=array([1.0, 0.0])),
    ],
)
def test_set_current_value(design_space, current_x):
    design_space.filter(["x1", "x2"])
    design_space.set_current_value(current_x)
    x_n = design_space.get_current_value(normalize=True)
    assert (x_n == 0.5).all()


def test_set_current_value_with_malformed_mapping_arg(design_space):
    """Check that setting the current value from a malformed mapping raises an error."""
    design_space.filter("x1")
    with pytest.raises(
        Exception,
        match="The variable x1 of size 1 cannot be set with an array of size 2.",
    ):
        design_space.set_current_value({"x1": array([1.0, 1.0])})


def test_set_current_value_with_malformed_opt_arg(design_space):
    """Check that setting the current value from a malformed optimization result raises
    an error."""
    with pytest.raises(
        Exception,
        match="Invalid x_opt, dimension mismatch: {} != 1".format(
            design_space.dimension
        ),
    ):
        design_space.set_current_value(OptimizationResult(x_opt=array([1.0])))


def test_set_current_value_with_malformed_current_x(design_space):
    """Check that setting the current value from a float raises an error."""
    with pytest.raises(
        TypeError,
        match=(
            "The current design value should be either an array, "
            "a dictionary of arrays or an optimization result; "
            "got <class 'float'> instead."
        ),
    ):
        design_space.set_current_value(1.0)


def test_read_from_txt():
    """Check that a variable name is correct when reading a txt file."""
    ds = DesignSpace.read_from_txt(CURRENT_DIR / "design_space_4.txt")
    assert ds.variables_names == ["x_shared"]


def test_integer_variable_set_current_x(design_space):
    """Check that an integer value is correctly set."""
    design_space.filter("x3")
    x_i = array([0], dtype=int32)
    design_space.set_current_value(x_i)
    x_i_conv = design_space.dict_to_array(design_space.array_to_dict(x_i))
    assert x_i_conv.dtype == x_i.dtype
    assert x_i_conv == x_i


def test_integer_variable_round_vect(design_space):
    """Check that a float value is correctly rounded to the closest integer."""
    design_space.filter("x3")
    assert design_space.round_vect(array([1.2])) == 1
    assert design_space.round_vect(array([1.9])) == 2

    assert (design_space.round_vect(array([[1.9], [1.2]])) == array([[2], [1]])).all()

    vector = array([1.2])
    rounded_vector = design_space.round_vect(vector)
    assert rounded_vector == array([1.0])
    assert vector != rounded_vector

    rounded_vector = design_space.round_vect(vector, copy=False)
    assert rounded_vector == array([1.0])
    assert vector == rounded_vector


@pytest.mark.parametrize("copy", [True, False])
def test_filter_by_variables_names(design_space, copy):
    """Check that the design space can be filtered by variables dimensions."""
    design_space_with_x5 = design_space.filter("x5", copy=copy)
    if not copy:
        design_space_with_x5 = design_space

    assert not design_space_with_x5.__contains__("x4")
    assert design_space_with_x5.__contains__("x5")

    if copy:
        assert design_space_with_x5 is not design_space


def test_filter_with_an_unknown_variable(design_space):
    """Check that filtering a design space with an unknown name raises an error."""
    with pytest.raises(ValueError, match="Variable 'unknown_x' is not known."):
        design_space.filter("unknown_x")


def test_filter_by_variables_dimensions(design_space):
    """Check that the design space can be filtered by variables dimensions."""
    design_space.filter_dim("x5", [0])
    with pytest.raises(ValueError, match="Dimension 1 of variable 'x5' is not known."):
        design_space.filter_dim("x5", [1])


def test_extend():
    """Test the extension of a design space with another."""
    design_space = DesignSpace()
    design_space.add_variable("x1", var_type="float", l_b=-1.0, u_b=0.0, value=-0.5)
    other = DesignSpace()
    other.add_variable("x2", size=3, var_type="float", l_b=-1.0, u_b=0.0, value=-0.5)
    design_space.extend(other)
    assert design_space.__contains__("x2")
    assert design_space.get_size("x2") == other.get_size("x2")
    assert (design_space.get_type("x2") == other.get_type("x2")).all()
    assert (design_space.get_lower_bound("x2") == other.get_lower_bound("x2")).all()
    assert (design_space.get_upper_bound("x2") == other.get_upper_bound("x2")).all()
    assert (
        design_space.get_current_value(["x2"]) == other.get_current_value(["x2"])
    ).all()


def test_active_bounds():
    """Check whether active bounds are correctly identified."""

    design_space = DesignSpace()
    design_space.add_variable("x", l_b=0.0, u_b=2.0)
    design_space.add_variable("y", l_b=-2.0, u_b=2.0)
    design_space.add_variable("z")
    lb_1, ub_1 = design_space.get_active_bounds(
        {"x": array([0.0]), "y": array([2.0]), "z": array([2.0])}
    )

    lb_2, ub_2 = design_space.get_active_bounds(array([1e-12, 2.0 - 1e-12, 1e-12]))

    assert lb_1 == lb_2
    assert lb_1["x"] == [True]
    assert lb_1["y"] == [False]
    assert not lb_1["z"][0]
    assert ub_1 == ub_2
    assert ub_1["y"] == [True]
    assert ub_1["x"] == [False]
    assert not ub_1["z"][0]

    with pytest.raises(
        TypeError, match="Expected dict or array for x_vec argument; got <class 'str'>."
    ):
        design_space.get_active_bounds("test")

    with pytest.raises(
        KeyError,
        match=re.escape("There is no current value for the design variables: x, y, z."),
    ):
        design_space.get_active_bounds()


@pytest.mark.parametrize("index,expected", [(0, "x"), (1, "z!0"), (2, "z!1")])
def test_get_indexed_variables_names(index, expected):
    """Check the variables names obtained with get_indexed_variables_names()."""
    design_space = DesignSpace()
    design_space.add_variable("x")
    design_space.add_variable("z", size=2)
    assert design_space.get_indexed_variables_names()[index] == expected


@pytest.mark.parametrize(
    "name,lower_bound,upper_bound",
    [("x6", -inf, 2.0), ("x7", 0.0, inf)],
)
def test_bounds(design_space, name, lower_bound, upper_bound):
    """Check that bounds are correctly retrieved."""
    assert design_space.get_lower_bound(name) == lower_bound
    assert design_space.get_upper_bound(name) == upper_bound
    assert design_space.get_lower_bounds([name]) == lower_bound
    assert design_space.get_upper_bounds([name]) == upper_bound


def test_bounds_set_lower_bound_with_nan(design_space):
    """Check that setting lower bound with nan raises an error."""
    with pytest.raises(ValueError, match="Value nan of variable 'x6' is NaN."):
        design_space.set_lower_bound("x6", float("nan"))


def test_bounds_set_lower_bound_with_inconsistent_size(design_space):
    """Check that setting lower bound with inconsistent sized value raises an error."""
    with pytest.raises(
        ValueError, match="The lower bounds of 'x6' should be of size 1."
    ):
        design_space.set_lower_bound("x6", ones(2))


def test_bounds_set_upper_bound_with_nan(design_space):
    """Check that setting upper bound with nan raises an error."""
    with pytest.raises(ValueError, match="Value nan of variable 'x6' is NaN."):
        design_space.set_upper_bound("x6", float("nan"))


def test_bounds_set_upper_bound_with_inconsistent_size(design_space):
    """Check that setting upper bound with inconsistent sized value raises an error."""
    with pytest.raises(
        ValueError, match="The upper bounds of 'x6' should be of size 1."
    ):
        design_space.set_upper_bound("x6", ones(2))


def test_bounds_check_value(design_space):
    """Check that a nan value is correctly handled as a nan and raises an error."""
    with pytest.raises(ValueError, match="Value nan of variable 'x6' is NaN."):
        design_space._check_value(array([float("nan")]), "x6")


def test_normalization():
    """Check the normalization of design variables."""
    design_space = DesignSpace()
    design_space.add_variable(
        "x_1", size=2, l_b=array([None, 0.0]), u_b=array([0.0, None])
    )
    design_space.add_variable("x_2", l_b=0.0, u_b=10.0)
    design_space.add_variable(
        "x_3", var_type=DesignVariableType.INTEGER, l_b=0.0, u_b=10.0
    )
    # Test the normalization policies:
    assert not design_space.normalize["x_1"][0]
    assert not design_space.normalize["x_1"][1]
    assert design_space.normalize["x_2"]
    assert design_space.normalize["x_3"]
    # Test the normalization:
    design_space.set_current_value(array([-10.0, 10.0, 5.0, 5]))
    current_x_norm = design_space.get_current_value(normalize=True)
    ref_current_x_norm = array([-10.0, 10.0, 0.5, 0.5])
    assert norm(current_x_norm - ref_current_x_norm) == pytest.approx(0.0)

    unnorm_curent_x = design_space.unnormalize_vect(current_x_norm)
    current_x = design_space.get_current_value()
    assert norm(unnorm_curent_x - current_x) == pytest.approx(0.0)

    x_2d = ones((5, 4))
    x_u = design_space.unnormalize_vect(x_2d)
    assert (x_u == array([1.0, 1.0, 10.0, 10.0] * 5).reshape((5, 4))).all()

    x_n = design_space.normalize_vect(x_2d)
    assert (x_n == array([1.0, 1.0, 0.1, 0.1] * 5).reshape((5, 4))).all()


def test_normalize_vect_with_integer(design_space):
    """Check that an integer vector is correctly normalized."""
    design_space.filter("x8")
    assert design_space.normalize_vect(ones(1))[0] == 0.0


@pytest.mark.parametrize(
    "vect,get_item",
    [
        (ones(1) * 0, lambda x: x[0]),
        (array([[0.0], [0.0]]), lambda x: x[0][0]),
        (array([[0.0], [0.0]]), lambda x: x[1][0]),
    ],
)
def test_unnormalize_vect_with_integer(design_space, vect, get_item):
    """Check that an integer vector is correctly unnormalized."""
    design_space.filter("x8")
    assert get_item(design_space.unnormalize_vect(vect)) == 1.0


def test_norm_policy():
    """Check the normalization policy."""
    design_space = DesignSpace()
    design_space.add_variable(
        "x_1", size=2, l_b=array([None, 0.0]), u_b=array([0.0, None])
    )

    with pytest.raises(ValueError, match="Variable 'foo' is not known."):
        design_space._add_norm_policy("foo")

    size = design_space.variables_sizes.pop("x_1")
    with pytest.raises(ValueError, match="The size of variable 'x_1' is not set."):
        design_space._add_norm_policy("x_1")

    design_space.variables_sizes["x_1"] = size
    design_space.variables_types.pop("x_1")
    with pytest.raises(
        ValueError, match="The components types of variable 'x_1' are not set."
    ):
        design_space._add_norm_policy("x_1")

    design_space.add_variable("x_c", l_b=array([0.0]), u_b=array([0.0]))
    assert not design_space.normalize["x_c"]

    design_space.add_variable("x_e", l_b=array([0.0]), u_b=array([0.0]))
    design_space.variables_types["x_e"] = array(["toto"])
    with pytest.raises(
        ValueError, match="The normalization policy for type toto is not implemented."
    ):
        design_space._add_norm_policy("x_e")

    design_space.variables_types.pop("x_e")
    with pytest.raises(
        ValueError, match="The components types of variable 'x_e' are not set."
    ):
        design_space._add_norm_policy("x_e")


def test_current_x():
    names = ["x_1", "x_2"]
    sizes = {"x_1": 1, "x_2": 2}
    l_b = {"x_1": 0.5, "x_2": (None, 2.0)}
    u_b = {"x_1": None, "x_2": (4.0, 5.0)}
    var_types = {
        "x_1": DesignVariableType.FLOAT,
        "x_2": DesignVariableType.INTEGER,
    }
    x_0 = np.array([0.5, 4.0, 4.0])
    # create the design space
    design_space = DesignSpace()

    # fill the design space
    for name in names:
        design_space.add_variable(
            name,
            size=sizes[name],
            var_type=var_types[name],
            l_b=l_b[name],
            u_b=u_b[name],
        )

    design_space.set_current_value(x_0)
    design_space.check()

    expected = re.escape("Expected current_x variables: ['x_1', 'x_2']; got ['x_1'].")
    with pytest.raises(ValueError, match=expected):
        design_space.set_current_value({"x_1": array([0.0])})

    with pytest.raises(
        ValueError,
        match=re.escape(
            "The component x_1[0] of the given array (-999.5) is lower "
            "than the lower bound (0.5) by 1.0e+03."
        ),
    ):
        design_space.set_current_value(x_0 - 1000.0)

    """
    Design Space: 3 scalar variables
    Variable   Type     Lower  Current  Upper
    x_1        float    0.5    0.5      inf
    x_2!0      integer  -inf   4        4
    x_2!1      integer  2      4        5
    """

    assert design_space.get_type("x_1") == np.array([DesignVariableType.FLOAT.value])
    assert design_space.get_type("x_3") is None

    design_space.set_current_variable("x_1", np.array([5.0]))
    assert design_space.get_current_value(as_dict=True)["x_1"][0] == 5.0

    with pytest.raises(ValueError, match="Variable 'x_3' is not known."):
        design_space.set_current_variable("x_3", 1.0)

    with pytest.raises(
        ValueError,
        match=re.escape("The bounds of variable 'error'[0] are not valid: [1.]!<[0.]."),
    ):
        design_space.add_variable("error", l_b=1.0, u_b=0.0)

    design_space = DesignSpace()
    design_space.add_variable("x", l_b=0.0, u_b=2.0)
    design_space.set_current_value({"x": None})
    assert not design_space.has_current_value()


def get_sobieski_design_space():
    """Return the design space for the Sobieski problem."""
    names = [
        "x_shared",
        "x_1",
        "x_2",
        "x_3",
        "y_14",
        "y_32",
        "y_31",
        "y_24",
        "y_34",
        "y_23",
        "y_21",
        "y_12",
    ]
    problem = SobieskiProblem()
    def_inputs = problem.get_default_inputs_equilibrium(names)

    ref_ds = DesignSpace()
    for name in names:
        value = def_inputs[name]
        l_b, u_b = problem.get_bounds_by_name([name])
        size = value.size
        ref_ds.add_variable(
            name, size=size, var_type="float", l_b=l_b, u_b=u_b, value=value
        )

    return ref_ds


def test_read_write(tmp_wd):
    """Check that read_from_txt and export_to_txt works correctly."""
    ref_ds = get_sobieski_design_space()
    f_path = Path("sobieski_design_space.txt")
    ref_ds.export_to_txt(f_path)
    read_ds = DesignSpace.read_from_txt(f_path)
    read_ds.get_lower_bounds()
    check_ds(ref_ds, read_ds, f_path)

    ds = DesignSpace.read_from_txt(TEST_INFILE)
    assert not ds.has_current_value()
    for i in range(1, 9):
        testfile = CURRENT_DIR / f"design_space_fail_{i}.txt"
        with pytest.raises(ValueError):
            DesignSpace.read_from_txt(testfile)

    for i in range(1, 4):
        testfile = CURRENT_DIR / f"design_space_{i}.txt"
        header = None
        if i == 2:
            header = ["name", "value", "lower_bound", "type", "upper_bound"]
        DesignSpace.read_from_txt(testfile, header=header)

    ds = DesignSpace.read_from_txt(TEST_INFILE)
    ds.set_lower_bound("x_shared", None)
    ds.set_upper_bound("x_shared", None)

    out_f = Path("table.txt")
    ds.export_to_txt(out_f, sortby="upper_bound")
    assert out_f.exists()


def test_dict_to_array():
    design_space = DesignSpace()
    design_space.add_variable("x", l_b=0.0, u_b=2.0)
    design_space.add_variable("y", l_b=-2.0, u_b=2.0)

    with pytest.raises(TypeError, match="x_dict values must be ndarray."):
        design_space.dict_to_array({"x": 1.0}, variable_names=["x"])

    with pytest.raises(KeyError, match="'y'"):
        design_space.dict_to_array({"x": array([1.0])})


def check_ds(ref_ds, read_ds, f_path):
    """
    :param ref_ds: param read_ds:
    :param f_path:
    :param read_ds:
    """
    assert f_path.exists()
    assert read_ds.variables_names == ref_ds.variables_names

    err = read_ds.get_lower_bounds() - ref_ds.get_lower_bounds()
    assert norm(err) == pytest.approx(0.0)

    err = read_ds.get_upper_bounds() - ref_ds.get_upper_bounds()
    assert norm(err) == pytest.approx(0.0)

    err = read_ds.get_current_value() - ref_ds.get_current_value()
    assert norm(err) == pytest.approx(0.0)

    type_read = [t for name in read_ds.variables_names for t in read_ds.get_type(name)]

    type_ref = [t for name in read_ds.variables_names for t in ref_ds.get_type(name)]

    assert type_read == type_ref

    for name in ref_ds.variables_names:
        assert name in read_ds.variables_names

    ref_str = str(ref_ds)
    assert ref_str == str(read_ds)
    assert len(ref_str) > 1000
    assert len(ref_str.split("\n")) > 20


def test_hdf5_export(tmp_wd):
    """Tests the export of a Design space in the HDF5 format."""
    ref_ds = get_sobieski_design_space()
    f_path = Path("_sobieski_design_space.h5")
    ref_ds.export_hdf(f_path)
    read_ds = DesignSpace(f_path)
    check_ds(ref_ds, read_ds, f_path)


def test_import_error_with_missing_file():
    """Check that a missing HDF file cannot be imported."""
    with pytest.raises(Exception):
        DesignSpace(hdf_file="dummy.h5")


def test_fail_import():
    """Check that a malformed HDF file cannot be imported."""
    with pytest.raises(Exception):
        DesignSpace().import_hdf(FAIL_HDF)


@pytest.fixture(scope="module")
def table_template() -> str:
    return """
+------+-------------+-------+-------------+-------+
| name | lower_bound | value | upper_bound | type  |
+------+-------------+-------+-------------+-------+
| x    |     -inf    |  None |     inf     | float |
| y{index_0} |     -inf    |  None |     inf     | float |
| y{index_1} |     -inf    |  None |     inf     | float |
+------+-------------+-------+-------------+-------+
""".strip()


@pytest.fixture(scope="module")
def design_space_2() -> DesignSpace:
    """Return a design space with scalar and vectorial variables."""
    design_space = DesignSpace()
    design_space.add_variable("x")
    design_space.add_variable("y", size=2)
    return design_space


@pytest.mark.parametrize(
    "with_index,indexes", ((True, ("[0]", "[1]")), (False, ("   ", "   ")))
)
def test_get_pretty_table(table_template, design_space_2, with_index, indexes):
    """Check that a design space is correctly rendered."""
    assert (
        table_template.format(index_0=indexes[0], index_1=indexes[1])
        == design_space_2.get_pretty_table(with_index=with_index).get_string()
    )


@pytest.mark.parametrize("name", ("", "foo"))
def test_str(table_template, design_space_2, name):
    """Check that a design space is correctly rendered."""
    if name:
        design_space_2.name = name
    table_template = f"Design space: {name}\n" + table_template
    assert table_template.format(index_0="[0]", index_1="[1]") == str(design_space_2)


@pytest.mark.parametrize(
    "normalized,expected", [(False, [-1, 0.5, 2]), (True, [0, 0.5, 1])]
)
def test_project_into_bounds(design_space, normalized, expected):
    """Tests the projection onto the design space bounds."""
    design_space.filter("x9")
    x_p = design_space.project_into_bounds([-2, 0.5, 3], normalized=normalized)
    assert norm(x_p - expected) == pytest.approx(0.0)


def test_contains(design_space):
    """Check the DesignSpace.__contains__."""
    assert "x1" in design_space
    assert "unknown_name" not in design_space


def test_len(design_space):
    """Check the length of a design space."""
    assert len(design_space) == len(design_space.variables_names)


def test_getitem(design_space):
    assert design_space["x21"] == DesignVariable(
        var_type=DesignVariableType.FLOAT.value,
        value=array([0.5]),
        size=1,
        l_b=array([-inf]),
        u_b=array([inf]),
    )

    assert design_space["x22"].value is None


def test_getitem_with_name_out_of_design_space(design_space):
    """Check that getitem with an unknown variable name raises an error."""
    expected = "Variable 'foo' is not known."
    with pytest.raises(KeyError, match=expected):
        design_space["foo"]


@pytest.mark.parametrize(
    "names,expected",
    [
        (["x10"], [0, 1, 2]),
        (["x11"], [3, 4]),
        (["x12"], [5]),
        (["x10", "x11"], [0, 1, 2, 3, 4]),
        (["x10", "x12"], [0, 1, 2, 5]),
        (["x11", "x12"], [3, 4, 5]),
    ],
)
def test_get_variables_indexes(design_space, names, expected):
    """Test the variables indexes getter."""
    design_space.filter(["x10", "x11", "x12"])
    assert (design_space.get_variables_indexes(names) == array(expected)).all()


def test_gradient_normalization(design_space):
    """Check that the normalization of the gradient performs well."""
    design_space.filter(["x18", "x19"])
    x_vect = array([0.5, 1.5])
    assert array_equal(
        design_space.unnormalize_vect(x_vect, minus_lb=False, no_check=False),
        design_space.normalize_grad(x_vect),
    )


def test_gradient_unnormalization(design_space):
    """Check that the unnormalization of the gradient performs well."""
    design_space.filter(["x18", "x19"])
    x_vect = array([0.5, 1.5])
    assert array_equal(
        design_space.normalize_vect(x_vect, minus_lb=False),
        design_space.unnormalize_grad(x_vect),
    )


def test_vartype_passed_as_bytes(design_space):
    """Check that a variable type passed as bytes is properly decoded."""
    assert design_space.variables_types["x20"] == DesignVariableType.FLOAT.value


@pytest.mark.parametrize(
    "name,kind", [("x13", "f"), ("x14", "i"), ("x16", "f"), ("x17", "i")]
)
def test_current_x_various_types(design_space, name, kind):
    """Check that set_current_value handles various types of data."""
    design_space.filter(["x13", "x14", "x15", "x16", "x17"])
    design_space.set_current_value(
        {
            "x13": array([0.5]),
            "x14": array([2.0]),
            "x15": None,
            "x16": array([1.0, 2.0]),
            "x17": array([1, 2]),
        }
    )
    assert design_space._current_value[name].dtype.kind == kind


def test_current_x_with_missing_variable(design_space):
    design_space.filter(["x13", "x14", "x15", "x16", "x17"])
    design_space.set_current_value(
        {
            "x13": array([0.5]),
            "x14": array([2.0]),
            "x15": None,
            "x16": array([1.0, 2.0]),
            "x17": array([1, 2]),
        }
    )
    assert design_space._current_value["x15"] is None


def test_design_space_name():
    """Check the naming of a design space."""
    assert DesignSpace().name is None
    assert DesignSpace(name="my_name").name == "my_name"


@pytest.fixture(scope="module")
def design_space_for_normalize_vect() -> DesignSpace:
    """A design space to check normalize_vect."""
    design_space = DesignSpace()
    design_space.add_variable(
        "x_1", 2, DesignSpace.FLOAT, array([None, 0.0]), array([0.0, None])
    )
    design_space.add_variable("x_2", 1, DesignSpace.FLOAT, 0.0, 10.0)
    design_space.add_variable("x_3", 1, DesignSpace.INTEGER, 0.0, 10.0)
    return design_space


@pytest.mark.parametrize("use_out", [False, True])
@pytest.mark.parametrize(
    "input_vec, ref",
    [
        (np.array([-10, -20, 5, 5]), np.array([-10, -20, 0.5, 0.5])),
        (np.array([-10.0, -20, 5.0, 5]), np.array([-10, -20, 0.5, 0.5])),
    ],
)
def test_normalize_vect(design_space_for_normalize_vect, input_vec, ref, use_out):
    """Test that the normalization is correctly computed whether the input values are
    floats or integers."""
    out = array([0.0, 0.0, 0.0, 0.0]) if use_out else None
    result = design_space_for_normalize_vect.normalize_vect(input_vec, out=out)
    assert result == pytest.approx(ref)
    assert (id(result) == id(out)) is use_out


@pytest.mark.parametrize(
    "input_vec, ref",
    [
        (np.array([-10, -20, 0, 1]), np.array([-10, -20, 0, 10])),
        (np.array([-10.0, -20, 0.5, 1]), np.array([-10, -20, 5, 10])),
    ],
)
def test_unnormalize_vect(input_vec, ref):
    """Test that the unnormalization is correctly computed whether the input values are
    floats or integers."""
    design_space = DesignSpace()
    design_space.add_variable(
        "x_1", 2, DesignSpace.FLOAT, array([None, 0.0]), array([0.0, None])
    )
    design_space.add_variable("x_2", 1, DesignSpace.FLOAT, 0.0, 10.0)
    design_space.add_variable("x_3", 1, DesignSpace.INTEGER, 0.0, 10.0)

    assert design_space.unnormalize_vect(input_vec) == pytest.approx(ref)


def test_unnormalize_vect_logging(caplog):
    """Check the warning logged when unnormalizing a vector."""
    design_space = DesignSpace()
    design_space.add_variable("x")  # unbounded variable
    design_space.add_variable("y", 2, l_b=-3.0, u_b=4.0)  # bounded variable
    design_space.unnormalize_vect(array([2.0, -5.0, 6.0]))
    msg = "All components of the normalized vector should be between 0 and 1; "
    msg += f"lower bounds violated: {array([-5.0])}; "
    msg += f"upper bounds violated: {array([6.0])}."
    assert ("gemseo.algos.design_space", logging.WARNING, msg) in caplog.record_tuples


def test_iter():
    """Check that a DesignSpace can be iterated."""
    design_space = DesignSpace()
    design_space.add_variable("x1")
    design_space.add_variable("x2", size=2)
    assert [name for name in design_space] == ["x1", "x2"]


def test_delitem():
    """Check that an item can be deleted with DesignSpace.__del__."""
    design_space = DesignSpace()
    design_space.add_variable("x1")
    assert design_space
    del design_space["x1"]
    assert not design_space


def test_ineq():
    """Check that DesignSpace cannot be equal to any object other than a DesignSpace."""
    design_space = DesignSpace()
    assert design_space != 1


def test_setitem():
    """Check that DesignSpace.__setitem__ works."""
    design_space = DesignSpace()
    design_space.add_variable(
        "x1", size=2, var_type=design_space.INTEGER, l_b=-1, u_b=1, value=0
    )

    new_design_space = DesignSpace()
    new_design_space["x1"] = design_space["x1"]
    assert design_space == new_design_space

    new_design_space.set_lower_bound("x1", array([0, 0]))
    assert design_space != new_design_space

    design_space.set_lower_bound("x1", array([0, 0]))
    assert design_space == new_design_space

    design_space.add_variable("x2")
    assert design_space != new_design_space

    new_design_space.add_variable("x3")
    assert design_space != new_design_space


def test_transform():
    """Check that transformation and inverse transformation works correctly."""
    parameter_space = DesignSpace()
    parameter_space.add_variable("x", l_b=0.0, u_b=2.0)
    vector = array([1.0])
    transformed_vector = parameter_space.transform_vect(vector)
    assert transformed_vector == array([0.5])
    untransformed_vector = parameter_space.untransform_vect(transformed_vector)
    assert vector == untransformed_vector


def test_setitem_from_dict():
    """Check that DesignSpace.__setitem__ works from an user dictionary."""
    design_space = DesignSpace()
    design_space["x"] = DesignVariable(l_b=2.0)
    assert design_space["x"].l_b == array([2.0])


def test_rename_variable():
    """Check the renaming of a variable."""
    design_variable = DesignVariable(2, "integer", 0.0, 2.0, array([1.0, 2.0]))

    design_space = DesignSpace()
    design_space["x"] = design_variable
    design_space.rename_variable("x", "y")

    other_design_space = DesignSpace()
    other_design_space["y"] = design_variable

    assert design_space == other_design_space


def test_rename_unknown_variable():
    """Check that a value error is raised when renaming of an unknown variable."""
    design_space = DesignSpace()
    with pytest.raises(ValueError, match="The variable x is not in the design space."):
        design_space.rename_variable("x", "y")


@pytest.mark.parametrize(
    "variables,expected",
    [
        (
            {
                "int": {
                    "size": 2,
                    "var_type": [DesignVariableType.INTEGER] * 2,
                    "value": array([1, 2]),
                },
                "float": {
                    "size": 1,
                    "var_type": DesignVariableType.FLOAT,
                    "value": array([1.0]),
                },
            },
            True,
        ),
        (
            {
                "float_1": {
                    "size": 2,
                    "var_type": [DesignVariableType.FLOAT] * 2,
                    "value": array([1, 2]),
                },
                "float_2": {
                    "size": 1,
                    "var_type": DesignVariableType.FLOAT,
                    "value": array([1.0]),
                },
            },
            False,
        ),
        (
            {
                "int_1": {
                    "size": 2,
                    "var_type": [DesignVariableType.INTEGER] * 2,
                    "value": array([1, 2]),
                },
                "int_2": {
                    "size": 1,
                    "var_type": DesignVariableType.INTEGER,
                    "value": array([1]),
                },
            },
            True,
        ),
    ],
)
def test_has_integer_variables(variables, expected):
    """Test that the correct bool is returned by the _has_integer_variables method."""
    design_space = DesignSpace()
    for key, val in variables.items():
        design_space.add_variable(
            key, size=val["size"], var_type=val["var_type"], value=val["value"]
        )

    assert design_space.has_integer_variables() == expected


@pytest.fixture(scope="module")
def design_space_with_complex_value() -> DesignSpace:
    """A design space with a float variable whose value is complex."""
    design_space = DesignSpace()
    design_space.add_variable("x")
    design_space.set_current_value({"x": array([1.0 + 0j])})
    return design_space


@pytest.mark.parametrize("cast", [False, True])
def test_get_current_x_no_complex(design_space_with_complex_value, cast):
    """Check that the complex value of a float variable is converted to float."""
    current_x = design_space_with_complex_value.get_current_value(complex_to_real=cast)
    assert (current_x.dtype.kind == "c") is not cast


DTYPE = DesignSpace._DesignSpace__DEFAULT_COMMON_DTYPE
FLOAT = DesignSpace.FLOAT
INT = DesignSpace.INTEGER


@pytest.mark.parametrize(
    "current_x,current_x_array,dtype,a_type,has_current_value",
    [
        ({"a": array([1])}, array([]), DTYPE, FLOAT, False),
        ({"a": array([1.0]), "b": array([2])}, array([1.0, 2.0]), float64, FLOAT, True),
        ({"a": array([1]), "b": array([2])}, array([1, 2]), int32, INT, True),
    ],
)
def test_current_x_setter(current_x, current_x_array, dtype, a_type, has_current_value):
    """Check that _current_value.setter updates both __current_value and metadata."""
    design_space = DesignSpace()
    design_space.add_variable("a", var_type=a_type)
    design_space.add_variable("b", var_type=design_space.INTEGER)

    design_space._current_value = current_x
    assert design_space._current_value == current_x
    assert design_space._DesignSpace__common_dtype == dtype
    assert design_space._DesignSpace__has_current_value is has_current_value


def test_cast_to_var_type(design_space: DesignSpace):
    """Test that a given value is cast to var_type in add_variable.

    Args:
        design_space: fixture that returns the design space for testing.
    """
    design_space.filter(["x23"])
    assert design_space.get_current_value() == array([1.0], dtype=np.float64)


@pytest.mark.parametrize("normalize", [True, False])
def test_normalization_casting(design_space: DesignSpace, normalize: bool):
    """Test that integer variable keep their type after unnormalization."""
    design_space.filter(["x14"])
    problem = OptimizationProblem(design_space)
    problem.objective = MDOFunction(lambda x: x, "f")
    out = problem.evaluate_functions(normalize=normalize)
    assert out[0]["f"] == array([2])
    assert out[0]["f"].dtype == int32


@pytest.fixture(scope="module")
def design_space_to_check_membership() -> DesignSpace:
    """A design space to test the method check_membership."""
    design_space = DesignSpace()
    design_space.add_variable("x", var_type=DesignSpace.INTEGER, l_b=-2, u_b=-1)
    design_space.add_variable("y", size=2, l_b=1.0, u_b=2.0)
    return design_space


@pytest.mark.parametrize(
    "x_vect,variable_names,error,error_msg",
    [
        (
            [0, 0],
            None,
            TypeError,
            (
                "The input vector should be an array or a dictionary; "
                "got a <class 'list'> instead."
            ),
        ),
        (array([-1, 1, 1]), None, None, None),
        (zeros(4), None, ValueError, "The array should be of size 3; got 4."),
        (array([-1, 1, 1]), ["x", "y"], None, None),
        (array([1, 1, -1]), ["y", "x"], None, None),
        (
            array([1, 1, 1]),
            None,
            ValueError,
            (
                "The components [0] of the given array ([1]) are greater "
                "than the upper bound ([-1.]) by [2.]."
            ),
        ),
        (
            array([-1, 1, 3]),
            None,
            ValueError,
            (
                "The components [2] of the given array ([3]) are greater "
                "than the upper bound ([2.]) by [1.]."
            ),
        ),
        ({"x": array([-1]), "y": array([1, 1])}, None, None, None),
        ({"x": array([-1]), "y": array([1, 1])}, ["x", "y"], None, None),
        ({"x": array([-1]), "y": array([1, 1])}, ["y", "x"], None, None),
        ({"x": array([-1]), "y": array([1, 1])}, ["x"], None, None),
        ({"x": array([-1]), "y": array([1, 1])}, ["y"], None, None),
        ({"x": array([-1]), "y": None}, None, None, None),
        (
            {"x": array([-1.5]), "y": None},
            None,
            ValueError,
            "The variable x is of type integer; got x[0] = -1.5.",
        ),
        (
            {"x": array([-1]), "y": array([1, 1, 1])},
            None,
            ValueError,
            "The variable y of size 2 cannot be set with an array of size 3.",
        ),
        (
            {"x": array([-1]), "y": array([1, 0])},
            None,
            ValueError,
            (
                "The component y[1] of the given array (0) is lower "
                "than the lower bound (1.0) by 1.0e+00."
            ),
        ),
        (
            {"x": array([-1]), "y": array([1, 3])},
            None,
            ValueError,
            (
                "The component y[1] of the given array (3) is greater "
                "than the upper bound (1.0) by 1.0e+00."
            ),
        ),
    ],
)
def test_check_membership(
    design_space_to_check_membership, x_vect, variable_names, error, error_msg
):
    """Check the method check_membership."""
    ds = design_space_to_check_membership

    if error is None:
        ds.check_membership(x_vect, variable_names)
        if variable_names is None and isinstance(x_vect, ndarray):
            assert_equal(ds._DesignSpace__lower_bounds_array, array([-2, 1, 1]))
            assert_equal(ds._DesignSpace__upper_bounds_array, array([-1, 2, 2]))
    else:
        with pytest.raises(error, match=re.escape(error_msg)):
            ds.check_membership(x_vect, variable_names)


@pytest.mark.parametrize(
    "l_b,expected_lb", [(-5, array([-5, -5])), (array([-5, -inf]), array([-5, -inf]))]
)
@pytest.mark.parametrize(
    "u_b,expected_ub", [(5, array([5, 5])), (array([5, inf]), array([5, inf]))]
)
def test_infinity_bounds_for_int(l_b, u_b, expected_lb, expected_ub):
    """Check that integer variables can handle -/+ infinity bounds.

    Args:
        l_b: The lower bounds.
        u_b: The upper bounds.
        expected_ub: The expected upper bounds.
        expected_lb: The expected lower bounds.
    """
    ds = DesignSpace()
    ds.add_variable("x", 2, l_b=l_b, u_b=u_b, var_type=DesignVariableType.INTEGER)
    assert array_equal(ds._lower_bounds["x"], expected_lb)
    assert array_equal(ds._upper_bounds["x"], expected_ub)


@pytest.fixture(scope="module")
def fbb_design_space() -> DesignSpace:
    """Foo-bar-baz (fbb) design space for test_get_current_value()."""
    design_space = DesignSpace()
    design_space.add_variable("foo", l_b=1.0, value=1.0)
    design_space.add_variable("bar", size=2, l_b=1.0, u_b=3.0, value=2.0)
    design_space.add_variable("baz", l_b=1.0, u_b=3.0, value=3.0)
    design_space.set_current_variable("baz", array([3.0 + 0.5j]))
    return design_space


@pytest.mark.parametrize("names", [None, ["baz", "foo"]])
@pytest.mark.parametrize("cast", [False, True])
@pytest.mark.parametrize("normalize", [False, True])
@pytest.mark.parametrize("as_dict", [False, True])
def test_get_current_value(fbb_design_space, names, cast, normalize, as_dict):
    """Check get_current_value() with all the combinations of its arguments."""
    result = fbb_design_space.get_current_value(
        variable_names=names,
        complex_to_real=cast,
        normalize=normalize,
        as_dict=as_dict,
    )
    if normalize:
        expected = {
            "foo": array([1.0]),
            "bar": array([0.5, 0.5]),
            "baz": array([1.0 + 0.25j]),
        }
    else:
        expected = {
            "foo": array([1.0]),
            "bar": array([2.0, 2.0]),
            "baz": array([3.0 + 0.5j]),
        }

    names = names or ["foo", "bar", "baz"]
    expected = {k: v for k, v in expected.items() if k in names}

    if cast:
        expected = {k: v.real for k, v in expected.items()}

    if not as_dict:
        expected = fbb_design_space.dict_to_array(expected, variable_names=names)

    assert_equal(result, expected)


@pytest.mark.parametrize("as_dict", [True, False])
def test_get_current_value_empty_names(as_dict):
    design_space = DesignSpace()
    assert not design_space.get_current_value(variable_names=[], as_dict=as_dict)


def test_get_current_value_bad_names():
    design_space = DesignSpace()
    match = "There are no such variables named: bar, foo."
    with pytest.raises(ValueError, match=match):
        design_space.get_current_value(variable_names=["foo", "bar"])


@pytest.mark.parametrize(
    "l_b,u_b,value",
    [
        (None, None, array([0, 0])),
        (array([1, 2]), None, array([1, 2])),
        (array([1, 2]), array([2, 4]), array([1, 3])),
        (None, array([1, 2]), array([1, 2])),
    ],
)
def test_initialize_missing_current_values(l_b, u_b, value):
    """Check the initialization of the missing current values."""
    design_space = DesignSpace()
    design_space.add_variable(
        "x", size=2, var_type=design_space.INTEGER, l_b=l_b, u_b=u_b
    )
    design_space.initialize_missing_current_values()
    assert_equal(design_space["x"].value, value)


def test_get_current_value_order():
    """Check that the order of variables is correctly handled in get_current_value."""
    design_space = DesignSpace()
    design_space.add_variable("x", value=0.0)
    design_space.add_variable("y", value=1.0)
    assert_equal(design_space.get_current_value(), array([0.0, 1.0]))
    assert_equal(
        design_space.get_current_value(variable_names=["x", "y"]),
        array([0.0, 1.0]),
    )
    assert_equal(
        design_space.get_current_value(variable_names=["y", "x"]),
        array([1.0, 0.0]),
    )


def test_export_import_with_none_value(tmp_wd):
    """Check that a design space exported without default value can be imported."""
    design_space = DesignSpace()
    design_space.add_variable("x")
    design_space.export_to_txt("foo.txt")
    txt_design_space = DesignSpace.read_from_txt("foo.txt")
    assert txt_design_space == design_space
