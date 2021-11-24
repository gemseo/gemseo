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
#    INITIAL AUTHORS - initial API and implementation and/or initial
#                           documentation
#        :author: Charlie Vanaret
#    OTHER AUTHORS   - MACROSCOPIC CHANGES

from __future__ import division, unicode_literals

import logging
import re

import numpy as np
import pytest
from numpy import array, array_equal, inf, int32, ones
from numpy.linalg import norm

from gemseo.algos.design_space import DesignSpace, DesignVariable, DesignVariableType
from gemseo.algos.opt_result import OptimizationResult
from gemseo.problems.sobieski.core import SobieskiProblem
from gemseo.third_party.prettytable.prettytable import PY2
from gemseo.utils.py23_compat import Path
from gemseo.utils.string_tools import MultiLineString

CURRENT_DIR = Path(__file__).parent
TEST_INFILE = CURRENT_DIR / "design_space.txt"
FAIL_HDF = CURRENT_DIR / "fail.hdf5"


@pytest.fixture
def design_space():
    """The main design space to be used by the test function.

    Feel free to add new variables.
    """
    ds = DesignSpace()
    ds.add_variable("x1", size=1, var_type=DesignVariableType.FLOAT, l_b=0.0, u_b=2.0)
    ds.add_variable("x2", size=1, var_type=DesignVariableType.FLOAT, l_b=-2.0, u_b=2.0)
    ds.add_variable("x3", size=1, var_type=DesignVariableType.INTEGER, l_b=0, u_b=2)
    ds.add_variable("x4", size=1, var_type="float", l_b=-1.0, u_b=0.0, value=-0.5)
    ds.add_variable("x5", size=3, var_type="float", l_b=-1.0, u_b=0.0, value=-0.5)
    ds.add_variable("x6", size=1, var_type=DesignVariableType.FLOAT, l_b=None, u_b=2.0)
    ds.add_variable("x7", size=1, var_type=DesignVariableType.FLOAT, l_b=0.0, u_b=None)
    ds.add_variable("x8", size=1, var_type=DesignVariableType.INTEGER, l_b=1, u_b=1)
    ds.add_variable("x9", size=3, var_type=DesignVariableType.FLOAT, l_b=-1.0, u_b=2.0)
    ds.add_variable("x10", size=3)
    ds.add_variable("x11", size=2)
    ds.add_variable("x12", size=1)
    ds.add_variable("x13", var_type=DesignVariableType.FLOAT, value=array([0.5]))
    ds.add_variable("x14", var_type=DesignVariableType.INTEGER, value=array([2.0]))
    ds.add_variable("x15", value=None)
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
    with pytest.raises(ValueError, match="Value nan of variable 'varname' is nan."):
        design_space.add_variable(name="varname", **{arg: float("nan")})


@pytest.mark.parametrize("arg,side", [("l_b", "lower"), ("u_b", "upper")])
def test_add_variable_with_inconsistent_bound_size(design_space, arg, side):
    """Check that using bounds with inconsistent size raises an."""
    with pytest.raises(
        ValueError, match="The {} bounds of 'varname' should be of size 3.".format(side)
    ):
        design_space.add_variable(name="varname", size=3, **{arg: [0.0, 0.0]})


def test_add_variable_with_upper_bounds_lower_than_lower_ones(design_space):
    """Check that using upper bounds lower than lower ones raises an error."""
    with pytest.raises(
        ValueError,
        match=re.escape(
            "The bounds of variable 'varname'[1] are not valid: [0.]!<[1.]."
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
        match=re.escape(
            "Cannot compute normalized current value since "
            "The design space has no current value for 'varname'.."
        ),
    ):
        design_space.get_current_x_normalized()


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
    design_space.set_current_x(current_x)
    x_n = design_space.get_current_x_normalized()
    assert (x_n == 0.5).all()


def test_set_current_value_with_malformed_mapping_arg(design_space):
    """Check that setting the current value from a malformed mapping raises an error."""
    design_space.filter("x1")
    with pytest.raises(
        Exception, match="The component 'x1' of the given array should have size 1."
    ):
        design_space.set_current_x({"x1": array([1.0, 1.0])})


def test_set_current_value_with_malformed_opt_arg(design_space):
    """Check that setting the current value from a malformed optimization result raises
    an error."""
    with pytest.raises(
        Exception,
        match="Invalid x_opt, dimension mismatch: {} != 1".format(
            design_space.dimension
        ),
    ):
        design_space.set_current_x(OptimizationResult(x_opt=array([1.0])))


def test_set_current_value_with_malformed_current_x(design_space):
    """Check that setting the current value from a float raises an error."""
    if PY2:
        keyword = "type"
    else:
        keyword = "class"
    with pytest.raises(
        TypeError,
        match=(
            "The current point should be either an array, "
            "a dictionary of arrays or an optimization result; "
            "got <{} 'float'> instead.".format(keyword)
        ),
    ):
        design_space.set_current_x(1.0)


def test_read_from_txt():
    """Check that a variable name is correct when reading a txt file."""
    ds = DesignSpace.read_from_txt(CURRENT_DIR / "design_space_4.txt")
    assert ds.variables_names == ["x_shared"]


def test_integer_variable_set_current_x(design_space):
    """Check that an integer value is correctly set."""
    design_space.filter("x3")
    x_i = array([0], dtype=int32)
    design_space.set_current_x(x_i)
    x_i_conv = design_space.dict_to_array(design_space.array_to_dict(x_i))
    assert x_i_conv.dtype == x_i.dtype
    assert x_i_conv == x_i


def test_integer_variable_round_vect(design_space):
    """Check that a float value is correctly rounded to the closest integer."""
    design_space.filter("x3")
    assert design_space.round_vect(array([1.2])) == 1
    assert design_space.round_vect(array([1.9])) == 2
    expected = design_space.round_vect(array([[1.9], [1.2]]))
    assert (expected == array([[2], [1]])).all()


def test_integer_variable_round_vect_with_malformed_value(design_space):
    """Check that round a 3d value raises an error."""
    design_space.filter("x3")
    with pytest.raises(
        ValueError, match="The array to be unnormalized must be 1d or 2d; got 3d."
    ):
        design_space.round_vect(array([[[1.0]]]))


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
    design_space.add_variable(
        "x1", size=1, var_type="float", l_b=-1.0, u_b=0.0, value=-0.5
    )
    other = DesignSpace()
    other.add_variable("x2", size=3, var_type="float", l_b=-1.0, u_b=0.0, value=-0.5)
    design_space.extend(other)
    assert design_space.__contains__("x2")
    assert design_space.get_size("x2") == other.get_size("x2")
    assert (design_space.get_type("x2") == other.get_type("x2")).all()
    assert (design_space.get_lower_bound("x2") == other.get_lower_bound("x2")).all()
    assert (design_space.get_upper_bound("x2") == other.get_upper_bound("x2")).all()
    assert (design_space.get_current_x(["x2"]) == other.get_current_x(["x2"])).all()


def test_active_bounds():
    """Check whether active bounds are correctly identified."""

    design_space = DesignSpace()
    design_space.add_variable(
        "x", size=1, var_type=DesignVariableType.FLOAT, l_b=0.0, u_b=2.0
    )
    design_space.add_variable(
        "y", size=1, var_type=DesignVariableType.FLOAT, l_b=-2.0, u_b=2.0
    )
    design_space.add_variable("z", size=1, var_type=DesignVariableType.FLOAT)
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

    if PY2:
        keyword1 = "type"
        keyword2 = "unicode"
    else:
        keyword1 = "class"
        keyword2 = "str"

    with pytest.raises(
        TypeError,
        match=(
            "Expected dict or array for x_vec argument; got <{} '{}'>.".format(
                keyword1, keyword2
            )
        ),
    ):
        design_space.get_active_bounds("test")

    with pytest.raises(
        KeyError, match="The design space has no current value for 'x'."
    ):
        design_space.get_active_bounds()


@pytest.mark.parametrize("index,expected", [(0, "x"), (1, "z!0"), (2, "z!1")])
def test_get_indexed_variables_names(index, expected):
    """Check the variables names obtained with get_indexed_variables_names()."""
    design_space = DesignSpace()
    design_space.add_variable("x", size=1)
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
    with pytest.raises(ValueError, match="Value nan of variable 'x6' is nan."):
        design_space.set_lower_bound("x6", float("nan"))


def test_bounds_set_lower_bound_with_inconsistent_size(design_space):
    """Check that setting lower bound with inconsistent sized value raises an error."""
    with pytest.raises(
        ValueError, match="The lower bounds of 'x6' should be of size 1."
    ):
        design_space.set_lower_bound("x6", ones(2))


def test_bounds_set_upper_bound_with_nan(design_space):
    """Check that setting upper bound with nan raises an error."""
    with pytest.raises(ValueError, match="Value nan of variable 'x6' is nan."):
        design_space.set_upper_bound("x6", float("nan"))


def test_bounds_set_upper_bound_with_inconsistent_size(design_space):
    """Check that setting upper bound with inconsistent sized value raises an error."""
    with pytest.raises(
        ValueError, match="The upper bounds of 'x6' should be of size 1."
    ):
        design_space.set_upper_bound("x6", ones(2))


def test_bounds_check_value(design_space):
    """Check that a nan value is correctly handled as a nan and raises an error."""
    with pytest.raises(ValueError, match="Value nan of variable 'x6' is nan."):
        design_space._check_value(array([float("nan")]), "x6")


def test_normalization():
    """Check the normalization of design variables."""
    design_space = DesignSpace()
    design_space.add_variable(
        "x_1",
        size=2,
        var_type=DesignVariableType.FLOAT,
        l_b=array([None, 0.0]),
        u_b=array([0.0, None]),
    )
    design_space.add_variable(
        "x_2", size=1, var_type=DesignVariableType.FLOAT, l_b=0.0, u_b=10.0
    )
    design_space.add_variable(
        "x_3",
        size=1,
        var_type=DesignVariableType.INTEGER,
        l_b=0.0,
        u_b=10.0,
    )
    # Test the normalization policies:
    assert not design_space.normalize["x_1"][0]
    assert not design_space.normalize["x_1"][1]
    assert design_space.normalize["x_2"]
    assert design_space.normalize["x_3"]
    # Test the normalization:
    design_space.set_current_x(array([-10.0, 10.0, 5.0, 5]))
    current_x_norm = design_space.get_current_x_normalized()
    ref_current_x_norm = array([-10.0, 10.0, 0.5, 0.5])
    assert norm(current_x_norm - ref_current_x_norm) == pytest.approx(0.0)

    unnorm_curent_x = design_space.unnormalize_vect(current_x_norm)
    current_x = design_space.get_current_x()
    assert norm(unnorm_curent_x - current_x) == pytest.approx(0.0)

    with pytest.raises(ValueError):
        design_space.normalize_vect(ones((2, 2, 2)))

    x_2d = ones((5, 4))
    x_u = design_space.unnormalize_vect(x_2d)
    assert (x_u == array([1.0, 1.0, 10.0, 10.0] * 5).reshape((5, 4))).all()

    x_n = design_space.normalize_vect(x_2d)
    assert (x_n == array([1.0, 1.0, 0.1, 0.1] * 5).reshape((5, 4))).all()

    with pytest.raises(
        ValueError, match="The array to be normalized must be 1d or 2d."
    ):
        design_space.normalize_vect(ones((2, 2, 2)))

    with pytest.raises(
        ValueError, match="The array to be unnormalized must be 1d or 2d, got 3d."
    ):
        design_space.unnormalize_vect(ones((2, 2, 2)))


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
        "x_1",
        size=2,
        var_type=DesignVariableType.FLOAT,
        l_b=array([None, 0.0]),
        u_b=array([0.0, None]),
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

    design_space.add_variable(
        "x_c",
        size=1,
        var_type=DesignVariableType.FLOAT,
        l_b=array([0.0]),
        u_b=array([0.0]),
    )
    assert not design_space.normalize["x_c"]

    design_space.add_variable(
        "x_e",
        size=1,
        var_type=DesignVariableType.FLOAT,
        l_b=array([0.0]),
        u_b=array([0.0]),
    )
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

    design_space.set_current_x(x_0)
    design_space.check()
    design_space.check_membership(2 * ones(3))

    with pytest.raises(
        ValueError, match=re.escape("The dimension of the input array (2) should be 3.")
    ):
        design_space.check_membership(2 * ones(2))

    if PY2:
        keyword = "type"
    else:
        keyword = "class"

    with pytest.raises(
        TypeError,
        match=(
            "The input vector should be an array or a dictionary; "
            "got <{} 'list'> instead.".format(keyword)
        ),
    ):
        design_space.check_membership([2.0] * 3)

    with pytest.raises(
        ValueError,
        match=re.escape(
            "The component 'x_2'!0 of the given array (6.0) is greater "
            "than the upper bound (4.0) by -2.0e+00."
        ),
    ):
        design_space.check_membership(6 * ones(3))

    with pytest.raises(
        ValueError,
        match=(
            "'x_2'!0 of the given array is not an integer "
            "while variable is of type integer! Value = 2.5"
        ),
    ):
        design_space.check_membership({"x_1": ones(1), "x_2": 2.5 * ones(2)})

    expected = re.escape("Expected current_x variables: ['x_1', 'x_2']; got ['x_1'].")
    if PY2:
        with pytest.raises(ValueError):
            design_space.set_current_x({"x_1": 0.0})
    else:
        with pytest.raises(ValueError, match=expected):
            design_space.set_current_x({"x_1": 0.0})

    with pytest.raises(
        ValueError,
        match=re.escape(
            "The component x_1!0 of the given array (-999.5) is lower "
            "than the lower bound (0.5) by 1.0e+03."
        ),
    ):
        design_space.set_current_x(x_0 - 1000.0)

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
    assert design_space.get_current_x_dict()["x_1"][0] == 5.0

    with pytest.raises(ValueError, match="Variable 'x_3' is not known."):
        design_space.set_current_variable("x_3", 1.0)

    with pytest.raises(
        ValueError,
        match=re.escape("The bounds of variable 'error'[0] are not valid: [0.]!<[1.]."),
    ):
        design_space.add_variable("error", l_b=1.0, u_b=0)

    design_space = DesignSpace()
    design_space.add_variable(
        "x", size=1, var_type=DesignVariableType.FLOAT, l_b=0.0, u_b=2.0
    )
    design_space.set_current_x({"x": None})
    assert not design_space.has_current_x()


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
    f_path = tmp_wd / "sobieski_design_space.txt"
    ref_ds.export_to_txt(f_path)
    read_ds = DesignSpace.read_from_txt(f_path)
    read_ds.get_lower_bounds()
    check_ds(ref_ds, read_ds, f_path)

    ds = DesignSpace.read_from_txt(TEST_INFILE)
    assert not ds.has_current_x()
    for i in range(1, 9):
        testfile = CURRENT_DIR / "design_space_fail_{}.txt".format(i)
        with pytest.raises(ValueError):
            DesignSpace.read_from_txt(testfile)

    for i in range(1, 4):
        testfile = CURRENT_DIR / "design_space_{}.txt".format(i)
        header = None
        if i == 2:
            header = ["name", "value", "lower_bound", "type", "upper_bound"]
        DesignSpace.read_from_txt(testfile, header=header)

    ds = DesignSpace.read_from_txt(TEST_INFILE)
    ds.set_lower_bound("x_shared", None)
    ds.set_upper_bound("x_shared", None)

    out_f = tmp_wd / "table.txt"
    ds.export_to_txt(out_f, sortby="upper_bound")
    assert out_f.exists()


def test_dict_to_array():
    design_space = DesignSpace()
    design_space.add_variable(
        "x", size=1, var_type=DesignVariableType.FLOAT, l_b=0.0, u_b=2.0
    )
    design_space.add_variable(
        "y", size=1, var_type=DesignVariableType.FLOAT, l_b=-2.0, u_b=2.0
    )

    with pytest.raises(TypeError, match="x_dict values must be ndarray."):
        design_space.dict_to_array({"x": 1.0})

    with pytest.raises(KeyError, match="'y'"):
        design_space.dict_to_array({"x": array([1.0])})

    x = design_space.dict_to_array({"x": array([1.0])}, False)
    assert x == 1.0


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

    err = read_ds.get_current_x() - ref_ds.get_current_x()
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
    f_path = tmp_wd / "_sobieski_design_space.h5"
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


def test_get_pretty_table():
    """Check that a design space is correctly rendered."""
    design_space = DesignSpace()
    design_space.add_variable("x")
    msg = MultiLineString()
    msg.add("+------+-------------+-------+-------------+-------+")
    msg.add("| name | lower_bound | value | upper_bound | type  |")
    msg.add("+------+-------------+-------+-------------+-------+")
    msg.add("| x    |     -inf    |  None |     inf     | float |")
    msg.add("+------+-------------+-------+-------------+-------+")
    assert str(msg) == design_space.get_pretty_table().get_string()


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
    """Check that set_current_x handles various types of data."""
    design_space.filter(["x13", "x14", "x15", "x16", "x17"])
    design_space.set_current_x(
        {
            "x13": array([0.5]),
            "x14": array([2.0]),
            "x15": None,
            "x16": array([1.0, 2.0]),
            "x17": array([1, 2]),
        }
    )
    assert design_space._current_x[name].dtype.kind == kind


def test_current_x_with_missing_variable(design_space):
    design_space.filter(["x13", "x14", "x15", "x16", "x17"])
    design_space.set_current_x(
        {
            "x13": array([0.5]),
            "x14": array([2.0]),
            "x15": None,
            "x16": array([1.0, 2.0]),
            "x17": array([1, 2]),
        }
    )
    assert design_space._current_x["x15"] is None


def test_design_space_name():
    """Check the naming of a design space."""
    assert DesignSpace().name is None
    assert DesignSpace(name="my_name").name == "my_name"


@pytest.mark.parametrize(
    "input_vec, ref",
    [
        (np.array([-10, -20, 5, 5]), np.array([-10, -20, 0.5, 0.5])),
        (np.array([-10.0, -20, 5.0, 5]), np.array([-10, -20, 0.5, 0.5])),
    ],
)
def test_normalize_vect(input_vec, ref):
    """Test that the normalization is correctly computed whether the input values are
    floats or integers."""
    design_space = DesignSpace()
    design_space.add_variable(
        "x_1", 2, DesignSpace.FLOAT, array([None, 0.0]), array([0.0, None])
    )
    design_space.add_variable("x_2", 1, DesignSpace.FLOAT, 0.0, 10.0)
    design_space.add_variable("x_3", 1, DesignSpace.INTEGER, 0.0, 10.0)

    assert design_space.normalize_vect(input_vec) == pytest.approx(ref)


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
    design_space.add_variable("x", 1)  # unbounded variable
    design_space.add_variable("y", 2, l_b=-3.0, u_b=4.0)  # bounded variable
    design_space.unnormalize_vect(array([2.0, -5.0, 6.0]))
    msg = "All components of the normalized vector should be between 0 and 1."
    msg += " Lower bounds violated: {}.".format(array([-5.0]))
    msg += " Upper bounds violated: {}.".format(array([6.0]))
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
