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
"""Tests for the categorical variables of a design space."""

from __future__ import annotations

import pickle
from copy import deepcopy
from pathlib import Path

import h5py
import pytest
from numpy import array
from numpy import float64
from numpy import int64
from numpy.testing import assert_array_equal
from scipy.sparse import csr_array

from gemseo.space.design import DesignSpace
from gemseo.space.variable import CategoricalVariable
from gemseo.util.testing.helper import assert_exception


@pytest.fixture
def categorical_design_space() -> DesignSpace:
    """A design space mixing a categorical variable and a real one."""
    design_space = DesignSpace()
    design_space.add_categorical_variable(
        "material", ("steel", "aluminium", "titanium"), value="aluminium"
    )
    design_space.add_real_variable("x", lower_bound=0.0, upper_bound=2.0, value=1.0)
    return design_space


def test_add_categorical_variable(categorical_design_space) -> None:
    """Check the addition of a categorical variable with a value."""
    variable = categorical_design_space.variables["material"]
    assert variable.type == DesignSpace.DesignVariableType.CATEGORICAL
    assert variable.categories == ("steel", "aluminium", "titanium")
    # The current value is the position of the label.
    assert_array_equal(
        categorical_design_space.get_current_value(["material"]), array([1])
    )
    assert categorical_design_space.get_current_value().dtype == float64
    assert categorical_design_space.get_current_value(["material"]).dtype == int64


def test_add_categorical_variable_without_value() -> None:
    """Check that a categorical variable can be added without a value."""
    design_space = DesignSpace()
    design_space.add_categorical_variable("material", ["steel", "aluminium"])
    assert design_space._current_value["material"] is None
    design_space.initialize_missing_current_values()
    assert_array_equal(design_space.get_current_value(), array([0]))


@pytest.mark.parametrize(
    "kwargs",
    [
        pytest.param(
            {"categories": ["steel", "aluminium"], "value": "gold"}, id="unknown-label"
        ),
        pytest.param({"categories": ["steel", "aluminium"], "value": 1}, id="integer"),
        pytest.param(
            {"categories": ["steel", "aluminium"], "value": ["steel", "gold"]},
            id="several-labels",
        ),
        pytest.param({"categories": ["steel", "steel"]}, id="duplicated-categories"),
    ],
)
def test_add_categorical_variable_with_invalid_arguments(kwargs, snapshot) -> None:
    """Check that a categorical variable with invalid arguments is not registered."""
    design_space = DesignSpace()
    with assert_exception(ValueError, snapshot):
        design_space.add_categorical_variable("material", **kwargs)

    assert "material" not in design_space


@pytest.mark.parametrize(
    ("value", "names", "expected"),
    [
        pytest.param(
            {"material": "titanium", "x": array([0.5])},
            ["material"],
            array([2]),
            id="label",
        ),
        pytest.param(
            {"material": array(["steel"]), "x": array([0.5])},
            ["material"],
            array([0]),
            id="label-array",
        ),
        # Coordinates are accepted as well, in a mapping or in a vector.
        pytest.param(
            {"material": array([2]), "x": array([0.5])},
            None,
            array([2, 0.5]),
            id="coordinate-mapping",
        ),
        pytest.param(array([1.0, 1.5]), None, array([1, 1.5]), id="vector"),
    ],
)
def test_set_current_value_with_labels(
    categorical_design_space, value, names, expected
) -> None:
    """Check that the current value of a categorical variable can be set by labels."""
    categorical_design_space.set_current_value(value)
    assert_array_equal(categorical_design_space.get_current_value(names), expected)


def test_set_current_value_does_not_modify_the_mapping(
    categorical_design_space,
) -> None:
    """Check that setting labels leaves the mapping of the caller untouched."""
    value = {"material": "titanium", "x": array([0.5])}
    categorical_design_space.set_current_value(value)
    assert value["material"] == "titanium"


@pytest.mark.parametrize("material", ["gold", array([3]), array([0.5])])
def test_set_current_value_with_an_invalid_categorical_value(
    categorical_design_space, material, snapshot
) -> None:
    """Check that an invalid label or coordinate is rejected without side effect."""
    with assert_exception(ValueError, snapshot):
        categorical_design_space.set_current_value({
            "material": material,
            "x": array([0.5]),
        })

    assert_array_equal(categorical_design_space.get_current_value(), array([1, 1.0]))


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        pytest.param(array(["titanium"]), array([2]), id="label-array"),
        pytest.param("steel", array([0]), id="label"),
        pytest.param(array([1]), array([1]), id="coordinate"),
        pytest.param(None, None, id="none"),
    ],
)
def test_set_current_variable_with_labels(
    categorical_design_space, value, expected
) -> None:
    """Check that the current value of a single variable can be set by labels."""
    categorical_design_space.set_current_variable("material", value)
    if expected is None:
        assert categorical_design_space._current_value["material"] is None
    else:
        assert_array_equal(
            categorical_design_space.get_current_value(["material"]), expected
        )


def test_reference_value(categorical_design_space) -> None:
    """Check that the reference value has labels while the current one has positions."""  # noqa: E501
    reference_value = categorical_design_space.reference_value
    assert_array_equal(reference_value["material"], array(["aluminium"]))
    assert_array_equal(reference_value["x"], array([1.0]))
    current_value = categorical_design_space.get_current_value(as_dict=True)
    assert_array_equal(current_value["material"], array([1]))


def test_reference_value_without_current_value() -> None:
    """Check that the reference value is empty when a categorical value is missing."""
    design_space = DesignSpace()
    design_space.add_categorical_variable("material", ["steel"])
    assert design_space.reference_value == {}


@pytest.mark.parametrize(
    ("method_name", "args"),
    [
        pytest.param("set_lower_bound", ("material", 0), id="set-lower-bound"),
        pytest.param("set_upper_bound", ("material", 2), id="set-upper-bound"),
    ],
)
def test_categorical_variable_has_no_bounds(
    categorical_design_space, method_name, args, snapshot
) -> None:
    """Check that the bounds of a categorical variable cannot be read or written."""
    with assert_exception(TypeError, snapshot):
        getattr(categorical_design_space, method_name)(*args)


def test_bounds_of_the_coordinates(categorical_design_space) -> None:
    """Check that the bounds of the full vector are those of the coordinates."""
    assert_array_equal(categorical_design_space.get_lower_bounds(), array([0, 0.0]))
    assert_array_equal(categorical_design_space.get_upper_bounds(), array([2, 2.0]))
    assert_array_equal(
        categorical_design_space.get_lower_bounds(["x"]),
        array([0.0]),
    )
    # The same holds when the names are given.
    assert_array_equal(
        categorical_design_space.get_lower_bounds(["material", "x"]), array([0, 0.0])
    )
    assert_array_equal(
        categorical_design_space.get_upper_bounds(["material"]), array([2])
    )


def test_bounds_as_dict_leave_out_categorical_variables(
    categorical_design_space,
) -> None:
    """Check that the dictionary views of the bounds leave out categorical variables."""
    lower_bounds = categorical_design_space.get_lower_bounds(as_dict=True)
    upper_bounds = categorical_design_space.get_upper_bounds(as_dict=True)
    assert list(lower_bounds) == ["x"]
    assert list(upper_bounds) == ["x"]
    assert_array_equal(upper_bounds["x"], array([2.0]))
    # Also when the names are given.
    assert not categorical_design_space.get_upper_bounds(["material"], as_dict=True)

    # The dictionaries can be fed back.
    for name, bound in lower_bounds.items():
        categorical_design_space.set_lower_bound(name, bound - 1)
    for name, bound in upper_bounds.items():
        categorical_design_space.set_upper_bound(name, bound + 1)
    assert_array_equal(categorical_design_space.get_lower_bounds(["x"]), [-1.0])
    assert_array_equal(categorical_design_space.get_upper_bounds(["x"]), [3.0])


def test_categorical_variable_has_no_active_bounds(categorical_design_space) -> None:
    """Check that a categorical variable has no active bound."""
    lower, upper = categorical_design_space.get_active_bounds(
        array([0.0, 2.0]), tol=1e-8
    )
    assert_array_equal(lower["material"], array([False]))
    assert_array_equal(upper["material"], array([False]))
    assert_array_equal(lower["x"], array([False]))
    assert_array_equal(upper["x"], array([True]))


@pytest.mark.parametrize(
    "value",
    [
        array([0.0, 1.0]),
        array([2.0, 2.0]),
        {"material": array([1]), "x": array([0.5])},
        array([[0.0, 1.0], [1.0, 1.0]]),
    ],
)
def test_categorical_check_membership(categorical_design_space, value) -> None:
    """Check the membership of valid values."""
    categorical_design_space.check_membership(value)


@pytest.mark.parametrize(
    "value",
    [
        array([3.0, 1.0]),
        array([-1.0, 1.0]),
        array([0.5, 1.0]),
        array([0.0, 3.0]),
        array([[0.0, 1.0], [3.0, 1.0]]),
        {"material": array([3]), "x": array([0.5])},
        {"material": array([0.5]), "x": array([0.5])},
        {"material": array([1]), "x": array([2.5])},
    ],
)
def test_categorical_check_membership_error(
    categorical_design_space, value, snapshot
) -> None:
    """Check the membership of invalid values."""
    with assert_exception(ValueError, snapshot):
        categorical_design_space.check_membership(value)


@pytest.mark.parametrize(
    ("n_categories", "unit_value", "position"),
    [
        (3, 0.0, 0),
        (3, 0.2, 0),
        (3, 0.33, 0),
        (3, 0.34, 1),
        (3, 0.5, 1),
        (3, 0.66, 1),
        (3, 0.67, 2),
        (3, 1.0, 2),
        (2, 0.0, 0),
        (2, 0.49, 0),
        (2, 0.5, 1),
        (2, 1.0, 1),
        (1, 0.0, 0),
        (1, 0.7, 0),
        (1, 1.0, 0),
    ],
)
def test_categorical_cells(n_categories, unit_value, position) -> None:
    """Check that each category owns a cell of the unit interval.

    The mapping is not an affine map of the bounds
    but a partition of the unit interval into cells of the same measure.
    """
    design_space = DesignSpace()
    design_space.add_categorical_variable("c", [str(i) for i in range(n_categories)])
    design_space.add_real_variable("x", lower_bound=0.0, upper_bound=10.0, value=1.0)
    design_space.set_current_value({"c": array([0]), "x": array([1.0])})
    with design_space._prepare_untransformation(True):
        samples = design_space.untransform_vect(
            array([[unit_value, 0.5]]), no_check=True
        )
        assert_array_equal(samples, array([[position, 5.0]]))
        # The normalization of a point gives the centre of its cell,
        # which the denormalization maps back to the same point.
        unit_point = design_space.transform_vect(samples[0])
        assert unit_point[0] == pytest.approx((position + 0.5) / n_categories)
        assert unit_point[1] == pytest.approx(0.5)
        assert_array_equal(
            design_space.untransform_vect(unit_point, no_check=True), samples[0]
        )


def test_categorical_cells_only_variable() -> None:
    """Check the denormalization of a design space with a categorical variable only."""
    design_space = DesignSpace()
    design_space.add_categorical_variable("c", ["a", "b", "c"], "a")
    with design_space._prepare_untransformation(True):
        samples = design_space.untransform_vect(
            array([[0.0], [0.5], [1.0]]), no_check=True
        )

    assert_array_equal(samples, array([[0], [1], [2]]))


def test_categorical_variable_is_not_normalized_by_default(
    categorical_design_space,
) -> None:
    """Check that the positions are left as is outside the DOE context."""
    assert_array_equal(
        categorical_design_space.name_to_normalization_mask["material"], [False]
    )
    assert_array_equal(
        categorical_design_space.normalize_vect(array([2.0, 1.0])), array([2.0, 0.5])
    )


def test_check_membership_of_many_points_is_vectorized(
    categorical_design_space, monkeypatch
) -> None:
    """Check that valid points are checked without looping over the variables."""

    def fail(*args, **kwargs) -> None:
        raise AssertionError

    monkeypatch.setattr("gemseo.space._design.checking.check_membership_dict", fail)
    categorical_design_space.check_membership(array([[0.0, 0.5], [2.0, 2.0]]))
    # The integer variables are vectorized too.
    design_space = DesignSpace()
    design_space.add_integer_variable("n", lower_bound=0, upper_bound=3)
    design_space.add_categorical_variable("material", ("steel", "aluminium"))
    design_space.check_membership(array([[0.0, 1.0], [3.0, 0.0]]))
    monkeypatch.undo()


@pytest.mark.parametrize(
    "points",
    [
        [[0.0, 0.5], [1.5, 1.0]],
        [[0.0, 0.5], [1.0, 2.5]],
        [[0.0, 0.5], [1.0, 0.5], [2.0, -0.5]],
    ],
)
def test_check_membership_of_many_invalid_points(
    categorical_design_space, points, snapshot
) -> None:
    """Check that an invalid point is still reported variable by variable."""
    with assert_exception(ValueError, snapshot):
        categorical_design_space.check_membership(array(points))


def test_check_membership_of_a_three_dimensional_array(
    categorical_design_space, snapshot
) -> None:
    """Check that the components are checked along the last axis of any array."""
    valid_points = array([[[0.0, 0.5], [1.0, 1.0]], [[2.0, 0.0], [1.0, 2.0]]])
    categorical_design_space.check_membership(valid_points)

    invalid_points = valid_points.copy()
    invalid_points[1, 0, 0] = 1.5
    with assert_exception(ValueError, snapshot):
        categorical_design_space.check_membership(invalid_points)


def test_check_membership_of_many_non_integer_points(snapshot) -> None:
    """Check that a non-integer value of an integer variable is not vectorized."""
    design_space = DesignSpace()
    design_space.add_integer_variable("n", lower_bound=0, upper_bound=3)
    design_space.add_categorical_variable("material", ("steel", "aluminium"))
    with assert_exception(ValueError, snapshot):
        design_space.check_membership(array([[1.0, 1.0], [1.5, 0.0]]))


@pytest.mark.parametrize("method_name", ["normalize_vect", "denormalize_vect"])
def test_sparse_point_with_a_categorical_variable(
    categorical_design_space, method_name, snapshot
) -> None:
    """Check that a sparse point cannot be (de)normalized with categories."""
    with (
        categorical_design_space._prepare_untransformation(True),
        assert_exception(TypeError, snapshot),
    ):
        getattr(categorical_design_space, method_name)(csr_array([[0.0, 1.0]]))


def test_sparse_gradient_with_a_categorical_variable(categorical_design_space) -> None:
    """Check that a sparse gradient can be normalized with a categorical variable."""
    gradient = categorical_design_space.normalize_grad(csr_array([[3.0, 4.0]]))
    assert_array_equal(gradient.toarray(), array([[3.0, 8.0]]))


@pytest.mark.parametrize("with_value", [True, False])
def test_categorical_variable_pretty_table(
    categorical_design_space, with_value, snapshot
) -> None:
    """Check that the pretty table shows the label of a categorical variable."""
    if with_value:
        design_space = categorical_design_space
    else:
        design_space = DesignSpace()
        design_space.add_categorical_variable("material", ["steel"])

    assert str(design_space) == snapshot


@pytest.fixture
def categorical_io_design_space() -> DesignSpace:
    """A design space mixing categorical variables with the other kinds."""
    design_space = DesignSpace()
    design_space.add_real_variable(
        "x", size=2, lower_bound=0.0, upper_bound=1.0, value=(0.25, 0.5)
    )
    design_space.add_categorical_variable(
        "material", ("aluminium", "stainless steel", "titanium"), "stainless steel"
    )
    design_space.add_integer_variable("n", lower_bound=1, upper_bound=10, value=4)
    design_space.add_discrete_variable("t", [0.72, 0.45, 0.55], value=0.55)
    design_space.add_categorical_variable("process", ("cast", "forged"))
    return design_space


def assert_same_design_space(actual: DesignSpace, expected: DesignSpace) -> None:
    """Check that two design spaces have the same variables and current values."""
    assert list(actual.variables) == list(expected.variables)
    for name, variable in expected.variables.items():
        actual_variable = actual.variables[name]
        assert actual_variable.type == variable.type
        assert actual_variable.size == variable.size
        if isinstance(variable, CategoricalVariable):
            assert actual_variable.categories == variable.categories

        actual_value = actual._current_value[name]
        expected_value = expected._current_value[name]
        if expected_value is None:
            assert actual_value is None
        else:
            assert_array_equal(actual_value, expected_value)


@pytest.mark.parametrize("with_value", [True, False])
@pytest.mark.parametrize(
    "file_name",
    ["ds.h5", "to_hdf"],
)
def test_categorical_variable_round_trip(
    categorical_io_design_space, tmp_wd, file_name, with_value
) -> None:
    """Check that a design space with categorical variables is exported and read."""
    design_space = categorical_io_design_space
    if not with_value:
        design_space.set_current_variable("material", None)

    if file_name == "to_hdf":
        design_space.to_hdf("ds.h5")
        new_design_space = DesignSpace.from_hdf("ds.h5")
    else:
        design_space.to_file(file_name)
        new_design_space = DesignSpace.from_file(file_name)

    assert_same_design_space(new_design_space, design_space)
    material = new_design_space.variables["material"]
    assert material.categories == ("aluminium", "stainless steel", "titanium")
    if with_value:
        assert_array_equal(new_design_space.get_current_value(["material"]), [1])
    else:
        assert new_design_space._current_value["material"] is None


def test_categorical_variable_hdf_storage(categorical_io_design_space, tmp_wd) -> None:
    """Check what is stored in the HDF file for a categorical variable."""
    categorical_io_design_space.to_hdf("ds.h5")
    with h5py.File("ds.h5") as h5file:
        group = h5file["design_space"]["material"]
        assert set(group) == {"categories", "size", "var_type", "value"}
        assert group["categories"].asstr()[()].tolist() == [
            "aluminium",
            "stainless steel",
            "titanium",
        ]
        assert group["value"].asstr()[()].tolist() == ["stainless steel"]
        assert group["var_type"][0].decode() == "categorical"
        assert group["size"][()] == 1


def test_categorical_variable_hdf_append(categorical_io_design_space, tmp_wd) -> None:
    """Check that a categorical variable can be exported again in the same file."""
    design_space = categorical_io_design_space
    design_space.to_hdf("ds.h5")
    design_space.set_current_variable("material", "titanium")
    design_space.set_current_variable("process", "forged")
    design_space.to_hdf("ds.h5", append=True)
    new_design_space = DesignSpace.from_hdf("ds.h5")
    assert_same_design_space(new_design_space, design_space)
    assert_array_equal(new_design_space.get_current_value(["material"]), [2])

    design_space.set_current_variable("material", None)
    design_space.to_hdf("ds.h5", append=True)
    new_design_space = DesignSpace.from_hdf("ds.h5")
    assert new_design_space._current_value["material"] is None
    assert_same_design_space(new_design_space, design_space)

    design_space.to_hdf("ds.h5", append=True, hdf_node_path="node")
    assert_same_design_space(
        DesignSpace.from_hdf("ds.h5", hdf_node_path="node"), design_space
    )


def test_categorical_variable_hdf_append_with_other_categories(
    categorical_io_design_space, tmp_wd, snapshot
) -> None:
    """Check that the categories cannot change when appending to an HDF file."""
    categorical_io_design_space.to_hdf("ds.h5")
    other_design_space = DesignSpace()
    other_design_space.add_real_variable("x", 2, 0.0, 1.0, (0.25, 0.5))
    # The categories are the same ones, in another order.
    other_design_space.add_categorical_variable(
        "material", ("titanium", "stainless steel", "aluminium")
    )
    other_design_space.add_integer_variable("n", lower_bound=1, upper_bound=10)
    other_design_space.add_discrete_variable("t", [0.72, 0.45, 0.55])
    other_design_space.add_categorical_variable("process", ("cast", "forged"))

    with assert_exception(ValueError, snapshot):
        other_design_space.to_hdf("ds.h5", append=True)


@pytest.mark.parametrize(
    "label",
    [
        "a b",
        " leading and trailing ",
        "tab\there",
        "a|b",
        'say "hi"',
        '"',
        "#hash",
        "x,y",
        "line\nbreak",
        "None",
        "acier inoxydable é",
        "",
    ],
)
def test_categorical_variable_hdf_labels(label, tmp_wd) -> None:
    """Check that any label survives an HDF export followed by an import."""
    design_space = DesignSpace()
    design_space.add_real_variable("x", lower_bound=0.0, upper_bound=1.0, value=0.5)
    design_space.add_categorical_variable("c", ("other", label, "more"), label)
    design_space.add_categorical_variable("d", (label,))
    design_space.add_integer_variable("n", lower_bound=0, upper_bound=3, value=0)
    design_space.to_hdf("ds.h5")
    new_design_space = DesignSpace.from_hdf("ds.h5")
    assert_same_design_space(new_design_space, design_space)
    variable = new_design_space.variables["c"]
    assert variable.decode(new_design_space._current_value["c"])[0] == label


@pytest.fixture
def two_categorical_design_space() -> DesignSpace:
    """A design space with two categorical variables."""
    design_space = DesignSpace()
    design_space.add_real_variable("x", lower_bound=0.0, upper_bound=1.0)
    design_space.add_categorical_variable("material", ("steel", "aluminium"))
    design_space.add_categorical_variable("colour", ("red", "blue"))
    return design_space


@pytest.mark.parametrize("method", ["to_csv", "to_file"])
def test_categorical_variable_cannot_be_exported_to_csv(
    two_categorical_design_space, tmp_wd, method, snapshot
) -> None:
    """Check that a design space with categorical variables is refused by CSV."""
    with assert_exception(ValueError, snapshot):
        getattr(two_categorical_design_space, method)("ds.csv")

    assert not Path("ds.csv").exists()


def test_categorical_variable_cannot_be_imported_from_csv(tmp_wd, snapshot) -> None:
    """Check that a CSV file with a categorical variable is refused."""
    Path("ds.csv").write_text(
        "name lower_bound value upper_bound type\n"
        "x 0.0 0.5 1.0 real\n"
        "material None None None categorical\n"
    )
    with assert_exception(ValueError, snapshot):
        DesignSpace.from_csv("ds.csv")


def _add_categories_to_a_real_variable(h5file: h5py.File) -> None:
    """Store categories for the real variable ``x``.

    Args:
        h5file: The HDF file of the design space, opened in append mode.
    """
    h5file["design_space"]["x"].create_dataset(
        "categories", data=array(["a", "b"], dtype=h5py.string_dtype())
    )


def _remove_the_categories(h5file: h5py.File) -> None:
    """Remove the categories of the categorical variable ``material``.

    Args:
        h5file: The HDF file of the design space, opened in append mode.
    """
    del h5file["design_space"]["material"]["categories"]


@pytest.mark.parametrize(
    "alter",
    [
        pytest.param(
            _add_categories_to_a_real_variable, id="categories-of-a-real-variable"
        ),
        pytest.param(_remove_the_categories, id="no-categories"),
    ],
)
def test_hdf_invalid_file(categorical_io_design_space, tmp_wd, alter, snapshot) -> None:
    """Check that an invalid HDF file is rejected."""
    categorical_io_design_space.to_hdf("ds.h5")
    with h5py.File("ds.h5", "a") as h5file:
        alter(h5file)

    with assert_exception(ValueError, snapshot):
        DesignSpace.from_hdf("ds.h5")


def test_categorical_variable_to_scalar_variables() -> None:
    """Check that splitting into scalar variables keeps a categorical variable."""
    design_space = DesignSpace()
    design_space.add_categorical_variable("c", ["a", "b"], "b")
    design_space.add_real_variable("x", size=2, lower_bound=0.0, upper_bound=1.0)
    scalar_design_space = design_space.to_scalar_variables()

    assert scalar_design_space.variables["c"] == design_space.variables["c"]
    assert_array_equal(scalar_design_space.get_current_value(["c"]), array([1]))
    assert list(scalar_design_space) == ["c", "x[0]", "x[1]"]


def test_categorical_variable_copy_and_pickle(categorical_design_space) -> None:
    """Check that copying and unpickling preserve the categories."""
    for design_space in (
        deepcopy(categorical_design_space),
        pickle.loads(pickle.dumps(categorical_design_space)),
    ):
        assert design_space == categorical_design_space
        assert design_space.variables["material"].categories == (
            "steel",
            "aluminium",
            "titanium",
        )
        assert design_space.reference_value["material"] == "aluminium"


def test_categorical_variable_extend_and_filter(categorical_design_space) -> None:
    """Check that a categorical variable can be added to another space."""
    design_space = DesignSpace()
    design_space.add_variables_from(categorical_design_space, "material")
    assert_array_equal(design_space.get_current_value(), array([1]))
    other_design_space = DesignSpace()
    other_design_space.extend(categorical_design_space)
    assert other_design_space == categorical_design_space

    categorical_design_space.filter_dimensions("material", [0])
    assert categorical_design_space.variables["material"].categories == (
        "steel",
        "aluminium",
        "titanium",
    )


def test_categorical_variable_and_rounding(categorical_design_space) -> None:
    """Check that rounding leaves the categorical variable alone."""
    assert_array_equal(
        categorical_design_space.round_vect(array([2.0, 0.3])), array([2.0, 0.3])
    )
