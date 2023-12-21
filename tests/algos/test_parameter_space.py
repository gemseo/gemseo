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
#        :author: Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import annotations

import re

import pytest
from numpy import allclose
from numpy import arange
from numpy import array
from numpy import array_equal
from numpy import concatenate
from numpy import ndarray
from numpy.testing import assert_array_equal
from openturns import NormalCopula

from gemseo.algos.design_space import DesignSpace
from gemseo.algos.parameter_space import ParameterSpace
from gemseo.algos.parameter_space import RandomVariable
from gemseo.algos.parameter_space import RandomVector
from gemseo.datasets.io_dataset import IODataset


def test_constructor():
    """Check that a ParameterSpace is empty after initialization."""
    space = ParameterSpace()
    assert not space.is_deterministic("x")
    assert not space.is_uncertain("x")


def test_add_variable():
    """Check that add_variable adds a deterministic variable."""
    space = ParameterSpace()
    space.add_variable("x")
    assert space.is_deterministic("x")
    assert not space.is_uncertain("x")
    assert "x" not in space.distributions


def test_add_random_variable():
    """Check that add_random_variable adds a random variable."""
    space = ParameterSpace()
    space.add_variable("x")
    space.add_random_variable("y", "SPNormalDistribution", mu=0.0, sigma=1.0)
    assert not space.is_deterministic("y")
    assert space.is_uncertain("y")
    assert space.variable_names == ["x", "y"]
    assert space.uncertain_variables == ["y"]
    assert space.deterministic_variables == ["x"]
    assert "y" in space.distributions


@pytest.fixture()
def mixed_space():
    """A parameter space containing both deterministic and uncertain variables."""
    space = ParameterSpace()
    space.add_variable("x1")
    space.add_variable("x2", value=0.0, l_b=0.0, u_b=1.0)
    space.add_random_variable("y", "SPNormalDistribution", mu=0.0, sigma=1.0)
    return space


def test_to_design_space(mixed_space):
    """Check the conversion of a ParameterSpace into a DesignSpace."""
    design_space = mixed_space.to_design_space()
    assert isinstance(design_space, DesignSpace)
    assert design_space.variable_names == ["x1", "x2", "y"]
    for name in ["x1", "x2"]:
        assert design_space.get_type(name) == mixed_space.get_type(name)
        assert design_space.get_size(name) == mixed_space.get_size(name)
        assert design_space.get_lower_bound(name) == mixed_space.get_lower_bound(name)
        assert design_space.get_upper_bound(name) == mixed_space.get_upper_bound(name)
        assert design_space._current_value.get(name) == mixed_space._current_value.get(
            name
        )

    assert (
        design_space.get_lower_bound("y")[0]
        == mixed_space.distributions["y"].math_lower_bound[0]
    )
    assert (
        design_space.get_upper_bound("y")[0]
        == mixed_space.distributions["y"].math_upper_bound[0]
    )
    assert (
        design_space.get_current_value(["y"])[0]
        == mixed_space.distributions["y"].mean[0]
    )


def test_extract_deterministic_space(mixed_space):
    """Check the extraction of the deterministic part."""
    deterministic_space = mixed_space.extract_deterministic_space()
    assert isinstance(deterministic_space, DesignSpace)
    assert deterministic_space.variable_names == ["x1", "x2"]


def test_extract_uncertain_space(mixed_space):
    """Check the extraction of the uncertain part."""
    uncertain_space = mixed_space.extract_uncertain_space()
    assert uncertain_space.variable_names == ["y"]
    assert uncertain_space.uncertain_variables == ["y"]


def test_extract_uncertain_space_as_design_space(mixed_space):
    """Check the extraction of the uncertain part as a design space."""
    uncertain_space = mixed_space.extract_uncertain_space(as_design_space=True)
    assert uncertain_space.variable_names == ["y"]
    assert isinstance(uncertain_space, DesignSpace)
    assert (
        uncertain_space.get_lower_bound("y")[0]
        == mixed_space.distributions["y"].math_lower_bound[0]
    )
    assert (
        uncertain_space.get_upper_bound("y")[0]
        == mixed_space.distributions["y"].math_upper_bound[0]
    )
    assert (
        uncertain_space.get_current_value(["y"])[0]
        == mixed_space.distributions["y"].mean[0]
    )


def test_remove_variable():
    """Check that remove_variable removes correctly variables (deterministic+random)."""
    space = ParameterSpace()
    space.add_variable("x1")
    space.add_variable("x2")
    space.add_random_variable("y1", "SPNormalDistribution", mu=0.0, sigma=1.0)
    space.add_random_variable("y2", "SPNormalDistribution", mu=0.0, sigma=1.0)
    space.remove_variable("x2")
    assert space.variable_names == ["x1", "y1", "y2"]
    assert space.uncertain_variables == ["y1", "y2"]
    space.remove_variable("y1")
    assert space.variable_names == ["x1", "y2"]
    assert space.uncertain_variables == ["y2"]
    assert "y1" not in space.distributions


def test_compute_samples():
    """Check that compute_samples works correctly."""
    space = ParameterSpace()
    space.add_variable("x1")
    space.add_variable("x2")
    space.add_random_variable("y1", "SPNormalDistribution", mu=0.0, sigma=1.0)
    space.add_random_variable("y2", "SPNormalDistribution", mu=0.0, sigma=1.0, size=3)
    sample = space.compute_samples(2)
    assert len(sample) == 2
    assert isinstance(sample, ndarray)
    assert sample.shape == (2, 4)
    sample = space.compute_samples(2, True)
    assert len(sample) == 2
    for idx in [0, 1]:
        assert "x1" not in sample[idx]
        assert "x2" not in sample[idx]
        assert isinstance(sample[idx]["y1"], ndarray)
        assert isinstance(sample[idx]["y2"], ndarray)
        assert len(sample[idx]["y1"]) == 1
        assert len(sample[idx]["y2"]) == 3


def test_evaluate_cdf():
    """Check that evaluate_cdf works correctly."""
    space = ParameterSpace()
    space.add_variable("x1")
    space.add_variable("x2")
    space.add_random_variable("y1", "SPNormalDistribution", mu=0.0, sigma=1.0)
    space.add_random_variable("y2", "SPNormalDistribution", mu=0.0, sigma=1.0, size=3)
    cdf = space.evaluate_cdf({"y1": array([0.0]), "y2": array([0.0] * 3)})
    inv_cdf = space.evaluate_cdf({"y1": array([0.5]), "y2": array([0.5] * 3)}, True)
    with pytest.raises(TypeError):
        space.evaluate_cdf(array([0.5] * 4), True)
    assert isinstance(cdf, dict)
    assert isinstance(inv_cdf, dict)
    assert allclose(cdf["y1"], array([0.5]), 1e-3)
    assert allclose(cdf["y2"], array([0.5] * 3), 1e-3)
    assert allclose(inv_cdf["y1"], array([0.0]), 1e-3)
    assert allclose(inv_cdf["y2"], array([0.0] * 3), 1e-3)


def test_range():
    """Check that range works correctly."""
    space = ParameterSpace()
    space.add_variable("x1")
    space.add_variable("x2")
    space.add_random_variable("y1", "SPUniformDistribution", minimum=0.0, maximum=2.0)
    space.add_random_variable("y2", "SPNormalDistribution", mu=0.0, sigma=2.0, size=3)
    expectation = array([0.0, 2.0])
    assert allclose(expectation, space.get_range("y1")[0], 1e-3)
    assert allclose(expectation, space.get_support("y1")[0], 1e-3)
    rng = space.get_range("y2")
    assert len(rng) == 3
    assert allclose(rng[0][1], -rng[0][0])
    assert allclose(rng[1][1], -rng[1][0])
    assert allclose(rng[2][1], -rng[2][0])


@pytest.fixture(scope="module")
def parameter_space():
    space = ParameterSpace()
    space.add_variable("x1")
    space.add_variable("x2")
    space.add_random_variable("y1", "SPUniformDistribution", minimum=0.0, maximum=2.0)
    space.add_random_variable("y2", "SPNormalDistribution", mu=0.0, sigma=2.0, size=3)
    return space


@pytest.mark.parametrize("one_dim", [True, False])
def test_normalize(parameter_space, one_dim):
    """Check that normalize works correctly with both 1D and 2D arrays."""
    vector = array([0.5] * 6) if one_dim else array([0.5] * 12).reshape((2, 6))

    u_vector = parameter_space.normalize_vect(vector, use_dist=True)
    values = [0.5] * 2 + [0.25] + [0.598706] * 3
    expectation = array(values) if one_dim else array([values, values])
    assert allclose(u_vector, expectation, 1e-3)


@pytest.mark.parametrize("one_dim", [True, False])
def test_unnormalize(parameter_space, one_dim):
    """Check that unnormalize works correctly with both 1D and 2D arrays."""
    values = [0.5] * 2 + [0.25] + [0.598706] * 3
    u_vector = array(values) if one_dim else array([values, values])
    vector = parameter_space.unnormalize_vect(u_vector, use_dist=True)
    values = [0.5] * 6
    expectation = array(values) if one_dim else array([values, values])
    assert allclose(vector, expectation, 1e-3)


def test_str_and_tabularview():
    """Check that str and unnormalize_vect work correctly."""
    space = ParameterSpace()
    space.add_variable("x")
    space.add_random_variable("y", "SPNormalDistribution", mu=0.0, sigma=1.0)
    space.add_random_variable("z", "SPUniformDistribution", minimum=0.0, maximum=1.0)
    assert "Parameter space" in str(space)
    tabular_view = space.get_tabular_view()
    assert "Parameter space" in tabular_view
    assert space._TRANSFORMATION in tabular_view
    assert space._SUPPORT in tabular_view
    assert space._MEAN in tabular_view
    assert space._STANDARD_DEVIATION in tabular_view
    assert space._RANGE in tabular_view


def test_unnormalize_vect():
    """Check that unnormalize_vect works correctly."""
    space = ParameterSpace()
    space.add_random_variable(
        "x", "SPTriangularDistribution", minimum=0.0, mode=0.5, maximum=2.0
    )
    assert allclose(
        space.unnormalize_vect(array([0.5]), use_dist=True), array([2.0 - 1.5**0.5])
    )
    assert space.unnormalize_vect(array([0.5]))[0] == 1.0


def test_normalize_vect():
    """Check that normalize_vect works correctly."""
    space = ParameterSpace()
    space.add_random_variable(
        "x", "SPTriangularDistribution", minimum=0.0, mode=0.5, maximum=2.0
    )
    assert allclose(
        space.normalize_vect(array([2.0 - 1.5**0.5]), use_dist=True), array([0.5])
    )
    assert space.normalize_vect(array([1.0]))[0] == 0.5


def test_evaluate_cdf_raising_errors():
    """Check that evaluate_cdf raises errors."""
    space = ParameterSpace()
    space.add_random_variable(
        "x", "SPTriangularDistribution", minimum=0.0, mode=0.5, maximum=2.0
    )

    expected = (
        r"obj must be a dictionary whose keys are the variables "
        "names and values are arrays whose dimensions are the "
        r"variables ones and components are in \[0, 1\]"
    )

    with pytest.raises(TypeError, match=expected):
        space.evaluate_cdf({"x": 1}, inverse=True)

    with pytest.raises(ValueError, match=expected):
        space.evaluate_cdf({"x": array([0.5] * 2)}, inverse=True)

    with pytest.raises(ValueError, match=expected):
        space.evaluate_cdf({"x": array([1.5])}, inverse=True)


@pytest.fixture()
def io_dataset() -> IODataset:
    """An input-output dataset."""
    inputs = arange(50).reshape(10, 5)
    outputs = arange(20).reshape(10, 2)
    data = concatenate([inputs, outputs], axis=1)
    variables = ["in_1", "in_2", "out_1"]
    variable_names_to_n_components = {"in_1": 2, "in_2": 3, "out_1": 2}
    variable_names_to_group_names = {
        "in_1": "inputs",
        "in_2": "inputs",
        "out_1": "outputs",
    }
    return IODataset.from_array(
        data, variables, variable_names_to_n_components, variable_names_to_group_names
    )


def test_init_from_dataset_default(io_dataset):
    """Check the default initialization from a dataset.

    Args:
        io_dataset (Dataset): An input-output dataset.
    """
    parameter_space = ParameterSpace.init_from_dataset(io_dataset)
    for name in ["in_1", "in_2", "out_1"]:
        assert name in parameter_space
        assert (parameter_space[name].var_type == "float").all()
        assert name in parameter_space.deterministic_variables
    assert parameter_space["in_1"].size == 2
    ref = io_dataset.get_view(variable_names="in_1").to_numpy().min(0)
    assert (parameter_space["in_1"].l_b == ref).all()
    ref = io_dataset.get_view(variable_names="in_1").to_numpy().max(0)
    assert (parameter_space["in_1"].u_b == ref).all()
    ref = (
        io_dataset.get_view(variable_names="in_1").to_numpy().max(0)
        + io_dataset.get_view(variable_names="in_1").to_numpy().min(0)
    ) / 2.0
    assert (parameter_space["in_1"].value == ref).all()
    assert parameter_space["in_2"].size == 3
    assert parameter_space["out_1"].size == 2


def test_init_from_dataset_uncertain(io_dataset):
    """Check the initialization from a dataset with uncertain variables.

    Args:
        io_dataset (Dataset): An input-output dataset.
    """
    parameter_space = ParameterSpace.init_from_dataset(
        io_dataset, uncertain={"in_1": True}
    )
    assert "in_1_0" in parameter_space.uncertain_variables
    assert "in_1_1" in parameter_space.uncertain_variables
    assert "in_2_0" not in parameter_space.uncertain_variables


def test_init_from_dataset_group(io_dataset):
    """Check the initialization from a dataset when groups are specified.

    Args:
        io_dataset (Dataset): An input-output dataset.
    """
    parameter_space = ParameterSpace.init_from_dataset(
        io_dataset, groups=[io_dataset.INPUT_GROUP]
    )
    for name in ["in_1", "in_2"]:
        assert name in parameter_space
    assert "out_1" not in parameter_space

    parameter_space = ParameterSpace.init_from_dataset(
        io_dataset, groups=[io_dataset.OUTPUT_GROUP]
    )
    for name in ["in_1", "in_2"]:
        assert name not in parameter_space
    assert "out_1" in parameter_space


def test_gradient_normalization():
    parameter_space = ParameterSpace()
    parameter_space.add_variable("x", l_b=-1.0, u_b=2.0)
    parameter_space.add_random_variable(
        "y", "OTUniformDistribution", minimum=1.0, maximum=3
    )
    x_vect = array([0.5, 1.5])
    assert array_equal(
        parameter_space.unnormalize_vect(x_vect, minus_lb=False),
        parameter_space.normalize_grad(x_vect),
    )


def test_gradient_unnormalization():
    parameter_space = ParameterSpace()
    parameter_space.add_variable("x", l_b=-1.0, u_b=2.0)
    parameter_space.add_variable("y", l_b=1.0, u_b=3.0)
    x_vect = array([0.5, 1.5])
    assert array_equal(
        parameter_space.normalize_vect(x_vect, minus_lb=False, use_dist=True),
        parameter_space.unnormalize_grad(x_vect),
    )


def test_parameter_space_name():
    """Check the naming of a parameter space."""
    assert ParameterSpace().name == ""
    assert ParameterSpace(name="my_name").name == "my_name"


def test_getitem_keyerror():
    """Check that getting an unknown item raises a KeyError."""
    parameter_space = ParameterSpace()
    with pytest.raises(KeyError, match="Variable 'x' is not known."):
        parameter_space["x"]


def test_getitem():
    """Check that an item can be correctly get from a ParameterSpace."""
    parameter_space = ParameterSpace()
    parameter_space.add_variable("x", l_b=0, u_b=1)
    parameter_space.add_random_variable("u", "SPNormalDistribution", mu=1.0, sigma=2.0)
    assert parameter_space["x"].l_b[0] == 0.0
    assert parameter_space["u"].parameters["mu"] == 1.0


def test_setitem():
    """Check that an item can be correctly passed to a ParameterSpace."""
    parameter_space = ParameterSpace()
    parameter_space.add_variable("x", l_b=0, u_b=1)
    parameter_space.add_random_variable("u", "SPNormalDistribution", mu=1.0, sigma=2.0)

    new_parameter_space = ParameterSpace()
    new_parameter_space["x"] = parameter_space["x"]
    new_parameter_space["u"] = parameter_space["u"]

    assert new_parameter_space["x"].l_b[0] == 0.0
    assert new_parameter_space["u"].parameters["mu"] == 1.0

    assert new_parameter_space == parameter_space


def test_transform():
    """Check that transformation and inverse transformation works correctly."""
    parameter_space = ParameterSpace()
    parameter_space.add_random_variable("x", "SPNormalDistribution")
    vector = array([0.0])
    transformed_vector = parameter_space.transform_vect(vector)
    assert transformed_vector == array([0.5])
    untransformed_vector = parameter_space.untransform_vect(transformed_vector)
    assert vector == untransformed_vector


def test_rename_variable():
    """Check the renaming of a variable."""
    design_variable = DesignSpace.DesignVariable(
        2, "integer", 0.0, 2.0, array([1.0, 2.0])
    )
    random_variable = RandomVariable(
        "SPNormalDistribution", 2, {"mu": 0.5, "sigma": 2.0}
    )
    random_vector = RandomVector(
        "SPNormalDistribution", 2, {"mu": [0.5, 1], "sigma": [2.0]}
    )

    parameter_space = ParameterSpace()
    parameter_space["x"] = design_variable
    parameter_space["u"] = random_variable
    parameter_space["z"] = random_vector
    parameter_space.rename_variable("x", "y")
    parameter_space.rename_variable("u", "v")

    other_parameter_space = ParameterSpace()
    other_parameter_space["y"] = design_variable
    other_parameter_space["v"] = random_variable
    other_parameter_space["z"] = random_vector

    assert parameter_space == other_parameter_space


@pytest.mark.parametrize(("first", "second"), [("SP", "OT"), ("OT", "SP")])
def test_mix_different_distribution_families(first, second):
    """Check that a ParameterSpace cannot mix distributions from different families."""
    parameter_space = ParameterSpace()
    parameter_space.add_random_variable("x", f"{first}UniformDistribution")
    with pytest.raises(
        ValueError,
        match=f"A parameter space cannot mix {first} and {second} distributions.",
    ):
        parameter_space.add_random_variable("y", f"{second}UniformDistribution")


def test_copula():
    """Check build_composed_distribution."""
    parameter_space = ParameterSpace()
    parameter_space.add_random_variable("x", "OTNormalDistribution")
    parameter_space.add_random_variable("y", "OTNormalDistribution", 2)
    parameter_space.build_composed_distribution(NormalCopula(3))
    assert (
        parameter_space.distribution.distribution.getCopula().getName()
        == "NormalCopula"
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"minimum": [0], "mode": [1, 2], "maximum": [3, 4, 5]},
        {"minimum": [0, 1], "mode": [2, 3, 4]},
        {"size": 3, "minimum": [0, 1]},
    ],
)
def test_random_vector_consistency(kwargs):
    """Check the error when adding a random vector with inconsistent parameter sizes."""
    text = "The lengths of the distribution parameter collections are not consistent."
    parameter_space = ParameterSpace()
    with pytest.raises(ValueError, match=re.escape(text)):
        parameter_space.add_random_vector("x", "SPTriangularDistribution", **kwargs)


@pytest.mark.parametrize(
    ("kwargs", "samples"),
    [
        ({"variable_value": [1, 2]}, array([[1, 2]] * 4)),
        ({"variable_value": [1, 2], "size": 2}, array([[1, 2]] * 4)),
        ({"variable_value": [1]}, array([[1]] * 4)),
        ({"variable_value": [1], "size": 2}, array([[1, 1]] * 4)),
    ],
)
def test_ot_random_vector(kwargs, samples):
    """Check add_random_vector with different settings.

    Use OpenTURNS.
    """
    parameter_space = ParameterSpace()
    parameter_space.add_random_vector("x", "OTDiracDistribution", **kwargs)
    assert_array_equal(parameter_space.compute_samples(4), samples)


@pytest.mark.parametrize(
    ("kwargs", "upper_bound"),
    [
        ({"maximum": [1, 2]}, [1, 2]),
        ({"maximum": [1, 2], "size": 2}, [1, 2]),
        ({"maximum": [1]}, [1]),
        ({"maximum": [1], "size": 2}, [1, 1]),
    ],
)
def test_sp_random_vector(kwargs, upper_bound):
    """Check add_random_vector with different settings.

    Use SciPy.
    """
    parameter_space = ParameterSpace()
    parameter_space.add_random_vector("x", "SPUniformDistribution", **kwargs)
    assert_array_equal(
        parameter_space.distribution.math_upper_bound, array(upper_bound)
    )


@pytest.mark.parametrize(
    ("kwargs", "samples"),
    [
        ({"interfaced_distribution_parameters": ([1, 2],)}, array([[1, 2]] * 4)),
        (
            {"interfaced_distribution_parameters": ([1, 2],), "size": 2},
            array([[1, 2]] * 4),
        ),
        ({"interfaced_distribution_parameters": ([1],)}, array([[1]] * 4)),
        (
            {"interfaced_distribution_parameters": ([1],), "size": 2},
            array([[1, 1]] * 4),
        ),
    ],
)
@pytest.mark.parametrize("use_parameters", [False, True])
def test_ot_random_vector_interfaced_distribution(kwargs, samples, use_parameters):
    """Check add_random_vector with interfaced_distribution and different settings.

    Use OpenTURNS.
    """
    if use_parameters:
        kwargs["parameters"] = kwargs.pop("interfaced_distribution_parameters")
    parameter_space = ParameterSpace()
    parameter_space.add_random_vector(
        "x", "OTDistribution", interfaced_distribution="Dirac", **kwargs
    )
    assert_array_equal(parameter_space.compute_samples(4), samples)


@pytest.mark.parametrize(
    ("kwargs", "upper_bound"),
    [
        (
            {"interfaced_distribution_parameters": {"scale": [1, 2]}},
            [1, 2],
        ),
        (
            {"interfaced_distribution_parameters": {"scale": [1, 2]}, "size": 2},
            [1, 2],
        ),
        ({"interfaced_distribution_parameters": {"scale": [1]}}, [1]),
        (
            {"interfaced_distribution_parameters": {"scale": [1]}, "size": 2},
            [1, 1],
        ),
    ],
)
@pytest.mark.parametrize("use_parameters", [False, True])
def test_sp_random_vector_interfaced_distribution(kwargs, upper_bound, use_parameters):
    """Check add_random_vector with interfaced_distribution.

    Use SciPy.
    """
    if use_parameters:
        kwargs["parameters"] = kwargs.pop("interfaced_distribution_parameters")
    parameter_space = ParameterSpace()
    parameter_space.add_random_vector(
        "x", "SPDistribution", interfaced_distribution="uniform", **kwargs
    )
    assert_array_equal(
        parameter_space.distribution.math_upper_bound, array(upper_bound)
    )


@pytest.mark.parametrize("method", ["add_random_vector", "add_random_variable"])
def test_parameters_and_interfaced_distribution_parameters(method):
    """Check that parameters and interfaced_distribution_parameters cannot be used at
    the same time."""
    parameter_space = ParameterSpace()
    with pytest.raises(
        ValueError,
        match=re.escape(
            "'interfaced_distribution_parameters' is the new name of 'parameters' "
            "which will be removed in the next major release; "
            "you cannot use both names at the same time; "
            "please use 'interfaced_distribution_parameters'."
        ),
    ):
        getattr(parameter_space, method)(
            "x",
            "SPDistribution",
            interfaced_distribution="uniform",
            interfaced_distribution_parameters={"scale": [1], "loc": [2]},
            parameters={"scale": [1], "loc": [2]},
        )


@pytest.mark.parametrize(
    ("obj", "args", "expected"),
    [
        ("variable", (2,), RandomVector),
        ("variable", (), RandomVariable),
        ("vector", (2,), RandomVector),
        ("vector", (), RandomVariable),
    ],
)
def test_random_vector_getitem(obj, args, expected):
    """Check the type object returned by __getitem__ depending on the size."""
    parameter_space = ParameterSpace()
    add_random_obj = getattr(parameter_space, f"add_random_{obj}")
    add_random_obj("x", "SPUniformDistribution", *args)
    assert isinstance(parameter_space["x"], expected)


@pytest.mark.parametrize(
    ("distribution", "interfaced_distribution", "interfaced_distribution_parameters"),
    [
        ("OTDistribution", "Uniform", ()),
        ("OTDistribution", "Uniform", (2, 4)),
        ("SPDistribution", "uniform", {}),
        ("SPDistribution", "uniform", {"scale": 2, "loc": 2}),
    ],
)
@pytest.mark.parametrize("use_parameters", [False, True])
def test_random_variable_interfaced_distribution(
    distribution,
    interfaced_distribution,
    interfaced_distribution_parameters,
    use_parameters,
):
    """Test adding a random variable from an interfaced distribution."""
    parameter = ParameterSpace()
    keyword = "parameters" if use_parameters else "interfaced_distribution_parameters"
    kwargs = {keyword: interfaced_distribution_parameters}
    parameter.add_random_variable(
        "x", distribution, interfaced_distribution=interfaced_distribution, **kwargs
    )
    marginal = parameter.distributions["x"].marginals[0]
    assert marginal.distribution_name == interfaced_distribution
    assert marginal.parameters == interfaced_distribution_parameters


def test_string_representation():
    """Check the string representation of a parameter space."""
    parameter_space = ParameterSpace()
    parameter_space.add_variable("a")
    parameter_space.add_random_variable("b", "OTUniformDistribution")
    parameter_space.add_random_variable("c", "OTUniformDistribution", 2)
    parameter_space.add_random_vector("d", "OTUniformDistribution", maximum=[2, 3, 4])
    expected = """Parameter space:
+------+-------------+-------+-------------+-------+-------------------------------+
| Name | Lower bound | Value | Upper bound | Type  |          Distribution         |
+------+-------------+-------+-------------+-------+-------------------------------+
| a    |     -inf    |  None |     inf     | float |                               |
| b    |      0      |  0.5  |      1      | float | Uniform(lower=0.0, upper=1.0) |
| c[0] |      0      |  0.5  |      1      | float | Uniform(lower=0.0, upper=1.0) |
| c[1] |      0      |  0.5  |      1      | float | Uniform(lower=0.0, upper=1.0) |
| d[0] |      0      |   1   |      2      | float |  Uniform(lower=0.0, upper=2)  |
| d[1] |      0      |  1.5  |      3      | float |  Uniform(lower=0.0, upper=3)  |
| d[2] |      0      |   2   |      4      | float |  Uniform(lower=0.0, upper=4)  |
+------+-------------+-------+-------------+-------+-------------------------------+"""  # noqa: E501
    assert str(parameter_space) == repr(parameter_space) == expected

    expected = """+------+-------------+-------+-------------+-------+-------------------------------+
| name | lower_bound | value | upper_bound | type  |          distribution         |
+------+-------------+-------+-------------+-------+-------------------------------+
| a    |     -inf    |  None |     inf     | float |                               |
| b    |      0      |  0.5  |      1      | float | Uniform(lower=0.0, upper=1.0) |
| c[0] |      0      |  0.5  |      1      | float | Uniform(lower=0.0, upper=1.0) |
| c[1] |      0      |  0.5  |      1      | float | Uniform(lower=0.0, upper=1.0) |
| d[0] |      0      |   1   |      2      | float |  Uniform(lower=0.0, upper=2)  |
| d[1] |      0      |  1.5  |      3      | float |  Uniform(lower=0.0, upper=3)  |
| d[2] |      0      |   2   |      4      | float |  Uniform(lower=0.0, upper=4)  |
+------+-------------+-------+-------------+-------+-------------------------------+"""  # noqa: E501
    assert (
        str(parameter_space.get_pretty_table(with_index=True, capitalize=False))
        == expected
    )

    parameter_space.remove_variable("a")
    expected = """Parameter space:
+------+-------------+-------+-------------+-------+-------------------------------+
| Name | Lower bound | Value | Upper bound | Type  |          Distribution         |
+------+-------------+-------+-------------+-------+-------------------------------+
| b    |      0      |  0.5  |      1      | float | Uniform(lower=0.0, upper=1.0) |
| c[0] |      0      |  0.5  |      1      | float | Uniform(lower=0.0, upper=1.0) |
| c[1] |      0      |  0.5  |      1      | float | Uniform(lower=0.0, upper=1.0) |
| d[0] |      0      |   1   |      2      | float |  Uniform(lower=0.0, upper=2)  |
| d[1] |      0      |  1.5  |      3      | float |  Uniform(lower=0.0, upper=3)  |
| d[2] |      0      |   2   |      4      | float |  Uniform(lower=0.0, upper=4)  |
+------+-------------+-------+-------------+-------+-------------------------------+"""
    assert repr(parameter_space) == expected

    expected = """Uncertain space:
+------+-------------------------------+
| Name |          Distribution         |
+------+-------------------------------+
|  b   | Uniform(lower=0.0, upper=1.0) |
| c[0] | Uniform(lower=0.0, upper=1.0) |
| c[1] | Uniform(lower=0.0, upper=1.0) |
| d[0] |  Uniform(lower=0.0, upper=2)  |
| d[1] |  Uniform(lower=0.0, upper=3)  |
| d[2] |  Uniform(lower=0.0, upper=4)  |
+------+-------------------------------+"""
    assert str(parameter_space) == expected

    parameter_space.add_random_variable(
        "e", "OTNormalDistribution", transformation="x+2"
    )
    expected = """Uncertain space:
+------+-------------------------------+--------------------+
| Name |      Initial distribution     | Transformation(x)= |
+------+-------------------------------+--------------------+
|  b   | Uniform(lower=0.0, upper=1.0) |         x          |
| c[0] | Uniform(lower=0.0, upper=1.0) |         x          |
| c[1] | Uniform(lower=0.0, upper=1.0) |         x          |
| d[0] |  Uniform(lower=0.0, upper=2)  |         x          |
| d[1] |  Uniform(lower=0.0, upper=3)  |         x          |
| d[2] |  Uniform(lower=0.0, upper=4)  |         x          |
|  e   |   Normal(mu=0.0, sigma=1.0)   |       (x)+2        |
+------+-------------------------------+--------------------+"""  # noqa: E501
    assert str(parameter_space) == expected


def test_existing_variable():
    """Check that one cannot add a random variable twice."""
    parameter_space = ParameterSpace()
    parameter_space.add_random_variable("a", "OTUniformDistribution")
    with pytest.raises(ValueError, match=re.escape("The variable 'a' already exists.")):
        parameter_space.add_random_variable("a", "OTUniformDistribution")
