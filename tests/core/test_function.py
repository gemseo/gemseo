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
# Copyright 2023 Capgemini Engineering
# Contributors:
#    INITIAL AUTHORS - initial API and implementation and/or
#                       initial documentation
#        :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import annotations

import re
from operator import add
from operator import itemgetter
from operator import mul
from operator import truediv
from unittest import mock

import pytest
from numpy import allclose
from numpy import array
from numpy import cos
from numpy import eye
from numpy import matmul
from numpy import ndarray
from numpy import ones
from numpy import sin
from numpy import zeros
from numpy.linalg import norm
from numpy.testing import assert_array_equal
from scipy.sparse import csr_array

from gemseo.algos.design_space import DesignSpace
from gemseo.algos.evaluation_counter import EvaluationCounter
from gemseo.algos.problem_function import ProblemFunction
from gemseo.core.mdo_functions.concatenate import Concatenate
from gemseo.core.mdo_functions.convex_linear_approx import ConvexLinearApprox
from gemseo.core.mdo_functions.function_restriction import FunctionRestriction
from gemseo.core.mdo_functions.mdo_function import MDOFunction
from gemseo.core.mdo_functions.set_pt_from_database import SetPtFromDatabase
from gemseo.core.mdo_functions.taylor_polynomials import compute_linear_approximation
from gemseo.core.mdo_functions.taylor_polynomials import compute_quadratic_approximation
from gemseo.problems.optimization.power_2 import Power2
from gemseo.utils.derivatives.approximation_modes import ApproximationMode
from gemseo.utils.pickle import from_pickle
from gemseo.utils.pickle import to_pickle


@pytest.fixture(scope="module")
def sinus() -> MDOFunction:
    """The sinus function."""
    return MDOFunction(sin, "sin")


@pytest.fixture(scope="module")
def sinus_eq_output_names() -> MDOFunction:
    """The sinus function of type ConstraintType.EQ with output_names."""
    return MDOFunction(
        sin,
        "sin",
        output_names=array(["sin"]),
        f_type=MDOFunction.ConstraintType.EQ,
    )


@pytest.mark.parametrize("x", [0, 1])
def test_call(sinus, x) -> None:
    """Check MDOFunction.__call__()."""
    assert sinus.evaluate(x) == sin(x)


def test_output_names_error() -> None:
    """Check that TypeError is raised when output_names has a wrong type."""
    with pytest.raises(TypeError):
        # TypeError: 'float' object is not iterable
        MDOFunction(sin, "sin", output_names=1.3)


def test_f_type(sinus, sinus_eq_output_names) -> None:
    """Check that the sum of a ConstraintType.EQ function with another function has
    FunctionType.EQ."""
    assert (sinus + sinus_eq_output_names).f_type == MDOFunction.ConstraintType.EQ
    assert (sinus_eq_output_names + sinus).f_type == MDOFunction.ConstraintType.EQ


@pytest.mark.parametrize(("operator", "symbol"), [(mul, "*"), (add, "+")])
def test_operation_error(sinus, operator, symbol) -> None:
    """Check that errors are raised with operations mixing MDOFunction and operators."""
    with pytest.raises(
        TypeError,
        match=re.escape(
            f"Unsupported {symbol} operator for MDOFunction and <class 'str'>."
        ),
    ):
        operator(sinus, "foo")


def test_init_from_dict_repr() -> None:
    """Check that initializing a MDOFunction with an unknown arg raised an error."""
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Cannot initialize MDOFunction attribute: foo, "
            "allowed ones are: dim, expr, f_type, input_names, name, output_names, "
            "special_repr."
        ),
    ):
        MDOFunction.init_from_dict_repr(foo="sin")


@pytest.fixture(scope="module")
def get_full_sin_func():
    """"""
    return MDOFunction(
        sin, name="F", f_type="obj", jac=cos, expr="sin(x)", input_names=["x"]
    )


def test_check_format() -> None:
    """Xxx."""
    MDOFunction(sin, f_type="obj", name=None, jac=cos, expr="sin(x)", input_names="x")


def test_func_error(sinus) -> None:
    """Check func() with a string argument."""
    with pytest.raises(TypeError):
        sinus.evaluate("toto")


@pytest.mark.parametrize("jacobian_type_1", ["dense", "sparse"])
@pytest.mark.parametrize("jacobian_type_2", ["dense", "sparse"])
def test_mdo_functions_algebra(jacobian_type_1, jacobian_type_2) -> None:
    """Test algebraic operations with mdo_functions."""
    array_ = array if jacobian_type_1 == "dense" else csr_array
    f = MDOFunction(
        lambda x: norm(x) ** 2,
        name="f",
        jac=lambda x: 2 * array_([x.tolist()]),
        expr="f(x)",
        input_names=["x"],
        dim=1,
    )

    array_ = array if jacobian_type_2 == "dense" else csr_array
    g = MDOFunction(
        lambda x: sum(cos(x)),
        name="cos",
        jac=lambda x: -sin(array_([x.tolist()])),
        expr="cos(x)",
        input_names=["x"],
        dim=1,
    )

    x = array([1.0, 2.0])

    f_x = norm(x) ** 2
    g_x = sum(cos(x))

    df_x = 2 * x
    dg_x = -sin(x)

    h = f + g
    assert allclose(h.evaluate(x), f_x + g_x)
    assert allclose(h.jac(x).data, (df_x + dg_x).data)
    h.check_grad(x, ApproximationMode.CENTERED_DIFFERENCES)

    h = f - g
    assert allclose(h.evaluate(x), f_x - g_x)
    assert allclose(h.jac(x).data, (df_x - dg_x).data)
    h.check_grad(x, ApproximationMode.CENTERED_DIFFERENCES)

    h = f * g
    assert allclose(h.evaluate(x), f_x * g_x)
    assert allclose(h.jac(x).data, (g_x * df_x + f_x * dg_x).data)
    h.check_grad(x, ApproximationMode.CENTERED_DIFFERENCES)

    h = f / g
    assert allclose(h.evaluate(x), f_x / g_x)
    assert allclose(h.jac(x).data, (g_x * df_x - f_x * dg_x).data / g_x**2)
    h.check_grad(x, ApproximationMode.CENTERED_DIFFERENCES)

    h = f + 3.0
    assert allclose(h.evaluate(x), f_x + 3.0)
    assert allclose(h.jac(x).data, df_x.data)
    h.check_grad(x, ApproximationMode.CENTERED_DIFFERENCES)

    h = f - 3.0
    assert allclose(h.evaluate(x), f_x - 3.0)
    assert allclose(h.jac(x).data, df_x.data)
    h.check_grad(x, ApproximationMode.CENTERED_DIFFERENCES)

    h = f * 3.0
    assert allclose(h.evaluate(x), f_x * 3.0)
    assert allclose(h.jac(x).data, (df_x * 3.0).data)
    h.check_grad(x, ApproximationMode.CENTERED_DIFFERENCES)

    h = f / 3.0
    assert allclose(h.evaluate(x), f_x / 3.0)
    assert allclose(h.jac(x).data, (df_x / 3.0).data)
    h.check_grad(x, ApproximationMode.CENTERED_DIFFERENCES)

    h = -f
    assert allclose(h.evaluate(x), -f_x)
    assert allclose(h.jac(x).data, (-df_x).data)
    h.check_grad(x, ApproximationMode.CENTERED_DIFFERENCES)


def test_todict_fromdict() -> None:
    """Check to_dict() and init_from_dict_repr()."""
    original_function = MDOFunction(
        sin,
        name="sin",
        jac=lambda x: array(cos(x)),
        expr="sin(x)",
        input_names=["x"],
        f_type=MDOFunction.ConstraintType.EQ,
        dim=1,
    )
    original_function_dict = original_function.to_dict()
    for name in MDOFunction.DICT_REPR_ATTR:
        if name != "special_repr":
            assert name in original_function_dict
            assert len(str(original_function_dict[name])) > 0

    new_function = MDOFunction.init_from_dict_repr(**original_function_dict)
    for name in MDOFunction.DICT_REPR_ATTR:
        assert getattr(new_function, name) == getattr(original_function, name)


def test_repr_1(get_full_sin_func) -> None:
    """Xxx."""
    assert str(get_full_sin_func) == "F(x) = sin(x)"


def test_repr_2() -> None:
    """Xxx."""
    g = MDOFunction(
        sin,
        name="G",
        f_type="ineq",
        jac=cos,
        expr="sin(x)",
        input_names=["x", "y"],
    )
    assert str(g) == "sin(x) <= 0.0"


def test_repr_3() -> None:
    """Xxx."""
    h = MDOFunction(sin, name="H", input_names=["x", "y", "x_shared"])
    assert str(h) == "H(x, y, x_shared)"


def test_repr_4() -> None:
    """Xxx."""
    g = MDOFunction(
        sin,
        name="G",
        expr="sin(x)",
        input_names=["x"],
    )
    i = MDOFunction(sin, name="I", input_names=["y"], expr="sin(y)")
    assert str(g + i) == "[G+I](x, y) = sin(x)+sin(y)"
    assert str(g - i) == "[G-I](x, y) = sin(x)-sin(y)"


def test_repr_5(get_full_sin_func) -> None:
    """Test the representation of many sign changes on an MDOFunction."""
    sin = get_full_sin_func
    minus_sin = -sin
    plus_sin = -minus_sin
    assert str(plus_sin) == "--F(x) = -(-(sin(x)))"


def test_wrong_jac_shape() -> None:
    f = MDOFunction(sin, name="sin", jac=lambda x: array([cos(x), 1.0]))
    with pytest.raises(ValueError):
        f.check_grad(array([0.0]))


@pytest.mark.parametrize(
    "function",
    [
        MDOFunction(lambda x: norm(x) ** 2, "f", jac=lambda x: 2.0 * x, dim=2),
        MDOFunction(
            lambda x: array([norm(x) ** 2, -(norm(x) ** 2)]),
            "f",
            jac=lambda x: array([2.0 * x, -2.0 * x]),
        ),
    ],
)
def test_restriction(function) -> None:
    """Test the restriction of a function."""
    x_vect = array([1.0, 2.0, 3.0])
    sub_x_vect = x_vect[array([0, 2])]
    restriction = FunctionRestriction(array([1]), array([2.0]), 3, function, "f_y")
    assert allclose(restriction.evaluate(sub_x_vect), function.evaluate(x_vect))
    assert allclose(
        restriction.jac(sub_x_vect), function.jac(x_vect)[..., array([0, 2])]
    )
    restriction.check_grad(sub_x_vect, error_max=1e-6)


def test_linearization() -> None:
    """Test the linearization of a function."""
    function = MDOFunction(
        lambda x: 0.5 * array([norm(x) ** 2, -(norm(x) ** 2)]),
        "f",
        jac=lambda x: array([x, -x]),
        dim=2,
    )
    linearization = compute_linear_approximation(function, array([1.0, 1.0, -2.0]))
    assert allclose(linearization.evaluate(array([2.0, 2.0, 2.0])), array([-3.0, 3.0]))
    linearization.check_grad(array([2.0, 2.0, 2.0]))


def test_convex_linearization() -> None:
    """Test the convex linearization of a function."""
    # Vectorial function
    function = MDOFunction(
        lambda x: 0.5 * array([norm(x) ** 2, -(norm(x) ** 2)]),
        "f",
        jac=lambda x: array([x, -x]),
        dim=2,
    )
    # The convex linearization (exact w.r.t. x_1) at (1, 1, -2) should be
    # [  0.5*x_1^2 + 5/2 +    (x_2-1) + 8/(x_3+2) ]
    # [ -0.5*x_1^2 - 5/2 +  1/(x_2-1) + 2*(x_3+2) ]
    convex_lin = ConvexLinearApprox(
        array([1.0, 1.0, -2.0]), function, array([False, True, True])
    )
    assert allclose(convex_lin.evaluate(array([2.0, 2.0, 2.0])), array([7.5, 4.5]))

    # Check the Jacobian of the convex linearization
    convex_lin.check_grad(array([2.0, 2.0, 2.0]), error_max=1e-6)

    # Scalar function (N.B. scalar value and 1-dimensional Jacobian matrix)
    function = MDOFunction(
        lambda x: 0.5 * norm(x) ** 2,
        "f",
        jac=lambda x: x,
        dim=1,
    )
    convex_lin = ConvexLinearApprox(
        array([1.0, 1.0, -2.0]), function, array([False, True, True])
    )
    value = convex_lin.evaluate(array([2.0, 2.0, 2.0]))
    assert isinstance(value, float)
    assert allclose(value, 7.5)
    gradient = convex_lin.jac(array([2.0, 2.0, 2.0]))
    assert len(gradient.shape) == 1
    convex_lin.check_grad(array([2.0, 2.0, 2.0]), error_max=1e-6)


@pytest.fixture(scope="module")
def function_for_quadratic_approximation() -> MDOFunction:
    """A function to check quadratic approximation."""
    return MDOFunction(
        lambda x: 0.5 * norm(x) ** 2,
        "f",
        jac=lambda x: x,
        input_names=["x_0", "x_1", "x_2"],
    )


@pytest.mark.parametrize("x_vect", [(ones(2), ones(3)), (eye(3), ones(2))])
def test_quadratic_approximation_error(
    function_for_quadratic_approximation, x_vect
) -> None:
    """Test the second-order polynomial of a function with inconsistent input."""
    with pytest.raises(
        ValueError, match=re.escape("Hessian approximation must be a square ndarray.")
    ):
        compute_quadratic_approximation(function_for_quadratic_approximation, *x_vect)


def test_quadratic_approximation(function_for_quadratic_approximation) -> None:
    """Test the second-order polynomial of a function."""
    approx = compute_quadratic_approximation(
        function_for_quadratic_approximation, ones(3), eye(3)
    )
    assert approx.evaluate(zeros(3)) == pytest.approx(0.0)
    assert allclose(approx.jac(zeros(3)), zeros(3))
    approx.check_grad(zeros(3), error_max=1e-6)


def test_concatenation() -> None:
    """Test the concatenation of functions."""
    dim = 2
    f = MDOFunction(lambda x: norm(x) ** 2, "f", jac=lambda x: 2 * x, dim=1)
    g = MDOFunction(lambda x: x, "g", jac=lambda x: eye(dim), dim=dim)
    h = Concatenate([f, g], "h")
    x_vect = ones(dim)
    assert allclose(h.evaluate(x_vect), array([2.0, 1.0, 1.0]))
    assert allclose(h.jac(x_vect), array([[2.0, 2.0], [1.0, 0.0], [0.0, 1.0]]))
    h.check_grad(x_vect, error_max=1e-6)


@pytest.mark.parametrize("normalize", [False, True])
def test_set_pt_from_database(normalize) -> None:
    problem = Power2()
    problem.preprocess_functions(is_function_input_normalized=normalize)
    x = zeros(3)
    problem.evaluate_functions(design_vector=x, design_vector_is_normalized=normalize)
    function = MDOFunction(sum, problem.objective.name)
    function.set_pt_from_database(
        problem.database, problem.design_space, normalize=normalize, jac=False
    )
    function.evaluate(x)


def test_linear_approximation() -> None:
    """Tests the linear approximation of a standard MDOFunction."""
    # Define the (affine) function f(x, y) = [1 2] [x] + [5]
    #                                        [3 4] [y]   [6]
    mat = array([[1.0, 2.0], [3.0, 4.0]])
    vec = array([5.0, 6.0])

    def func(x_vect):
        return vec + matmul(mat, x_vect)

    def jac(_):
        return mat

    function = MDOFunction(
        func,
        "f",
        jac=jac,
        expr="[1 2] [x] + [5]\n[3 4] [y]   [6]",
        input_names=["x", "y"],
    )

    # Get a linear approximation of the MDOFunction
    linear_approximation = compute_linear_approximation(function, array([7.0, 8.0]))
    assert (linear_approximation.coefficients == mat).all()
    assert (linear_approximation.value_at_zero == vec).all()


@pytest.fixture
def function():
    return MDOFunction(lambda x: x, "n", expr="e", special_repr="a_special_repr")


@pytest.mark.parametrize(
    ("neg", "neg_after", "value", "expected_n", "expected_e", "expected_sr"),
    [
        (False, True, 1.0, "[n+1.0]", "e+1.0", "a_special_repr+1.0"),
        (True, True, 1.0, "-[n+1.0]", "-(e+1.0)", "-(a_special_repr+1.0)"),
        (False, True, -1.0, "[n-1.0]", "e-1.0", "a_special_repr-1.0"),
        (True, True, -1.0, "-[n-1.0]", "-(e-1.0)", "-(a_special_repr-1.0)"),
        (False, False, 1.0, "[n+1.0]", "e+1.0", "a_special_repr+1.0"),
        (True, False, 1.0, "[-n+1.0]", "-(e)+1.0", "-(a_special_repr)+1.0"),
        (False, False, -1.0, "[n-1.0]", "e-1.0", "a_special_repr-1.0"),
        (True, False, -1.0, "[-n-1.0]", "-(e)-1.0", "-(a_special_repr)-1.0"),
        (
            False,
            False,
            array([1.0, 1.0]),
            "[n+offset]",
            "e+offset",
            "a_special_repr+offset",
        ),
        (
            True,
            False,
            array([1.0, 1.0]),
            "[-n+offset]",
            "-(e)+offset",
            "-(a_special_repr)+offset",
        ),
        (
            True,
            True,
            array([1.0, 1.0]),
            "-[n+offset]",
            "-(e+offset)",
            "-(a_special_repr+offset)",
        ),
    ],
)
def test_offset_name_and_expr(
    function, neg, neg_after, value, expected_n, expected_e, expected_sr
) -> None:
    """Check the name and expression of a function after 1) __neg__ and 2) offset."""
    if neg_after:
        function = function.offset(value)

        if neg:
            function = -function
    else:
        if neg:
            function = -function

        function = function.offset(value)

    assert function.name == expected_n
    assert function.expr == expected_e
    assert function.special_repr == expected_sr


def test_expects_normalized_inputs(function) -> None:
    """Check the inputs normalization expectation."""
    assert not function.expects_normalized_inputs


@pytest.fixture(scope="module")
def design_space():
    """A design space."""
    return mock.Mock()


@pytest.fixture(scope="module")
def problem():
    """An optimization problem."""
    return mock.Mock()


@pytest.fixture(scope="module")
def database():
    """A database."""
    return mock.Mock()


@pytest.mark.parametrize("normalize", [False, True])
def test_expect_normalized_inputs_from_database(
    function, design_space, database, normalize
) -> None:
    """Check the inputs normalization expectation."""
    func = SetPtFromDatabase(database, design_space, function, normalize=normalize)
    assert func.expects_normalized_inputs == normalize


def test_activate_counters() -> None:
    """Check that the function counter is active by default."""
    func = MDOFunction(lambda x: x, "func")
    func = ProblemFunction(
        func,
        (func.func,),
        (func.func,),
        False,
        None,
        EvaluationCounter(),
        False,
        DesignSpace(),
    )
    assert func.n_calls == 0
    func.evaluate(array([1.0]))
    assert func.n_calls == 1


def test_deactivate_counters() -> None:
    """Check that the function counter is set to None when deactivated."""
    enable_statistics = ProblemFunction.enable_statistics

    ProblemFunction.enable_statistics = False

    func = MDOFunction(lambda x: x, "func")
    func = ProblemFunction(
        func,
        (func.func,),
        (func.func,),
        False,
        None,
        EvaluationCounter(),
        False,
        DesignSpace(),
    )
    assert not func.n_calls

    with pytest.raises(
        RuntimeError, match=re.escape("The function counters are disabled.")
    ):
        func.n_calls = 1

    ProblemFunction.enable_statistics = enable_statistics


def test_get_indexed_name(function) -> None:
    """Check the indexed function name."""
    assert function.get_indexed_name(3) == "n[3]"


@pytest.mark.parametrize("fexpr", [None, "x**2"])
@pytest.mark.parametrize("gexpr", [None, "x**3"])
@pytest.mark.parametrize(
    ("op", "op_name", "func", "jac"),
    [(mul, "*", 32, 80), (truediv, "/", 0.5, -0.25)],
)
def test_multiplication_by_function(fexpr, gexpr, op, op_name, func, jac) -> None:
    """Check the multiplication of a function by a function or its inverse."""
    f = MDOFunction(
        lambda x: x**2, "f", jac=lambda x: 2 * x, input_names=["x"], expr=fexpr
    )
    g = MDOFunction(
        lambda x: x**3, "g", jac=lambda x: 3 * x**2, input_names=["x"], expr=gexpr
    )

    f_op_g = op(f, g)
    suffix = ""
    if fexpr and gexpr:
        suffix = f" = {fexpr}{op_name}{gexpr}"

    assert repr(f_op_g) == f"[f{op_name}g](x)" + suffix
    assert f_op_g.evaluate(2) == func
    assert f_op_g.jac(2) == jac


@pytest.mark.parametrize("expr", [None, "x**2"])
@pytest.mark.parametrize(
    ("op", "op_name", "func", "jac"), [(mul, "*", 16, 24), (truediv, "/", 4, 6)]
)
def test_multiplication_by_scalar(expr, op, op_name, func, jac) -> None:
    """Check the multiplication of a function by a scalar or its inverse."""
    f = MDOFunction(
        lambda x: x**3, "f", jac=lambda x: 3 * x**2, input_names=["x"], expr=expr
    )
    f_op_2 = op(f, 2)
    suffix = ""
    if expr:
        suffix = f" = 2*{expr}" if op_name == "*" else f" = {expr}/2"

    if op_name == "*":
        assert repr(f_op_2) == "2*f(x)" + suffix
    else:
        assert repr(f_op_2) == "[f/2](x)" + suffix

    assert f_op_2.evaluate(2) == func
    assert f_op_2.jac(2) == jac


@pytest.fixture(scope="module")
def multidimensional_function() -> MDOFunction:
    return MDOFunction(itemgetter(slice(-1)), "f", jac=lambda x: eye(x.size)[:-1, :])


def test_multiplication_by_array(multidimensional_function):
    """Check the multiplication of a function by an array."""
    product = multidimensional_function * array([2.0, 3.0])
    inputs = array([4.0, 5.0, 6.0])
    assert (product.evaluate(inputs) == [8.0, 15.0]).all()
    assert (product.jac(inputs) == array([[2, 0, 0], [0, 3, 0]])).all()


def test_division_by_array(multidimensional_function):
    """Check the division of a function by an array."""
    quotient = multidimensional_function / array([2.0, 5.0])
    inputs = array([4.0, 3.0, 6.0])
    assert (quotient.evaluate(inputs) == array([2.0, 0.6])).all()
    assert (quotient.jac(inputs) == array([[0.5, 0, 0], [0, 0.2, 0]])).all()


@pytest.mark.parametrize(
    ("expr_1", "expr_2", "op", "expected"),
    [
        ("1+x", "x", mul, "[f*g](x) = (1+x)*x"),
        ("1-x", "-x+3", mul, "[f*g](x) = (1-x)*(-x+3)"),
        ("(1+(x-4))", "x", truediv, "[f/g](x) = (1+(x-4))/x"),
        ("((x-4)-9)", "x", truediv, "[f/g](x) = ((x-4)-9)/x"),
    ],
)
def test_repr_mult_sum(expr_1, expr_2, op, expected) -> None:
    """Test the str repr of the product of two functions."""
    f = MDOFunction(simple_function, "f", expr=expr_1, input_names=["x"])
    g = MDOFunction(simple_function, "g", expr=expr_2, input_names=["x"])
    assert repr(op(f, g)) == expected


def simple_function(x: ndarray) -> ndarray:
    """An identity function.

    Serialization tests require explicitly defined functions instead of lambdas.

    Args:
        x: The input data.

    Returns:
        The input data.
    """
    return x


@pytest.mark.parametrize("enable_statistics", [True, False])
@pytest.mark.parametrize(
    ("mdo_function", "kwargs", "value"),
    [
        (
            MDOFunction,
            {
                "func": sin,
                "f_type": "obj",
                "name": "obj",
                "jac": cos,
                "expr": "sin(x)",
                "input_names": ["x"],
            },
            array([1.0]),
        ),
    ],
)
def test_serialize_deserialize(
    enable_statistics, mdo_function, kwargs, value, tmp_wd
) -> None:
    """Test the serialization/deserialization method.

    Args:
        enable_statistics: Whether to enable the statistics.
        mdo_function: The ``MDOFunction`` to be tested.
        kwargs: The keyword arguments to instantiate the ``MDOFunction``.
        value: The value to evaluate the ``MDOFunction``.
        tmp_wd: Fixture to move into a temporary work directory.
    """
    function = mdo_function(**kwargs)
    function = ProblemFunction(
        function,
        (sum,),
        (sum,),
        False,
        None,
        EvaluationCounter(),
        False,
        DesignSpace(),
    )
    out_file = "function1.o"
    function.enable_statistics = enable_statistics
    function.evaluate(value)
    to_pickle(function, out_file)
    serialized_func = from_pickle(out_file)

    if enable_statistics:
        assert function.n_calls == serialized_func.n_calls
        serialized_func.evaluate(value)
        assert serialized_func._n_calls.value == 2
    else:
        assert not serialized_func.n_calls

    s_func_u_dict = serialized_func.__dict__
    ok = True
    for k in function.__dict__:
        if k not in s_func_u_dict:
            ok = False
    assert ok


@pytest.mark.parametrize(
    ("force_real", "expected"), [(False, array([1j])), (True, array([0.0]))]
)
def test_force_real(force_real, expected) -> None:
    """Verify the use of force_real."""
    f = MDOFunction(lambda x: x, "f", force_real=force_real)
    assert f.evaluate(array([1j])) == expected


@pytest.mark.parametrize(
    ("ft1", "ft2", "ft"), [("", "", ""), ("obj", "", "obj"), ("", "obj", "obj")]
)
def test_f_type_sum_two_functions(ft1, ft2, ft) -> None:
    """Verify the f_type of the sum of two functions."""
    f = MDOFunction(lambda x: x, "f1", f_type=ft1) + MDOFunction(
        lambda x: x, "f2", f_type=ft2
    )
    assert f.f_type == ft


@pytest.mark.parametrize("f_type", ["", "obj"])
def test_f_type_sum_function_and_number(f_type) -> None:
    """Verify the f_type of the sum of a function and a number."""
    f = MDOFunction(lambda x: x, "f", f_type=f_type) + 1.0
    assert f.f_type == f_type


@pytest.mark.parametrize(
    ("f_out", "g_out", "h_out"),
    [
        ([], [], []),
        (["a"], [], []),
        ([], ["b"], []),
        (["a"], ["b"], ["a", "b"]),
    ],
)
def test_concatenate(f_out, g_out, h_out) -> None:
    """Check ``Concatenate.output_names``."""
    f = Concatenate(
        [
            MDOFunction(lambda x: x, "f", output_names=f_out),
            MDOFunction(lambda x: x, "g", output_names=g_out),
        ],
        "h",
    )
    assert f.output_names == h_out


@pytest.mark.parametrize(
    ("f_type", "input_names", "expr", "neg", "expected"),
    [
        (MDOFunction.FunctionType.NONE, [], "", False, "f"),
        (MDOFunction.FunctionType.NONE, [], "2*x", False, "f = 2*x"),
        (MDOFunction.FunctionType.NONE, ["y", "x"], "2*x", False, "f(y, x) = 2*x"),
        (MDOFunction.FunctionType.NONE, ["y", "x"], "", False, "f(y, x)"),
        (MDOFunction.FunctionType.NONE, [], "", True, "-f"),
        (MDOFunction.FunctionType.NONE, [], "2*x", True, "-f = -(2*x)"),
        (MDOFunction.FunctionType.NONE, ["y", "x"], "2*x", True, "-f(y, x) = -(2*x)"),
        (MDOFunction.FunctionType.NONE, ["y", "x"], "", True, "-f(y, x)"),
        (MDOFunction.FunctionType.INEQ, [], "", False, "f <= 0.0"),
        (MDOFunction.FunctionType.INEQ, [], "2*x", False, "2*x <= 0.0"),
        (MDOFunction.FunctionType.INEQ, ["y", "x"], "2*x", False, "2*x <= 0.0"),
        (MDOFunction.FunctionType.INEQ, ["y", "x"], "", False, "f(y, x) <= 0.0"),
        (MDOFunction.FunctionType.INEQ, [], "", True, "-f <= 0.0"),
        (MDOFunction.FunctionType.INEQ, [], "2*x", True, "-(2*x) <= 0.0"),
        (MDOFunction.FunctionType.INEQ, ["y", "x"], "2*x", True, "-(2*x) <= 0.0"),
        (MDOFunction.FunctionType.INEQ, ["y", "x"], "", True, "-f(y, x) <= 0.0"),
    ],
)
def test_default_repr(f_type, input_names, expr, neg, expected) -> None:
    """Check default_repr.

    Special cases:
    - input names or not,
    - expression or not,
    - inequality type or not,
    - negation operator or not.
    """
    f = MDOFunction(lambda x: x, "f", f_type=f_type, input_names=input_names, expr=expr)
    if neg:
        f = -f
    assert f.default_repr == expected


@pytest.mark.parametrize(("method", "n_calls"), [("func", 0), ("evaluate", 1)])
def test_func(method, n_calls):
    """Check that the property func is an alias of _func."""
    f = MDOFunction(lambda x: 2 * x, "f")
    f = ProblemFunction(
        f,
        (f.func,),
        (f.func,),
        False,
        None,
        EvaluationCounter(),
        False,
        DesignSpace(),
    )
    assert f.n_calls == 0
    assert_array_equal(getattr(f, method)(array([2])), array([4]))
    assert f.n_calls == n_calls
