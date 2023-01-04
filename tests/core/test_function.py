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
#    INITIAL AUTHORS - initial API and implementation and/or
#                       initial documentation
#        :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import annotations

import math
import re
from operator import add
from operator import mul
from operator import truediv
from unittest import mock

import numpy as np
import pytest
from gemseo.core.mdofunctions.concatenate import Concatenate
from gemseo.core.mdofunctions.mdo_function import MDOFunction
from gemseo.core.mdofunctions.norm_db_function import NormDBFunction
from gemseo.core.mdofunctions.norm_function import NormFunction
from gemseo.core.mdofunctions.set_pt_from_database import SetPtFromDatabase
from gemseo.problems.analytical.power_2 import Power2
from numpy import allclose
from numpy import array
from numpy import eye
from numpy import matmul
from numpy import ndarray
from numpy import ones
from numpy import zeros
from numpy.linalg import norm


@pytest.fixture(scope="module")
def sinus() -> MDOFunction:
    """The sinus function."""
    return MDOFunction(math.sin, "sin")


@pytest.fixture(scope="module")
def sinus_eq_outvars() -> MDOFunction:
    """The sinus function of type TYPE_EQ with outvars."""
    return MDOFunction(
        math.sin, "sin", outvars=array(["sin"]), f_type=MDOFunction.TYPE_EQ
    )


@pytest.mark.parametrize("x", [0, 1])
def test_call(sinus, x):
    """Check MDOFunction.__call__()."""
    assert sinus(x) == math.sin(x)


def test_outvars_error():
    """Check that TypeError is raised when outvars has a wrong type."""
    with pytest.raises(TypeError):
        # TypeError: 'float' object is not iterable
        MDOFunction(math.sin, "sin", outvars=1.3)


def test_has_outvars(sinus, sinus_eq_outvars):
    """Check outvars()."""
    assert not sinus.outvars
    assert sinus_eq_outvars.has_outvars()


def test_f_type(sinus, sinus_eq_outvars):
    """Check that the sum of a TYPE_EQ function with another function has TYPE_EQ."""
    assert (sinus + sinus_eq_outvars).f_type == MDOFunction.TYPE_EQ
    assert (sinus_eq_outvars + sinus).f_type == MDOFunction.TYPE_EQ


@pytest.mark.parametrize("operator,symbol", [(mul, "*"), (add, "+")])
def test_operation_error(sinus, operator, symbol):
    """Check that errors are raised with operations mixing MDOFunction and operators."""
    with pytest.raises(
        TypeError,
        match=re.escape(
            f"Unsupported {symbol} operator for MDOFunction and <class 'str'>."
        ),
    ):
        operator(sinus, "foo")


def test_init_from_dict_repr():
    """Check that initializing a MDOFunction with an unknown arg raised an error."""
    with pytest.raises(
        ValueError,
        match=re.escape(
            "Cannot initialize MDOFunction attribute: foo, "
            "allowed ones are: args, dim, expr, f_type, name, special_repr."
        ),
    ):
        MDOFunction.init_from_dict_repr(foo="sin")


@pytest.fixture(scope="module")
def get_full_sin_func():
    """"""
    return MDOFunction(
        math.sin, name="F", f_type="obj", jac=math.cos, expr="sin(x)", args=["x"]
    )


def test_f_type_error():
    """Check that an exception is raised when passing a wrong function type."""
    with pytest.raises(
        Exception,
        match=re.escape(
            "MDOFunction type must be among ['obj', 'eq', 'ineq', 'obs']; "
            "got Xobj instead."
        ),
    ):
        MDOFunction(math.sin, "F", f_type="Xobj")


def test_check_format():
    """xxx."""
    MDOFunction(
        math.sin, f_type="obj", name=None, jac=math.cos, expr="sin(x)", args="x"
    )


def test_func_error(sinus):
    """Check func() with a string argument."""
    with pytest.raises(Exception):
        sinus.func("toto")


def test_has_expr(sinus):
    """Check has_expr()."""
    assert not sinus.has_expr()


def test_has_args(sinus):
    """Check has_args()."""
    assert not sinus.has_args()


def test_add_sub_neg():
    """"""
    f = MDOFunction(
        np.sin,
        name="sin",
        jac=lambda x: np.array(np.cos(x)),
        expr="sin(x)",
        args=["x"],
        f_type=MDOFunction.TYPE_EQ,
        dim=1,
    )
    g = MDOFunction(
        np.cos,
        name="cos",
        jac=lambda x: -np.array(np.sin(x)),
        expr="cos(x)",
        args=["x"],
    )

    h = f + g
    k = f - g
    mm = f * g
    mm_c = f * 3.5
    x = 1.0
    assert h(x) == math.sin(x) + math.cos(x)
    assert h.jac(x) == math.cos(x) - math.sin(x)

    n = -f
    assert n(x) == -math.sin(x)
    assert n.jac(x) == -math.cos(x)

    assert k(x) == math.sin(x) - math.cos(x)
    assert k.jac(x) == math.cos(x) + math.sin(x)

    assert mm.jac(x) == math.cos(x) ** 2 + -math.sin(x) ** 2

    fplu = f + 3.5
    fmin = f - 5.0

    for func in [f, g, h, k, mm, mm_c, fplu, fmin]:
        func.check_grad(np.array([x]), "ComplexStep")


def test_todict_fromdict():
    """Check to_dict() and init_from_dict_repr()."""
    original_function = MDOFunction(
        np.sin,
        name="sin",
        jac=lambda x: np.array(np.cos(x)),
        expr="sin(x)",
        args=["x"],
        f_type=MDOFunction.TYPE_EQ,
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


def test_repr_1(get_full_sin_func):
    """xxx."""
    assert str(get_full_sin_func) == "F(x) = sin(x)"


def test_repr_2():
    """xxx."""
    g = MDOFunction(
        math.sin,
        name="G",
        f_type="ineq",
        jac=math.cos,
        expr="sin(x)",
        args=["x", "y"],
    )
    assert str(g) == "sin(x) <= 0.0"


def test_repr_3():
    """xxx."""
    h = MDOFunction(math.sin, name="H", args=["x", "y", "x_shared"])
    assert str(h) == "H(x, y, x_shared)"


def test_repr_4():
    """xxx."""
    g = MDOFunction(
        math.sin,
        name="G",
        expr="sin(x)",
        args=["x"],
    )
    i = MDOFunction(math.sin, name="I", args=["y"], expr="sin(y)")
    assert str(g + i) == "G+I(x, y) = sin(x)+sin(y)"
    assert str(g - i) == "G-I(x, y) = sin(x)-sin(y)"


def test_wrong_jac_shape():
    f = MDOFunction(np.sin, name="sin", jac=lambda x: np.array([np.cos(x), 1.0]))
    with pytest.raises(ValueError):
        f.check_grad(array([0.0]))


@pytest.mark.parametrize(
    "function",
    [
        MDOFunction(lambda x: norm(x) ** 2, "f", jac=lambda x: 2.0 * x, dim=2),
        MDOFunction(
            lambda x: array([norm(x) ** 2, -norm(x) ** 2]),
            "f",
            jac=lambda x: array([2.0 * x, -2.0 * x]),
        ),
    ],
)
def test_restriction(function):
    """Test the restriction of a function."""
    x_vect = array([1.0, 2.0, 3.0])
    sub_x_vect = x_vect[array([0, 2])]
    restriction = function.restrict(array([1]), array([2.0]), 3, "f_y")
    assert allclose(restriction(sub_x_vect), function(x_vect))
    assert allclose(
        restriction.jac(sub_x_vect), function.jac(x_vect)[..., array([0, 2])]
    )
    restriction.check_grad(sub_x_vect, error_max=1e-6)


def test_linearization():
    """Test the linearization of a function."""
    function = MDOFunction(
        lambda x: 0.5 * array([norm(x) ** 2, -norm(x) ** 2]),
        "f",
        jac=lambda x: array([x, -x]),
        dim=2,
    )
    linearization = function.linear_approximation(array([1.0, 1.0, -2.0]))
    assert allclose(linearization(array([2.0, 2.0, 2.0])), array([-3.0, 3.0]))
    linearization.check_grad(array([2.0, 2.0, 2.0]))


def test_convex_linearization():
    """Test the convex linearization of a function."""
    # Vectorial function
    function = MDOFunction(
        lambda x: 0.5 * array([norm(x) ** 2, -norm(x) ** 2]),
        "f",
        jac=lambda x: array([x, -x]),
        dim=2,
    )
    # The convex linearization (exact w.r.t. x_1) at (1, 1, -2) should be
    # [  0.5*x_1^2 + 5/2 +    (x_2-1) + 8/(x_3+2) ]
    # [ -0.5*x_1^2 - 5/2 +  1/(x_2-1) + 2*(x_3+2) ]
    convex_lin = function.convex_linear_approx(
        array([1.0, 1.0, -2.0]), array([False, True, True])
    )
    assert allclose(convex_lin(array([2.0, 2.0, 2.0])), array([7.5, 4.5]))

    # Check the Jacobian of the convex linearization
    convex_lin.check_grad(array([2.0, 2.0, 2.0]), error_max=1e-6)

    # Scalar function (N.B. scalar value and 1-dimensional Jacobian matrix)
    function = MDOFunction(
        lambda x: 0.5 * norm(x) ** 2,
        "f",
        jac=lambda x: x,
        dim=1,
    )
    convex_lin = function.convex_linear_approx(
        array([1.0, 1.0, -2.0]), array([False, True, True])
    )
    value = convex_lin(array([2.0, 2.0, 2.0]))
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
        args=["x_0", "x_1", "x_2"],
    )


@pytest.mark.parametrize("x_vect", [(ones(2), ones(3)), (eye(3), ones(2))])
def test_quadratic_approximation_error(function_for_quadratic_approximation, x_vect):
    """Test the second-order polynomial of a function with inconsistent input."""
    with pytest.raises(
        ValueError, match="Hessian approximation must be a square ndarray."
    ):
        function_for_quadratic_approximation.quadratic_approx(*x_vect)


def test_quadratic_approximation(function_for_quadratic_approximation):
    """Test the second-order polynomial of a function."""
    approx = function_for_quadratic_approximation.quadratic_approx(ones(3), eye(3))
    assert approx(zeros(3)) == pytest.approx(0.0)
    assert allclose(approx.jac(zeros(3)), zeros(3))
    approx.check_grad(zeros(3), error_max=1e-6)


def test_concatenation():
    """Test the concatenation of functions."""
    dim = 2
    f = MDOFunction(lambda x: norm(x) ** 2, "f", jac=lambda x: 2 * x, dim=1)
    g = MDOFunction(lambda x: x, "g", jac=lambda x: eye(dim), dim=dim)
    h = MDOFunction.concatenate([f, g], "h")
    x_vect = ones(dim)
    assert allclose(h(x_vect), array([2.0, 1.0, 1.0]))
    assert allclose(h.jac(x_vect), array([[2.0, 2.0], [1.0, 0.0], [0.0, 1.0]]))
    h.check_grad(x_vect, error_max=1e-6)


@pytest.mark.parametrize("normalize", [False, True])
def test_set_pt_from_database(normalize):
    problem = Power2()
    problem.preprocess_functions(is_function_input_normalized=normalize)
    x = np.zeros(3)
    problem.evaluate_functions(x, normalize=normalize)
    function = MDOFunction(np.sum, problem.objective.name)
    function.set_pt_from_database(
        problem.database, problem.design_space, normalize=normalize, jac=False
    )
    function(x)


def test_linear_approximation():
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
        func, "f", jac=jac, expr="[1 2] [x] + [5]\n[3 4] [y]   [6]", args=["x", "y"]
    )

    # Get a linear approximation of the MDOFunction
    linear_approximation = function.linear_approximation(array([7.0, 8.0]))
    assert (linear_approximation.coefficients == mat).all()
    assert (linear_approximation.value_at_zero == vec).all()


@pytest.fixture
def function():
    return MDOFunction(lambda x: x, "n", expr="e")


@pytest.mark.parametrize(
    "neg,neg_after,value,expected_n,expected_e",
    [
        (False, True, 1.0, "n + 1.0", "e + 1.0"),
        (True, True, 1.0, "-n - 1.0", "-e - 1.0"),
        (False, True, -1.0, "n - 1.0", "e - 1.0"),
        (True, True, -1.0, "-n + 1.0", "-e + 1.0"),
        (False, False, 1.0, "n + 1.0", "e + 1.0"),
        (True, False, 1.0, "-n + 1.0", "-e + 1.0"),
        (False, False, -1.0, "n - 1.0", "e - 1.0"),
        (True, False, -1.0, "-n - 1.0", "-e - 1.0"),
        (False, False, array([1.0, 1.0]), "n + offset", "e + offset"),
        (True, False, array([1.0, 1.0]), "-n + offset", "-e + offset"),
        (True, True, array([1.0, 1.0]), "-n - offset", "-e - offset"),
    ],
)
def test_offset_name_and_expr(function, neg, neg_after, value, expected_n, expected_e):
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


def test_expects_normalized_inputs(function):
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
):
    """Check the inputs normalization expectation."""
    func = SetPtFromDatabase(database, design_space, function, normalize=normalize)
    assert func.expects_normalized_inputs == normalize


@pytest.mark.parametrize("normalize", [False, True])
def test_expect_normalized_inputs_normfunction(function, problem, normalize):
    """Check the inputs normalization expectation."""
    func = NormFunction(function, normalize, False, problem)
    assert func.expects_normalized_inputs == normalize


@pytest.mark.parametrize("normalize", [False, True])
def test_expect_normalized_inputs_normdbfunction(function, problem, normalize):
    """Check the inputs normalization expectation."""
    func = NormDBFunction(function, normalize, False, problem)
    assert func.expects_normalized_inputs == normalize


def test_activate_counters():
    """Check that the function counter is active by default."""
    func = MDOFunction(lambda x: x, "func")
    assert func.n_calls == 0
    func(array([1.0]))
    assert func.n_calls == 1


def test_deactivate_counters():
    """Check that the function counter is set to None when deactivated."""
    activate_counters = MDOFunction.activate_counters

    MDOFunction.activate_counters = False

    func = MDOFunction(lambda x: x, "func")
    assert func.n_calls is None

    with pytest.raises(RuntimeError, match="The function counters are disabled."):
        func.n_calls = 1

    MDOFunction.activate_counters = activate_counters


def test_get_indexed_name(function):
    """Check the indexed function name."""
    assert function.get_indexed_name(3) == "n!3"


@pytest.mark.parametrize("fexpr", [None, "x**2"])
@pytest.mark.parametrize("gexpr", [None, "x**3"])
@pytest.mark.parametrize(
    "op,op_name,func,jac", [(mul, "*", 32, 80), (truediv, "/", 0.5, -1.0 / 9)]
)
def test_multiplication_by_function(fexpr, gexpr, op, op_name, func, jac):
    """Check the multiplication of a function by a function or its inverse."""
    f = MDOFunction(lambda x: x**2, "f", jac=lambda x: 2 * x, args=["x"], expr=fexpr)
    g = MDOFunction(
        lambda x: x**3, "g", jac=lambda x: 3 * x**2, args=["x"], expr=gexpr
    )

    f_op_g = op(f, g)
    suffix = ""
    if fexpr and gexpr:
        suffix = f" = {fexpr}{op_name}{gexpr}"

    assert repr(f_op_g) == f"f{op_name}g(x)" + suffix
    assert f_op_g(2) == func
    assert f_op_g.jac(2) == jac


@pytest.mark.parametrize("expr", [None, "x**2"])
@pytest.mark.parametrize(
    "op,op_name,func,jac", [(mul, "*", 16, 24), (truediv, "/", 4, 6)]
)
def test_multiplication_by_scalar(expr, op, op_name, func, jac):
    """Check the multiplication of a function by a scalar or its inverse."""
    f = MDOFunction(
        lambda x: x**3, "f", jac=lambda x: 3 * x**2, args=["x"], expr=expr
    )
    f_op_2 = op(f, 2)
    suffix = ""
    if expr:
        if op_name == "*":
            suffix = f" = 2*{expr}"
        else:
            suffix = f" = {expr}/2"

    if op_name == "*":
        assert repr(f_op_2) == "2*f(x)" + suffix
    else:
        assert repr(f_op_2) == "f/2(x)" + suffix

    assert f_op_2(2) == func
    assert f_op_2.jac(2) == jac


def simple_function(x: ndarray) -> ndarray:
    """An identity function.

    Serialization tests require explicitly defined functions instead of lambdas.

    Args:
        x: The input data.

    Returns:
        The input data.
    """
    return x


@pytest.mark.parametrize("activate_counters", [True, False])
@pytest.mark.parametrize(
    "mdo_function, kwargs, value",
    [
        (
            MDOFunction,
            {
                "func": math.sin,
                "f_type": "obj",
                "name": "obj",
                "jac": math.cos,
                "expr": "sin(x)",
                "args": ["x"],
            },
            array([1.0]),
        ),
        (
            NormFunction,
            {
                "orig_func": MDOFunction(simple_function, "f"),
                "normalize": True,
                "round_ints": False,
                "optimization_problem": Power2(),
            },
            array([1.0, 1.0, 1.0]),
        ),
    ],
)
def test_serialize_deserialize(activate_counters, mdo_function, kwargs, value, tmp_wd):
    """Test the serialization/deserialization method.

    Args:
        activate_counters: Whether to activate the function counters.
        mdo_function: The ``MDOFunction`` to be tested.
        kwargs: The keyword arguments to instantiate the ``MDOFunction``.
        value: The value to evaluate the ``MDOFunction``.
        tmp_wd: Fixture to move into a temporary work directory.
    """
    function = mdo_function(**kwargs)
    out_file = "function1.o"
    function.activate_counters = activate_counters
    function(value)
    function.serialize(out_file)
    serialized_func = mdo_function.deserialize(out_file)

    if activate_counters:
        assert function.n_calls == 1
    else:
        assert function.n_calls is None

    s_func_u_dict = serialized_func.__dict__
    ok = True
    for k, _ in function.__dict__.items():
        if k not in s_func_u_dict:
            ok = False
    assert ok


@pytest.mark.parametrize(
    "force_real,expected", [(False, array([1j])), (True, array([0.0]))]
)
def test_force_real(force_real, expected):
    """Verify the use of force_real."""
    f = MDOFunction(lambda x: x, "f", force_real=force_real)
    assert f.evaluate(array([1j])) == expected


@pytest.mark.parametrize(
    "ft1,ft2,ft", [("", "", ""), ("obj", "", "obj"), ("", "obj", "obj")]
)
def test_f_type_sum_two_functions(ft1, ft2, ft):
    """Verify the f_type of the sum of two functions."""
    f = MDOFunction(lambda x: x, "f1", f_type=ft1) + MDOFunction(
        lambda x: x, "f2", f_type=ft2
    )
    assert f.f_type == ft


@pytest.mark.parametrize("f_type", ["", "obj"])
def test_f_type_sum_function_and_number(f_type):
    """Verify the f_type of the sum of a function and a number."""
    f = MDOFunction(lambda x: x, "f", f_type=f_type) + 1.0
    assert f.f_type == f_type


@pytest.mark.parametrize(
    "f_out,g_out,h_out",
    [
        (None, None, []),
        (["a"], None, []),
        (None, ["b"], []),
        (["a"], ["b"], ["a", "b"]),
    ],
)
def test_concatenate(f_out, g_out, h_out):
    """Check ``Concatenate.outvars``."""
    f = Concatenate(
        [
            MDOFunction(lambda x: x, "f", outvars=f_out),
            MDOFunction(lambda x: x, "g", outvars=g_out),
        ],
        "h",
    )
    assert f.outvars == h_out
