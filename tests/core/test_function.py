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
import unittest
from operator import mul
from operator import truediv
from unittest import mock

import numpy as np
import pytest
from gemseo.core.mdofunctions.concatenate import Concatenate
from gemseo.core.mdofunctions.function_generator import MDOFunctionGenerator
from gemseo.core.mdofunctions.mdo_function import MDOFunction
from gemseo.core.mdofunctions.mdo_linear_function import MDOLinearFunction
from gemseo.core.mdofunctions.norm_db_function import NormDBFunction
from gemseo.core.mdofunctions.norm_function import NormFunction
from gemseo.core.mdofunctions.set_pt_from_database import SetPtFromDatabase
from gemseo.problems.analytical.power_2 import Power2
from gemseo.problems.sobieski.disciplines import SobieskiMission
from gemseo.utils.data_conversion import concatenate_dict_of_arrays_to_array
from gemseo.utils.data_conversion import update_dict_of_arrays_from_array
from numpy import allclose
from numpy import array
from numpy import eye
from numpy import matmul
from numpy import ndarray
from numpy import ones
from numpy import zeros
from numpy.linalg import norm
from scipy import optimize


class TestMdofunction(unittest.TestCase):
    """"""

    def test_init(self):
        """"""
        MDOFunction(math.sin, "sin")

    def test_call(self):
        """"""
        f = MDOFunction(math.sin, "sin")
        for x in range(100):
            assert f(x) == math.sin(x)

        self.assertRaises(TypeError, MDOFunction, math.sin, "sin", outvars=1.3)
        f2 = MDOFunction(
            math.sin, "sin", outvars=array(["sin"]), f_type=MDOFunction.TYPE_EQ
        )
        assert f2.has_outvars()
        f3 = f + f2
        assert f3.f_type == MDOFunction.TYPE_EQ

        def mult(x, y):
            return x * y

        self.assertRaises(TypeError, mult, f, "toto")

        self.assertRaises(ValueError, MDOFunction.init_from_dict_repr, toto="sin")

    def get_full_sin_func(self):
        """"""
        return MDOFunction(
            math.sin, name="F", f_type="obj", jac=math.cos, expr="sin(x)", args=["x"]
        )

    def test_check_format(self):
        """"""

        self.assertRaises(
            Exception,
            MDOFunction,
            math.sin,
            name="F",
            f_type="Xobj",
            jac=math.cos,
            expr="sin(x)",
            args=["x"],
        )

        MDOFunction(
            math.sin, f_type="obj", name=None, jac=math.cos, expr="sin(x)", args="x"
        )

        f = MDOFunction(
            math.sin, f_type="obj", name="obj", jac=math.cos, expr="sin(x)", args=["x"]
        )
        self.assertRaises(Exception, f.func, "toto")

        f = MDOFunction(math.sin, f_type="obj", name="obj", jac=math.cos, args=["x"])
        self.assertFalse(f.has_expr())

        f = MDOFunction(math.sin, f_type="obj", name="obj", jac=math.cos, expr="sin(x)")
        self.assertFalse(f.has_args())

    def test_add_sub_neg(self):
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

    def test_apply_operator(self):
        f = MDOFunction(np.sin, name="sin")
        self.assertRaises(TypeError, f.__add__, self)

    def test_todict_fromdict(self):
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
        repr_dict = f.to_dict()
        for k in MDOFunction.DICT_REPR_ATTR:
            if k != "special_repr":
                assert k in repr_dict
                assert len(str(repr_dict[k])) > 0

        f_2 = MDOFunction.init_from_dict_repr(**repr_dict)

        for name in MDOFunction.DICT_REPR_ATTR:
            assert hasattr(f_2, name)
            assert getattr(f_2, name) == getattr(f, name)

    def test_all_args(self):
        """"""
        self.get_full_sin_func()

    def test_opt_bfgs(self):
        """"""
        f = MDOFunction(
            math.sin, name="F", f_type="obj", jac=math.cos, expr="sin(x)", args=["x"]
        )
        x0 = np.zeros(1)

        # This is powerful !
        opt = optimize.fmin_bfgs(f, x0)
        self.assertAlmostEqual(opt[0], -math.pi / 2, 4)

    def test_repr(self):
        """"""
        f = self.get_full_sin_func()
        assert str(f) == "F(x) = sin(x)"
        g = MDOFunction(
            math.sin,
            name="G",
            f_type="ineq",
            jac=math.cos,
            expr="sin(x)",
            args=["x", "y"],
        )
        assert str(g) == "G(x, y) = sin(x)"
        h = MDOFunction(math.sin, name="H", args=["x", "y", "x_shared"])
        assert str(h) == "H(x, y, x_shared)"

        i = MDOFunction(math.sin, name="I", args=["y"], expr="sin(y)")
        assert str(g + i) == "G+I(x, y) = sin(x)+sin(y)"
        assert str(g - i) == "G-I(x, y) = sin(x)-sin(y)"

    def test_wrong_jac_shape(self):
        f = MDOFunction(np.sin, name="sin", jac=lambda x: np.array([np.cos(x), 1.0]))
        self.assertRaises(ValueError, f.check_grad, array([0.0]))

    def test_restriction(self):
        """Test the restriction of a function."""
        frozen_indexes = array([1])
        active_indexes = array([0, 2])
        frozen_values = array([2.0])
        x_vect = array([1.0, 2.0, 3.0])
        sub_x_vect = x_vect[active_indexes]
        # Scalar function
        function = MDOFunction(
            lambda x: norm(x) ** 2, "f", jac=lambda x: 2.0 * x, dim=2
        )
        restriction = function.restrict(frozen_indexes, frozen_values, 3, "f_y")
        self.assertAlmostEqual(restriction(sub_x_vect), function(x_vect))
        assert allclose(
            restriction.jac(sub_x_vect), function.jac(x_vect)[active_indexes]
        )
        restriction.check_grad(sub_x_vect, error_max=1e-6)
        # Multi-valued function
        function = MDOFunction(
            lambda x: array([norm(x) ** 2, -norm(x) ** 2]),
            "f",
            jac=lambda x: array([2.0 * x, -2.0 * x]),
        )
        restriction = function.restrict(frozen_indexes, frozen_values, 3, "f_y")
        assert allclose(restriction(sub_x_vect), function(x_vect))
        assert allclose(
            restriction.jac(sub_x_vect), function.jac(x_vect)[:, active_indexes]
        )
        restriction.check_grad(sub_x_vect, error_max=1e-6)

    def test_linearization(self):
        """Test the linearization of a function."""
        function = MDOFunction(
            lambda x: 0.5 * array([norm(x) ** 2, -norm(x) ** 2]),
            "f",
            jac=lambda x: array([x, -x]),
            dim=2,
        )
        linearization = function.linear_approximation(array([1.0, 1.0, -2.0]))
        assert allclose(linearization(array([2.0, 2.0, 2.0])), array([-3.0, 3.0]))

        # Check the Jacobian of the convex linearization
        linearization.check_grad(array([2.0, 2.0, 2.0]))

    def test_convex_linearization(self):
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

    def test_quadratic_approximation(self):
        """Test the second-order polynomial of a function."""
        dim = 3
        args = (f"x_{i}" for i in range(dim))
        function = MDOFunction(
            lambda x: 0.5 * norm(x) ** 2, "f", jac=lambda x: x, args=args
        )
        x_vect = ones(dim)
        hessian_approx = eye(dim)
        self.assertRaises(ValueError, function.quadratic_approx, ones(dim - 1), x_vect)
        self.assertRaises(
            ValueError, function.quadratic_approx, hessian_approx, ones(dim - 1)
        )
        approx = function.quadratic_approx(x_vect, hessian_approx)
        self.assertAlmostEqual(approx(zeros(dim)), 0.0)
        assert allclose(approx.jac(zeros(dim)), zeros(dim))
        approx.check_grad(zeros(dim), error_max=1e-6)

    def test_concatenation(self):
        """Test the concatenation of functions."""
        dim = 2
        f = MDOFunction(lambda x: norm(x) ** 2, "f", jac=lambda x: 2 * x, dim=1)
        g = MDOFunction(lambda x: x, "g", jac=lambda x: eye(dim), dim=dim)
        h = MDOFunction.concatenate([f, g], "h")
        x_vect = ones(dim)
        assert allclose(h(x_vect), array([2.0, 1.0, 1.0]))
        assert allclose(h.jac(x_vect), array([[2.0, 2.0], [1.0, 0.0], [0.0, 1.0]]))
        h.check_grad(x_vect, error_max=1e-6)


class TestMdofunctiongenerator(unittest.TestCase):
    """"""

    def test_update_dict_from_val_arr(self):
        """"""
        x = np.zeros(2)
        d = {"x": x}
        out_d = update_dict_of_arrays_from_array(d, [], x)
        assert (out_d["x"] == x).all()

        args = [d, ["x"], np.ones(4)]
        self.assertRaises(Exception, update_dict_of_arrays_from_array, *args)
        args = [d, ["x"], np.ones(1)]
        self.assertRaises(Exception, update_dict_of_arrays_from_array, *args)

    def test_get_values_array_from_dict(self):
        """"""
        x = np.zeros(2)
        data_dict = {"x": x}
        out_x = concatenate_dict_of_arrays_to_array(data_dict, ["x"])
        assert (out_x == x).all()
        out_x = concatenate_dict_of_arrays_to_array(data_dict, [])
        assert out_x.size == 0

    def test_get_function(self):
        """"""
        sr = SobieskiMission()
        gen = MDOFunctionGenerator(sr)
        gen.get_function(None, None)
        args = [["x_shared"], ["y_4"]]
        gen.get_function(*args)
        args = ["x_shared", "y_4"]
        gen.get_function(*args)
        args = [["toto"], ["y_4"]]
        self.assertRaises(Exception, gen.get_function, *args)
        args = [["x_shared"], ["toto"]]
        self.assertRaises(Exception, gen.get_function, *args)

    def test_instanciation(self):
        """"""
        MDOFunctionGenerator(None)

    def test_range_discipline(self):
        """"""
        sr = SobieskiMission()
        gen = MDOFunctionGenerator(sr)
        range_f_z = gen.get_function(["x_shared"], ["y_4"])
        x_shared = sr.default_inputs["x_shared"]
        range_ = range_f_z(x_shared).real
        range_f_z2 = gen.get_function("x_shared", ["y_4"])
        range2 = range_f_z2(x_shared).real

        assert range_ == range2

    def test_grad_ko(self):
        """"""
        sr = SobieskiMission()
        gen = MDOFunctionGenerator(sr)
        range_f_z = gen.get_function(["x_shared"], ["y_4"])
        x_shared = sr.default_inputs["x_shared"]
        range_f_z.check_grad(x_shared, step=1e-5, error_max=1e-4)
        self.assertRaises(
            Exception, range_f_z.check_grad, x_shared, step=1e-5, error_max=1e-20
        )

        self.assertRaises(ValueError, range_f_z.check_grad, x_shared, method="toto")

    def test_wrong_default_inputs(self):
        sr = SobieskiMission()
        sr.default_inputs = {"y_34": array([1])}
        gen = MDOFunctionGenerator(sr)
        range_f_z = gen.get_function(["x_shared"], ["y_4"])
        self.assertRaises(ValueError, range_f_z, array([1.0]))

    def test_wrong_jac(self):
        sr = SobieskiMission()

        def _compute_jacobian_short(inputs, outputs):
            SobieskiMission._compute_jacobian(sr, inputs, outputs)
            sr.jac["y_4"]["x_shared"] = sr.jac["y_4"]["x_shared"][:, :1]

        sr._compute_jacobian = _compute_jacobian_short
        gen = MDOFunctionGenerator(sr)
        range_f_z = gen.get_function(["x_shared"], ["y_4"])
        self.assertRaises(ValueError, range_f_z.jac, sr.default_inputs["x_shared"])

    def test_wrong_jac2(self):
        sr = SobieskiMission()

        def _compute_jacobian_long(inputs, outputs):
            SobieskiMission._compute_jacobian(sr, inputs, outputs)
            sr.jac["y_4"]["x_shared"] = ones((1, 20))

        sr._compute_jacobian = _compute_jacobian_long
        gen = MDOFunctionGenerator(sr)
        range_f_z = gen.get_function(["x_shared"], ["y_4"])
        self.assertRaises(ValueError, range_f_z.jac, sr.default_inputs["x_shared"])

    def test_set_pt_from_database(self):
        for normalize in (True, False):
            pb = Power2()
            pb.preprocess_functions(is_function_input_normalized=normalize)
            x = np.zeros(3)
            pb.evaluate_functions(x, normalize=normalize)
            func = MDOFunction(np.sum, pb.objective.name)
            func.set_pt_from_database(
                pb.database, pb.design_space, normalize=normalize, jac=False
            )
            func(x)


class TestMDOLinearFunction(unittest.TestCase):
    def test_inputs(self):
        """Tests the formatting of the passed inputs."""
        coeffs_as_list = [1.0, 2.0]
        coeffs_as_vec = array(coeffs_as_list)
        coeffs_as_mat = array([coeffs_as_list])
        self.assertRaises(ValueError, MDOLinearFunction, coeffs_as_list, "f")
        MDOLinearFunction(coeffs_as_mat, "f")
        func = MDOLinearFunction(coeffs_as_vec, "f")
        assert (func.coefficients == coeffs_as_mat).all()
        self.assertRaises(
            ValueError,
            MDOLinearFunction,
            coeffs_as_mat,
            "f",
            value_at_zero=array([0.0, 0.0]),
        )
        MDOLinearFunction(coeffs_as_mat, "f", value_at_zero=array([0.0]))
        func = MDOLinearFunction(coeffs_as_mat, "f")
        assert (func.value_at_zero == array([0.0])).all()

    def test_args_generation(self):
        """Tests the generation of arguments strings."""
        # No arguments strings passed
        func = MDOLinearFunction(array([1.0, 2.0, 3.0]), "f")
        args = [
            MDOLinearFunction.DEFAULT_ARGS_BASE
            + MDOLinearFunction.INDEX_PREFIX
            + str(i)
            for i in range(3)
        ]
        assert func.args == args
        # Not enough arguments strings passed
        func = MDOLinearFunction(array([1.0, 2.0, 3.0]), "f", args=["u", "v"])
        assert func.args == args
        # Only one argument string passed
        func = MDOLinearFunction(array([1.0, 2.0, 3.0]), "f", args=["u"])
        args = ["u" + MDOLinearFunction.INDEX_PREFIX + str(i) for i in range(3)]
        assert func.args == args
        # Enough arguments strings passed
        func = MDOLinearFunction(array([1.0, 2.0, 3.0]), "f", args=["u1", "u2", "v"])
        assert func.args == ["u1", "u2", "v"]

    def test_linear_function(self):
        """Tests the MDOLinearFunction class."""
        coefs = np.array([0.0, 0.0, -1.0, 2.0, 1.0, 0.0, -9.0])
        linear_fun = MDOLinearFunction(coefs, "f")
        coeffs_str = (MDOFunction.COEFF_FORMAT_1D.format(coeff) for coeff in (2, 9))
        expr = "-x!2 + {}*x!3 + x!4 - {}*x!6".format(*coeffs_str)
        assert linear_fun.expr == expr
        assert linear_fun(np.ones(coefs.size)) == -7.0
        # Jacobian
        jac = linear_fun.jac(np.array([]))
        for i in range(jac.size):
            assert jac[i] == coefs[i]

    def test_nd_expression(self):
        """Tests multi-valued MDOLinearFunction literal expression."""
        coefficients = array([[1.0, 2.0], [3.0, 4.0]])
        value_at_zero = array([5.0, 6.0])
        func = MDOLinearFunction(
            coefficients, "f", args=["x", "y"], value_at_zero=value_at_zero
        )
        coeffs_str = (
            MDOFunction.COEFF_FORMAT_ND.format(coeff) for coeff in (1, 2, 5, 3, 4, 6)
        )
        expr = "[{} {}][x] + [{}]\n[{} {}][y]   [{}]".format(*coeffs_str)
        assert func.expr == expr

    def test_mult_linear_function(self):
        """Tests the multiplication of a standard MDOFunction and an
        MDOLinearFunction."""
        sqr = MDOFunction(
            lambda x: x[0] ** 2.0,
            name="sqr",
            jac=lambda x: 2.0 * x[0],
            expr="x_0**2.",
            args=["x"],
            dim=1,
        )

        coefs = np.array([2.0])
        linear_fun = MDOLinearFunction(coefs, "f")

        prod = sqr * linear_fun
        x_array = np.array([4.0])
        assert prod(x_array) == 128.0

        numerical_jac = prod.jac(x_array)
        assert numerical_jac[0] == 96.0

        sqr_eq = MDOFunction(
            lambda x: x[0] ** 2.0,
            name="sqr",
            jac=lambda x: 2.0 * x[0],
            expr="x_0**2.",
            args=["x"],
            dim=1,
            f_type="eq",
        )
        prod = sqr * sqr_eq

    def test_linear_restriction(self):
        """Tests the restriction of an MDOLinear function."""
        coefficients = array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        value_at_zero = array([7.0, 8.0])
        function = MDOLinearFunction(
            coefficients, "f", args=["x", "y", "z"], value_at_zero=value_at_zero
        )
        frozen_indexes = array([1, 2])
        frozen_values = array([1.0, 2.0])
        restriction = function.restrict(frozen_indexes, frozen_values)
        assert (restriction.coefficients == array([[1.0], [4.0]])).all()
        assert (restriction.value_at_zero == array([15.0, 25.0])).all()
        assert restriction.args == ["x"]
        coeffs_str = (MDOFunction.COEFF_FORMAT_ND.format(val) for val in (1, 15, 4, 25))
        expr = "[{}][x] + [{}]\n[{}]      [{}]".format(*coeffs_str)
        assert restriction.expr == expr

    def test_linear_approximation(self):
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
        x_vect = array([7.0, 8.0])
        linear_approximation = function.linear_approximation(x_vect)
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
