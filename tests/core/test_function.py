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
#    INITIAL AUTHORS - initial API and implementation and/or
#                       initial documentation
#        :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES

from __future__ import absolute_import, division, print_function, unicode_literals

import math
import unittest
from builtins import range, str

import numpy as np
from future import standard_library
from future.utils import with_metaclass
from numpy import array, ones
from scipy import optimize

from gemseo import SOFTWARE_NAME
from gemseo.api import configure_logger
from gemseo.core.function import (
    MDOFunction,
    MDOFunctionGenerator,
    MDOLinearFunction,
    SingleInstancePerAttributeId,
)
from gemseo.problems.analytical.power_2 import Power2
from gemseo.problems.sobieski.wrappers import SobieskiMission
from gemseo.third_party.junitxmlreq import link_to
from gemseo.utils.data_conversion import DataConversion

standard_library.install_aliases()


LOGGER = configure_logger(SOFTWARE_NAME)


class Test_SingleInstancePerAttribute(unittest.TestCase):
    """ """

    def test_fail(self):
        """ """

        class DummyClassFail(with_metaclass(SingleInstancePerAttributeId, object)):
            """ """

            def __init__(self):
                pass

        self.assertRaises(Exception, DummyClassFail)

    def test_single_instance(self):
        """ """

        class DummyClass(with_metaclass(SingleInstancePerAttributeId, object)):
            """ """

            def __init__(self, arg):
                self.arg = arg

        obj1 = object()
        obj2 = object()
        d1 = DummyClass(obj1)
        d1_bis = DummyClass(obj1)
        d2 = DummyClass(obj2)
        assert d1 == d1_bis
        assert d2 != d1


class Test_MDOFunction(unittest.TestCase):
    """ """

    def test_init(self):
        """ """
        MDOFunction(math.sin, "sin")

    def test_call(self):
        """ """
        f = MDOFunction(math.sin, "sin")
        for x in range(100):
            assert f(x) == math.sin(x)

        self.assertRaises(TypeError, MDOFunction, math.sin, 1)
        self.assertRaises(TypeError, MDOFunction, math.sin, "sin", jac="cos")
        self.assertRaises(TypeError, MDOFunction, math.sin, "sin", expr=math.sin)
        self.assertRaises(TypeError, MDOFunction, math.sin, "sin", dim=1.3)

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
        """ """
        return MDOFunction(
            math.sin, name="F", f_type="obj", jac=math.cos, expr="sin(x)", args=["x"]
        )

    def test_check_format(self):
        """ """

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
        """ """
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

        LOGGER.info(mm)

        fplu = f + 3.5
        fmin = f - 5.0

        for func in [f, g, h, k, mm, mm_c, fplu, fmin]:
            func.check_grad(np.array([x]), "ComplexStep")

    def test_apply_operator(self):
        f = MDOFunction(np.sin, name="sin")
        self.assertRaises(TypeError, f.__add__, self)

    def test_todict_fromdict(self):
        """ """
        f = MDOFunction(
            np.sin,
            name="sin",
            jac=lambda x: np.array(np.cos(x)),
            expr="sin(x)",
            args=["x"],
            f_type=MDOFunction.TYPE_EQ,
            dim=1,
        )
        repr_dict = f.get_data_dict_repr()
        for k in MDOFunction.DICT_REPR_ATTR:
            assert k in repr_dict
            assert len(str(repr_dict[k])) > 0

        f_2 = MDOFunction.init_from_dict_repr(**repr_dict)

        for name in MDOFunction.DICT_REPR_ATTR:
            assert hasattr(f_2, name)
            assert getattr(f_2, name) == getattr(f, name)

    def test_all_args(self):
        """ """
        self.get_full_sin_func()

    def test_opt_bfgs(self):
        """ """
        f = MDOFunction(
            math.sin, name="F", f_type="obj", jac=math.cos, expr="sin(x)", args=["x"]
        )
        x0 = np.zeros(1)

        # This is powerful !
        opt = optimize.fmin_bfgs(f, x0)
        self.assertAlmostEqual(opt[0], -math.pi / 2, 4)

    def test_repr(self):
        """ """
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


class Test_MDOFunctionGenerator(unittest.TestCase):
    """ """

    @link_to("Req-MDO-1.8")
    def test_update_dict_from_val_arr(self):
        """ """
        x = np.zeros(2)
        d = {"x": x}
        out_d = DataConversion.update_dict_from_array(d, data_names=[], values_array=x)
        assert (out_d["x"] == x).all()

        args = [d, ["x"], np.ones(4)]
        self.assertRaises(Exception, DataConversion.update_dict_from_array, *args)
        args = [d, ["x"], np.ones(1)]
        self.assertRaises(Exception, DataConversion.update_dict_from_array, *args)

    def test_get_values_array_from_dict(self):
        """ """
        x = np.zeros(2)
        data_dict = {"x": x}
        out_x = DataConversion.dict_to_array(data_dict, data_names=["x"])
        assert (out_x == x).all()
        out_x = DataConversion.dict_to_array(data_dict, data_names=[])
        assert out_x.size == 0

    @link_to("Req-MDO-1.8")
    def test_get_function(self):
        """ """
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
        """ """
        MDOFunctionGenerator(None)

    def test_Singleton(self):
        """ """
        sr = SobieskiMission()
        gen = MDOFunctionGenerator(sr)
        instances = type(type(gen)).instances
        inst_len = len(instances)
        MDOFunctionGenerator(sr)
        assert inst_len == len(instances)
        MDOFunctionGenerator(sr)
        assert inst_len == len(instances)

    @link_to("Req-MDO-1.8")
    def test_range_discipline(self):
        """ """
        sr = SobieskiMission()
        gen = MDOFunctionGenerator(sr)
        range_f_Z = gen.get_function(["x_shared"], ["y_4"])
        x_shared = sr.default_inputs["x_shared"]
        Range = range_f_Z(x_shared).real
        range_f_Z2 = gen.get_function("x_shared", ["y_4"])
        Range2 = range_f_Z2(x_shared).real

        assert Range == Range2

    def test_grad_ko(self):
        """ """
        sr = SobieskiMission()
        gen = MDOFunctionGenerator(sr)
        range_f_Z = gen.get_function(["x_shared"], ["y_4"])
        x_shared = sr.default_inputs["x_shared"]
        range_f_Z.check_grad(x_shared, step=1e-5, error_max=1e-4)
        self.assertRaises(
            Exception, range_f_Z.check_grad, x_shared, step=1e-5, error_max=1e-20
        )

        self.assertRaises(ValueError, range_f_Z.check_grad, x_shared, method="toto")

    def test_wrong_default_inputs(self):
        sr = SobieskiMission()
        sr.default_inputs = {"y_34": [1]}
        gen = MDOFunctionGenerator(sr)
        range_f_Z = gen.get_function(["x_shared"], ["y_4"])
        self.assertRaises(ValueError, range_f_Z, array([1.0]))

    def test_wrong_jac(self):
        sr = SobieskiMission()

        def _compute_jacobian_short(inputs, outputs):
            SobieskiMission._compute_jacobian(sr, inputs, outputs)
            sr.jac["y_4"]["x_shared"] = sr.jac["y_4"]["x_shared"][:, :1]

        sr._compute_jacobian = _compute_jacobian_short
        gen = MDOFunctionGenerator(sr)
        range_f_Z = gen.get_function(["x_shared"], ["y_4"])
        self.assertRaises(ValueError, range_f_Z.jac, sr.default_inputs["x_shared"])

    def test_wrong_jac2(self):
        sr = SobieskiMission()

        def _compute_jacobian_long(inputs, outputs):
            SobieskiMission._compute_jacobian(sr, inputs, outputs)
            sr.jac["y_4"]["x_shared"] = ones((1, 20))

        sr._compute_jacobian = _compute_jacobian_long
        gen = MDOFunctionGenerator(sr)
        range_f_Z = gen.get_function(["x_shared"], ["y_4"])
        self.assertRaises(ValueError, range_f_Z.jac, sr.default_inputs["x_shared"])

    def test_set_pt_from_database(self):
        for normalize in (True, False):
            pb = Power2()
            pb.preprocess_functions(normalize=normalize)
            x = np.zeros(3)
            pb.evaluate_functions(x, normalize=normalize)
            func = MDOFunction(np.sum, pb.objective.name)
            func.set_pt_from_database(
                pb.database, pb.design_space, normalize=normalize, jac=False
            )
            func(x)


class Test_MDOLinearFunction(unittest.TestCase):
    def test_linear_function(self):
        """Test the MDOLinearFunction class"""
        coefs = np.array([0.0, 0.0, -1.0, 2.0, 1.0, 0.0, -9.0])
        linear_fun = MDOLinearFunction(coefs, "f")
        assert linear_fun.expr == "-x_2 + 2.0*x_3 + x_4 - 9.0*x_6"
        assert linear_fun(np.ones(coefs.size)) == -7.0
        # Jacobian
        jac = linear_fun.jac(np.array([]))
        for i in range(jac.size):
            assert jac[0, i] == coefs[i]

    def test_mult_linear_function(self):
        """Test the multiplication of a standard MDOFunction and
        an MDOLinearFunction


        """
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
        assert numerical_jac[0, 0] == 96.0

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
