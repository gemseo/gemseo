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

import numpy as np
import pytest
from gemseo.core.mdofunctions.function_generator import MDOFunctionGenerator
from gemseo.problems.sobieski.disciplines import SobieskiMission
from gemseo.utils.data_conversion import concatenate_dict_of_arrays_to_array
from gemseo.utils.data_conversion import update_dict_of_arrays_from_array
from numpy import array
from numpy import ones


def test_update_dict_from_val_arr():
    """"""
    x = np.zeros(2)
    d = {"x": x}
    out_d = update_dict_of_arrays_from_array(d, [], x)
    assert (out_d["x"] == x).all()

    args = [d, ["x"], np.ones(4)]
    with pytest.raises(Exception):
        update_dict_of_arrays_from_array(*args)

    args = [d, ["x"], np.ones(1)]
    with pytest.raises(Exception):
        update_dict_of_arrays_from_array(*args)


def test_get_values_array_from_dict():
    """"""
    x = np.zeros(2)
    data_dict = {"x": x}
    out_x = concatenate_dict_of_arrays_to_array(data_dict, ["x"])
    assert (out_x == x).all()
    out_x = concatenate_dict_of_arrays_to_array(data_dict, [])
    assert out_x.size == 0


def test_get_function():
    """"""
    sr = SobieskiMission()
    gen = MDOFunctionGenerator(sr)
    gen.get_function(None, None)
    args = [["x_shared"], ["y_4"]]
    gen.get_function(*args)
    args = ["x_shared", "y_4"]
    gen.get_function(*args)
    args = [["toto"], ["y_4"]]
    with pytest.raises(Exception):
        gen.get_function(*args)
    args = [["x_shared"], ["toto"]]
    with pytest.raises(Exception):
        gen.get_function(*args)


def test_instanciation():
    """"""
    MDOFunctionGenerator(None)


def test_range_discipline():
    """"""
    sr = SobieskiMission()
    gen = MDOFunctionGenerator(sr)
    range_f_z = gen.get_function(["x_shared"], ["y_4"])
    x_shared = sr.default_inputs["x_shared"]
    range_ = range_f_z(x_shared).real
    range_f_z2 = gen.get_function("x_shared", ["y_4"])
    range2 = range_f_z2(x_shared).real

    assert range_ == range2


def test_grad_ko():
    """"""
    sr = SobieskiMission()
    gen = MDOFunctionGenerator(sr)
    range_f_z = gen.get_function(["x_shared"], ["y_4"])
    x_shared = sr.default_inputs["x_shared"]
    range_f_z.check_grad(x_shared, step=1e-5, error_max=1e-4)
    with pytest.raises(Exception):
        range_f_z.check_grad(x_shared, step=1e-5, error_max=1e-20)
    with pytest.raises(ValueError):
        range_f_z.check_grad(x_shared, method="toto")


def test_wrong_default_inputs():
    sr = SobieskiMission()
    sr.default_inputs = {"y_34": array([1])}
    gen = MDOFunctionGenerator(sr)
    range_f_z = gen.get_function(["x_shared"], ["y_4"])
    with pytest.raises(ValueError):
        range_f_z(array([1.0]))


def test_wrong_jac():
    sr = SobieskiMission()

    def _compute_jacobian_short(inputs, outputs):
        SobieskiMission._compute_jacobian(sr, inputs, outputs)
        sr.jac["y_4"]["x_shared"] = sr.jac["y_4"]["x_shared"][:, :1]

    sr._compute_jacobian = _compute_jacobian_short
    gen = MDOFunctionGenerator(sr)
    range_f_z = gen.get_function(["x_shared"], ["y_4"])
    with pytest.raises(ValueError):
        range_f_z.jac(sr.default_inputs["x_shared"])


def test_wrong_jac2():
    sr = SobieskiMission()

    def _compute_jacobian_long(inputs, outputs):
        SobieskiMission._compute_jacobian(sr, inputs, outputs)
        sr.jac["y_4"]["x_shared"] = ones((1, 20))

    sr._compute_jacobian = _compute_jacobian_long
    gen = MDOFunctionGenerator(sr)
    range_f_z = gen.get_function(["x_shared"], ["y_4"])
    with pytest.raises(ValueError):
        range_f_z.jac(sr.default_inputs["x_shared"])
