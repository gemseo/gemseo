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

import re

import numpy as np
import pytest
from numpy import array
from numpy import ones
from numpy.testing import assert_equal

from gemseo.algos.design_space import DesignSpace
from gemseo.core.mdo_functions.discipline_adapter_generator import (
    DisciplineAdapterGenerator,
)
from gemseo.core.mdo_functions.function_from_discipline import FunctionFromDiscipline
from gemseo.disciplines.analytic import AnalyticDiscipline
from gemseo.formulations.disciplinary_opt import DisciplinaryOpt
from gemseo.problems.mdo.sobieski.disciplines import SobieskiMission
from gemseo.utils.data_conversion import concatenate_dict_of_arrays_to_array


def test_get_values_array_from_dict() -> None:
    """"""
    x = np.zeros(2)
    data_dict = {"x": x}
    out_x = concatenate_dict_of_arrays_to_array(data_dict, ["x"])
    assert (out_x == x).all()
    out_x = concatenate_dict_of_arrays_to_array(data_dict, [])
    assert out_x.size == 0


def test_get_function() -> None:
    """"""
    sr = SobieskiMission()
    gen = DisciplineAdapterGenerator(sr)
    gen.get_function(None, None)
    args = [["x_shared"], ["y_4"]]
    gen.get_function(*args)
    args = [["toto"], ["y_4"]]
    with pytest.raises(
        ValueError,
        match=re.escape(
            "['toto'] are not names of inputs in the discipline SobieskiMission; "
            "expected names among ['x_shared', 'y_14', 'y_24', 'y_34']."
        ),
    ):
        gen.get_function(*args)
    args = [["x_shared"], ["toto"]]
    with pytest.raises(
        ValueError,
        match=re.escape(
            "['toto'] are not names of outputs in the discipline SobieskiMission; "
            "expected names among ['y_4']."
        ),
    ):
        gen.get_function(*args)


def test_instanciation() -> None:
    """"""
    DisciplineAdapterGenerator(None)


def test_range_discipline() -> None:
    """"""
    sr = SobieskiMission()
    gen = DisciplineAdapterGenerator(sr)
    range_f_z = gen.get_function(["x_shared"], ["y_4"])
    x_shared = sr.io.input_grammar.defaults["x_shared"]
    range_ = range_f_z.evaluate(x_shared).real
    range_f_z2 = gen.get_function(["x_shared"], ["y_4"])
    range2 = range_f_z2.evaluate(x_shared).real

    assert range_ == range2


def test_grad_ko() -> None:
    """"""
    sr = SobieskiMission()
    gen = DisciplineAdapterGenerator(sr)
    range_f_z = gen.get_function(["x_shared"], ["y_4"])
    x_shared = sr.io.input_grammar.defaults["x_shared"]
    range_f_z.check_grad(x_shared, step=1e-5, error_max=1e-4)
    with pytest.raises(ValueError):
        range_f_z.check_grad(x_shared, step=1e-5, error_max=1e-20)
    with pytest.raises(ImportError):
        range_f_z.check_grad(x_shared, approximation_mode="toto")


def test_wrong_default_inputs() -> None:
    sr = SobieskiMission()
    sr.io.input_grammar.defaults = {"y_34": array([1])}
    gen = DisciplineAdapterGenerator(sr)
    range_f_z = gen.get_function(["x_shared"], ["y_4"])
    with pytest.raises(ValueError):
        range_f_z.evaluate(array([1.0]))


def test_wrong_jac() -> None:
    class SM(SobieskiMission):
        def _compute_jacobian(self, inputs, outputs) -> None:
            super()._compute_jacobian(inputs, outputs)
            self.jac["y_4"]["x_shared"] = self.jac["y_4"]["x_shared"][:, :1]

    sr = SM()
    gen = DisciplineAdapterGenerator(sr)
    range_f_z = gen.get_function(["x_shared"], ["y_4"])
    with pytest.raises(ValueError):
        range_f_z.jac(sr.io.input_grammar.defaults["x_shared"])


def test_wrong_jac2() -> None:
    class SM(SobieskiMission):
        def _compute_jacobian(self, inputs, outputs) -> None:
            super()._compute_jacobian(inputs, outputs)
            self.jac["y_4"]["x_shared"] = ones((1, 20))

    sr = SM()
    gen = DisciplineAdapterGenerator(sr)
    range_f_z = gen.get_function(["x_shared"], ["y_4"])
    with pytest.raises(ValueError):
        range_f_z.jac(sr.io.input_grammar.defaults["x_shared"])


@pytest.mark.parametrize(
    ("input_names", "expected_output_value"),
    [
        ((), array(8)),
        (["a", "b"], array(8)),
        (["a"], array(2)),
        (["b"], array(6)),
    ],
)
def test_function_from_discipline_input_names(input_names, expected_output_value):
    """Check the input_names argument of FunctionFromDiscipline."""
    discipline = AnalyticDiscipline({"y": "2*a+3*b"})
    design_space = DesignSpace()
    design_space.add_variable("a")
    design_space.add_variable("b")
    formulation = DisciplinaryOpt([discipline], "y", design_space)
    function = FunctionFromDiscipline(["y"], formulation, input_names=input_names)
    assert_equal(function.evaluate(array([1, 2])), expected_output_value)
