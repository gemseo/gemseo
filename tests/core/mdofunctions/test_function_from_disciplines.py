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
from __future__ import annotations

from gemseo.algos.design_space import DesignSpace
from gemseo.core.mdo_functions.function_from_discipline import FunctionFromDiscipline
from gemseo.disciplines.analytic import AnalyticDiscipline
from gemseo.formulations.disciplinary_opt import DisciplinaryOpt


def test_design_space_copy():
    """Verify that FunctionFromDiscipline uses a copy of DesignSpace.variable_sizes."""
    design_space = DesignSpace()
    design_space.add_variable("a")
    formulation = DisciplinaryOpt([AnalyticDiscipline({"f": "a"})], "f", design_space)
    function = FunctionFromDiscipline(["f"], formulation)
    function.discipline_adapter._DisciplineAdapter__input_names_to_sizes["b"] = 1
    assert "b" not in design_space.variable_sizes
