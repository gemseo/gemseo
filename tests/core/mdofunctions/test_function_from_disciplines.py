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

import pytest
from numpy import array

from gemseo.algos.design_space import DesignSpace
from gemseo.algos.optimization_problem import OptimizationProblem
from gemseo.core.functions.function_from_discipline import FunctionFromDiscipline
from gemseo.disciplines.analytic import AnalyticDiscipline
from gemseo.formulations.disciplinary_opt import DisciplinaryOpt


@pytest.mark.parametrize("use_discipline", [False, True])
def test_design_space_copy(use_discipline):
    """Verify that FunctionFromDiscipline uses a copy of DesignSpace.variable_sizes."""
    design_space = DesignSpace()
    design_space.add_variable("a")
    evaluation_problem = OptimizationProblem(design_space)
    discipline = AnalyticDiscipline({"f": "2*a"})
    formulation = DisciplinaryOpt(evaluation_problem, [discipline])
    evaluation_problem.objective = formulation.create_objective(["f"])
    kwargs = {"discipline": discipline} if use_discipline else {}
    function = FunctionFromDiscipline(["f"], formulation, **kwargs)
    assert function.evaluate(array([3.0])) == 6.0
    function.discipline_adapter._DisciplineAdapter__input_name_to_size["b"] = 1
    assert "b" not in design_space.variable_sizes
