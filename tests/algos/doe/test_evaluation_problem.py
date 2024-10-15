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

import re
from math import prod

from numpy import array
from numpy.testing import assert_equal

from gemseo.algos.design_space import DesignSpace
from gemseo.algos.doe.custom_doe.custom_doe import CustomDOE
from gemseo.algos.evaluation_problem import EvaluationProblem
from gemseo.core.mdo_functions.mdo_function import MDOFunction


def test_default(caplog):
    """Check that a DOELibrary can handle an EvaluationProblem."""
    design_space = DesignSpace()
    design_space.add_variable("x", size=2)

    evaluation_problem = EvaluationProblem(design_space)
    evaluation_problem.add_observable(MDOFunction(sum, "sum"))
    evaluation_problem.add_observable(MDOFunction(prod, "prod"))

    custom_doe = CustomDOE()
    custom_doe.execute(evaluation_problem, samples=array([[2.0, 3.0], [4.0, 5.0]]))

    get_function_history = evaluation_problem.database.get_function_history
    assert_equal(get_function_history("sum"), array([5.0, 9.0]))
    assert_equal(get_function_history("prod"), array([6.0, 20.0]))
    result = "\n".join([line[2] for line in caplog.record_tuples])
    expected_result = r"""^Evaluation problem:
   Evaluate the functions: prod, sum
   over the design space:
      \+------\+-------------\+-------\+-------------\+-------\+
      \| Name \| Lower bound \| Value \| Upper bound \| Type  \|
      \+------\+-------------\+-------\+-------------\+-------\+
      \| x\[0\] \|     -inf    \|  None \|     inf     \| float \|
      \| x\[1\] \|     -inf    \|  None \|     inf     \| float \|
      \+------\+-------------\+-------\+-------------\+-------\+
Running the algorithm CustomDOE:
    50%\|█████     \| 1\/2 \[\d+:\d+<(?:\d+:\d+|\?), (?:\s*\d+\.\d+|\?) it\/sec\]
   100%\|██████████\| 2\/2 \[\d+:\d+<(?:\d+:\d+|\?), (?:\s*\d+\.\d+|\?) it\/sec\]$"""
    assert re.match(expected_result, result)
