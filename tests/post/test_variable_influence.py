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
#    INITIAL AUTHORS - API and implementation and/or documentation
#       :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import annotations

from pathlib import Path

import pytest
from numpy import array
from numpy import eye

from gemseo.algos.design_space import DesignSpace
from gemseo.algos.doe.diagonal_doe.diagonal_doe import DiagonalDOE
from gemseo.algos.doe.diagonal_doe.settings.diagonal_doe_settings import (
    DiagonalDOE_Settings,
)
from gemseo.algos.optimization_problem import OptimizationProblem
from gemseo.core.functions.array_function import ArrayFunction
from gemseo.post import VariableInfluence_Settings
from gemseo.post.factory import POST_FACTORY
from gemseo.post.variable_influence import VariableInfluence
from gemseo.problems.mdo.sobieski.core.design_space import SobieskiDesignSpace
from gemseo.problems.mdo.sobieski.disciplines import SobieskiStructure
from gemseo.scenarios.mdo import MDOScenario
from gemseo.utils.testing.helpers import assert_exception

POWER_HDF5_PATH = Path(__file__).parent / "power2_opt_pb.h5"
SSBJ_HDF5_PATH = Path(__file__).parent / "mdf_backup.h5"


def test_variable_influence(tmp_wd) -> None:
    """Test the variable influence post-processing.

    Args:
        tmp_wd : Fixture to move into a temporary directory.
    """
    problem = OptimizationProblem.from_hdf(POWER_HDF5_PATH)
    post = POST_FACTORY.execute(
        problem, VariableInfluence_Settings(file_path="var_infl")
    )
    assert len(post.output_file_paths) == 1
    for outf in post.output_file_paths:
        assert Path(outf).exists()

    # THIS CODE SEEMS WRONG: THE SENSITIVITIES ARE NOT COMPUTED WRT DESIGN VARIABLES.
    # database = problem.database
    # database.filter(["pow2", "@pow2"])
    # problem.constraints = []
    # for k in list(database.keys()):
    #     v = database.pop(k)
    #     v["@pow2"] = repeat(v["@pow2"], 60)
    #     database[repeat(k.wrapped, 60)] = v
    #
    # post = factory.execute(problem, "VariableInfluence", file_path="var_infl2")
    # assert len(post.output_file_paths) == 1
    # for outf in post.output_file_paths:
    #     assert Path(outf).exists()


def test_variable_influence_doe(tmp_wd, snapshot) -> None:
    """Test the variable influence post-processing on a DOE.

    Args:
        tmp_wd : Fixture to move into a temporary directory.
    """
    disc = SobieskiStructure()
    design_space = SobieskiDesignSpace()
    inputs = [name for name in disc.io.input_grammar if not name.startswith("c_")]
    design_space.filter(inputs)
    doe_scenario = MDOScenario([disc], design_space)
    doe_scenario.add_objective("y_12")
    doe_scenario.execute(DiagonalDOE_Settings(n_samples=10, eval_jac=False))
    with assert_exception(ValueError, snapshot):
        doe_scenario.post_process(
            VariableInfluence_Settings(file_path="doe", save=True)
        )


def test_variable_influence_ssbj(tmp_wd) -> None:
    """Test the variable influence post-processing on the SSBJ problem.

    Args:
        tmp_wd : Fixture to move into a temporary directory.
    """
    problem = OptimizationProblem.from_hdf(SSBJ_HDF5_PATH)
    post = POST_FACTORY.execute(
        problem,
        VariableInfluence_Settings(
            file_path="ssbj",
            log_scale=True,
            absolute_value=False,
            level=0.98,
            save_var_files=True,
        ),
    )
    assert len(post.output_file_paths) == 14
    for outf in post.output_file_paths:
        assert Path(outf).exists()


@pytest.mark.parametrize(
    "use_standardized_objective",
    [True, False],
)
def test_common_scenario(
    use_standardized_objective, common_problem, snapshot_matplotlib
) -> None:
    """Check VariableInfluence with objective, standardized or not."""
    common_problem.use_standardized_objective = use_standardized_objective
    opt = VariableInfluence(common_problem)
    opt.execute(VariableInfluence_Settings(save=False))


@pytest.mark.parametrize(
    "size",
    [10, 20, 21, 30],
)
def test_visible_labels(size, snapshot_matplotlib) -> None:
    """A dummy optimization problem to check post-processors."""
    design_space = DesignSpace()
    design_space.add_variable("x", size=size, lower_bound=0, upper_bound=1, value=0.5)
    problem = OptimizationProblem(design_space)
    func = ArrayFunction(sum, name="obj", jac=lambda x: array([1.0] * size))
    problem.objective = func
    problem.minimize_objective = False
    func = ArrayFunction(lambda x: x * 0.5, name="eq", jac=lambda x: 0.5 * eye(size))
    problem.add_constraint(func, constraint_type=ArrayFunction.ConstraintType.EQ)
    doe = DiagonalDOE()
    doe.execute(problem, settings=DiagonalDOE_Settings(n_samples=size, eval_jac=True))
    post = VariableInfluence(problem)
    post.execute(VariableInfluence_Settings(save=False))
