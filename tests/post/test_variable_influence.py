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

import re
from pathlib import Path

import pytest

from gemseo.algos.optimization_problem import OptimizationProblem
from gemseo.post.factory import PostFactory
from gemseo.post.variable_influence import VariableInfluence
from gemseo.problems.mdo.sobieski.core.design_space import SobieskiDesignSpace
from gemseo.problems.mdo.sobieski.disciplines import SobieskiStructure
from gemseo.scenarios.doe_scenario import DOEScenario
from gemseo.utils.testing.helpers import image_comparison

POWER_HDF5_PATH = Path(__file__).parent / "power2_opt_pb.h5"
SSBJ_HDF5_PATH = Path(__file__).parent / "mdf_backup.h5"


def test_variable_influence(tmp_wd) -> None:
    """Test the variable influence post-processing.

    Args:
        tmp_wd : Fixture to move into a temporary directory.
    """
    factory = PostFactory()
    problem = OptimizationProblem.from_hdf(POWER_HDF5_PATH)
    post = factory.execute(problem, post_name="VariableInfluence", file_path="var_infl")
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


def test_variable_influence_doe(tmp_wd) -> None:
    """Test the variable influence post-processing on a DOE.

    Args:
        tmp_wd : Fixture to move into a temporary directory.
    """
    disc = SobieskiStructure()
    design_space = SobieskiDesignSpace()
    inputs = [name for name in disc.io.input_grammar if not name.startswith("c_")]
    design_space.filter(inputs)
    doe_scenario = DOEScenario(
        [disc], "y_12", design_space, formulation_name="DisciplinaryOpt"
    )
    doe_scenario.execute(algo_name="DiagonalDOE", n_samples=10, eval_jac=False)
    with pytest.raises(
        ValueError, match=re.escape("No gradients to plot at current iteration.")
    ):
        doe_scenario.post_process(
            post_name="VariableInfluence",
            file_path="doe",
            save=True,
        )


def test_variable_influence_ssbj(tmp_wd) -> None:
    """Test the variable influence post-processing on the SSBJ problem.

    Args:
        tmp_wd : Fixture to move into a temporary directory.
    """
    factory = PostFactory()
    problem = OptimizationProblem.from_hdf(SSBJ_HDF5_PATH)
    post = factory.execute(
        problem,
        post_name="VariableInfluence",
        file_path="ssbj",
        log_scale=True,
        absolute_value=False,
        level=0.98,
        save_var_files=True,
    )
    assert len(post.output_file_paths) == 14
    for outf in post.output_file_paths:
        assert Path(outf).exists()


TEST_PARAMETERS = {
    "standardized": (True, ["VariableInfluence_standardized"]),
    "unstandardized": (False, ["VariableInfluence_unstandardized"]),
}


@pytest.mark.parametrize(
    ("use_standardized_objective", "baseline_images"),
    TEST_PARAMETERS.values(),
    indirect=["baseline_images"],
    ids=TEST_PARAMETERS.keys(),
)
@image_comparison(None)
def test_common_scenario(
    use_standardized_objective, baseline_images, common_problem
) -> None:
    """Check VariableInfluence with objective, standardized or not."""
    opt = VariableInfluence(common_problem)
    common_problem.use_standardized_objective = use_standardized_objective
    opt.execute(save=False)
