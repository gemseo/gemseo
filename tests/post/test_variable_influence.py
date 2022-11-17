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
from gemseo.algos.opt_problem import OptimizationProblem
from gemseo.core.doe_scenario import DOEScenario
from gemseo.post.post_factory import PostFactory
from gemseo.post.variable_influence import VariableInfluence
from gemseo.problems.sobieski.disciplines import SobieskiProblem
from gemseo.problems.sobieski.disciplines import SobieskiStructure
from gemseo.utils.testing import image_comparison
from numpy import repeat

POWER_HDF5_PATH = Path(__file__).parent / "power2_opt_pb.h5"
SSBJ_HDF5_PATH = Path(__file__).parent / "mdf_backup.h5"


def test_variable_influence(tmp_wd, pyplot_close_all):
    """Test the variable influence post-processing.

    Args:
        tmp_wd : Fixture to move into a temporary directory.
        pyplot_close_all : Fixture that prevents figures aggregation
            with matplotlib pyplot.
    """
    factory = PostFactory()
    problem = OptimizationProblem.import_hdf(POWER_HDF5_PATH)
    post = factory.execute(problem, "VariableInfluence", file_path="var_infl")
    assert len(post.output_files) == 1
    for outf in post.output_files:
        assert Path(outf).exists()
    database = problem.database
    database.filter(["pow2", "@pow2"])
    problem.constraints = []
    for k in list(database.keys()):
        v = database.pop(k)
        v["@pow2"] = repeat(v["@pow2"], 60)
        database[repeat(k.wrapped, 60)] = v

    post = factory.execute(problem, "VariableInfluence", file_path="var_infl2")
    assert len(post.output_files) == 1
    for outf in post.output_files:
        assert Path(outf).exists()


def test_variable_influence_doe(tmp_wd, pyplot_close_all):
    """Test the variable influence post-processing on a DOE.

    Args:
        tmp_wd : Fixture to move into a temporary directory.
        pyplot_close_all : Fixture that prevents figures aggregation
            with matplotlib pyplot.
    """
    disc = SobieskiStructure()
    design_space = SobieskiProblem().design_space
    inputs = [name for name in disc.get_input_data_names() if not name.startswith("c_")]
    design_space.filter(inputs)
    doe_scenario = DOEScenario([disc], "DisciplinaryOpt", "y_12", design_space)
    doe_scenario.execute(
        {
            "algo": "DiagonalDOE",
            "n_samples": 10,
            "algo_options": {"eval_jac": False},
        }
    )
    with pytest.raises(ValueError, match="No gradients to plot at current iteration."):
        doe_scenario.post_process(
            "VariableInfluence",
            file_path="doe",
            save=True,
        )


def test_variable_influence_ssbj(tmp_wd, pyplot_close_all):
    """Test the variable influence post-processing on the SSBJ problem.

    Args:
        tmp_wd : Fixture to move into a temporary directory.
        pyplot_close_all : Fixture that prevents figures aggregation
            with matplotlib pyplot.
    """
    factory = PostFactory()
    problem = OptimizationProblem.import_hdf(SSBJ_HDF5_PATH)
    post = factory.execute(
        problem,
        "VariableInfluence",
        file_path="ssbj",
        log_scale=True,
        absolute_value=False,
        level=0.98,
        save_var_files=True,
    )
    assert len(post.output_files) == 14
    for outf in post.output_files:
        assert Path(outf).exists()


TEST_PARAMETERS = {
    "standardized": (True, ["VariableInfluence_standardized"]),
    "unstandardized": (False, ["VariableInfluence_unstandardized"]),
}


@pytest.mark.parametrize(
    "use_standardized_objective, baseline_images",
    TEST_PARAMETERS.values(),
    indirect=["baseline_images"],
    ids=TEST_PARAMETERS.keys(),
)
@image_comparison(None)
def test_common_scenario(
    use_standardized_objective, baseline_images, common_problem, pyplot_close_all
):
    """Check VariableInfluence with objective, standardized or not."""
    opt = VariableInfluence(common_problem)
    common_problem.use_standardized_objective = use_standardized_objective
    opt.execute(save=False)
