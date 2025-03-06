"""Scripts."""

from __future__ import annotations

from pathlib import Path

from gemseo import create_discipline
from gemseo import create_scenario
from gemseo.algos.opt.scipy_local.settings.slsqp import SLSQP_Settings
from gemseo.core.mdo_functions.mdo_function import MDOFunction
from gemseo.formulations.mdf_settings import MDF_Settings
from gemseo.mda.gauss_seidel_settings import MDAGaussSeidel_Settings
from gemseo.problems.mdo.sobieski.core.design_space import SobieskiDesignSpace

ROOT_PATH = Path("doc_src/_examples/post_process/algorithms/")


def save_optimization_history(gallery_conf, fname) -> None:
    "Execute and save converged MDF scenario on the Sobieski's SSBJ problem."
    if Path.exists(ROOT_PATH / "sobieski_mdf_scenario.h5"):
        return

    disciplines = create_discipline([
        "SobieskiPropulsion",
        "SobieskiAerodynamics",
        "SobieskiMission",
        "SobieskiStructure",
    ])

    formulation_settings = MDF_Settings(
        main_mda_name="MDAGaussSeidel",
        main_mda_settings=MDAGaussSeidel_Settings(
            max_mda_iter=30,
            tolerance=1e-10,
            warm_start=True,
            use_lu_fact=True,
        ),
    )

    scenario = create_scenario(
        disciplines,
        "y_4",
        design_space=SobieskiDesignSpace(),
        maximize_objective=True,
        formulation_settings_model=formulation_settings,
    )

    for name in ["g_1", "g_2", "g_3"]:
        scenario.add_constraint(name, constraint_type=MDOFunction.ConstraintType.INEQ)

    scenario.execute(SLSQP_Settings(max_iter=20))
    scenario.save_optimization_history(ROOT_PATH / "sobieski_mdf_scenario.h5")
