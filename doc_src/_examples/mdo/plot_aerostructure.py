# Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com
#
# This work is licensed under a BSD 0-Clause License.
#
# Permission to use, copy, modify, and/or distribute this software
# for any purpose with or without fee is hereby granted.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL
# WARRANTIES WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL
# THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT, INDIRECT,
# OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING
# FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT,
# NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION
# WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
# Contributors:
#    INITIAL AUTHORS - API and implementation and/or documentation
#        :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""
MDO formulations for a toy example in aerostructure
===================================================
"""

from __future__ import annotations

from gemseo import configure_logger
from gemseo import create_discipline
from gemseo import create_scenario
from gemseo import generate_n2_plot
from gemseo.problems.mdo.aerostructure.aerostructure_design_space import (
    AerostructureDesignSpace,
)
from gemseo.settings.mda import MDAChain_Settings
from gemseo.settings.opt import NLOPT_COBYLA_Settings
from gemseo.settings.opt import NLOPT_SLSQP_Settings

configure_logger()

# Passed to algo settings
cobyla_settings = NLOPT_COBYLA_Settings(
    max_iter=7,
    xtol_rel=1e-8,
    xtol_abs=1e-8,
    ftol_rel=1e-8,
    ftol_abs=1e-8,
    ineq_tolerance=1e-5,
    eq_tolerance=1e-3,
)

slsqp_settings = NLOPT_SLSQP_Settings(
    max_iter=10,
    xtol_rel=1e-8,
    xtol_abs=1e-8,
    ftol_rel=1e-8,
    ftol_abs=1e-8,
    ineq_tolerance=1e-5,
    eq_tolerance=1e-3,
)
# %%
# Create discipline
# -----------------
# First, we create disciplines (aero, structure, mission) with dummy formulas
# using the :class:`.AnalyticDiscipline` class.

aero_formulas = {
    "drag": "0.1*((sweep/360)**2 + 200 + thick_airfoils**2-thick_airfoils -4*displ)",
    "forces": "10*sweep + 0.2*thick_airfoils-0.2*displ",
    "lift": "(sweep + 0.2*thick_airfoils-2.*displ)/3000.",
}
aerodynamics = create_discipline(
    "AnalyticDiscipline", name="Aerodynamics", expressions=aero_formulas
)
struc_formulas = {
    "mass": "4000*(sweep/360)**3 + 200000 + 100*thick_panels +200.0*forces",
    "reserve_fact": "-3*sweep -6*thick_panels+0.1*forces+55",
    "displ": "2*sweep + 3*thick_panels-2.*forces",
}
structure = create_discipline(
    "AnalyticDiscipline", name="Structure", expressions=struc_formulas
)
mission_formulas = {"range": "8e11*lift/(mass*drag)"}
mission = create_discipline(
    "AnalyticDiscipline", name="Mission", expressions=mission_formulas
)

disciplines = [aerodynamics, structure, mission]

# %%
# We can see that structure and aerodynamics are strongly coupled:
generate_n2_plot(disciplines, save=False, show=True)

# %%
# Create an MDO scenario with MDF formulation
# -------------------------------------------
# Then, we create an MDO scenario based on the MDF formulation
design_space = AerostructureDesignSpace()
scenario = create_scenario(
    disciplines,
    "range",
    design_space,
    maximize_objective=True,
    formulation_name="MDF",
)
scenario.add_constraint("reserve_fact", constraint_type="ineq", value=0.5)
scenario.add_constraint("lift", value=0.5)
scenario.execute(slsqp_settings)
scenario.post_process(post_name="OptHistoryView", save=False, show=True)

# %%
# Create an MDO scenario with bilevel formulation
# -----------------------------------------------
# Then, we create an MDO scenario based on the bilevel formulation
design_space_ref = AerostructureDesignSpace()

# %%
# Create the aeronautics sub-scenario
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# For this purpose, we create a first sub-scenario to maximize the range
# with respect to the thick airfoils, based on the aerodynamics discipline.
aero_scenario = create_scenario(
    [aerodynamics, mission],
    "range",
    design_space_ref.filter(["thick_airfoils"], copy=True),
    maximize_objective=True,
    formulation_name="DisciplinaryOpt",
)
aero_scenario.set_algorithm(slsqp_settings)

# %%
# Create the structure sub-scenario
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
# We create a second sub-scenario to maximize the range
# with respect to the thick panels, based on the structure discipline.
struct_scenario = create_scenario(
    [structure, mission],
    "range",
    design_space_ref.filter(["thick_panels"], copy=True),
    maximize_objective=True,
    formulation_name="DisciplinaryOpt",
)
struct_scenario.set_algorithm(slsqp_settings)

# %%
# Create the system scenario
# ^^^^^^^^^^^^^^^^^^^^^^^^^^
# Lastly, we build a system scenario to maximize the range with respect to
# the sweep, which is a shared variable, based on the previous sub-scenarios.
design_space_system = design_space_ref.filter(["sweep"], copy=True)
system_scenario = create_scenario(
    [aero_scenario, struct_scenario, mission],
    "range",
    design_space_system,
    formulation_name="BiLevel",
    maximize_objective=True,
    main_mda_settings=MDAChain_Settings(inner_mda_name="MDAJacobi", tolerance=1e-8),
)
system_scenario.add_constraint("reserve_fact", constraint_type="ineq", value=0.5)
system_scenario.add_constraint("lift", value=0.5)
system_scenario.execute(cobyla_settings)

system_scenario.post_process(post_name="OptHistoryView", save=False, show=True)
