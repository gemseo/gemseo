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
from __future__ import annotations

from gemseo import create_discipline
from gemseo import create_scenario
from gemseo import get_available_formulations
from gemseo import get_available_scenario_types
from gemseo.problems.mdo.sellar.sellar_design_space import SellarDesignSpace

get_available_scenario_types()

###

disciplines = create_discipline(["Sellar1", "Sellar2", "SellarSystem"])

###

design_space = SellarDesignSpace()

###

objective_name = "obj"
formulation = "MDF"

###

get_available_formulations()

###

scenario_type = "MDO"

###

scenario = create_scenario(
    disciplines,
    objective_name,
    design_space,
    formulation_name=formulation,
    scenario_type=scenario_type,
)

###

print(scenario.get_optim_variable_names())
print(scenario.design_space)
scenario.xdsmize(monitor=True, log_workflow_status=True)

###

scenario.execute(algo_name="SLSQP", max_iter=100)

###

opt_results = scenario.optimization_result
print(f"The solution of P is (x*,f(x*)) = ({opt_results.x_opt}, {opt_results.f_opt})")
scenario.print_execution_metrics()

scenario.log_me()
