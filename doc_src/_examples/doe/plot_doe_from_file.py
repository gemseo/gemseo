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
#        :author: Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""
Use a design of experiments from a file
=======================================
"""
from __future__ import annotations

from gemseo.api import create_design_space
from gemseo.api import create_discipline
from gemseo.api import create_scenario

#######################################################################################
# Let us consider a discipline implementing the function :math:`y=a*b`
discipline = create_discipline("AnalyticDiscipline", expressions={"y": "a*b"})

#######################################################################################
# where :math:`a,b\in\{1,2,\ldots,10\}`:
design_space = create_design_space()
design_space.add_variable("a", 1, design_space.INTEGER, 1, 10)
design_space.add_variable("b", 1, design_space.INTEGER, 1, 10)

#######################################################################################
# We want to evaluate this discipline over this design space
# by using the input samples defined in the file "doe.txt":
f = open("doe.txt")
print(f.read())

#######################################################################################
# In this file,
# rows are points and columns are variables
# whose order must be consistent with that of the design space.
# In this example,
# we can see that the first input value is defined by :math:`a=1` and :math:`b=2`.

#######################################################################################
# For that, we can create a scenario and execute it with a :class:`.CustomDOE`,
# with the option "doe_file".
# We could also change the delimiter (default: ',')  or skip the first rows in the file.
scenario = create_scenario(
    [discipline], "DisciplinaryOpt", "y", design_space, scenario_type="DOE"
)
scenario.execute({"algo": "CustomDOE", "algo_options": {"doe_file": "doe.txt"}})

#######################################################################################
# We can display the content of the database as a dataframe
# and check the values of the output,
# which should be the product of :math:`a` and :math:`b`.
opt_problem = scenario.formulation.opt_problem
dataset = opt_problem.export_to_dataset(name="custom_sampling", opt_naming=False)
print(dataset.export_to_dataframe())
