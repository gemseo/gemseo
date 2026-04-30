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
"""# Tutorial - Use namespaces to run the same discipline in multiple contexts

## Goal

In this tutorial, you will learn to model and simulate an ensemble of similar objects
using the same discipline definition — without duplicating any code.

The motivating scenario is to design a launcher
that must carry several satellites into different orbits.
Each satellite has the same mass model (structure mass + propellant mass),
but different propellant budgets depending on its target orbit.
The goal is to compute the total payload mass the launcher has to deliver in orbit.

By the end of this tutorial, you will know how to:

- understand that GEMSEO forbids multiple disciplines with the same output name,
- use **namespaces** to differentiate instances of the same discipline,
- share inputs across instances automatically,
- aggregate namespaced outputs into a single result,
- assemble the full workflow and execute it.
"""

from __future__ import annotations

from numpy import array

from gemseo import generate_n2_plot
from gemseo.disciplines.analytic import AnalyticDiscipline
from gemseo.disciplines.linear_combination import LinearCombination
from gemseo.mda.chain import MDAChain
from gemseo.utils.discipline import check_disciplines_consistency

# %%
# ## Step 1 - Create the satellite mass discipline
#
# A satellite mass is the sum of its structural (bus) mass and its propellant mass.
# This is our base discipline: one input per mass component, one output.
satellite = AnalyticDiscipline(
    {"mass": "structure_mass + propellant_mass"}, name="Satellite mass"
)

# %%
# Let's verify it works on a single satellite:
satellite.execute({
    "structure_mass": array([500.0]),
    "propellant_mass": array([120.0]),
})
print(f"Single satellite mass: {satellite.io.data['mass']} kg")

# %%
# ## Step 2 - GEMSEO forbids duplicate outputs
#
# Suppose we naively instantiate three copies of this discipline,
# one per satellite, and try to use them together in a workflow.
satellites_no_ns = [
    AnalyticDiscipline({"mass": "structure_mass + propellant_mass"}),
    AnalyticDiscipline({"mass": "structure_mass + propellant_mass"}),
    AnalyticDiscipline({"mass": "structure_mass + propellant_mass"}),
]

# %%
# GEMSEO enforces that each output is produced by exactly one discipline.
# Let's verify this rule:
try:
    check_disciplines_consistency(satellites_no_ns, log_message=False, raise_error=True)
except ValueError as err:
    print(err)

# %%
# We cannot build a valid workflow with three disciplines all producing `"mass"`.
# We need a way to disambiguate each instance.
#
# ## Step 3 - Add namespaces to differentiate the instances
#
# A namespace is a prefix added to selected input or output variable names.
# The separator between namespace and name is `":"` by default,
# so adding namespace `"sat1"` to `"mass"` gives `"sat1:mass"`.
#
# For each satellite, we namespace:
#
# - the **input** `propellant_mass`: each satellite has its own propellant budget,
# - the **output** `mass`: each satellite produces a distinct mass value.
#
# We leave `structure_mass` without a namespace.
# This means all satellite instances share the same structural mass —
# they all use the same satellite bus design.
n_satellites = 3
satellites = []
for i in range(n_satellites):
    discipline = AnalyticDiscipline(
        {"mass": "structure_mass + propellant_mass"}, name=f"Satellite{i + 1}"
    )

    ns = f"sat{i + 1}"
    discipline.add_namespace_to_input("propellant_mass", ns)
    discipline.add_namespace_to_output("mass", ns)

    satellites.append(discipline)


# %%
# ## Step 4 - Inspect the grammar
#
# Let's confirm that the namespaces have been applied correctly.
for disc in satellites:
    print(
        f"{disc.name}:  "
        f"inputs={list(disc.input_grammar.names)}  "
        f"outputs={list(disc.output_grammar.names)}"
    )

# %%
# As expected:
#
# - `structure_mass` appears as-is in every discipline: it is shared.
# - `sat1:propellant_mass`, `sat2:propellant_mass`, `sat3:propellant_mass`
#   are distinct, so each satellite can be given its own propellant value.
# - `sat1:mass`, `sat2:mass`, `sat3:mass` are distinct outputs —
#   GEMSEO will now accept all three disciplines in the same workflow.
#
# ## Step 5 - Aggregate the masses
#
# We need one more discipline that sums all satellite masses
# into a single `total_mass` output.
# The built-in
# [LinearCombination][gemseo.disciplines.linear_combination.LinearCombination]
# discipline is ideal for this.
satellite_mass_variable_names = ["sat1:mass", "sat2:mass", "sat3:mass"]

aggregator = LinearCombination(
    satellite_mass_variable_names,
    "total_mass",
    input_coefficients=dict.fromkeys(satellite_mass_variable_names, 1.0),
)

# %%
# ## Step 6 - Assemble the workflow and visualize it
#
# All disciplines are now consistent: no output is produced twice.
# We can visualize the data flow with an N2 diagram.
# The shared input `structure_mass` will not appear,
# but each `sat_i:mass` flows only from its discipline to the aggregator.
all_disciplines = [*satellites, aggregator]
generate_n2_plot(all_disciplines, save=False, show=True)

# %%
# ## Step 7 - Execute
#
# We assemble the disciplines into an
# [MDAChain][gemseo.mda.chain.MDAChain]
# and execute with realistic values.
# The three satellites target different orbits, hence different propellant budgets.
chain = MDAChain(all_disciplines)
chain.execute({
    "structure_mass": array([500.0]),  # kg — shared satellite bus
    "sat1:propellant_mass": array([120.0]),  # kg — first orbit slot
    "sat2:propellant_mass": array([200.0]),  # kg — second orbit slot
    "sat3:propellant_mass": array([80.0]),  # kg — third orbit slot
})
print(f"Total payload mass: {chain.io.data['total_mass']} kg")
# (500+120) + (500+200) + (500+80) = 1900 kg

# %%
# ## Key takeaways
#
# - GEMSEO requires every output to be produced by exactly one discipline.
#   Naively duplicating a discipline breaks this rule.
# - **Namespaces** solve this by prefixing selected variable names:
#   [add_namespace_to_input()][gemseo.core.discipline.base_discipline.BaseDiscipline.add_namespace_to_input]
#   and
#   [add_namespace_to_output()][gemseo.core.discipline.base_discipline.BaseDiscipline.add_namespace_to_output].
# - The discipline code itself never changes.
#   Namespaces are applied after instantiation,
#   keeping the model reusable in any number of contexts.
