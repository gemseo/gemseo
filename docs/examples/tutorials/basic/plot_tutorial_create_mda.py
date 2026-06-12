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
#    INITIAL AUTHORS - initial API and implementation and/or initial
#                           documentation
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""# Tutorial - The Multidisciplinary Design Analysis

## Goal

This tutorial will guide you to create and execute a Multidisciplinary Design
Analysis (MDA).

In cases where a set of disciplines is strongly coupled,
one must solve the underlying coupled system to get consistent variables values.
Even in the absence of strong coupling,
an MDA is relevant
because it allows disciplines
to be executed in the correct order.
GEMSEO offers different algorithms dedicated to this problem.

You will first create Sobieski coupled disciplines and
learn how to visualize their couplings.
Then, you will create and execute a Gauss-Seidel MDA,
and display the decrease of the coupling residuals.
"""

from __future__ import annotations

from gemseo import create_discipline
from gemseo import create_mda
from gemseo import generate_coupling_graph
from gemseo import generate_n2_plot
from gemseo.settings.mda import MDAGaussSeidel_Settings

# %%
# ## Step 1 - Create Sobieski disciplines
#
# Sobieski disciplines are already implemented in GEMSEO so to be used as examples.
# Therefore, we just need to use the
# [create_discipline][gemseo.create_discipline] high level API function.
#
# !!! warning
#     Any [Discipline][gemseo.core.discipline.discipline.Discipline] provided to
#     a [BaseMDA][gemseo.mda.base.BaseMDA]
#     with strong couplings **must** define its
#     [default_input_data][gemseo.core.discipline.discipline.Discipline.default_input_data].
#     Otherwise, the execution will fail.

disciplines = create_discipline([
    "SobieskiStructure",
    "SobieskiPropulsion",
    "SobieskiAerodynamics",
    "SobieskiMission",
])

# %%
# ## Step 2 - Visualize the couplings
#
# GEMSEO offers different ways to visualize the couplings between different disciplines.
#
# ### The coupling graph
#
# In the coupling graph, disciplines are shown as nodes
# and the couplings are represented as edges.
# This Directed Graph helps to see whether the execution of disciplines is sequential
# or need complex executions to solve strong couplings.
#
# !!! note
#     By setting file_path to an empty string, the generated file is not written to
#     disk.
generate_coupling_graph(disciplines, file_path="")

# %%
# A quick look shows that the *Propulsion*, *Structure*, and *Aerodynamics* disciplines
# are strongly coupled, while the *Mission* discipline can be run afterward.
# This information may be hard to get when considering many disciplines.
# This is why you can generate the condensed coupling graph (which is a directed
# acyclic graph (DAG)),
# that shows the disciplines that are highly coupled.
generate_coupling_graph(disciplines, file_path="", full=False)

# %%
# ### The N2 diagram
#
# The coupling graph may become hard to analyse, even when condensed.
# The N2 diagram comes as a standard to represent the disciplines and their couplings.
# GEMSEO simply generates a N2 diagram from a list of disciplines.
#
# !!! note
#     To understand everything on the N2 diagram, please refer to
#     [N2 chart visualization][concept-n2-chart].
generate_n2_plot(disciplines, save=False, show=True)

# %%
# Step 2 - Create an MDA
#
# GEMSEO offers different methods to solve higly coupled disciplines.
# They have different properties,
# such as allowing the execution of the disciplines in a multi-process environment,
# or to focus on sequential executions.
# In this tutorial, the focus is given to the Gauss-Seidel algorithm.
# It executes the disciplines sequentially, and the solution given at every iteration
# is physically feasible.
# It means that even if the algorithm does not converge,
# the solution of your problem has a physical meaning.
#
# !!! note
#     You are advised to start with Gauss-Seidel MDAs when you are starting a new study.
#
# !!! note
#     Different options can be passed to the algorithm by using the dedicated settings.
gauss_seidel_mda = create_mda(
    "MDAGaussSeidel",
    disciplines,
    settings_model=MDAGaussSeidel_Settings(max_mda_iter=15),
)

# %%
# !!! how-to
#     Here, we create an MDA with the 4 disciplines,
#     even if the *Mission* is not highly coupled with the other 3 disciplines.
#     It is sub-optimal, since the *Mission* discipline will be executed many times.
#     You may want to create
#     a [DisciplineChain][gemseo.core.chains.chain.DisciplineChain]
#     of the MDA (3 disciplines) and the *Mission* discipline, as explained in the
#     [Chain disciplines][chain-disciplines] how-to.
#     Or you can simply explore the
#     [Instantiate an MDAChain][manage-nested-coupling-systems] how-to.
#     to let GEMSEO determine the best workflow.
#
# ## Step 3 - Execution of the Gauss Seidel MDA
#
# In GEMSEO, an MDA is a
# [ProcessDiscipline][gemseo.core.process_discipline.ProcessDiscipline].
# Thus, it can be used just like a discipline.
output_data = gauss_seidel_mda.execute()
output_data
# %%
# ## Step 4 - Residuals
#
# The convergence of MDAs is computed thanks to residuals,
# as explained in [MDA Stopping criteria][concept-mda-stopping-criteria].
#
# The MDA algorithm will stop if one of the following criteria is fulfilled:
#
#     - The normalized residual norm is lower than a threshold.
#       In that case, the MDA has converged and the design is considered as feasible.
#     - The maximal number of iterations is reached.
#       In that case, the MDA has not converged and
#       the design is considered as not-feasible.
#       Increasing the number of iterations may solve the problem.
#       However, the MDA might never converge.
#
# The normed residual can be seen by
# [normalized_residual_norm][gemseo.mda.base.BaseMDA.normalized_residual_norm]
# attribute.
gauss_seidel_mda.normalized_residual_norm

# %%
# The evolution of its value can be visualized with the
# [plot_residual_history()][gemseo.mda.base.BaseMDA.plot_residual_history] method.
gauss_seidel_mda.plot_residual_history(logscale=[1e-8, 10.0], save=False, show=True)
# %%
# In our case, the MDA converged in 8 iterations:
# the disciplines have been executed 8 times to make the couplings converge.
# The given solution is a feasible and converged point.
#
# The values can be accessed
# by [residual_history][gemseo.mda.base.BaseMDA.residual_history].
gauss_seidel_mda.residual_history

# %%
# ## Step 5 - Additional output
#
# Like the discipline, the MDA has inputs and outputs.
# They are determined from the discipline inputs and outputs.
# You may have noticed that an additional output is set for the MDA,
# which is not in the disciplines outputs.
name = gauss_seidel_mda.NORMALIZED_RESIDUAL_NORM
name, output_data[name]

# %%
# This normalized residual norm can be used as a constraint in an MDO scenario,
# or can simply be retrieved as a scenario observable.
#
# ## Key takeaways
#
# In this tutorial, you've learned to:
#
# - generate graphes to show the couplings, either with
# [generate_coupling_graph()][gemseo.generate_coupling_graph] or
# [generate_n2_plot()][gemseo.generate_n2_plot];
# - create an MDA from highly coupled discipline by means of the
# [create_mda()][gemseo.create_mda] high-level function;
# - execute the MDA just like you would execute a discipline;
# - visualize the algorithm convergence thourgh the residuals.
# You can plot the residual history with the
# [plot_residual_history()][gemseo.mda.base.BaseMDA.plot_residual_history] method;
# - access the specific MDA output.
#
# ## How-to guides
#
# For further information,
# please refer to the following how-to guides:
#
# - [Instantiate an MDAChain][manage-nested-coupling-systems]
# - [MDA acceleration techniques][accelerate-mda-convergence]
# - [Create sequential hybrid MDAs][create-sequential-hybrid-mdas]
