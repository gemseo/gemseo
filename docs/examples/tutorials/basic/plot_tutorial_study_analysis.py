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
"""# Tutorial - Prototyping a multidisciplinary study

## Goal

In this tutorial,
you will learn how to design and visualize a multidiscipinary process
using N2 and XDSM diagrams,
**without implementing any actual computation**.

This approach is useful at the start of a project,
when you want to explore coupling structures and MDO formulations
before writing solver code.

GEMSEO provides two complementary paths to do this:

1. [DummyDiscipline][gemseo.utils.discipline.DummyDiscipline] —
   define disciplines that do nothing, from input and output names.
2. Excel-based study analysis —
   describe disciplines and scenarios in a spreadsheet,
   then generate diagrams with
   [CouplingStudyAnalysis][gemseo.utils.study_analyses.coupling_study_analysis.CouplingStudyAnalysis]
   or
   [MDOStudyAnalysis][gemseo.utils.study_analyses.mdo_study_analysis.MDOStudyAnalysis].

!!! tip "Web interface for non-programmers"
    If you prefer not to write Python or Excel files,
    you can use the interactive web application
    [GEMSEO Web Study](https://gemseo-web-study.streamlit.app/)
    to describe your disciplines graphically and generate N2 and XDSM diagrams
    directly in your browser.
"""

from __future__ import annotations

from gemseo import generate_n2_plot
from gemseo.algos.design_space import DesignSpace
from gemseo.scenarios.evaluation import EvaluationScenario
from gemseo.utils.discipline import DummyDiscipline
from gemseo.utils.study_analyses.coupling_study_analysis import CouplingStudyAnalysis
from gemseo.utils.study_analyses.mdo_study_analysis import MDOStudyAnalysis

# %%
# ## Step 1 — Define disciplines with DummyDiscipline
#
# A [DummyDiscipline][gemseo.utils.discipline.DummyDiscipline]
# is a placeholder discipline that declares input and output variable names
# but performs no computation.
# It is ideal for sketching MDO architectures early in a project,
# before any solver code exists.
#
# Create two disciplines by specifying their names
# and lists of input and output variable names:

d1 = DummyDiscipline("Discipline1", ["a", "b"], ["d", "e", "f"])
d2 = DummyDiscipline("Discipline2", ["d", "x", "z"], ["a", "b", "c"])

# %%
# Notice that `d2` produces `a`, `b`, `c`,
# and that the two first outputs are consumed by `d1`.
# On the other side,
# `d1` produces `d`, `e`, `f`, and `d2` needs `d` as input.
# These mutual dependencies define a **strong coupling**
# between the two disciplines.

# %%
# ## Step 2 — Visualize the coupling with an N2 chart
#
# An **N2 chart**, a.k.a. design structure matrix (DSM),
# shows which outputs of each discipline feed into the inputs of the others.
# Off-diagonal entries indicate coupling variables,
# while a circular pattern in the matrix reveals feedback loops
# that will require an MDA (Multi-Disciplinary Analysis) to converge.
# See [N2 chart][concept-n2-chart] for more details.
#
# Generate it from the list of disciplines:

generate_n2_plot([d1, d2], save=False, show=True)

# %%
# The chart reveals the loop:
# `d2` → `d1` → `d2` (linked by variables `a`, `b`, and `d`),
# confirming that an MDA will be needed to converge the coupling.
#
# !!! how-to
#     The N2 chart is also available as an interactive HTML file.
#     See [Generate the N2 chart][generate-the-n2-chart] for details.
#
# ## Step 3 — Visualize the MDO process with an XDSM
#
# An **XDSM** (eXtended Design Structure Matrix)
# provides a richer picture of the full MDO process.
# For an MDF formulation, it shows the optimizer loop,
# the MDA convergence loop, and the discipline execution order
# all in a single diagram.
#
# To generate an XDSM you first need a scenario:
# create a minimal [DesignSpace][gemseo.algos.design_space.DesignSpace]
# and an [EvaluationScenario][gemseo.scenarios.evaluation.EvaluationScenario].

design_space = DesignSpace()
design_space.add_variable("x")
design_space.add_variable("z")

scenario = EvaluationScenario(
    [d1, d2],
    design_space,
)
scenario.add_observable("c")
scenario.add_observable("f")
# %%
# Now generate the XDSM diagram.
# The resulting `xdsm.html` file can be opened in any browser:

scenario.xdsmize(show_html=True, save_html=False)

# %%
# ## Step 4 — Excel-based coupling study analysis
#
# Instead of writing Python, you can describe disciplines in an Excel file.
# Each sheet corresponds to one discipline (sheet name = discipline name)
# and must contain two columns: **Inputs** and **Outputs**.
#
# ```
# | Inputs |    | Outputs |
# |--------| -- |---------|
# | a      |    | d       |
# | b      |    | e       |
# |        |    | f       |
# ```
#
# This spreadsheet format is convenient for early-stage design reviews,
# letting domain experts contribute without writing any code.
#
# Pass the path to the Excel file to
# [CouplingStudyAnalysis][gemseo.utils.study_analyses.coupling_study_analysis.CouplingStudyAnalysis]:

coupling_study = CouplingStudyAnalysis("coupling_study.xlsx")
coupling_study.generate_n2(save=False, show=True)

# %%
# ## Step 5 — Excel-based MDO study analysis
#
# To also generate an XDSM from Excel,
# add a sheet named **Scenario** to your workbook.
# This sheet defines the optimization problem:
#
# | Design variables | Objective function | Constraints | Disciplines              | Formulation | Options | Options values |
# |------------------|--------------------|-------------|--------------------------|-------------|---------|----------------|
# | x                | f                  | g           | Discipline1, Discipline2 | MDF         |         |                |
#
# Pass this file to
# [MDOStudyAnalysis][gemseo.utils.study_analyses.mdo_study_analysis.MDOStudyAnalysis]:
mdo_study = MDOStudyAnalysis("mdo_study.xlsx")
mdo_study.generate_n2(save=False, show=True)
mdo_study.generate_xdsm(".")

# %%
# ## Step 6 — Command-line alternative
#
# The generation of the N2 chart and the XDSM is also available
# from the command line via the `gemseo-study` executable,
# using the same Excel files.
#
# ```bash
# # Coupling study — N2 only
# gemseo-study coupling_study.xlsx -t coupling -o outputs --height 5 --width 5
#
# # MDO study — N2 and XDSM
# gemseo-study mdo_study.xlsx -o outputs -h 5 -w 5 -x -p
# ```
#
# Key options:
#
# - `-t coupling` selects the coupling study type (default is `mdo`)
# - `-o outputs` sets the output directory
# - `-h` / `-w` set the N2 chart height and width in inches
# - `-x` generates an XDSM (MDO studies only)
# - `-p` also generates a PDF of the XDSM
#
# ## Key takeaways
#
# - A [DummyDiscipline][gemseo.utils.discipline.DummyDiscipline]
#   lets you sketch a multidiscipinary architecture
#   by declaring only input and output names, with no solver code required.
# - The **N2 chart** reveals coupling variables and feedback loops
#   between disciplines at a glance.
# - The **XDSM** diagram maps the full MDO process —
#   optimizer loop, MDA loop, and execution order —
#   for a chosen formulation such as MDF.
# - Excel-based study analyses
#   ([CouplingStudyAnalysis][gemseo.utils.study_analyses.coupling_study_analysis.CouplingStudyAnalysis],
#   [MDOStudyAnalysis][gemseo.utils.study_analyses.mdo_study_analysis.MDOStudyAnalysis])
#   let domain experts contribute to architecture design
#   without writing Python.
# - The `gemseo-study` command-line tool provides a scriptable interface
#   to generate N2 and XDSM diagrams from Excel files.
# - A web application is available to sketch MDO architectures:
#   [GEMSEO Web Study](https://gemseo-web-study.streamlit.app/)
#
# ## How-to guides
#
# For further information,
# please refer to the following how-to guides:
#
# - [Generate the N2 chart][generate-the-n2-chart],
# - [Generate an XDSM chart][generate-an-xdsm-chart],
