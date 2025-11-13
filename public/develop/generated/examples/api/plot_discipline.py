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
#        :author: Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""# Discipline."""

from __future__ import annotations

from numpy import array

from gemseo import create_discipline
from gemseo import generate_coupling_graph
from gemseo import generate_n2_plot
from gemseo import get_available_disciplines
from gemseo import get_discipline_inputs_schema
from gemseo import get_discipline_options_defaults
from gemseo import get_discipline_options_schema
from gemseo import get_discipline_outputs_schema
from gemseo.core.discipline import Discipline
from gemseo.utils.discipline import get_all_inputs
from gemseo.utils.discipline import get_all_outputs

# %%
# In this example, we will discover the different functions of the API
# related to disciplines, which are the GEMSEO' objects
# dedicated to the representation of an input-output process. All classes
# implementing disciplines inherit from [Discipline][gemseo.core.discipline.discipline.Discipline] which is an
# abstract class.
#
# ## Get available disciplines
#
#
# The [get_available_disciplines()][gemseo.get_available_disciplines] function
# can list the available disciplines:

get_available_disciplines()

# %%
# ## Create a discipline
#
# The [create_discipline()][gemseo.create_discipline] function can create a
# [Discipline][gemseo.core.discipline.discipline.Discipline] or a list of [Discipline][gemseo.core.discipline.discipline.Discipline]
# by using its class name. Specific `**options` can be provided in
# argument. E.g.
disciplines = create_discipline([
    "SobieskiPropulsion",
    "SobieskiAerodynamics",
    "SobieskiMission",
    "SobieskiStructure",
])
type(disciplines), type(disciplines[0]), isinstance(disciplines[0], Discipline)
# %%
# This function can also be used to create a particular [Discipline][gemseo.core.discipline.discipline.Discipline]
# from scratch, such as [AnalyticDiscipline][gemseo.disciplines.analytic.AnalyticDiscipline]
# or [AutoPyDiscipline][gemseo.disciplines.auto_py.AutoPyDiscipline]. E.g.
addition = create_discipline("AnalyticDiscipline", expressions={"y": "x1+x2"})
addition.execute({"x1": array([1.0]), "x2": array([2.0])})

# %%
# ## Get all inputs/outputs
#
# The [get_all_inputs()][gemseo.utils.discipline.get_all_inputs] function can list all the inputs
# of a list of disciplines, including the sub-disciplines if the
# argument `recursive` (default: `False`) is `True`,
# merging the input data from the discipline grammars. E.g.
get_all_inputs(disciplines)

# %%
# The [get_all_outputs()][gemseo.utils.discipline.get_all_outputs] function can list all the inputs
# of a list of disciplines, including the sub-disciplines if the
# argument `recursive` (default: `False`) is `True`,
# merging the input data from the discipline grammars. E.g.
get_all_outputs(disciplines)

# %%
# ## Get discipline schemas for inputs, outputs and options
#
#
# - The function [get_discipline_inputs_schema()][gemseo.get_discipline_inputs_schema] returns
#   the inputs of a discipline. E.g.
get_discipline_inputs_schema(disciplines[0])

# %%
# - The function [get_discipline_outputs_schema()][gemseo.get_discipline_outputs_schema] returns
#   the outputs of a discipline. E.g.
get_discipline_outputs_schema(disciplines[0])

# %%
# - The function [get_discipline_options_schema()][gemseo.get_discipline_options_schema] returns
#   the options of a discipline. E.g.
get_discipline_options_schema("SobieskiMission")

# %%
# - The function [get_discipline_options_defaults()][gemseo.get_discipline_options_defaults]
#   can get the default option values of a discipline. E.g.
get_discipline_options_defaults("SobieskiMission")

# %%
# ## Plot coupling structure
#
# The [generate_coupling_graph()][gemseo.generate_coupling_graph] function plots the
# coupling graph of a set of [Discipline][gemseo.core.discipline.discipline.Discipline]:
generate_coupling_graph(disciplines, file_path="full_coupling_graph.pdf")
generate_coupling_graph(
    disciplines, file_path="condensed_coupling_graph.pdf", full=False
)

# %%
# The [generate_n2_plot()][gemseo.generate_n2_plot] function plots the N2 diagram of
# a set of [Discipline][gemseo.core.discipline.discipline.Discipline]:
generate_n2_plot(disciplines, save=False, show=True)
