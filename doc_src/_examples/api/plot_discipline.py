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
"""
Discipline
==========
"""
from __future__ import annotations

from gemseo.api import configure_logger
from gemseo.api import create_discipline
from gemseo.api import generate_coupling_graph
from gemseo.api import generate_n2_plot
from gemseo.api import get_available_disciplines
from gemseo.api import get_discipline_inputs_schema
from gemseo.api import get_discipline_options_defaults
from gemseo.api import get_discipline_options_schema
from gemseo.api import get_discipline_outputs_schema
from gemseo.core.discipline import MDODiscipline
from gemseo.disciplines.utils import get_all_inputs
from gemseo.disciplines.utils import get_all_outputs
from numpy import array

configure_logger()


##########################################################################
# In this example, we will discover the different functions of the API
# related to disciplines, which are the |g|' objects
# dedicated to the representation of an input-output process. All classes
# implementing disciplines inherit from :class:`.MDODiscipline` which is an
# abstract class.
#
# Get available disciplines
# -------------------------
#
# The :meth:`~gemseo.api.get_available_disciplines` function
# can list the available disciplines:

get_available_disciplines()

##########################################################################
# Create a discipline
# -------------------
# The :meth:`~gemseo.api.create_discipline` function can create a
# :class:`.MDODiscipline` or a list of :class:`.MDODiscipline`
# by using its class name. Specific :code:`**options` can be provided in
# argument. E.g.
disciplines = create_discipline(
    [
        "SobieskiPropulsion",
        "SobieskiAerodynamics",
        "SobieskiMission",
        "SobieskiStructure",
    ]
)
print(type(disciplines))
print(type(disciplines[0]))
print(isinstance(disciplines[0], MDODiscipline))
##########################################################################
# This function can also be used to create a particular :class:`.MDODiscipline`
# from scratch, such as :class:`.AnalyticDiscipline`
# or :class:`.AutoPyDiscipline`. E.g.
addition = create_discipline("AnalyticDiscipline", expressions={"y": "x1+x2"})
print(addition.execute({"x1": array([1.0]), "x2": array([2.0])}))

##########################################################################
# Get all inputs/outputs
# ----------------------
# The :func:`~gemseo.disciplines.utils.get_all_inputs` function can list all the inputs
# of a list of disciplines, including the sub-disciplines if the
# argument :code:`recursive` (default: :code:`False`) is :code:`True`,
# merging the input data from the discipline grammars. E.g.
print(get_all_inputs(disciplines))

##########################################################################
# The :func:`~gemseo.disciplines.utils.get_all_outputs` function can list all the inputs
# of a list of disciplines, including the sub-disciplines if the
# argument :code:`recursive` (default: :code:`False`) is :code:`True`,
# merging the input data from the discipline grammars. E.g.
print(get_all_outputs(disciplines))

##########################################################################
# Get discipline schemas for inputs, outputs and options
# ------------------------------------------------------
#
# - The function :meth:`~gemseo.api.get_discipline_inputs_schema` returns
#   the inputs of a discipline. E.g.
print(get_discipline_inputs_schema(disciplines[0]))

##########################################################################
# - The function :meth:`~gemseo.api.get_discipline_outputs_schema` returns
#   the outputs of a discipline. E.g.
print(get_discipline_outputs_schema(disciplines[0]))

##########################################################################
# - The function :meth:`~gemseo.api.get_discipline_options_schema` returns
#   the options of a discipline. E.g.
print(get_discipline_options_schema("SobieskiMission"))

##########################################################################
# - The function :meth:`~gemseo.api.get_discipline_options_defaults`
#   can get the default option values of a discipline. E.g.
print(get_discipline_options_defaults("SobieskiMission"))

##########################################################################
# Plot coupling structure
# -----------------------
# The :meth:`~gemseo.api.generate_coupling_graph` function plots the
# coupling graph of a set of :class:`.MDODiscipline`:
generate_coupling_graph(disciplines, file_path="full_coupling_graph.pdf")
generate_coupling_graph(
    disciplines, file_path="condensed_coupling_graph.pdf", full=False
)

##########################################################################
# The :meth:`~gemseo.api.generate_n2_plot` function plots the N2 diagram of
# a set of :class:`.MDODiscipline`:
generate_n2_plot(disciplines, save=False, show=True)
