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
Scalable diagonal discipline
============================

Let us consider the
:class:`~gemseo.problems.sobieski.disciplines.SobieskiAerodynamics` discipline.
We want to build its :class:`.ScalableDiscipline` counterpart,
using a :class:`.ScalableDiagonalModel`

For that, we can use a 20-length :class:`.DiagonalDOE`
and test different sizes of variables or different settings
for the scalable diagonal discipline.
"""
from __future__ import annotations

from gemseo.api import configure_logger
from gemseo.api import create_discipline
from gemseo.api import create_scalable
from gemseo.api import create_scenario
from gemseo.problems.sobieski.core.problem import SobieskiProblem

###############################################################################
# Import
# ------

configure_logger()


###############################################################################
# Learning dataset
# ----------------
# The first step is to build an :class:`.AbstractFullCache` dataset
# from a :class:`.DiagonalDOE`.

###############################################################################
# Instantiate the discipline
# ~~~~~~~~~~~~~~~~~~~~~~~~~~
# For that, we instantiate the
# :class:`~gemseo.problems.sobieski.disciplines.SobieskiAerodynamics` discipline
# and set it up to cache all evaluations.
discipline = create_discipline("SobieskiAerodynamics")

###############################################################################
# Get the input space
# ~~~~~~~~~~~~~~~~~~~
# We also define the input space on which to sample the discipline.
input_space = SobieskiProblem().design_space
input_names = [name for name in discipline.get_input_data_names() if name != "c_4"]
input_space.filter(input_names)

###############################################################################
# Build the DOE scenario
# ~~~~~~~~~~~~~~~~~~~~~~
# Lastly, we sample the discipline by means of a :class:`.DOEScenario`
# relying on both discipline and input space.
# In order to build a diagonal scalable discipline,
# a :class:`.DiagonalDOE` must be used.
scenario = create_scenario(
    [discipline], "DisciplinaryOpt", "y_2", input_space, scenario_type="DOE"
)
for output_name in discipline.get_output_data_names():
    if output_name != "y_2":
        scenario.add_observable(output_name)
scenario.execute({"algo": "DiagonalDOE", "n_samples": 20})

###############################################################################
# Scalable diagonal discipline
# ----------------------------

###############################################################################
# Build the scalable discipline
# -----------------------------
# The second step is to build a :class:`.ScalableDiscipline`,
# using a :class:`.ScalableDiagonalModel` and the database
# converted to a :class:`.Dataset`.
dataset = scenario.export_to_dataset(opt_naming=False)
scalable = create_scalable("ScalableDiagonalModel", dataset)

###############################################################################
# Visualize the input-output dependencies
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# We can easily access the underlying :class:`.ScalableDiagonalModel`
# and plot the corresponding input-output dependency matrix
# where the level of gray and the number (in [0,100]) represent
# the degree of dependency between inputs and outputs.
# Input are on the left while outputs are at the top.
# More precisely, for a given output component located at the top of the graph,
# these degrees are contributions to the output component and they add up to 1.
# In other words, a degree expresses this contribution in percentage
# and for a given column, the elements add up to 100.
scalable.scalable_model.plot_dependency(save=False, show=True)

###############################################################################
# Visualize the 1D interpolations
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# For every output, we can also visualize a spline interpolation of the output
# samples over the diagonal of the input space.
scalable.scalable_model.plot_1d_interpolations(save=False, show=True)

###############################################################################
# Increased problem dimension
# ---------------------------
# We can repeat the construction of the scalable discipline for different sizes
# of variables and visualize the input-output dependency matrices.

###############################################################################
# Twice as many inputs
# ~~~~~~~~~~~~~~~~~~~~
# For example, we can increase the size of each input by a factor of 2.
sizes = {name: dataset.sizes[name] * 2 for name in input_names}
scalable = create_scalable("ScalableDiagonalModel", dataset, sizes)
scalable.scalable_model.plot_dependency(save=False, show=True)

###############################################################################
# Twice as many outputs
# ~~~~~~~~~~~~~~~~~~~~~
# Or we can increase the size of each output by a factor of 2.
sizes = {
    name: discipline.cache.names_to_sizes[name] * 2
    for name in discipline.get_output_data_names()
}
scalable = create_scalable("ScalableDiagonalModel", dataset, sizes)
scalable.scalable_model.plot_dependency(save=False, show=True)

###############################################################################
# Twice as many variables
# ~~~~~~~~~~~~~~~~~~~~~~~
# Or we can increase the size of each input and each output by a factor of 2.
names = input_names + list(discipline.get_output_data_names())
sizes = {name: dataset.sizes[name] * 2 for name in names}
scalable = create_scalable("ScalableDiagonalModel", dataset, sizes)
scalable.scalable_model.plot_dependency(save=False, show=True)

###############################################################################
# Binary IO dependencies
# ----------------------
# By default, any output component depends on any input component
# with a random level.
# We can also consider sparser input-output dependency
# by means of binary input-output dependency matrices.
# For that, we have to set the value of the fill factor
# which represents the part of connection between inputs and outputs.
# Then, a connection is represented by a black square
# while an absence of connection is presented by a white one.
# When the fill factor is equal to 1, any input is connected to any output.
# Conversely, when the fill factor is equal to 0,
# there is not a single connection between inputs and outputs.

###############################################################################
# Fill factor = 0.2
# ~~~~~~~~~~~~~~~~~
scalable = create_scalable("ScalableDiagonalModel", dataset, sizes, fill_factor=0.2)
scalable.scalable_model.plot_dependency(save=False, show=True)

###############################################################################
# Fill factor = 0.5
# ~~~~~~~~~~~~~~~~~
scalable = create_scalable("ScalableDiagonalModel", dataset, sizes, fill_factor=0.5)
scalable.scalable_model.plot_dependency(save=False, show=True)

###############################################################################
# Fill factor = 0.8
# ~~~~~~~~~~~~~~~~~
scalable = create_scalable("ScalableDiagonalModel", dataset, sizes, fill_factor=0.8)
scalable.scalable_model.plot_dependency(save=False, show=True)

###############################################################################
# Heterogeneous dependencies
# --------------------------
scalable = create_scalable(
    "ScalableDiagonalModel", dataset, sizes, fill_factor={"y_2": 0.2}
)
scalable.scalable_model.plot_dependency(save=False, show=True)

###############################################################################
# Group dependencies
# ------------------
scalable = create_scalable(
    "ScalableDiagonalModel", dataset, sizes, group_dep={"y_2": ["x_shared"]}
)
scalable.scalable_model.plot_dependency(save=False, show=True)
