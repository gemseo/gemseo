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
Self-Organizing Map
===================

In this example, we illustrate the use of the :class:`.SOM` plot
on the Sobieski's SSBJ problem.
"""

from __future__ import annotations

from gemseo import configure_logger
from gemseo import create_discipline
from gemseo import create_scenario
from gemseo.problems.mdo.sobieski.core.design_space import SobieskiDesignSpace

# %%
# Import
# ------
# The first step is to import some high-level functions
# and a method to get the design space.

configure_logger()

# %%
# Description
# -----------
#
# The :class:`.SOM` post-processing performs a Self Organizing Map
# clustering on the optimization history.
# A :class:`.SOM` is a 2D representation of a design of experiments
# which requires dimensionality reduction since it may be in a very high dimension.
#
# A :term:`SOM` is built by using an unsupervised artificial neural network
# :cite:`Kohonen:2001`.
# A map of size ``n_x.n_y`` is generated, where
# ``n_x`` is the number of neurons in the :math:`x` direction and ``n_y``
# is the number of neurons in the :math:`y` direction. The design space
# (whatever the dimension) is reduced to a 2D representation based on
# ``n_x.n_y`` neurons. Samples are clustered to a neuron when their design
# variables are close in terms of their L2 norm. A neuron is always located at the
# same place on a map. Each neuron is colored according to the average value for
# a given criterion. This helps to qualitatively analyze whether parts of the design
# space are good according to some criteria and not for others, and where
# compromises should be made. A white neuron has no sample associated with
# it: not enough evaluations were provided to train the SOM.
#
# SOM's provide a qualitative view of the :term:`objective function`, the
# :term:`constraints`, and of their relative behaviors.

# %%
# Create disciplines
# ------------------
# At this point, we instantiate the disciplines of Sobieski's SSBJ problem:
# Propulsion, Aerodynamics, Structure and Mission
disciplines = create_discipline([
    "SobieskiPropulsion",
    "SobieskiAerodynamics",
    "SobieskiStructure",
    "SobieskiMission",
])

# %%
# Create design space
# -------------------
# We also create the :class:`.SobieskiDesignSpace`.
design_space = SobieskiDesignSpace()

# %%
# Create and execute scenario
# ---------------------------
# The next step is to build an MDO scenario in order to maximize the range,
# encoded 'y_4', with respect to the design parameters, while satisfying the
# inequality constraints 'g_1', 'g_2' and 'g_3'. We can use the MDF formulation,
# the Monte Carlo DOE algorithm and 30 samples.
scenario = create_scenario(
    disciplines,
    "y_4",
    design_space,
    formulation_name="MDF",
    maximize_objective=True,
    scenario_type="DOE",
)
for constraint in ["g_1", "g_2", "g_3"]:
    scenario.add_constraint(constraint, constraint_type="ineq")
scenario.execute(algo_name="OT_MONTE_CARLO", n_samples=30)

# %%
# Post-process scenario
# ---------------------
# Lastly, we post-process the scenario by means of the
# :class:`.SOM` plot which performs a self organizing map
# clustering on optimization history.

# %%
# .. tip::
#
#    Each post-processing method requires different inputs and offers a variety
#    of customization options. Use the high-level function
#    :func:`.get_post_processing_options_schema` to print a table with
#    the options for any post-processing algorithm.
#    Or refer to our dedicated page:
#    :ref:`gen_post_algos`.

scenario.post_process(post_name="SOM", save=False, show=True)

# %%
# The following figure illustrates another :term:`SOM` on the Sobieski
# use case. The optimization method is a (costly) derivative free algorithm
# (``NLOPT_COBYLA``), indeed all the relevant information for the optimization
# is obtained at the cost of numerous evaluations of the functions. For
# more details, please read the paper by
# :cite:`kumano2006multidisciplinary` on wing MDO post-processing
# using SOM.
#
# .. figure:: /tutorials/ssbj/figs/MDOScenario_SOM_v100.png
#
#     SOM example on the Sobieski problem.
#
# A DOE may also be a good way to produce SOM maps.
# The following figure shows an example with 10000 points on
# the same test case. This produces more relevant SOM plots.
#
# .. figure:: /tutorials/ssbj/figs/som_fine.png
#
#     SOM example on the Sobieski problem with a 10 000 samples DOE.
