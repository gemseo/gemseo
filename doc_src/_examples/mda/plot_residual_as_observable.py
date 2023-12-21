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
#        :author: Loïc Cousin
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
r"""
MDA residuals
=============

This example illustrates how to retrieve the normed residual of a given MDA.

The residual of an MDA is a vector
defined by :math:`\mathrm{couplings}_k - \mathrm{couplings}_{k+1}`,
where :math:`k` and :math:`k+1` are two successive iterations of the MDA algorithm
and :math:`\mathrm{couplings}` is the coupling vector.

The normed residual is the normalized value of :math:`\norm{\mathrm{residuals}}`.
When the normed residual is smaller than a given tolerance value,
the MDA has converged.
It is a simple way to quantify the MDA convergence.

This normed residual can be seen as an MDA output.
Therefore, it can be used as a constraint in an MDO scenario,
or can simply be retrieved as a scenario observable.
It is a simple way to determine
whether a given solution has a feasible design (MDA convergence) or not.
"""

from __future__ import annotations

from gemseo import configure_logger
from gemseo import create_discipline
from gemseo import create_mda

configure_logger()

# %%
# Create and execute the MDA
# --------------------------
# We do not need to specify the inputs, the default inputs
# of the :class:`.MDA` will be used and computed from the
# default inputs of the disciplines.
#
# Here, we could have replaced :class:`.MDAGaussSeidel` by any other MDA.

disciplines = create_discipline([
    "SobieskiStructure",
    "SobieskiPropulsion",
    "SobieskiAerodynamics",
    "SobieskiMission",
])
mda = create_mda("MDAGaussSeidel", disciplines)
output_data = mda.execute()

# %%
# MDA convergence analysis
# ------------------------
# The MDA algorithm will stop if one of the following criteria is fulfilled:
#
#     - The normed residual is lower than the MDA tolerance.
#       This case appears when the design is feasible.
#     - The maximal number of iterations is reached.
#       In that case, the design is not feasible.
#
# The normed residual can be seen by the :class:`.MDA` attribute :attr:`~.MDA.normed_residual`.
mda.normed_residual

# %%
# The evolution of its value can be plotted with :meth:`~.MDA.plot_residual_history`,
# or accessed by :attr:`~.MDA.residual_history`.
#
# When an MDA is called more than once (by a DOE driver for instance),
# the :attr:`~.MDA.residual_history`
# stores the different values of the normed residual
# in a single list.
residual_history = mda.residual_history
residual_history

# %%
# The normed MDA residual can be seen as an MDA output,
# just like the couplings.
# To get the normed MDA residual,
# the key registered by :attr:`~.MDA.RESIDUALS_NORM` (``""MDA residuals norm"``) can be used.
f"The normed residual key is: {mda.RESIDUALS_NORM}."

# %%
# This normed residual can be used as a constraint in an MDO scenario,
# or can simply be retrieved as a scenario observable.
normed_residual = output_data[mda.RESIDUALS_NORM]
normed_residual
