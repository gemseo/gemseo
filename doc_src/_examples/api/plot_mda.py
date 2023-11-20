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
MDA
===

In this example, we will discover the different high-level functions
related to MDAs, which are the |g|' objects dedicated to the
feasibility of the multidisciplinary coupling. All classes
implementing MDAs inherit from :class:`.MDA` which is an abstract class.
"""

from __future__ import annotations

from gemseo import configure_logger
from gemseo import create_discipline
from gemseo import create_mda
from gemseo import get_available_mdas
from gemseo import get_mda_options_schema

configure_logger()


# %%
# Get available MDA
# -----------------
#
# The :func:`.get_available_mdas` function returns the list
# of MDAs available in |g| or in external modules
get_available_mdas()

# %%
# Get MDA options schema
# ----------------------
# For a given MDA algorithm, e.g. ``"MDAGaussSeidel"``,
# we can get the options; e.g.
get_mda_options_schema("MDAGaussSeidel")

# %%
# Create an MDA
# -------------
# The high-level function :func:`~gemseo.create_mda` can be used
# to create a scenario:
disciplines = create_discipline(["Sellar1", "Sellar2"])
mda = create_mda("MDAGaussSeidel", disciplines)
output_data = mda.execute()
output_data
