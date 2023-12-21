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
Formulation
===========

In this example, we will discover the different functions of the API
related to MDO formulations: their names, their options and their sub-options.
"""

from __future__ import annotations

from gemseo import configure_logger
from gemseo import get_available_formulations
from gemseo import get_formulation_options_schema
from gemseo import get_formulation_sub_options_schema
from gemseo import get_formulations_options_defaults
from gemseo import get_formulations_sub_options_defaults

configure_logger()


# %%
# Get available formulations
# --------------------------
#
# The :func:`.get_available_formulations` function returns the list
# of MDO formulations available in |g| or in external modules
get_available_formulations()

# %%
# Get formulation schemas for (sub-)options
# -----------------------------------------
# For a given MDO formulation, e.g. ``"MDF"``, we can:
#
# - get the options of an MDO formulation using the
#   :func:`.get_formulation_options_schema` function; e.g.
get_formulation_options_schema("MDF")

# %%
# - get the default option values using the
#   :func:`.get_formulations_options_defaults` function; e.g.
get_formulations_options_defaults("MDF")

# %%
# - get sub-options of an MDO formulation using the
#   :func:`.get_formulation_sub_options_schema` function; e.g.
get_formulation_sub_options_schema("MDF", main_mda_name="MDAGaussSeidel")

# %%
# - get the sub-option values using the
#   :func:`.get_formulations_sub_options_defaults` function; e.g.
get_formulations_sub_options_defaults("MDF", main_mda_name="MDAGaussSeidel")
