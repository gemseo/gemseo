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
DOE algorithms
==============

In this example, we will discover the different functions of the API
related to design of experiments.

"""

from __future__ import annotations

from gemseo import configure_logger
from gemseo import get_algorithm_options_schema
from gemseo import get_available_doe_algorithms

configure_logger()


# %%
# Get available DOE algorithms
# ----------------------------
#
# The :func:`.get_available_doe_algorithms` function returns the list
# of optimization algorithms available in |g| or in external modules
get_available_doe_algorithms()

# %%
# Get options schema
# ------------------
# For a given optimization algorithm, e.g. ``"DiagonalDOE"``,
# we can get the options; e.g.
get_algorithm_options_schema("DiagonalDOE")

# %%
# Or import its settings model and pass it directly with the keyword
# "algo_settings_model".
from gemseo.settings.doe import DiagonalDOE_Settings  # noqa: E402

settings_model = DiagonalDOE_Settings()

# %%
# See :ref:`algorithm_settings` for more information on how to use settings models.
