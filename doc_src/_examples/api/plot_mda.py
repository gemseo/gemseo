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

In this example, we will discover the different functions of the API
related to MDAs, which are the |g|' objects dedicated to the
feasibility of the multidisciplinary coupling. All classes
implementing MDAs inherit from :class:`.MDA` which is an abstract class.
"""
from __future__ import annotations

from gemseo.api import configure_logger
from gemseo.api import create_discipline
from gemseo.api import create_mda
from gemseo.api import get_available_mdas
from gemseo.api import get_mda_options_schema

configure_logger()


##########################################################################
# Get available MDA
# -----------------
#
# The :meth:`~gemseo.api.get_available_mdas` function returns the list
# of MDAs available in |g| or in external modules
print(get_available_mdas())

##########################################################################
# Get MDA options schema
# ----------------------
# For a given MDA algorithm, e.g. :code:`"MDAGaussSeidel"`,
# we can get the options; e.g.
print(get_mda_options_schema("MDAGaussSeidel"))

##########################################################################
# Create an MDA
# -------------
# The API function :meth:`~gemseo.api.create_mda` can be used
# to create a scenario:
disciplines = create_discipline(["Sellar1", "Sellar2"])
mda = create_mda("MDAGaussSeidel", disciplines)
output_data = mda.execute()
print(output_data)
