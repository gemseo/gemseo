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
Design space
============

In this example, we will discover the different functions of the API related to
design space, which is a key element to represent the space of parameters on
which a scenario will evaluate a list of disciplines.

"""
from __future__ import annotations

from gemseo.api import configure_logger
from gemseo.api import create_design_space
from gemseo.api import export_design_space
from gemseo.api import read_design_space
from numpy import array

configure_logger()


##########################################################################
# Create a design space
# ---------------------
#
# To create a standard :class:`.DesignSpace`,
# the API function :meth:`~gemseo.api.create_design_space` can be used.
#
# - This function does not take any argument.
# - This function returns an empty instance of :class:`.DesignSpace`.
design_space = create_design_space()
print(design_space)

##########################################################################
# Once built, we can add variables. E.g.

design_space.add_variable(
    "x", 2, l_b=array([0.0] * 2), u_b=array([1.0] * 2), value=array([0.5] * 2)
)
print(design_space)

##########################################################################
# Read a design space
# -------------------
# In presence of a design space specified in a CSV file,
# the API function :meth:`~gemseo.api.read_design_space` can be used.
#
# - Its first argument is the file path of the design space.
#   Its second argument is the list of fields available in the file
#   and is optional.
# - By default, the design space reads these information from the file.
# - This function returns an instance of :class:`.DesignSpace`.
design_space.export_to_txt("saved_design_space.txt")
loaded_design_space = read_design_space("saved_design_space.txt")

##########################################################################
# Write a design space
# --------------------
#
# To export an instance of :class:`.DesignSpace` into an hdf or txt file,
# the :meth:`~gemseo.api.export_design_space` API function can be used:
loaded_design_space.add_variable("y", l_b=-1, u_b=3, value=0.0)
export_design_space(loaded_design_space, "saved_design_space.txt")
print(read_design_space("saved_design_space.txt"))
