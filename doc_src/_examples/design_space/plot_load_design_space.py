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
#        :author: Jean-Christophe Giret
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""
DesignSpace import and export from disk
=======================================

In this example, we will see how to read, filter, and export a design space
from the disk.
"""
from __future__ import annotations

from gemseo.api import configure_logger
from gemseo.api import export_design_space
from gemseo.api import read_design_space

configure_logger()


###############################################################################
# Read a design space from a file
# -------------------------------
#
# The user can read a design space from a file using the
# :func:`.create_design_space` function.


design_space = read_design_space("design_space.txt")
print(design_space)


###############################################################################
# Filtering the design space
# --------------------------
#
# The user can filter the design space in order to only keep some variables. To
# do so, the user can use the :meth:`.DesignSpace.filter` method:

design_space.filter(["x1", "x2"])
print(design_space)

###############################################################################
# Export the design space
# -----------------------
#
# The user can export a :class:`.DesignSpace` instance by using the
# :func:`.export_design_space` function.


export_design_space(design_space, "new_design_space.txt")
