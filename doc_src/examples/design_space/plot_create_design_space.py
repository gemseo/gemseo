# -*- coding: utf-8 -*-
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
DesignSpace creation and manipulation
=====================================

In this example, we will see how to create and how to manipulate an instance of
:class:`.DesignSpace`.
"""
from __future__ import division, unicode_literals

from numpy import array

from gemseo.api import configure_logger, create_design_space

configure_logger()


###############################################################################
# Create a parameter space
# ------------------------
#
# The user can create an instance of the :class:`.DesignSpace` using the API
# and the :func:`.create_design_space` function.


design_space = create_design_space()


###############################################################################
# Add design variables
# --------------------
#
# The user can add new design variables using the :meth:`.DesignSpace.add_variable`. In
# the following example, we add the `x` variable in the design space. We also
# define the lower and upper bound of the variable.
# It is then possible to plot the :class:`.DesignSpace` instance either using a
# print statement or by using the logger.

design_space.add_variable("x", l_b=array([-2.0]), u_b=array([2.0]), value=array([0.0]))

print(design_space)

###############################################################################
# The user can also add design variables with dimension greater than one. To do
# that, the user can use the `size` keyword:

design_space.add_variable(
    "y", l_b=array([-2.0, -1.0]), u_b=array([2.0, 1.0]), value=array([0.0, 0.0]), size=2
)
print(design_space)

###############################################################################
# Remove design variables
# -----------------------
#
# The user can also remove a variable in the design space using the
# :meth:`.DesignSpace.remove_variable` method:

design_space.remove_variable("x")
print(design_space)
