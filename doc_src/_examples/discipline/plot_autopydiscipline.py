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
Create a discipline from a Python function
==========================================
"""
###############################################################################
# Import
# ------
from __future__ import annotations

from gemseo.api import configure_logger
from gemseo.api import create_discipline
from numpy import array
from numpy import empty

configure_logger()

###############################################################################
# Build a discipline from a simple Python function
# ------------------------------------------------
# Let's consider a simple Python function, e.g.:


def f(x=0.0, y=0.0):
    """A simple Python function."""
    z = x + 2 * y
    return z


###############################################################################
# Create and instantiate the discipline
# -------------------------------------
# Then, we can consider the
# :class:`.AutoPyDiscipline` class
# to convert it into an :class:`.MDODiscipline`.
# For that, we can use the :meth:`~gemseo.api.create_discipline` API function
# with :code:`'AutoPyDiscipline'` as first argument:
disc = create_discipline("AutoPyDiscipline", py_func=f)

###############################################################################
# The original Python function may or may not include default values for input
# arguments, however, if the resulting :class:`.AutoPyDiscipline` is going to be
# placed inside an :class:`.MDF`, a :class:`.BiLevel` formulation or an :class:`.MDA`
# with strong couplings, then the Python function **must** assign default values
# for its input arguments.

###############################################################################
# Execute the discipline
# ----------------------
# Then, we can execute it easily, either considering default inputs:
print(disc.execute())

###############################################################################
# or using new inputs:
print(disc.execute({"x": array([1.0]), "y": array([-3.2])}))

###############################################################################
# Optional arguments
# ------------------
# The optional arguments passed to the constructor are:
#
# - :code:`py_jac=None`: pointer to the jacobian function which must returned
#   a 2D numpy array (see below),
# - :code:`use_arrays=False`: if :code:`True`, the function is expected to take
#   arrays as inputs and give outputs as arrays,
# - :code:`write_schema=False`: if :code:`True`, write the json schema on the
#   disk.

###############################################################################
# Define the jacobian function
# ----------------------------
# Here is an example of jacobian function:


def dfdxy(x=0.0, y=0.0):
    """Jacobian function of f."""
    jac = empty((2, 1))
    jac[0, 0] = 1
    jac[1, 0] = 2
    return jac


###############################################################################
# that we can execute with default inputs for example:
print(dfdxy())
