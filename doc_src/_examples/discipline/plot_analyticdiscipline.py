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
Create a discipline from analytical expressions
===============================================
"""
from __future__ import annotations

from gemseo.api import configure_logger
from gemseo.api import create_discipline
from numpy import array

###############################################################################
# Import
# ------

configure_logger()

###############################################################################
# Introduction
# ------------
# A simple :class:`.MDODiscipline` can be created
# using analytic formulas, e.g. :math:`y_1=2x^2` and :math:`y_2=5+3x^2z^3`,
# thanks to the :class:`.AnalyticDiscipline` class
# which is a quick alternative to model a simple analytic MDO problem.


###############################################################################
# Create the dictionary of analytic outputs
# -----------------------------------------
# First of all, we have to define the output expressions in a dictionary
# where keys are output names and values are formula with :code:`string`
# format:
expressions = {"y_1": "2*x**2", "y_2": "5+3*x**2+z**3"}

###############################################################################
# Create the discipline
# ---------------------
# Then, we create and instantiate the corresponding
# :class:`.AnalyticDiscipline`,
# which is a particular :class:`.MDODiscipline`.
# For that, we use the API function :meth:`~gemseo.api.create_discipline` with:
#
# - :code:`discipline_name="AnalyticDiscipline"`,
# - :code:`name="analytic"`,
# - :code:`expressions=expr_dict`.
#
# In practice, we write:
disc = create_discipline("AnalyticDiscipline", expressions=expressions)

###############################################################################
# .. note::
#
#    |g| takes care of the grammars and
#    :meth:`!MDODiscipline._run` method generation
#    from the :code:`expressions` argument.
#    In the background, |g| considers that :code:`x` is a monodimensional
#    float input parameter and :code:`y_1` and :code:`y_2` are
#    monodimensional float output parameters.

###############################################################################
# Execute the discipline
# ----------------------
# Lastly, we can execute this discipline any other:
input_data = {"x": array([2.0]), "z": array([3.0])}
out = disc.execute(input_data)
print(("y_1 =", out["y_1"]))
print(("y_2 =", out["y_2"]))

###############################################################################
# About the analytic jacobian
# ---------------------------
# The discipline will provide analytic derivatives (Jacobian) automatically
# using the `sympy library <https://www.sympy.org/fr/>`_.
#
# This can be checked easily using
# :meth:`.MDODiscipline.check_jacobian`:
disc.check_jacobian(
    input_data, derr_approx=disc.FINITE_DIFFERENCES, step=1e-5, threshold=1e-3
)
