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
"""# Create a discipline from analytical expressions."""

from __future__ import annotations

from numpy import array

from gemseo import create_discipline

# %%
# ## Import
#


# %%
# ## Introduction
#
# A simple [Discipline][gemseo.core.discipline.discipline.Discipline] can be created
# using analytic formulas, e.g. $y_1=2x^2$ and $y_2=5+3x^2z^3$,
# thanks to the [AnalyticDiscipline][gemseo.disciplines.analytic.AnalyticDiscipline] class
# which is a quick alternative to model a simple analytic MDO problem.


# %%
# ## Create the dictionary of analytic outputs
#
# First of all, we have to define the output expressions in a dictionary
# where keys are output names and values are formula with `string`
# format:
expressions = {"y_1": "2*x**2", "y_2": "5+3*x**2+z**3"}

# %%
# ## Create the discipline
#
# Then, we create and instantiate the corresponding
# [AnalyticDiscipline][gemseo.disciplines.analytic.AnalyticDiscipline],
# which is a particular [Discipline][gemseo.core.discipline.discipline.Discipline].
# For that, we use the API function [create_discipline()][gemseo.create_discipline] with:
#
# - `discipline_name="AnalyticDiscipline"`,
# - `name="analytic"`,
# - `expressions=expr_dict`.
#
# In practice, we write:
disc = create_discipline("AnalyticDiscipline", expressions=expressions)

# %%
# !!! note
#
#     GEMSEO takes care of the grammars and `Discipline._run()` method generation from the `expressions` argument.
#     In the background, GEMSEO considers that `x` is a mono-dimensional
#     float input parameter and `y_1` and `y_2` are
#     mono-dimensional float output parameters.

# %%
# ## Execute the discipline
#
# Lastly, we can execute this discipline any other:
input_data = {"x": array([2.0]), "z": array([3.0])}
disc.execute(input_data)

# %%
# ## About the analytic jacobian
#
# The discipline will provide analytic derivatives (Jacobian) automatically
# using the [sympy library](https://www.sympy.org/fr/).
#
# This can be checked easily using [check_jacobian()][gemseo.core.discipline.discipline.Discipline.check_jacobian]:
disc.check_jacobian(
    input_data,
    derr_approx=disc.ApproximationMode.FINITE_DIFFERENCES,
    step=1e-5,
    threshold=1e-3,
)
