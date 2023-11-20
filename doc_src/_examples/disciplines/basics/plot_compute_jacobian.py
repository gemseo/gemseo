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
Compute the Jacobian of a discipline analytically
=================================================

In this example,
we will compute the Jacobians of some outputs of an :class:`.MDODiscipline`
with respect to some inputs, based on its analytical derivatives.
"""

from __future__ import annotations

from numpy import array

from gemseo.disciplines.analytic import AnalyticDiscipline

# %%
# First,
# we create a discipline, e.g. an :class:`.AnalyticDiscipline`:
discipline = AnalyticDiscipline({"y": "a**2+b", "z": "a**3+b**2"})

# %%
# We can execute it with its default input values:
discipline.execute()
discipline.local_data

# %%
# or with custom ones:
discipline.execute({"a": array([1.0])})
discipline.local_data

# %%
# Then,
# we use the method :meth:`.MDODiscipline.linearize` to compute the derivatives:
jacobian_data = discipline.linearize()
jacobian_data

# %%
# There is no Jacobian data
# because we need to set the input variables
# against which to compute the Jacobian of the output ones.
# For that,
# we use the method :meth:`~.MDODiscipline.add_differentiated_inputs`.
# We also need to set these output variables:
# with the method :meth:`~.MDODiscipline.add_differentiated_outputs`.
# For instance,
# we may want to only compute the derivative of ``"z"`` with respect to ``"a"``:
discipline.add_differentiated_inputs(["a"])
discipline.add_differentiated_outputs(["z"])
jacobian_data = discipline.linearize()
jacobian_data

# %%
# By default,
# |g| uses :attr:`.MDODiscipline.default_inputs` as input data
# for which to compute the Jacobian on.
# We can change them with ``input_data``:
jacobian_data = discipline.linearize(input_data={"a": array([1.0])})
jacobian_data

# %%
# We can also force the discipline to compute
# the derivatives of all the outputs with respect to all the inputs:
jacobian_data = discipline.linearize(compute_all_jacobians=True)
jacobian_data
