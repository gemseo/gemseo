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
Compute the Jacobian of a discipline with finite differences
============================================================

In this example,
we will compute the Jacobians of some outputs of a :class:`.MDODiscipline`
with respect to some inputs using the finite difference method.
"""
from __future__ import annotations

from gemseo.disciplines.auto_py import AutoPyDiscipline
from numpy import array


# %%
# First,
# we create a discipline, e.g. an :class:`.AutoPyDiscipline`:
def f(a=0.0, b=0.0):
    y = a**2 + b
    z = a**3 + b**2
    return y, z


discipline = AutoPyDiscipline(f)

# %%
# We can execute it with its default input values:
discipline.execute()
print(discipline.local_data)

# %%
# or with custom ones:
discipline.execute({"a": array([1.0])})
print(discipline.local_data)

# %%
# Then,
# we use the method :meth:`.MDODiscipline.linearize` to compute the derivatives:
jacobian_data = discipline.linearize()
print(jacobian_data)

# %%
# There is no Jacobian data
# because we need to set the input variables
# against which to compute the Jacobian of the output ones.
# For that,
# we use the method :meth:`~.MDODiscipline.add_differentiated_inputs`.
# We also need to set these output variables:
# with the method :meth:`~.MDODiscipline.add_differentiated_outputs`.
# For instance,
# we may want to compute the derivative of ``"z"`` with respect to ``"a"`` only:
discipline.add_differentiated_inputs(["a"])
discipline.add_differentiated_outputs(["z"])
jacobian_data = discipline.linearize()
print(jacobian_data)

# %%
# We can have a quick look at the values of these derivatives
# and verify that they are equal to the analytical results,
# up to the numerical precision.
#
# By default,
# |g| uses :attr:`.MDODiscipline.default_inputs` as input data
# for which to compute the Jacobian one.
# We can change them with ``input_data``:
jacobian_data = discipline.linearize(input_data={"a": array([1.0])})
print(jacobian_data)

# %%
# We can also force the discipline to compute
# the derivatives of all the outputs with respect to all the inputs:
jacobian_data = discipline.linearize(compute_all_jacobians=True)
print(jacobian_data)

# %%
# Lastly,
# we can change the approximation type to complex step and compare the results:
discipline.set_jacobian_approximation(
    jac_approx_type=discipline.ApproximationMode.COMPLEX_STEP
)
jacobian_data = discipline.linearize(compute_all_jacobians=True)
print(jacobian_data)
