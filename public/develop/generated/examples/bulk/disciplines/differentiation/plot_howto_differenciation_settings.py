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
# Change the differentiation settings

## Problem

You want to parametrize your differentiation method used in your
[Discipline][gemseo.core.discipline.discipline.Discipline].
You may also want to change the method.

## Solution

You can set differentiation settings with the
[set_jacobian_approximation()][gemseo.core.discipline.discipline.Discipline.set_jacobian_approximation]
method.

## Step-by-step guide
"""

from __future__ import annotations

from gemseo.disciplines.auto_py import AutoPyDiscipline


# %%
# ### 1. Create the discipline to linearize
#
# !!! note
#     You may see the how-to:
#     [Compute the Jacobian of a discipline][compute-the-jacobian-of-a-discipline]
def f(a=0.0, b=0.0):
    y = a**2 + b
    z = a**3 + b**2
    return y, z


discipline = AutoPyDiscipline(f)
jacobian_data = discipline.linearize(compute_all_jacobians=True)
jacobian_data

# %%
# ### 2. Change the jacobian approximation settings
#
# You can change the approximation type to complex step:
discipline.set_jacobian_approximation(
    jac_approx_type=discipline.ApproximationMode.COMPLEX_STEP
)
jacobian_data_complex_step = discipline.linearize(compute_all_jacobians=True)
jacobian_data_complex_step

# %%
# or to centered differences:
discipline.set_jacobian_approximation(
    jac_approx_type=discipline.ApproximationMode.CENTERED_DIFFERENCES,
    jax_approx_step=1e-2,
)
jacobian_data_centered_differences = discipline.linearize(compute_all_jacobians=True)
jacobian_data_centered_differences

# %%
# ## Summary
#
# You can parametrize the differentiation method used for your
# [Discipline][gemseo.core.discipline.discipline.Discipline]
# by using the
# [set_jacobian_approximation()][gemseo.core.discipline.discipline.Discipline.set_jacobian_approximation]
# method.
