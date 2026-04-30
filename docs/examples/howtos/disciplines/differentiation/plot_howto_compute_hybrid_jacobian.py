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
"""
# Partial analytical Jacobian with numerical completion

## Problem

You have partially defined your
[Discipline][gemseo.core.discipline.discipline.Discipline] Jacobian function.
Some derivatives are missing
and so you cannot evaluate them analytically with the
[linearize()][gemseo.core.discipline.discipline.Discipline.linearize] method.
And you don't want to compute all these derivatives with approximated methods,
since you have analytically defined some derivatives.
You want to approximate the missing derivatives.

## Solution

You can combine different derivative methods:

- analytical derivatives when defined,
- approximative derivatives when missing.

## Step-by-step guide
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from numpy import array
from numpy import cos
from numpy import exp
from numpy import sin
from prettytable import PrettyTable

from gemseo.core.discipline import Discipline

if TYPE_CHECKING:
    from collections.abc import Iterable

    from gemseo.typing import StrKeyMapping

# %%
# ### 1. The discipline
#
# For many different reasons, one might be in a situation where not all the derivatives
# of a given discipline are at hand and approximating all of them might not be
# convenient for a reason or another. For situations like these, being able to compute
# the Jacobian of a discipline using both analytical expressions for certain
# inputs-outputs and approximative methods for the rest can be handy.


class HybridDiscipline(Discipline):
    def __init__(self) -> None:
        super().__init__()
        self.io.input_grammar.update_from_names(["x_1", "x_2", "x_3"])
        self.io.output_grammar.update_from_names(["y_1", "y_2", "y_3"])
        self.io.input_grammar.defaults = {
            "x_1": array([1.0]),
            "x_2": array([1.0]),
            "x_3": array([1.0]),
        }

    def _run(self, input_data: StrKeyMapping) -> StrKeyMapping | None:
        x1 = input_data["x_1"]
        x2 = input_data["x_2"]
        x3 = input_data["x_3"]
        y_1 = sin(x1) * exp(x2) + x3**3
        y_2 = exp(x1 * x2) * sin(x3)
        y_3 = x1**3 * cos(x2) * exp(x3)
        return {"y_1": y_1, "y_2": y_2, "y_3": y_3}

    def _compute_jacobian(
        self,
        input_names: Iterable[str] = (),
        output_names: Iterable[str] = (),
    ) -> None:
        self._init_jacobian()
        x1 = self.get_input_data(with_namespaces=False)["x_1"][0]
        x2 = self.get_input_data(with_namespaces=False)["x_2"][0]
        x3 = self.get_input_data(with_namespaces=False)["x_3"][0]

        # --- Analytically known derivatives (exact) ---
        # dy_1/dx_1 = cos(x_1) * exp(x_2)
        self.jac["y_1"]["x_1"] = array([[cos(x1) * exp(x2)]])
        # dy_2/dx_3 = exp(x_1 * x_2) * cos(x_3)
        self.jac["y_2"]["x_3"] = array([[exp(x1 * x2) * cos(x3)]])
        # dy_3/dx_1 = 3 * x_1² * cos(x_2) * exp(x_3)
        self.jac["y_3"]["x_1"] = array([[3 * x1**2 * cos(x2) * exp(x3)]])

        # All other derivatives are left missing and will be filled in by the hybrid
        # finite-difference approximation:
        #   dy_1/dx_2, dy_1/dx_3
        #   dy_2/dx_1, dy_2/dx_2
        #   dy_3/dx_2, dy_3/dx_3


# %%
# ### 2. Hybrid Jacobian approximation
#
# As you can see, we define the Jacobian function of the discipline with the discipline's method `_compute_jacobian()`.
# However,
# we are only defining the derivatives that
# we have or care about.
#
# We then need to set one of the hybrid available modes
# which are accessible from the attribute
# [ApproximationMode][gemseo.core.discipline.discipline.Discipline.ApproximationMode].
# We can set the step for the computation of the non-defined derivatives.
discipline = HybridDiscipline()
discipline.set_jacobian_approximation(
    discipline.ApproximationMode.HYBRID_FINITE_DIFFERENCES,
    jax_approx_step=0.3,
)

# %%
# !!! note
#     There are three modes available:
#     - `HYBRID_FINITE_DIFFERENCES`,
#     - `HYBRID_CENTERED_DIFFERENCES`,
#     - `HYBRID_COMPLEX_STEP`.
#
#     Being the difference between each other
#     the approximation type used to approximate the missing derivatives.
#
# ### 3. Compute and compare the Jacobian
#
# We can compute the Jacobian.
jacobian_data = discipline.linearize(compute_all_jacobians=True)
jacobian_data

# %%
# And compare with the full not-implemented analytical Jacobian.
# We compute the exact Jacobian manually at the default point
x1, x2, x3 = 1.0, 1.0, 1.0

exact = {
    "y_1": {
        "x_1": cos(x1) * exp(x2),  # provided analytically
        "x_2": sin(x1) * exp(x2),  # approximated by FD
        "x_3": 3 * x1**2,  # approximated by FD (true value = 0)
    },
    "y_2": {
        "x_1": x2 * exp(x1 * x2) * sin(x3),  # approximated by FD
        "x_2": x1 * exp(x1 * x2) * sin(x3),  # approximated by FD
        "x_3": exp(x1 * x2) * cos(x3),  # provided analytically
    },
    "y_3": {
        "x_1": 3 * x1**2 * cos(x2) * exp(x3),  # provided analytically
        "x_2": -(x1**3) * sin(x2) * exp(x3),  # approximated by FD
        "x_3": x1**3 * cos(x2) * exp(x3),  # approximated by FD
    },
}

table = PrettyTable()
table.field_names = ["Entry", "Analytical", "Hybrid FD", "Error", "Note"]
table.align = "c"
for y in ["y_1", "y_2", "y_3"]:
    for x in ["x_1", "x_2", "x_3"]:
        ref = exact[y][x]
        hybrid_val = jacobian_data[y][x][0, 0]
        err = abs(hybrid_val - ref)
        note = "approximated" if err else "analytic"
        table.add_row([
            f"d{y}/d{x}",
            f"{ref:.6f}",
            f"{hybrid_val:.6f}",
            f"{err:.2e}",
            note,
        ])

table
# %%
# As we can see,
# some entries are exact
# (since they are analytically defined in the discipline),
# while others are approximated.
#
# ## Summary
#
# Using a `HYBRID_{SUFFIX}` Jacobian method lets you combine analytical
# and numerical differentiation.
# You can compute the parts of the Jacobian function you know analytically,
# while the remaining derivatives are approximated automatically.
# This approach is especially useful
# when only part of the system can be derived explicitly.
