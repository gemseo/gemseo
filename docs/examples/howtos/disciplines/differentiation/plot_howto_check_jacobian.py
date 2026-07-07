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
r"""# Check the Jacobian of a discipline

## Problem

You have implemented the Jacobian of a
[Discipline][gemseo.core.discipline.discipline.Discipline]
and you want to check whether it is correct.

## Solution

The Jacobian can be checked against a numerical approximation
with the [check_jacobian()][gemseo.check_jacobian] function.

## Step-by-step guide
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from numpy import array

from gemseo import check_jacobian
from gemseo.core.discipline import Discipline
from gemseo.enums import ApproximationMode

if TYPE_CHECKING:
    from collections.abc import Iterable

    from gemseo.typing import StrKeyMapping


# %%
# ### 1. Create the discipline
#
# We consider a discipline computing $f(x,y)=2x+3y$ and $g(x,y)=x^3+y$,
# and introduce a bug in the Jacobian of $g$:
# we code $\frac{\partial g}{\partial x}=x^2$ instead of the correct $3x^2$.
class BuggedDiscipline(Discipline):
    def __init__(self) -> None:
        super().__init__()
        self.input_grammar.update_from_names(["x", "y"])
        self.output_grammar.update_from_names(["f", "g"])
        self.default_input_data = {"x": array([1.0]), "y": array([1.0])}

    def _run(self, input_data: StrKeyMapping) -> StrKeyMapping:
        x = input_data["x"]
        y = input_data["y"]
        return {"f": 2 * x + 3 * y, "g": x**3 + y}

    def _compute_jacobian(
        self,
        input_names: Iterable[str] = (),
        output_names: Iterable[str] = (),
    ) -> None:
        x = self.io.input_data["x"]
        self._init_jacobian()
        self.jac["f"]["x"][:] = 2.0
        self.jac["f"]["y"][:] = 3.0
        self.jac["g"]["x"][:] = x**2  # this is wrong: it should be 3 * x**2.
        self.jac["g"]["y"][:] = 1.0


# %%
# ### 2. Check the implemented Jacobian
#
# Call [check_jacobian()][gemseo.check_jacobian] from your discipline
discipline = BuggedDiscipline()
check_jacobian(discipline, plot_result=True)

# %%
# The function returns `False`: the Jacobian is wrong.
# At $x=1$, the correct value is $\frac{\partial g}{\partial x}=3x^2=3$
# while the discipline returns $x^2=1$;
# the function logs that `∂g/∂x` is wrong by about 50%
# (its error is the absolute difference normalized by the approximated value plus one).
# The graph confirms it:
# the provided components (blue dots) and the approximated ones (red dots)
# overlap everywhere except for the $\frac{\partial g}{\partial x}$ component.

# %%
# ## Summary
#
# The implementation of the Jacobian of a discipline can be checked against a numerical
# approximation with the [check_jacobian()][gemseo.check_jacobian] function.
# This function returns `False` and, with `plot_result=True`, draws the provided and
# approximated Jacobian components so that the wrong ones can be spotted.

# %%
# ## One step further
#
# ### Choose the approximation method
#
# The numerical approximation method is set via the `approximation_mode` argument
# of `check_jacobian`,
# either `ApproximationMode.FINITE_DIFFERENCES` (default),
# `ApproximationMode.CENTERED_DIFFERENCES` or `ApproximationMode.COMPLEX_STEP`,
# and the discretization step via its `approximation_step` argument.
#
# `ApproximationMode.FINITE_DIFFERENCES` uses forward differences.
# Forward differences and complex step are first-order methods using one sampling point
# per input, with a truncation error decreasing linearly with the step.
# Centered differences are second-order, using twice as many points.
# Complex step is immune to numerical cancellation, so an arbitrarily small step can be
# used, but it requires a discipline supporting complex inputs.
check_jacobian(
    discipline,
    approximation_mode=ApproximationMode.COMPLEX_STEP,
    step=1e-30,
)

# %%
# ### Set the step automatically
#
# For finite and centered differences,
# the step is a trade-off between truncation and numerical errors.
# Setting `approximation_step` to `None`
# computes the step minimizing the total error automatically.
check_jacobian(
    discipline,
    approximation_mode=ApproximationMode.CENTERED_DIFFERENCES,
    step=None,
)
