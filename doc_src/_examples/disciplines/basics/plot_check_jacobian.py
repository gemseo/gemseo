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
Check the Jacobian of a discipline
==================================

In this example,
the Jacobian of an :class:`.MDODiscipline` is checked by derivative approximation.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from numpy import array
from numpy import exp

from gemseo import configure_logger
from gemseo.core.discipline import MDODiscipline

if TYPE_CHECKING:
    from collections.abc import Iterable

configure_logger()


# %%
# First,
# we create a discipline computing
# :math:`f(x,y)=e^{-(1-x)^2-(1-y)^2}`
# and
# :math:`g(x,y)=x^2+y^2-1`
# and introduce an error in the implementation of
# :math:`\frac{\partial f(x,y)}{\partial x}`.
class BuggedDiscipline(MDODiscipline):
    def __init__(self):
        super().__init__()
        self.input_grammar.update_from_names(["x", "y"])
        self.output_grammar.update_from_names(["f", "g"])
        self.default_inputs = {"x": array([0.0]), "y": array([0.0])}

    def _run(self) -> None:
        x, y = self.get_inputs_by_name(["x", "y"])
        self.local_data["f"] = exp(-((1 - x) ** 2) - (1 - y) ** 2)
        self.local_data["g"] = x**2 + y**2 - 1

    def _compute_jacobian(
        self,
        inputs: Iterable[str] | None = None,
        outputs: Iterable[str] | None = None,
    ) -> None:
        x, y = self.get_inputs_by_name(["x", "y"])
        self._init_jacobian()
        g_jac = self.jac["g"]
        g_jac["x"][:] = 2 * x
        g_jac["y"][:] = 2 * y
        f_jac = self.jac["f"]
        aux = 2 * exp(-((1 - x) ** 2) - (1 - y) ** 2)
        f_jac["x"][:] = aux  # this is wrong.
        f_jac["y"][:] = aux * (1 - y)


# %%
# We want to check if the implemented Jacobian is correct.
# For practical applications where Jacobians are needed, this is not a simple task.
# GEMSEO automates such tests thanks to the :meth:`.MDODiscipline.check_jacobian` method.
#
# Finite differences (default)
# ----------------------------
discipline = BuggedDiscipline()
discipline.check_jacobian(
    input_data={"x": array([0.0]), "y": array([1.0])},
    show=True,
    plot_result=True,
    step=1e-1,
)

# %%
# The step here is chosen big enough to underline the truncation error.
# From this graph, we can see that almost all the provided components  of the Jacobians
# (blue dots) are close but distinct from the approximated by finite differences using
# a step of 0.1 (red dots). This kind of graph can be used to spot implementation
# mistakes in fact we can already spot a large mistake in the wrong components.
#

# The ``derr_approx`` argument can be either ``finite_differences``, ``centered_differences`` or
# ``complex_step``.

# Centered differences
# --------------------
discipline.check_jacobian(
    input_data={"x": array([0.0]), "y": array([1.0])},
    derr_approx=discipline.ApproximationMode.CENTERED_DIFFERENCES,
    show=True,
    plot_result=True,
    step=1e-1,
)

# With the same step the truncation error is in this case much smaller.


# Complex step
# ------------
discipline.check_jacobian(
    input_data={"x": array([0.0]), "y": array([1.0])},
    derr_approx=discipline.ApproximationMode.COMPLEX_STEP,
    show=True,
    plot_result=True,
    step=1e-1,
)

# With the same step the truncation error is also smaller than finite differences.
# This confirms again that an implementation mistake was done.

# Advantages and drawbacks of each method
# ---------------------------------------
# Finite differnces and complex are first-order methods, they use one
# sampling point per input and the truncation error goes down linearly with the step.
# Centered differences are second-order methods which use twice as many points as finite
# differences and complex step. Complex step derivatives are less prone to numerical
# cancellation errors so that a tiny step can be used. On the other hand complex step is
# not compatible with discipline not supporting complex inputs.

discipline.check_jacobian(
    input_data={"x": array([0.0]), "y": array([1.0])},
    derr_approx=discipline.ApproximationMode.COMPLEX_STEP,
    show=True,
    plot_result=True,
    step=1e-10,
)
# %%
# Automatic time step
# -------------------
# Finite differences and centered differences steps
# need to be chosen as a trade between truncation and numerical errors.
# For this reason, the ``auto_set_step`` option can be used to automatically compute the step
# where the total error is minimized.

discipline.check_jacobian(
    input_data={"x": array([0.0]), "y": array([1.0])},
    derr_approx=discipline.ApproximationMode.CENTERED_DIFFERENCES,
    show=True,
    plot_result=True,
    auto_set_step=True,
)
