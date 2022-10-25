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
#    INITIAL AUTHORS - API and implementation and/or documentation
#        :author: François Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""
Analytical test case # 1
========================
"""
#############################################################################
# In this example, we consider a simple optimization problem to illustrate
# algorithms interfaces and :class:`.MDOFunction`.
#
# Imports
# -----------------------------
from __future__ import annotations

from gemseo.api import configure_logger
from gemseo.core.mdofunctions.mdo_function import MDOFunction
from numpy import cos
from numpy import exp
from numpy import ones
from numpy import sin
from scipy import optimize

configure_logger()


#############################################################################
# Define the objective function
# -----------------------------
# We define the objective function :math:`f(x)=\sin(x)-\exp(x)`
# using a :class:`.MDOFunction` defined by the sum of :class:`.MDOFunction` s.

f_1 = MDOFunction(sin, name="f_1", jac=cos, expr="sin(x)")
f_2 = MDOFunction(exp, name="f_2", jac=exp, expr="exp(x)")
objective = f_1 - f_2

#############################################################################
# .. seealso::
#
#    The following operators are implemented: :math:`+`, :math:`-`
#    and :math:`*`. The minus operator is also defined.
#

print("Objective function = ", objective)

#############################################################################
# Minimize the objective function
# -------------------------------
# We want to minimize this objective function over :math:`[-2,2]`,
# starting from 1.
# We use scipy.optimize for illustration.
#
# .. note::
#
#    :class:`.MDOFunction` objects are callable like a Python function.
#

x_0 = -ones(1)
opt = optimize.fmin_l_bfgs_b(objective, x_0, fprime=objective.jac, bounds=[(-0.2, 2.0)])

print("Optimum = ", opt)
