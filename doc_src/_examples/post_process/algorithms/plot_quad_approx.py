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
r"""
Quadratic approximations
========================

In this example, we illustrate the use of the :class:`.QuadApprox` plot
on the Sobieski's SSBJ problem.

The :class:`.QuadApprox` post-processing performs a quadratic approximation of a given
function from an optimization history and provide two plots.

The first plot shows an approximation of the Hessian matrix
:math:`\dfrac{\partial^2 f}{\partial x_i \partial x_j}` based on the *Symmetric Rank 1*
method (SR1) :cite:`Nocedal2006`. The color map uses a symmetric logarithmic (symlog)
scale. This plots the cross influence of the design variables on the objective function
or constraints. For instance, on the last figure, the maximal second-order sensitivity
is :math:`\dfrac{\partial^2 -y_4}{\partial^2 x_0} = 2.10^5`, which means that the
:math:`x_0` is the most influential variable. Then, the cross derivative
:math:`\dfrac{\partial^2 -y_4}{\partial x_0 \partial x_2} = 5.10^4` is positive and
relatively high compared to the previous one but the combined effects of :math:`x_0` and
:math:`x_2` are non-negligible in comparison.

The second plot represents the quadratic approximation of the objective around the
optimal solution : :math:`a_{i}(t)=0.5 (t-x^*_i)^2
\dfrac{\partial^2 f}{\partial x_i^2} + (t-x^*_i) \dfrac{\partial
f}{\partial x_i} + f(x^*)`, where :math:`x^*` is the optimal solution.
This approximation highlights the sensitivity of the :term:`objective function` with
respect to the :term:`design variables`: we notice that the design
variables :math:`x\_1, x\_5, x\_6` have little influence , whereas
:math:`x\_0, x\_2, x\_9` have a huge influence on the objective. This trend is also
noted in the diagonal terms of the :term:`Hessian` matrix
:math:`\dfrac{\partial^2 f}{\partial x_i^2}`.
"""

from __future__ import annotations

from gemseo import execute_post
from gemseo.settings.post import QuadApprox_Settings

execute_post(
    "sobieski_mdf_scenario.h5",
    settings_model=QuadApprox_Settings(function="-y_4", save=False, show=True),
)
