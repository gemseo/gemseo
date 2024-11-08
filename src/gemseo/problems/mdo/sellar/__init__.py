# Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License version 3 as published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program; if not, write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.
r"""A customizable version of the Sellar MDO problem.

:cite:`Sellar1996` proposed an MDO problem
which has become a classic for comparing MDO formulations:

.. math::

   \begin{aligned}
   \text{minimize the objective function }&obj=x_{1}^2 + x_{shared,2}+y_1^2+e^{-y_2} \\
   \text{with respect to the design variables }&x_{shared},\,x_{1} \\
   \text{subject to the general constraints }
   & c_1 = 3.16 - y_1^2 \leq 0\\
   & c_2 = y_2 - 24 \leq 0\\
   \text{subject to the bound constraints }
   & -10 \leq x_{shared,1} \leq 10\\
   & 0 \leq x_{shared,2} \leq 10\\
   & 0 \leq x_1 \leq 10.
   \end{aligned}

where :math:`c_1=3.16 - y_1^2`,
:math:`c_2=y_2 - 24`,

.. math::

    y_1 = \sqrt{x_{shared,1}^2 + x_{shared,2} + x_1 - 0.2\,y_2}

and

.. math::

    y_2 = |y_1| + x_{shared,1} + x_{shared,2}.

In :cite:`Sellar1996`,
all the design and coupling variables are scalar.

In |g|,
the local design variables and the coupling variables
are vectors of dimension :math:`n` (default: 1),
a second design variable :math:`x_2` intervenes in the objective expression
and a coefficient :math:`k` controls the strength of the coupling:

.. math::

   \begin{aligned}
   \text{minimize the objective function }&
   obj=(x_1^Tx_1+x_2^Tx_2+nx_{shared,2}+y_1^\atop y_1+e^{-y_2^T1_n})/n \\
   \text{with respect to the design variables }&x_{shared},\,x_{1},\,x_{2} \\
   \text{subject to the general constraints }
   & c_1=\alpha - y_1^2 \leq 0\\
   & c_2=y_2 - \beta \leq 0\\
   \text{subject to the bound constraints }
   & -10 \leq x_{shared,1} \leq 10\\
   & 0 \leq x_{shared,2} \leq 10\\
   & 0 \leq x_1 \leq 10\\
   & 0 \leq x_2 \leq 10\\
   \end{aligned}

where the coupling variables are

.. math::

    y_1 = \sqrt{x_{shared,1}^2 + x_{shared,2} + x_1 - \gamma ky_2}

and

.. math::

    y_2 = k|y_1| + x_{shared,1} + x_{shared,2} - x_2.

The original problem :cite:`Sellar1996` can be obtained
by taking :math:`k=1`, :math:`n=1`, :math:`x_2=0`,
:math:`\alpha=3.16`, :math:`\beta=24` and :math:`\gamma=0.2`.

This package implements three disciplines
to compute the different coupling variables, constraints and objective:

- :class:`.Sellar1`:
  this :class:`.Discipline` computes :math:`y_1`
  from :math:`y_2`, :math:`x_{shared,1}`, :math:`x_{shared,2}` and :math:`x_1`.
- :class:`.Sellar2`:
  this :class:`.Discipline` computes :math:`y_2`
  from :math:`y_1`, :math:`x_{shared,1}`, :math:`x_{shared,2}` and :math:`x_2`.
- :class:`.SellarSystem`:
  this :class:`.Discipline` computes both objective and constraints
  from :math:`y_1`, :math:`y_2`, :math:`x_1`, :math:`x_2` and :math:`x_{shared,2}`,

as well as a design space called :class:`.SellarDesignSpace`.
"""

from __future__ import annotations

# This allows to test a specific data converter.
WITH_2D_ARRAY = False
