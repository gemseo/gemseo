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
r"""The wing weight problem.

The function of the wing weight problem

.. math::

   f(A,\ell,\Lambda,N_z,q,S_w,t_c,W_{dg},W_{fw},W_p) = 0.036 S_w^{0.758} W_{fw}^{0.0035}
   \left(\frac{A}{\cos(\pi / 180 \Lambda)^2}\right)^{0.6} q^{0.006} \ell^{0.04}
   \left(\frac{100 t_c}{\cos(\pi / 180 \Lambda)}\right)^{-0.3} (N_z W_{dg})^{0.49}
   + S_w W_p

is commonly studied through the random input vector
:math:`W_w=f(A,\ell,\Lambda,N_z,q,S_w,t_c,W_{dg},W_{fw},W_p)` whose components are
independent random variables uniformly distributed:

- :math:`A\sim\mathcal{U}([6.0, 10.0])`, the aspect ratio (-),
- :math:`\ell\sim\mathcal{U}([0.5, 1.0])`, the taper ratio (-),
- :math:`\Lambda\sim\mathcal{U}([-10.0, 10.0])`, the quarter-chord sweep angle (deg),
- :math:`N_z\sim\mathcal{U}([2.5, 6.0])`, the ultimate load factor (-),
- :math:`q\sim\mathcal{U}([16, 45])`, the dynamic pressure at cruise (lb/ft^2),
- :math:`S_w\sim\mathcal{U}([150, 200])`, the wing area (ft^2),
- :math:`t_c\sim\mathcal{U}([0.08, 0.18])`, the airfoil thickness to chord ratio (-),
- :math:`W_{dg}\sim\mathcal{U}([1700, 2500])`, the flight design gross weight (lb),
- :math:`W_{fw}\sim\mathcal{U}([220, 300])`, the weight of fuel in the wing (lb),
- :math:`W_p\sim\mathcal{U}([0.025, 0.08])`, the paint weight (lb/ft^2).

The wing weight problem is presented in :cite:`forrester2008`
and the description given here is based on that of
`OpenTURNS <https://openturns.github.io/openturns/latest/usecases/use_case_wingweight.html>`__.
"""

from __future__ import annotations
