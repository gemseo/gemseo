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
r"""The welded beam problem with a catalog of materials.

!!! warning "Experimental"
    This package is protected until the catalog variables can be encoded:
    it is outside the public API and its modules are imported directly.

Deb and Goyal (1996) proposed a welded beam design problem
mixing a choice of material, a choice of welding,
discrete dimensions and a continuous one.
A beam of length $L$, welded to a support,
carries a load $F$ at its free end.
The weld has a thickness $h$ and a length $l$,
the bar a height $t$ and a thickness $b$:

$$
   \begin{aligned}
   \text{minimize the cost }
   & f = (1 + c_1)(w t + l) h^2 + c_2 t b (L + l) \\
   \text{with respect to }
   & \text{the material},\, h,\, l,\, t,\, b \\
   \text{subject to }
   & \tau \leq 0.577 S,\ \sigma \leq S,\ h \leq b,\
   P_c \geq F,\ \delta \leq \delta_{\max} \\
   & h, b \in \{0.0625 k,\ k = 1, \ldots, 32\},\
   t \in \{0.0625 k,\ k = 32, \ldots, 320\},\
   0.1 \leq l \leq 10
   \end{aligned}
$$

where $F = 6000$ lb, $L = 14$ in, $\delta_{\max} = 0.25$ in,
and the material sets the unit cost $c_1$ of the weld,
the unit cost $c_2$ of the bar, the allowable stress $S$,
the Young modulus $E$ and the shear modulus $G$:

| material | $c_1$ | $c_2$ | $S$ (psi) | $E$ (psi) | $G$ (psi) |
|---|---|---|---|---|---|
| steel | 0.1047 | 0.0481 | $3 \cdot 10^4$ | $3 \cdot 10^7$ | $1.2 \cdot 10^7$ |
| cast iron | 0.0489 | 0.0224 | $8 \cdot 10^3$ | $1.4 \cdot 10^7$ | $6 \cdot 10^6$ |
| aluminum | 0.5235 | 0.2405 | $5 \cdot 10^3$ | $10^7$ | $4 \cdot 10^6$ |
| brass | 0.5584 | 0.2566 | $8 \cdot 10^3$ | $1.6 \cdot 10^7$ | $6 \cdot 10^6$ |

The bending stress, the deflection and the buckling load of the bar are

$$
   \sigma = \frac{6 F L}{t^2 b},\quad
   \delta = \frac{4 F L^3}{E t^3 b},\quad
   P_c = \frac{4.013\, t b^3 \sqrt{E G}}{6 L^2}
   \left(1 - \frac{t}{4 L}\sqrt{\frac{E}{G}}\right),
$$

and the shear stress in the weld is

$$
   \tau = \sqrt{\tau'^2 + \tau''^2 + 2 \tau' \tau'' \cos\theta},\quad
   \tau' = \frac{F}{A},\quad
   \tau'' = \frac{F (L + l/2) R}{J},
$$

whose area $A$, polar moment of inertia $J$, radius $R$ and angle $\theta$
depend on the welding:

- two-sided, $w = 0$:
  $A = \sqrt{2} h l$,
  $J = A \left(\frac{(h + t)^2}{4} + \frac{l^2}{12}\right)$,
  $R = \frac{1}{2}\sqrt{l^2 + (h + t)^2}$,
  $\cos\theta = \frac{l}{2 R}$;
- four-sided, $w = 1$:
  $A = \sqrt{2} h (t + l)$,
  $J = \sqrt{2} h l \left(\frac{(h + t)^2}{4} + \frac{l^2}{12}\right)
  + \sqrt{2} h t \left(\frac{(h + l)^2}{4} + \frac{t^2}{12}\right)$,
  $R = \frac{1}{2}\max\left(\sqrt{l^2 + (h + t)^2}, \sqrt{t^2 + (h + l)^2}\right)
  = \frac{1}{2}\sqrt{\min(l, t)^2 + (h + \max(l, t))^2}$,
  $\cos\theta = \frac{\min(l, t)}{2 R}$.

In GEMSEO,
the welding is not a design variable but an option of the discipline,
since it changes the expressions of the problem,
and the five constraints are normalized:

$$
   \frac{\tau}{0.577 S} - 1 \leq 0,\quad
   \frac{\sigma}{S} - 1 \leq 0,\quad
   h - b \leq 0,\quad
   1 - \frac{P_c}{F} \leq 0,\quad
   \frac{\delta}{\delta_{\max}} - 1 \leq 0.
$$

The cost increases with $l$, $\tau$ decreases with $l$
and the other constraints do not depend on $l$,
so for given material, $h$, $t$ and $b$,
the optimal $l$ is the smallest one satisfying $\tau \leq 0.577 S$.
Enumerating the discrete values with this optimal $l$
gives the optimum of each welding, reached with steel:

| welding | $f^*$ | $h^*$ | $l^*$ | $t^*$ | $b^*$ |
|---|---|---|---|---|---|
| two-sided | 2.129332 | 0.25 | 4.40085668 | 8.25 | 0.25 |
| four-sided | 1.955301 | 0.1875 | 1.78210225 | 8.25 | 0.25 |

The four-sided optimum is the best known solution
reported by Datta and Figueira (2011).
The shear stress constraint is active at both optima.

!!! note "Not the classic welded beam"
    The two-sided steel case differs from the classic welded beam problem
    of Ragsdell and Phillips, Deb (1991) and Coello,
    whose allowable shear stress is 13600 psi instead of $0.577 S = 17310$ psi;
    its optima, 1.7249 with real dimensions and 2.3810 with discrete ones,
    do not apply here.

!!! note "Sources"
    - K. Deb and M. Goyal,
      A combined genetic adaptive search (GeneAS) for engineering design,
      Computer Science and Informatics 26(4), 1996.
    - S. Datta and J. R. Figueira,
      A real-integer-discrete-coded particle swarm optimization for design problems,
      Applied Soft Computing 11(4), 2011.
    - The properties of the materials were cross-checked with table 3,
      and the four-sided expressions with equation 21, of
      Frontiers in Mechanical Engineering, 2025,
      [doi:10.3389/fmech.2025.1547819](https://doi.org/10.3389/fmech.2025.1547819).
      The `bo_pr` implementation is not a reference:
      it computes $\tau' = \sqrt{F/A}$ and gives $G = 10^7$ psi for brass.

This package provides
the data of the problem (`data`),
among which the catalog of the materials (`data.catalog`)
and the initial point (`data.initial_material` and `data.initial_dimensions`),
the design space (`design_space.create_design_space`),
the weldings (`Welding`)
and the discipline computing the cost and the constraints
from the properties of the material (`discipline.WeldedBeam`).
"""

from __future__ import annotations

from enum import StrEnum


class Welding(StrEnum):
    """The welding of the beam to its support."""

    TWO_SIDED = "two_sided"
    FOUR_SIDED = "four_sided"
