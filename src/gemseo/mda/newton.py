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
# Contributors:
#    INITIAL AUTHORS - API and implementation and/or documentation
#        :author: Charlie Vanaret, Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""A set of Newton algorithm variants for solving MDAs.

Root finding methods include:

- `Newton-Raphson <https://en.wikipedia.org/wiki/Newton%27s_method>`__
- `quasi-Newton methods <https://en.wikipedia.org/wiki/Quasi-Newton_method>`__

Each of these methods is implemented by a class in this module.
All inherit from the common abstract MDARoot.
"""

from __future__ import annotations

from gemseo.mda.newton_raphson import MDANewtonRaphson  # noqa: F401
from gemseo.mda.quasi_newton import MDAQuasiNewton  # noqa: F401
from gemseo.mda.root import MDARoot  # noqa: F401

# TODO: API: remove.
