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
#    INITIAL AUTHORS - initial API and implementation and/or initial
#                         documentation
#        :author: Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""The default settings of the scalable problem."""

from __future__ import annotations

from typing import Final

DEFAULT_N_DISCIPLINES: Final[int] = 2
r"""The default number :math:`N` of scalable disciplines."""

DEFAULT_D_0: Final[int] = 1
r"""The default size of the shared design variable :math:`x_0`."""

DEFAULT_D_I: Final[int] = 1
r"""The default size of the local design variable :math:`x_i`."""

DEFAULT_P_I: Final[int] = 1
r"""The default size of the coupling variable :math:`y_i`."""
