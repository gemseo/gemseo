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
"""The variable names of the customizable Sellar MDO problem."""

from __future__ import annotations

from typing import Final

y_1: Final[str] = "y_1"
"""The name of the coupling variable computed by [Sellar1][gemseo.problem.mdo.sellar.sellar_1.Sellar1]."""  # noqa: E501

y_2: Final[str] = "y_2"
"""The name of the coupling variable computed by [Sellar2][gemseo.problem.mdo.sellar.sellar_2.Sellar2]."""  # noqa: E501

x_shared: Final[str] = "x_shared"
"""The name of the shared design variable."""

x_1: Final[str] = "x_1"
"""The name of the local design variable specific to [Sellar1][gemseo.problem.mdo.sellar.sellar_1.Sellar1]."""  # noqa: E501

x_2: Final[str] = "x_2"
"""The name of the local design variable specific to [Sellar2][gemseo.problem.mdo.sellar.sellar_2.Sellar2]."""  # noqa: E501

obj: Final[str] = "obj"
"""The name of the objective to minimize."""

c_1: Final[str] = "c_1"
"""The name of the constraint based on `"y_1"`."""

c_2: Final[str] = "c_2"
"""The name of the constraint based on `"y_2"`."""

alpha: Final[str] = "alpha"
"""The name of the tunable parameter in the constraint `"c_1"`."""

beta: Final[str] = "beta"
"""The name of the tunable parameter in the constraint `"c_2"`."""

gamma: Final[str] = "gamma"
"""The name of the tunable parameter in the discipline [Sellar1][gemseo.problem.mdo.sellar.sellar_1.Sellar1]."""  # noqa: E501
