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
r"""The Ishigami use case to benchmark and illustrate UQ algorithms.

The Isighami function
$f(x_1,_2,x_3) = \sin(x_1)+ 7\sin(x_2)^2 + 0.1x_3^4\sin(X_1)$
is commonly studied through the random variable $Y=f(X_1,X_2,X_3)$
where $X_1$, $X_2$ and $X_3$ are independent random variables
uniformly distributed over $[-\pi,\pi]$.

!!! quote "References"

    T. Ishigami and T. Homma.
    An importance quantification technique
    in uncertainty analysis for computer models.
    In First International Symposium on Uncertainty Modeling and Analysis, 1990.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Final

from gemseo.util.package_import import install_lazy_reexport

if TYPE_CHECKING:
    # static visibility for mypy / IDEs
    from gemseo.problem.uncertainty.ishigami.ishigami_discipline import (
        IshigamiDiscipline,  # noqa: F401
    )
    from gemseo.problem.uncertainty.ishigami.ishigami_function import IshigamiFunction  # noqa: F401
    from gemseo.problem.uncertainty.ishigami.ishigami_problem import IshigamiProblem  # noqa: F401
    from gemseo.problem.uncertainty.ishigami.ishigami_space import IshigamiSpace  # noqa: F401

# Class name -> defining submodule (lazy-loaded on attribute access).
_NAME_TO_LOCATION: Final[dict[str, str]] = {
    "IshigamiDiscipline": "ishigami_discipline",
    "IshigamiFunction": "ishigami_function",
    "IshigamiProblem": "ishigami_problem",
    "IshigamiSpace": "ishigami_space",
}

install_lazy_reexport(globals(), _NAME_TO_LOCATION)
