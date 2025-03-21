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
"""Settings for the SciPy linear programming algorithms."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import ClassVar
from typing import Literal

from pydantic import Field
from pydantic import model_validator

from gemseo.algos.opt.base_optimizer_settings import BaseOptimizerSettings

if TYPE_CHECKING:
    from typing_extensions import Self


class BaseSciPyLinProgSettings(BaseOptimizerSettings):
    """The SciPy linear programming library setting."""

    autoscale: bool = Field(
        default=False,
        description="""Whether to perform auto-scaling of the constraints.""",
    )

    disp: bool = Field(
        default=False,
        description="""Whether to print convergence messages.""",
    )

    presolve: bool = Field(
        default=True,
        description=(
            """Whether to perform a preliminary analysis on the problem before solving.

It attempts to detect infeasibility, unboundedness or problem simplifications."""
        ),
    )

    rr: bool = Field(
        default=True,
        description="""Whether to remove linearly dependent equality-constraints.""",
    )

    rr_method: Literal["SVD", "pivot", "ID"] | None = Field(
        default=None,
        description="""The method to remove redundancy, either 'SVD', 'pivot' or 'ID'.

If ``None``, use “SVD” if the matrix is nearly full rank. If not, uses
“pivot”. The behavior of this default is subject to change without prior notice.""",
    )

    _redundant_settings: ClassVar[list[str]] = ["maxiter", "tol"]

    @model_validator(mode="after")
    def __check_scaling(self) -> Self:
        """Use ``autoscale`` when ``scaling_threshold`` is set.

        The scaling of outputs by GEMSEO has no effect on ``ScipyLinprog`` because
        it does not scale the coefficients of linear functions. This validation ensures
        the ``autoscale`` setting is used instead.
        """
        if self.scaling_threshold is not None:
            self.autoscale = True
        return self
