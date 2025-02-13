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
"""Settings for the mNBI algorithm."""

from __future__ import annotations

from collections.abc import Sequence  # noqa: TC003
from pathlib import Path  # noqa: TC003
from typing import TYPE_CHECKING

from pydantic import Field
from pydantic import NonNegativeInt
from pydantic import PositiveInt
from pydantic import field_validator
from pydantic import model_validator

from gemseo.algos.opt.base_optimizer_settings import BaseOptimizerSettings
from gemseo.typing import StrKeyMapping  # noqa: TC001
from gemseo.utils.pydantic_ndarray import NDArrayPydantic  # noqa: TC001

if TYPE_CHECKING:
    from typing_extensions import Self


class MNBI_Settings(BaseOptimizerSettings):  # noqa: N801
    """The mNBI algorithm settings."""

    _TARGET_CLASS_NAME = "MNBI"

    normalize_design_space: bool = Field(
        default=False,
        description="""Whether to normalize the design space variables between 0 and 1.

The mNBI algorithm does not allow to normalize the design space at the top
level, only the sub-optimizations accept design space normalization. To do
this, pass the setting ``normalize_design_space`` to
``sub_optim_algo_settings``.""",
    )

    sub_optim_algo: str = Field(
        description=(
            "The optimization algorithm used to solve the sub-optimization problems."
        )
    )

    n_sub_optim: PositiveInt = Field(
        default=1,
        description="""The number of sub-optimizations points.

mNBI generates ``n_sub_optim`` points on the Pareto front between the
`n-objective` individual minima. This value must be strictly greater than
the number of objectives of the problem.""",
    )

    sub_optim_algo_settings: StrKeyMapping = Field(
        default_factory=dict,
        description="""The settings for the sub optimization algorithm.""",
    )

    sub_optim_max_iter: NonNegativeInt = Field(
        default=0,
        description="""The maximum number of iterations of the sub-optimization algorithms.

If 0, the ``max_iter`` value is used.""",  # noqa: E501
    )

    doe_algo: str = Field(
        default="PYDOE_FULLFACT",
        description="""
            The design of experiments algo for the target points on the Pareto front.

A ``fullfactorial`` DOE is used default as these tend to be low dimensions,
usually not more than 3 objectives for a given problem.
This setting is relevant only for problems with more than 2 objectives.""",  # noqa: E501
    )

    doe_algo_settings: StrKeyMapping = Field(
        default_factory=dict,
        description="""The settings for the DOE algorithm.""",
    )

    debug: bool = Field(
        default=False,
        description=(
            """Whether to output the sub-optimization optima in a database hdf file."""
        ),
    )

    debug_file_path: str | Path = Field(
        default="debug_history.h5",
        description="""The path to the debug file if debug mode is active.""",
    )

    skip_betas: bool = Field(
        default=True,
        description="""Whether to skip the sub-optimizations of relevant.

The sub-optimizations are skipped if they correspond to values of beta for
which the theoretical result has already been found.""",
    )

    custom_anchor_points: Sequence[NDArrayPydantic] = Field(
        default=(),
        description=(
            """The bounding points of the custom phi simplex for the optimization."""
        ),
    )

    custom_phi_betas: Sequence[NDArrayPydantic] = Field(
        default=(),
        description=(
            r"The custom values of :math:`\Phi \beta` to be used in the optimization."
        ),
    )

    n_processes: PositiveInt = Field(
        default=1,
        description=(
            "The maximum number of processes used to parallelize the sub-optimizations."
        ),
    )

    @field_validator("normalize_design_space")
    @classmethod
    def __validate_normalize(cls, normalize_design_space: bool) -> bool:
        """Check that the normalization is disabled for the top-level problem."""
        if normalize_design_space:
            message = (
                "The mNBI algo does not allow to normalize the design space at "
                "the top level, only the sub optimizations accept design space "
                "normalization. To do this, pass the setting "
                "``normalize_design_space`` to ``sub_optim_algo_settings``."
            )
            raise ValueError(message)
        return normalize_design_space

    @model_validator(mode="after")
    def __validate_sub_optim_max_iter(self) -> Self:
        """Check if a sub_optim_max_iter was passed, otherwise use max_iter."""
        if self.sub_optim_max_iter == 0:
            self.sub_optim_max_iter = self.max_iter
        return self
