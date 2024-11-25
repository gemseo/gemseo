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
"""Settings for MDAChain."""

from __future__ import annotations

from collections.abc import Sequence  # Noqa: TC003
from typing import ClassVar  # Noqa: TC003

from pydantic import Field

from gemseo.core.coupling_structure import CouplingStructure  # Noqa: TC001
from gemseo.mda.base_mda_settings import BaseMDASettings  # noqa: TC001
from gemseo.mda.base_parallel_mda_settings import BaseParallelMDASettings  # Noqa: TC001
from gemseo.mda.composed_mda_settings import ComposedMDASettings
from gemseo.typing import StrKeyMapping  # noqa: TC001


class MDAChain_Settings(BaseParallelMDASettings, ComposedMDASettings):  # noqa: N801
    """The settings for :class:`.MDAChain`."""

    chain_linearize: bool = Field(
        default=False,
        description="""Whether to linearize the chain of execution.

Otherwise,
linearize the overall MDA with base class method.
This last option is preferred to minimize computations in adjoint mode,
while in direct mode, linearizing the chain may be cheaper.""",
    )

    inner_mda_name: str = Field(
        default="MDAJacobi",
        description="""The class name of the inner-MDA.""",
    )

    inner_mda_settings: StrKeyMapping | BaseMDASettings = Field(
        default_factory=dict, description="The settings for the inner MDAs."
    )

    initialize_defaults: bool = Field(
        default=False,
        description="""Whether to create a :class:`.MDOInitializationChain`
to compute the eventually missing :attr:`.default_inputs` at the first
execution.""",
    )

    mdachain_parallel_settings: StrKeyMapping = Field(
        default_factory=dict,
        description="""The settings of the MDOParallelChain instances, if any.""",
    )

    mdachain_parallelize_tasks: bool = Field(
        default=False,
        description="""Whether to parallelize tasks, if relevant.""",
    )

    sub_coupling_structures: Sequence[CouplingStructure] = Field(
        default=(),
        description="""The coupling structures to be used by the inner MDAs.

If empty, they are created from ``disciplines``.""",
    )

    _settings_names_to_be_cascaded: ClassVar[Sequence[str]] = [
        "tolerance",
        "max_mda_iter",
        "log_convergence",
    ]
    """The settings that must be cascaded to the inner MDAs."""
