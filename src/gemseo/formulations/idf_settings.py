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
"""Settings of the IDF formulation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from pydantic import Field
from pydantic import PositiveInt
from pydantic import model_validator

from gemseo.formulations.base_formulation_settings import BaseFormulationSettings
from gemseo.mda.mda_chain import MDAChain
from gemseo.mda.mda_chain_settings import MDAChain_Settings  # noqa: TC001
from gemseo.typing import StrKeyMapping  # noqa: TC001

if TYPE_CHECKING:
    from typing_extensions import Self


class IDF_Settings(BaseFormulationSettings):  # noqa: N801
    """Settings of the :class:`.IDF` formulation."""

    _TARGET_CLASS_NAME = "IDF"

    include_weak_coupling_targets: bool = Field(
        default=True,
        description="""If ``True``,
all disciplines are executed in parallel,
and all couplings (weak and strong) are set as target variables in the design space.
This maximizes the exploitation of the parallelism
but leads to a larger design space,
so usually more iterations by the optimizer.
Otherwise,
the coupling graph is analyzed
and the disciplines are chained in sequence and in parallel to solve all weak couplings.
In this case,
only the strong couplings are used as target variables in the design space.
This reduces the size of the optimization problem,
so usually leads to less iterations.
The best option depends on the number of strong vs weak couplings,
the availability of gradients,
the availability of CPUs versus the number of disciplines,
so it is very context dependant.
Otherwise, IDF will consider only the strong coupling targets.""",
    )

    mda_chain_settings_for_start_at_equilibrium: StrKeyMapping | MDAChain_Settings = (
        Field(
            default_factory=dict,
            description="""The settings for the MDA when ``start_at_equilibrium=True``.

See detailed settings in :class:`.MDAChain`.""",
        )
    )

    n_processes: PositiveInt = Field(
        default=1,
        description="""The maximum simultaneous number of threads
if ``use_threading`` is ``True``, or processes otherwise,
used to parallelize the execution.""",
    )

    normalize_constraints: bool = Field(
        default=True,
        description=(
            "Whether the outputs of the coupling consistency constraints are scaled."
        ),
    )

    start_at_equilibrium: bool = Field(
        default=False,
        description="Whether an MDA is used to initialize the coupling variables.",
    )

    use_threading: bool = Field(
        default=True,
        description="""Whether to use threads instead of processes
to parallelize the execution when ``include_weak_coupling_targets`` is ``True``;
multiprocessing will copy (serialize) all the disciplines,
while threading will share all the memory.
This is important to note
if you want to execute the same discipline multiple times,
you shall use multiprocessing.""",
    )

    @model_validator(mode="after")
    def __mda_chain_settings_to_pydantic_model(self) -> Self:
        """Convert the MDA chain settings into a Pydantic model."""
        mda_chain_settings = self.mda_chain_settings_for_start_at_equilibrium
        if isinstance(mda_chain_settings, Mapping):
            self.mda_chain_settings_for_start_at_equilibrium = MDAChain.Settings(
                **mda_chain_settings
            )
        return self
