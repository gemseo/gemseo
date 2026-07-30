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
"""The disciplines computing array-based output data from array-based input data."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Final

from gemseo.util.package_import import install_lazy_reexport

if TYPE_CHECKING:
    # static visibility for mypy / IDEs
    from gemseo.core.discipline.discipline import Discipline  # noqa: F401
    from gemseo.discipline.analytic import AnalyticDiscipline  # noqa: F401
    from gemseo.discipline.array_based_function import (
        ArrayBasedFunctionDiscipline,  # noqa: F401
    )
    from gemseo.discipline.auto_py import AutoPyDiscipline  # noqa: F401
    from gemseo.discipline.chain.additive_chain import (
        AdditiveDisciplineChain,  # noqa: F401
    )
    from gemseo.discipline.chain.chain import DisciplineChain  # noqa: F401
    from gemseo.discipline.chain.initialization_chain import (
        InitializationDisciplineChain,  # noqa: F401
    )
    from gemseo.discipline.chain.parallel_chain import (
        ParallelDisciplineChain,  # noqa: F401
    )
    from gemseo.discipline.chain.warm_started_chain import (
        WarmStartedDisciplineChain,  # noqa: F401
    )
    from gemseo.discipline.concatenater import Concatenater  # noqa: F401
    from gemseo.discipline.constraint_aggregation import (
        ConstraintAggregation,  # noqa: F401
    )
    from gemseo.discipline.factory import DISCIPLINE_FACTORY  # noqa: F401
    from gemseo.discipline.linear_combination import LinearCombination  # noqa: F401
    from gemseo.discipline.ode.ode_discipline import ODEDiscipline  # noqa: F401
    from gemseo.discipline.remapping import RemappingDiscipline  # noqa: F401
    from gemseo.discipline.splitter import Splitter  # noqa: F401
    from gemseo.discipline.surrogate import SurrogateDiscipline  # noqa: F401
    from gemseo.discipline.taylor import TaylorDiscipline  # noqa: F401
    from gemseo.discipline.wrapper.disc_from_exe import DiscFromExe  # noqa: F401
    from gemseo.discipline.wrapper.filtering_discipline import (
        FilteringDiscipline,  # noqa: F401
    )
    from gemseo.discipline.wrapper.job_scheduler.lsf import LSF  # noqa: F401
    from gemseo.discipline.wrapper.job_scheduler.slurm import SLURM  # noqa: F401
    from gemseo.discipline.wrapper.retry_discipline import RetryDiscipline  # noqa: F401

# Exported name -> location (lazy-loaded on attribute access).
_NAME_TO_LOCATION: Final[dict[str, str]] = {
    "AdditiveDisciplineChain": "chain.additive_chain",
    "AnalyticDiscipline": "analytic",
    "ArrayBasedFunctionDiscipline": "array_based_function",
    "AutoPyDiscipline": "auto_py",
    "Concatenater": "concatenater",
    "ConstraintAggregation": "constraint_aggregation",
    "DiscFromExe": "wrapper.disc_from_exe",
    "DISCIPLINE_FACTORY": "factory",
    "Discipline": "gemseo.core.discipline.discipline:Discipline",
    "DisciplineChain": "chain.chain",
    "FilteringDiscipline": "wrapper.filtering_discipline",
    "InitializationDisciplineChain": "chain.initialization_chain",
    "LSF": "wrapper.job_scheduler.lsf",
    "LinearCombination": "linear_combination",
    "ODEDiscipline": "ode.ode_discipline",
    "ParallelDisciplineChain": "chain.parallel_chain",
    "RemappingDiscipline": "remapping",
    "RetryDiscipline": "wrapper.retry_discipline",
    "SLURM": "wrapper.job_scheduler.slurm",
    "Splitter": "splitter",
    "SurrogateDiscipline": "surrogate",
    "TaylorDiscipline": "taylor",
    "WarmStartedDisciplineChain": "chain.warm_started_chain",
}

install_lazy_reexport(globals(), _NAME_TO_LOCATION)
