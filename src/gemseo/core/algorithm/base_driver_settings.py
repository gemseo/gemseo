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
"""Settings for the driver library."""

from __future__ import annotations

from pydantic import Field
from pydantic import NonNegativeFloat
from pydantic import PositiveInt

from gemseo.core.algorithm.progress_bar_data.data import ProgressBarData
from gemseo.core.algorithm.progress_bar_data.factory import ProgressBarDataName
from gemseo.util.pydantic import BaseSettings


class BaseDriverSettings(BaseSettings):
    """The common parameters for all driver libraries."""

    enable_progress_bar: bool | None = Field(
        default=None,
        description=(
            """Whether to enable the progress bar in the optimization log.

If `None`, use the global value of `enable_progress_bar` (see the
`configure` function to change it globally)."""
        ),
    )

    eq_tolerance: NonNegativeFloat = Field(
        default=1e-2,
        description="""The tolerance on the equality constraints.""",
    )

    evaluate_observable_jacobian: bool = Field(
        default=False,
        description="Whether to evaluate the Jacobian of the observables.",
    )

    ineq_tolerance: NonNegativeFloat = Field(
        default=1e-4,
        description="""The tolerance on the inequality constraints.""",
    )

    log_problem: bool = Field(
        default=True,
        description="""Whether to log the definition and result of the problem.""",
    )

    max_input_space_dimension_to_log: PositiveInt = Field(
        default=40,
        description="The maximum dimension of an input space "
        "to be logged. "
        "If this number is higher than the dimension of the input space "
        "then the input space will not be logged.",
    )

    max_time: NonNegativeFloat = Field(
        default=0.0,
        description="""The maximum runtime in seconds, disabled if 0.""",
    )

    normalize_design_space: bool = Field(
        default=True,
        description=(
            """Whether to normalize the design space variables between 0 and 1."""
        ),
    )

    progress_bar_data_name: ProgressBarDataName = Field(
        default=ProgressBarData.__name__,
        description="""The name of a
[BaseProgressBarData][gemseo.core.algorithm.progress_bar_data.base.BaseProgressBarData]
class to define the data of an evaluation problem to be displayed
in the progress bar.""",
    )

    relax_discrete_variables: bool = Field(
        default=False,
        description="""Whether to relax the discrete variables
that the algorithm does not handle to float ones,
so that it can be run on them
and explores them continuously between their smallest and largest choices.
The optimum written back into the design space
is snapped to the nearest choice.
Otherwise,
such an algorithm raises an error on a discrete variable.
This has no effect on discrete variables the algorithm handles,
e.g. those of a DOE.""",
    )

    relax_integer_variables: bool = Field(
        default=False,
        description="""Whether to relax the integer variables
that the algorithm does not handle to float ones,
so that it can be run on them
and explores them continuously between their bounds.
The optimum written back into the design space is rounded.
Otherwise,
such an algorithm raises an error on an integer variable.
This has no effect on integer variables the algorithm handles,
e.g. those of a DOE.""",
    )

    reset_iteration_counters: bool = Field(
        default=True,
        description=(
            """Whether to reset the iteration counters before each execution."""
        ),
    )

    store_jacobian: bool = Field(
        default=True,
        description="""Whether to store the Jacobian matrices in the database.

This argument is ignored when the `use_database` option is `False`.
If a gradient-based algorithm is used,
this option cannot be set along with kkt options.""",
    )

    use_database: bool = Field(
        default=True,
        description="""Whether to wrap the functions in the database.""",
    )

    use_one_line_progress_bar: bool = Field(
        default=False,
        description="""Whether to log the progress bar on a single line.""",
    )
