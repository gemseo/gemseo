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
#                           documentation
#        :author: Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""The settings of the numerical methods of a maximum-entropy distribution."""

from __future__ import annotations

from pydantic import Field
from pydantic import PositiveFloat
from pydantic import PositiveInt

from gemseo.util.pydantic import BasePydanticModel


class MaximumEntropyNumericalSettings(BasePydanticModel):
    """The settings of the numerical methods computing the distribution parameters.

    Some parameters have no closed-form expression and are computed by a solver,
    whose tolerances are set here,
    together with the number of expansions of the bracket enclosing a tilt
    and the dimensionless tilts below and above which
    the moments of a tilted uniform distribution
    are computed by a series or an asymptotic expansion
    instead of their closed-form expressions.
    The default values deliver the parameters to double precision
    for the supports and the moments met in practice
    and rarely need to be changed.
    """

    tilt_tolerance: PositiveFloat = Field(
        default=1e-12,
        lt=0.5,
        description="""The distance from the mean to the middle of the support,
relative to the length of the support,
below which a tilted uniform distribution is treated as a uniform one.

This distance corresponds to a dimensionless tilt of about twelve times this value.
Treating the distribution as a uniform one moves the mean to the middle of the support,
hence by at most this value times the length of the support.
The value is less than 0.5,
as every mean lies within half of the length of the support from its middle
and would thus be treated as its middle.""",
    )

    solver_tolerance: PositiveFloat = Field(
        default=1e-12,
        le=1e-6,
        description="""The absolute tolerance of the solvers
computing the distribution parameters.

For the tilt,
it applies to the dimensionless tilt,
i.e. the tilt times the length of the support,
whereas the truncated normal fit uses it
as a relative tolerance on the steps of its natural parameters.
Three thresholds derive from it.
The first one is the relative slack
within which a standard deviation is treated
as the largest achievable one on a bounded support,
that largest one being computed by the tilt solver.
The second one,
the square root of this tolerance,
is the acceptance threshold of the moment residuals of the truncated normal fit
and hence bounds the accuracy of that fit.
The third one is the margin beyond which a truncation bound is negligible.
The value is at most 1e-6,
beyond which these derived thresholds would exceed the quantities that they police.""",
    )

    small_tilt: PositiveFloat = Field(
        default=1e-2,
        le=0.1,
        description="""The absolute dimensionless tilt,
i.e. the tilt times the length of the support,
below which the series expansions are used.

The value is at most 0.1,
as the series and the closed-form expressions are equally accurate near 0.05.
There,
the truncation error of the series,
whose first omitted terms are of order seven for the mean and six for the variance,
meets the cancellation error of the closed-form expressions,
which grows as the machine epsilon divided by the tilt and by its square respectively.
The series thus loses about two digits whenever the tilt doubles.""",
    )

    large_tilt: PositiveFloat = Field(
        default=50.0,
        ge=45.0,
        le=700.0,
        description="""The absolute dimensionless tilt
above which the asymptotic expansions are used.

The value is at least 45,
below which these expansions are less accurate than double precision,
the variance being off by 20% at a tilt of 5 and by 8e-7 at a tilt of 20.
It is at most 700,
beyond which the closed-form expression of the mean
would overflow for a negative tilt.""",
    )

    maximum_solver_iterations: PositiveInt = Field(
        default=100,
        description="""The maximum number of expansions of the bracket enclosing a tilt.

A tilt that cannot be bracketed within this number raises a `ValueError`.""",
    )

    # The bounds of the two tilts guarantee that the small one is less than
    # the large one, which no validator has to check.
