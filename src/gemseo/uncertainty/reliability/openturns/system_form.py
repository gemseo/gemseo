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
"""First-order reliability method (FORM) for an intersection of events."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import ClassVar
from typing import Final

from openturns import SystemFORM

from gemseo.uncertainty.reliability.openturns.form import OT_FORM
from gemseo.uncertainty.reliability.openturns.system_form_settings import (
    OT_SystemFORM_Settings,
)

if TYPE_CHECKING:
    from openturns import Analytical

_collinear_error_fragment: Final[str] = (
    "delta=rho12^2+rho13^2+rho23^2-2*rho12*rho13*rho23"
)
"""The fragment of the OpenTURNS error raised for a singular correlation matrix.

For an intersection of three elementary events,
`ot.SystemFORM` computes a trivariate normal probability
with the algorithm of A. Genz,
"Numerical computation of rectangular bivariate and trivariate normal
and t probabilities",
Statistics and Computing 14 (2004) 251-260,
https://doi.org/10.1023/B:STCO.0000035304.20635.31.
Its correlation matrix `R` is the Gram matrix
of the unit directions of the design points,
i.e. of the unit normals of the linearized limit states,
so it is positive semidefinite
and `delta = 1 - det R`,
which is `rho12^2+rho13^2+rho23^2-2*rho12*rho13*rho23`,
is at most 1.
OpenTURNS checks `delta <= 1`, i.e. `det R >= 0`
(https://github.com/openturns/openturns/blob/ab675309b0e8ebe62713b96f5b82b77fe3135599/lib/src/Uncertainty/Distribution/Normal3DCDF.cxx#L124).
This check fails, and `ot.SystemFORM` raises this error,
only when rounding makes `R` look indefinite,
i.e. when `det R` is about 0 because the three normals are linearly dependent,
e.g. coplanar.
The `ValueError` raised from this error catches some singular cases, not all of them:
when rounding keeps `det R >= 0`,
or for an intersection of more than three elementary events,
which OpenTURNS computes without this check,
the analysis returns a probability without any error,
and this probability may be wrong
(https://gitlab.com/gemseo/dev/gemseo/-/issues/2080).
"""


class OT_SystemFORM(OT_FORM):  # noqa: N801
    """The first-order reliability method (FORM) for a combination of events."""

    settings_class: ClassVar[type[OT_SystemFORM_Settings]] = OT_SystemFORM_Settings

    _algo_class: ClassVar[type[SystemFORM]] = SystemFORM

    support_elementary_event_combinations: ClassVar[bool] = True

    _use_multiform_result: ClassVar[bool] = True

    def _run_ot_algo(self, algo: Analytical, event_name: str) -> None:
        """Run the OpenTURNS algorithm.

        Args:
            algo: The OpenTURNS algorithm.
            event_name: The name of the event.

        Raises:
            ValueError: When the normals of the linearized limit states
                of three elementary events are linearly dependent,
                so that their correlation matrix is singular
                and rounding makes it look indefinite.
        """
        try:
            super()._run_ot_algo(algo, event_name)
        except TypeError as error:
            if _collinear_error_fragment not in str(error):
                raise

            msg = (
                f"The probability of the event '{event_name}' "
                f"cannot be computed with {self.__class__.__name__} because "
                "the normals of the linearized limit states "
                "of three of its elementary events are linearly dependent, "
                "so their correlation matrix is singular."
            )
            raise ValueError(msg) from error
