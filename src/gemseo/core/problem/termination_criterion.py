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
#    INITIAL AUTHORS - API and implementation and/or documentation
#        :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Generic termination criteria for drivers."""

from __future__ import annotations

from typing import ClassVar


class TerminationCriterion(Exception):  # noqa: N818
    """Stop driver for some reason."""

    _MESSAGE: ClassVar[str] = ""
    """The default message describing the termination criterion."""

    @property
    def message(self) -> str:
        """The message describing why the driver stopped.

        The first exception argument if any, the default message otherwise.
        """
        return str(self.args[0]) if self.args else self._MESSAGE


class FunctionIsNan(TerminationCriterion):  # noqa: N818
    """Stops driver when a function has NaN value or NaN Jacobian."""

    _MESSAGE: ClassVar[str] = (
        "Function value or gradient or constraint is NaN, "
        "and problem.stop_if_nan is set to True. "
    )


class DesvarIsNan(TerminationCriterion):  # noqa: N818
    """Stops driver when the design variables are nan."""

    _MESSAGE: ClassVar[str] = "Design variables are NaN. "


class MaxIterReachedException(TerminationCriterion):  # noqa: N818
    """Exception raised when the maximum number of iterations is reached."""

    _MESSAGE: ClassVar[str] = "Maximum number of iterations reached. "


class MaxTimeReached(TerminationCriterion):  # noqa: N818
    """Exception raised when the maximum execution time is reached."""

    _MESSAGE: ClassVar[str] = "Maximum time reached. "


class FtolReached(TerminationCriterion):  # noqa: N818
    """Exception raised when the f_tol_rel or f_tol_abs criteria is reached."""

    _MESSAGE: ClassVar[str] = (
        "Successive iterates of the objective function "
        "are closer than ftol_rel or ftol_abs. "
    )


class XtolReached(TerminationCriterion):  # noqa: N818
    """Exception raised when the x_tol_rel or x_tol_abs criteria is reached."""

    _MESSAGE: ClassVar[str] = (
        "Successive iterates of the design variables "
        "are closer than xtol_rel or xtol_abs. "
    )
