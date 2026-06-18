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
"""The settings of the QMC algorithm using the Faure sequence."""

from __future__ import annotations

from typing import ClassVar

from openturns import FaureSequence

from gemseo.uncertainty.reliability.openturns.base_qmc_settings import BaseOTQMCSettings


class OT_Faure_Settings(BaseOTQMCSettings):  # noqa: N801
    """The settings of the QMC algorithm using the Faure sequence."""

    _SEQUENCE_CLASS: ClassVar[type[FaureSequence]] = FaureSequence
