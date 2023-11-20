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
# Copyright 2022 IRT Saint Exupéry, https://www.irt-saintexupery.com
# Contributors:
#    INITIAL AUTHORS - API and implementation and/or documentation
#        :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Derivation modes for the GEMSEO processes."""

from __future__ import annotations

from strenum import StrEnum


class DerivationMode(StrEnum):
    """The derivation modes."""

    DIRECT = "direct"
    """The direct Jacobian accumulation, chain rule from inputs to outputs, or
    derivation of an MDA that solves one system per input."""

    REVERSE = "reverse"
    """The reverse Jacobian accumulation, chain rule from outputs to inputs."""

    ADJOINT = "adjoint"
    """The adjoint resolution mode for MDAs, solves one system per output."""

    AUTO = "auto"
    """Automatic switch between direct, reverse or adjoint depending on data sizes."""
