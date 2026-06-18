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
    """The derivation modes of the GEMSEO processes.

    `DIRECT` and `ADJOINT` apply to an MDA, which solves one linear system per input
    (direct) or per output (adjoint). `FORWARD` and `REVERSE` apply to a chain of
    disciplines, accumulating the chain rule from inputs to outputs (forward) or from
    outputs to inputs (reverse). `AUTO` lets GEMSEO switch automatically depending on
    the data sizes.

    The MDA-specific modes are gathered in
    [MDADerivationMode][gemseo.core.derivatives.jacobian_assembly.MDADerivationMode]
    and the chain-specific modes in
    [ChainDerivationMode][gemseo.core.chains.chain.ChainDerivationMode].
    """

    DIRECT = "direct"
    """The direct resolution mode for an MDA, solving one linear system per input."""

    ADJOINT = "adjoint"
    """The adjoint resolution mode for an MDA, solving one linear system per output."""

    FORWARD = "forward"
    """The forward chain rule for a chain, accumulating from inputs to outputs."""

    REVERSE = "reverse"
    """The reverse chain rule for a chain, accumulating from outputs to inputs."""

    AUTO = "auto"
    """Automatic switch between the modes depending on the data sizes."""
