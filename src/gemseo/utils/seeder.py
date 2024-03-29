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
"""A seed generator."""

from __future__ import annotations

SEED: int = 0
"""The default seed for random number generators."""


class Seeder:
    """A seed generator."""

    default_seed: int
    """The default seed."""

    def __init__(self, default_seed: int = SEED) -> None:
        """
        Args:
            default_seed: The initial default seed.
        """  # noqa: D205, D212
        self.default_seed = default_seed

    def get_seed(self, seed: int | None = None) -> int:
        """Return a seed.

        Args:
            seed: The seed to be returned.
                If ``None``,
                return ``initial_seed + i`` on the i-th call to this method,
                where ``initial_seed`` is the seed passed at instantiation.

        Returns:
            A seed.
        """
        self.default_seed += 1
        return self.default_seed if seed is None else seed
