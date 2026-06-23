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
"""Seeding utilities for the OpenTURNS random generator."""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING

from openturns import RandomGenerator

if TYPE_CHECKING:
    from collections.abc import Iterator


@contextmanager
def seed_ot_random_generator(seed: int | None) -> Iterator[bool]:
    """Temporarily seed the OpenTURNS random generator.

    On exit, the generator state is restored to what it was before entering,
    even if an exception is raised inside the context.

    Args:
        seed: The seed for reproducible results.
            If `None`, the generator is left untouched (no reseeding, no restoring).

    Yields:
        Whether the generator was reseeded, i.e. `seed` is not `None`.
    """
    if seed is None:
        yield False
        return

    state = RandomGenerator.GetState()
    RandomGenerator.SetSeed(seed)
    try:
        yield True
    finally:
        RandomGenerator.SetState(state)
