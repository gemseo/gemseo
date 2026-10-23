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
from openturns import ResourceMap

if TYPE_CHECKING:
    from collections.abc import Generator

    from openturns import RandomGeneratorState


def is_initialized(state: RandomGeneratorState) -> bool:
    """Check whether a state of the OpenTURNS random generator is initialized.

    The generator is initialized at its first use,
    so the state of an unused generator is a buffer of zeros but one word.
    Restoring this state would make the generator draw invalid numbers,
    e.g. -1 from the uniform distribution over $[0, 1]$.
    The buffer of an initialized state is filled with random words.

    Args:
        state: The state of the generator.

    Returns:
        Whether the state is initialized.
    """
    return sum(1 for word in state.getBuffer() if word) > 1


@contextmanager
def seed_ot_random_generator(seed: int | None) -> Generator[bool, None, None]:
    """Temporarily seed the OpenTURNS random generator.

    On exit, the generator state is restored to what it was before entering,
    even if an exception is raised inside the context;
    an unused generator is left in the state it would have been initialized to,
    i.e. the one given by the initial seed of the `ResourceMap`
    (`"RandomGenerator-InitialSeed"`).

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
        if is_initialized(state):
            RandomGenerator.SetState(state)
        else:
            # The initial seed of the ResourceMap gives the state
            # the generator would have been initialized to.
            RandomGenerator.SetSeed(
                ResourceMap.GetAsUnsignedInteger("RandomGenerator-InitialSeed")
            )
