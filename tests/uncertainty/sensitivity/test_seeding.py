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
from __future__ import annotations

import subprocess
import sys
from typing import TYPE_CHECKING

import pytest
from numpy import array
from numpy.testing import assert_array_equal
from openturns import RandomGenerator
from openturns import ResourceMap
from openturns import Uniform

from gemseo.uncertainty.sensitivity._seeding import is_initialized
from gemseo.uncertainty.sensitivity._seeding import seed_ot_random_generator

if TYPE_CHECKING:
    from collections.abc import Iterator

    from gemseo.util.typing import RealArray


@pytest.fixture
def restore_random_generator() -> Iterator[None]:
    """Restore the state of the OpenTURNS random generator after the test.

    The state of an unused generator cannot be restored,
    so the generator is then seeded with the initial seed of the `ResourceMap`,
    which gives the same state.
    """
    state = RandomGenerator.GetState()
    yield
    if is_initialized(state):
        RandomGenerator.SetState(state)
    else:
        RandomGenerator.SetSeed(
            ResourceMap.GetAsUnsignedInteger("RandomGenerator-InitialSeed")
        )


def draw_uniform_samples() -> RealArray:
    """Draw samples from the uniform distribution over [0, 1].

    Returns:
        The samples.
    """
    return array(Uniform(0.0, 1.0).getSample(1000)).ravel()


@pytest.mark.usefixtures("restore_random_generator")
def test_restore_state() -> None:
    """Check that the state of a used generator is restored on exit."""
    RandomGenerator.SetSeed(3)
    Uniform(0.0, 1.0).getSample(10)
    state = RandomGenerator.GetState()
    expected = draw_uniform_samples()
    RandomGenerator.SetState(state)
    with seed_ot_random_generator(1):
        draw_uniform_samples()

    assert_array_equal(draw_uniform_samples(), expected)


def test_restore_unused_generator() -> None:
    """Check that an unused generator is left in a valid state on exit.

    The state of a generator not used yet is not initialized
    and would make the generator draw invalid numbers if restored,
    e.g. -1 from the uniform distribution over [0, 1].
    On exit, the generator is instead in the state it would have been initialized to,
    i.e. the one given by the initial seed of the `ResourceMap`,
    which is 0 by default, as in the new process of this test.
    The test runs in a new process, whose generator is not used yet.
    """
    code = """
from numpy import array
from openturns import RandomGenerator
from openturns import Uniform
from gemseo.uncertainty.sensitivity._seeding import is_initialized
from gemseo.uncertainty.sensitivity._seeding import seed_ot_random_generator

assert not is_initialized(RandomGenerator.GetState())
with seed_ot_random_generator(1):
    Uniform(0.0, 1.0).getSample(10)

samples = array(Uniform(0.0, 1.0).getSample(1000))
RandomGenerator.SetSeed(0)
assert (samples == array(Uniform(0.0, 1.0).getSample(1000))).all()
"""
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.usefixtures("restore_random_generator")
def test_unused_generator_is_seeded_with_initial_seed(monkeypatch) -> None:
    """Check that an unused generator is seeded with the initial seed on exit.

    Unlike the test above, it runs in the current process,
    where the generator is already used,
    so `is_initialized` is patched to report its state as not initialized.
    The initial seed of the `ResourceMap` is set to a non-default value.
    """
    monkeypatch.setattr(
        "gemseo.uncertainty.sensitivity._seeding.is_initialized", lambda state: False
    )
    key = "RandomGenerator-InitialSeed"
    initial_seed = ResourceMap.GetAsUnsignedInteger(key)
    ResourceMap.SetAsUnsignedInteger(key, 42)
    try:
        RandomGenerator.SetSeed(42)
        expected = draw_uniform_samples()
        with seed_ot_random_generator(1):
            draw_uniform_samples()

        actual = draw_uniform_samples()
    finally:
        ResourceMap.SetAsUnsignedInteger(key, initial_seed)

    assert_array_equal(actual, expected)


@pytest.mark.usefixtures("restore_random_generator")
def test_initialized_state() -> None:
    """Check that the state of a seeded generator is initialized."""
    RandomGenerator.SetSeed(1)
    assert is_initialized(RandomGenerator.GetState())


@pytest.mark.usefixtures("restore_random_generator")
def test_no_seed() -> None:
    """Check that the generator is left untouched without seed."""
    RandomGenerator.SetSeed(3)
    state = RandomGenerator.GetState()
    expected = draw_uniform_samples()
    RandomGenerator.SetState(state)
    with seed_ot_random_generator(None) as reseeded:
        pass

    assert not reseeded
    assert_array_equal(draw_uniform_samples(), expected)
