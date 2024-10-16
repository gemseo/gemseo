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
"""Tests for SciPyDOE."""

from __future__ import annotations

import pytest
from numpy.testing import assert_equal
from packaging.version import parse as parse_version

from gemseo.algos.design_space import DesignSpace
from gemseo.algos.doe.scipy import scipy_doe
from gemseo.algos.doe.scipy.scipy_doe import SciPyDOE
from gemseo.algos.doe.scipy.scipy_doe import _MonteCarlo
from gemseo.utils.compatibility import scipy


@pytest.fixture
def library() -> SciPyDOE:
    """An instance of SciPyDOE."""
    return SciPyDOE("LHS")


def test_remove_recent_scipy_options(library) -> None:
    """Check that the method removing not yet available SciPy options works."""
    original_option_names = ["a", "b", "c"]
    option_names = original_option_names.copy()

    release = scipy.SCIPY_VERSION.release
    prev_release_name = f"{release[0]}.{release[1] - 1}.0"
    next_release_name = f"{release[0]}.{release[1] + 1}.0"

    library._SciPyDOE__remove_recent_scipy_options(
        option_names, "b", scipy.SCIPY_VERSION.public
    )
    assert option_names == original_option_names

    library._SciPyDOE__remove_recent_scipy_options(option_names, "b", prev_release_name)
    assert option_names == original_option_names

    library._SciPyDOE__remove_recent_scipy_options(option_names, "b", next_release_name)
    assert option_names == ["a", "c"]


def check_option_filtering(
    option_name, target_version, current_version, caplog
) -> None:
    """Check that the options not yet available in SciPy are correctly removed."""
    text = (
        f"Removed the option {option_name} "
        f"which is only available from SciPy {target_version}."
    )
    is_old_version = parse_version(current_version) < parse_version(target_version)
    assert (text in caplog.text) is is_old_version


@pytest.mark.parametrize("algo_name", ["Sobol", "Halton", "MC", "LHS", "PoissonDisk"])
@pytest.mark.parametrize("version", ["1.7", "1.8", "1.9", "1.10", "1.11", "1.12"])
@pytest.mark.parametrize("seed", [None, 3])
def test_generate_samples(algo_name, version, seed, caplog, monkeypatch) -> None:
    """Check the generation of samples."""
    dimension = 2
    n_samples = 3
    library = SciPyDOE(algo_name)
    options = library._validate_settings(n_samples=n_samples)
    options["seed"] = seed

    scipy_version = scipy_doe.SCIPY_VERSION
    scipy_doe.SCIPY_VERSION = parse_version(version)
    if scipy_version >= parse_version("1.12") and "centered" in options:
        del options["centered"]
    variables_space = DesignSpace()
    variables_space.add_variable("x", size=dimension)
    samples = library._generate_unit_samples(variables_space, **options)
    scipy_doe.SCIPY_VERSION = scipy_version

    if algo_name == "Sobol":
        check_option_filtering("bits", "1.9", version, caplog)
        check_option_filtering("optimization", "1.10", version, caplog)
    elif algo_name == "Halton":
        check_option_filtering("optimization", "1.10", version, caplog)
    elif algo_name == "LHS":
        check_option_filtering("scramble", "1.10", version, caplog)
        check_option_filtering("optimization", "1.8", version, caplog)
        check_option_filtering("strength", "1.8", version, caplog)
    elif algo_name == "PoissonDisk":
        check_option_filtering("optimization", "1.10", version, caplog)

    assert samples.shape == (n_samples, dimension)


def test_monte_carlo() -> None:
    """Check that the class _MonteCarlo works properly."""
    monte_carlo = _MonteCarlo(3, 4)
    samples = monte_carlo.random(2)
    assert samples.shape == (2, 3)

    monte_carlo.reset()
    new_samples = monte_carlo.random(2)
    assert_equal(samples, new_samples)

    monte_carlo.reset()
    monte_carlo.fast_forward(1)
    new_samples = monte_carlo.random(1)
    assert_equal(new_samples, samples[[1]])

    monte_carlo = _MonteCarlo(3, 5)
    new_samples = monte_carlo.random(2)
    with pytest.raises(AssertionError):
        assert_equal(samples, new_samples)
