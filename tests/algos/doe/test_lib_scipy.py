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

from unittest import mock

import packaging
import pytest
import scipy
from numpy.testing import assert_equal

from gemseo.algos.doe.lib_scipy import SciPyDOE
from gemseo.algos.doe.lib_scipy import _MonteCarlo


@pytest.fixture()
def library() -> SciPyDOE:
    """An instance of SciPyDOE."""
    return SciPyDOE()


def test_get_options(library):
    """Check that _get_options passed the right values to _process_options."""
    with mock.patch.object(library, "_process_options") as mock_method:
        library._get_options()

    assert mock_method.call_args.kwargs == {
        "max_time": 0,
        "eval_jac": False,
        "n_processes": 1,
        "wait_time_between_samples": 0.0,
        "n_samples": 1,
        "seed": None,
        "centered": False,
        "scramble": True,
        "radius": 0.05,
        "hypersphere": "volume",
        "ncandidates": 30,
        "optimization": None,
        "bits": None,
        "strength": 1,
    }


def test_remove_recent_scipy_options(library):
    """Check that the method removing not yet available SciPy options works."""
    original_option_names = ["a", "b", "c"]
    option_names = original_option_names.copy()

    _version = packaging.version.parse(scipy.__version__)
    release = _version.release
    prev_release_name = f"{release[0]}.{release[1] - 1}.0"
    next_release_name = f"{release[0]}.{release[1] + 1}.0"

    library._SciPyDOE__remove_recent_scipy_options(option_names, "b", _version.public)
    assert option_names == original_option_names

    library._SciPyDOE__remove_recent_scipy_options(option_names, "b", prev_release_name)
    assert option_names == original_option_names

    library._SciPyDOE__remove_recent_scipy_options(option_names, "b", next_release_name)
    assert option_names == ["a", "c"]


def check_option_filtering(option_name, target_version, current_version, caplog):
    """Check that the options not yet available in SciPy are correctly removed."""
    text = (
        f"Removed the option {option_name} "
        f"which is only available from SciPy {target_version}."
    )
    is_old_version = packaging.version.parse(current_version) < packaging.version.parse(
        target_version
    )
    assert (text in caplog.text) is is_old_version


@pytest.mark.parametrize("algo_name", ["Sobol", "Halton", "MC", "LHS", "PoissonDisk"])
@pytest.mark.parametrize("version", ["1.7", "1.8", "1.9", "1.10"])
@pytest.mark.parametrize("seed", [None, 3])
def test_generate_samples(library, algo_name, version, seed, caplog):
    """Check the generation of samples."""
    dimension = 2
    n_samples = 3
    library.algo_name = algo_name
    options = library._update_algorithm_options(n_samples=n_samples)
    options["seed"] = seed

    scipy_version = library._SciPyDOE__SCIPY_VERSION
    library._SciPyDOE__SCIPY_VERSION = packaging.version.parse(version)
    samples = library._generate_samples(dimension=dimension, **options)
    library._SciPyDOE__SCIPY_VERSION = scipy_version

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


def test_monte_carlo():
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


@pytest.mark.parametrize(
    "kwargs", [{}, {"optimization": SciPyDOE.Optimizer.NONE}, {"optimization": ""}]
)
def test_no_optimizer(library, kwargs):
    """Check that _get_options converts SciPyDOE.Optimizer.NONE to None."""
    library.init_options_grammar("HALTON")
    assert library._get_options(**kwargs)["optimization"] is None
