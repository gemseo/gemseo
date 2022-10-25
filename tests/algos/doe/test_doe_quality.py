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
"""Tests for DOE quality."""
from __future__ import annotations

from operator import ge
from operator import gt
from operator import le
from operator import lt

import pytest
from gemseo.algos.doe.doe_quality import compute_discrepancy
from gemseo.algos.doe.doe_quality import compute_mindist_criterion
from gemseo.algos.doe.doe_quality import compute_phip_criterion
from gemseo.algos.doe.doe_quality import DOEMeasures
from gemseo.algos.doe.doe_quality import DOEQuality
from numpy import array
from numpy import ndarray


@pytest.fixture(scope="module")
def doe_measures():
    """A set of quality measures."""
    return DOEMeasures(1.0, 2.0, 3.0)


@pytest.fixture(scope="module")
def samples() -> ndarray:
    """Some space-filling samples."""
    return array([[0.0], [0.4], [1.0]])


@pytest.fixture(scope="module")
def other_samples() -> ndarray:
    """Other samples."""
    return array([[0.0], [0.1], [1.0]])


def test_doe_measures_str(doe_measures):
    """Check DOEMeasures string representations."""
    assert (
        repr(doe_measures)
        == str(doe_measures)
        == "DOEMeasures(discrepancy=1.0, mindist=2.0, phip=3.0)"
    )


def test_doe_quality_str(samples, doe_measures):
    """Check DOEQuality string representations."""
    quality = DOEQuality(samples)
    # Mock the measures with custom values for test purposes.
    quality.measures = doe_measures
    assert (
        repr(quality)
        == str(quality)
        == "DOEMeasures(discrepancy=1.0, mindist=2.0, phip=3.0)"
    )


def test_doe_measures_fields(doe_measures):
    """Check the fields of DOEMeasures."""
    assert doe_measures.discrepancy == 1.0
    assert doe_measures.mindist == 2.0
    assert doe_measures.phip == 3.0


@pytest.mark.parametrize("phip_1", [0.0, 1.0])
@pytest.mark.parametrize("phip_2", [0.0, 1.0])
@pytest.mark.parametrize("mindist_1", [0.0, 1.0])
@pytest.mark.parametrize("mindist_2", [0.0, 1.0])
@pytest.mark.parametrize("discr_1", [0.0, 1.0])
@pytest.mark.parametrize("discr_2", [0.0, 1.0])
def test_measures_comparison(
    phip_1, phip_2, mindist_1, mindist_2, discr_1, discr_2, samples
):
    """Check the logical operators of DOEQuality."""
    quality_1 = DOEQuality(samples)
    quality_2 = DOEQuality(samples)
    # Mock the measures with custom values for test purposes.
    quality_1.measures = DOEMeasures(phip_1, mindist_1, discr_1)
    quality_2.measures = DOEMeasures(phip_2, mindist_2, discr_2)
    assert (quality_1 == quality_2) == (
        (phip_1, mindist_1, discr_1) == (phip_2, mindist_2, discr_2)
    )

    transformations = [lambda x: x, lambda x: -x, lambda x: x]
    for operator, other_operator in {lt: gt, le: ge, gt: lt, ge: le}.items():
        if (
            sum(
                operator(t(x), t(y))
                for x, y, t in zip(
                    quality_1.measures, quality_2.measures, transformations
                )
            )
            / 3
            >= 0.5
        ):
            assert other_operator(quality_1, quality_2)


def test_doe_quality_measures(samples, other_samples):
    """Check the values of the measures associated with DOEQuality."""
    quality_1 = DOEQuality(samples)
    quality_2 = DOEQuality(other_samples)
    assert quality_1 > quality_2
    assert quality_1.measures.phip < quality_2.measures.phip
    assert quality_1.measures.mindist > quality_2.measures.mindist
    assert quality_1.measures.discrepancy < quality_2.measures.discrepancy


def test_compute_phip_criterion_default(samples):
    """Check compute_phip_criterion with default power."""
    assert compute_phip_criterion(samples) == pytest.approx(2.5, abs=0.1)


def test_compute_phip_criterion_custom(samples):
    """Check compute_phip_criterion with custom power."""
    assert compute_phip_criterion(samples, power=5) == pytest.approx(2.6, abs=0.1)


def test_compute_mindist_criterion(samples):
    """Check compute_mindist_criterion."""
    assert compute_mindist_criterion(samples) == 0.4


def test_compute_discrepancy_default(samples):
    """Check compute_discrepancy."""
    assert compute_discrepancy(samples) == pytest.approx(0.03, abs=0.01)


def test_compute_discrepancy_method(samples):
    """Check compute_discrepancy with custom method."""
    assert compute_discrepancy(samples, type_name="MD") == pytest.approx(0.05, abs=0.01)


def test_compute_discrepancy_option(samples):
    """Check compute_discrepancy with custom option."""
    assert compute_discrepancy(
        samples, type_name="MD", iterative=True
    ) == pytest.approx(0.14, abs=0.01)
