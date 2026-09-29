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
# Contributors:
#    INITIAL AUTHORS - initial API and implementation and/or
#                      initial documentation
#        :author:  Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
from numpy import allclose
from numpy import array
from numpy import inf
from numpy import int_
from numpy.testing import assert_allclose
from openturns import BlockIndependentCopula
from openturns import ClaytonCopula
from openturns import CorrelationMatrix
from openturns import EmpiricalBernsteinCopula
from openturns import FrankCopula
from openturns import IndependentCopula
from openturns import JointDistribution
from openturns import MarginalDistribution
from openturns import MarshallOlkinCopula
from openturns import Normal
from openturns import NormalCopula
from openturns import RandomGenerator
from openturns import Sample
from openturns import StudentCopula
from scipy.stats import norm

from gemseo.uncertainty.distribution.openturns.joint import OTJointDistribution
from gemseo.uncertainty.distribution.openturns.joint_settings import (
    OTJointDistribution_Settings,
)
from gemseo.uncertainty.distribution.openturns.normal_settings import (
    OTNormalDistribution_Settings,
)
from gemseo.uncertainty.distribution.openturns.uniform_settings import (
    OTUniformDistribution_Settings,
)
from gemseo.util.testing.helper import assert_exception

if TYPE_CHECKING:
    from gemseo.uncertainty.distribution.openturns.base_settings import (
        BaseOTMarginalDistributionSettings,
    )


def _create_correlation_matrix(rho: float) -> CorrelationMatrix:
    """Create a 2x2 correlation matrix.

    Args:
        rho: The off-diagonal correlation coefficient.

    Returns:
        The correlation matrix.
    """
    correlation = CorrelationMatrix(2)
    correlation[0, 1] = rho
    return correlation


@pytest.fixture(scope="module")
def correlation_3d() -> CorrelationMatrix:
    """A 3x3 correlation matrix with distinct off-diagonal coefficients."""
    correlation = CorrelationMatrix(3)
    correlation[0, 1] = 0.2
    correlation[0, 2] = 0.5
    correlation[1, 2] = -0.3
    return correlation


@pytest.fixture(scope="module")
def joint_distribution() -> OTJointDistribution:
    """The distribution of a 2D Gaussian vector with independent components."""
    return OTJointDistribution(
        OTJointDistribution_Settings(
            marginal_settings=[
                OTNormalDistribution_Settings(),
                OTNormalDistribution_Settings(),
            ]
        )
    )


@pytest.mark.parametrize(
    ("settings", "expected"),
    [
        pytest.param(
            OTJointDistribution_Settings(
                marginal_settings=[OTNormalDistribution_Settings()]
            ),
            "Normal(mu=0.0, sigma=1.0)",
            id="one_marginal",
        ),
        pytest.param(
            OTJointDistribution_Settings(
                marginal_settings=[OTNormalDistribution_Settings()] * 2
            ),
            (
                "OTJointDistribution("
                "Normal(mu=0.0, sigma=1.0), Normal(mu=0.0, sigma=1.0); "
                "IndependentCopula(dimension = 2))"
            ),
            id="two_marginals_independent_copula",
        ),
        pytest.param(
            OTJointDistribution_Settings(
                marginal_settings=[OTNormalDistribution_Settings()] * 2,
                copula=NormalCopula(2),
            ),
            (
                "OTJointDistribution("
                "Normal(mu=0.0, sigma=1.0), Normal(mu=0.0, sigma=1.0); "
                "NormalCopula(R = [[ 1 0 ]\n [ 0 1 ]]))"
            ),
            id="two_marginals_normal_copula",
        ),
    ],
)
def test_repr(settings, expected) -> None:
    """Check the string representation of a joint probability distribution."""
    assert repr(OTJointDistribution(settings)) == expected


def test_str(joint_distribution) -> None:
    """Check that the string representation matches the repr one."""
    assert str(joint_distribution) == repr(joint_distribution)


@pytest.mark.parametrize("n_samples", [3, int_(3)])
def test_get_sample(joint_distribution, n_samples) -> None:
    sample = joint_distribution.compute_samples(n_samples)
    assert sample.shape == (3, 2)


def test_get_cdf(joint_distribution) -> None:
    result = joint_distribution.compute_cdf(array([0.0, 0.0]))
    assert allclose(result, array([0.5, 0.5]))


def test_get_inverse_cdf(joint_distribution) -> None:
    result = joint_distribution.compute_inverse_cdf(array([0.5, 0.5]))
    assert allclose(result, array([0.0, 0.0]))


def test_transform_independent(joint_distribution) -> None:
    """Without copula, the Rosenblatt transformation reduces to the marginal CDFs."""
    point = array([0.3, -0.7])
    transformed = joint_distribution.map_to_uniform(point)
    assert allclose(transformed, joint_distribution.compute_cdf(point))
    assert allclose(joint_distribution.map_from_uniform(transformed), point)


def test_transform_with_copula() -> None:
    """With a copula, the Rosenblatt transformation accounts for the dependence."""
    distribution = OTJointDistribution(
        OTJointDistribution_Settings(
            marginal_settings=[
                OTNormalDistribution_Settings(),
                OTNormalDistribution_Settings(),
            ],
            copula=NormalCopula(_create_correlation_matrix(0.8)),
        )
    )
    point = array([0.3, -0.7])
    transformed = distribution.map_to_uniform(point)
    expected = array(distribution.distribution.computeSequentialConditionalCDF(point))
    assert_allclose(transformed, expected)
    # The dependence makes the Rosenblatt transformation differ from the marginal CDFs.
    assert not allclose(transformed, distribution.compute_cdf(point))
    # The transformation is invertible.
    assert_allclose(distribution.map_from_uniform(transformed), point)


def test_mean(joint_distribution) -> None:
    assert allclose(joint_distribution.mean, array([0.0, 0.0]))


def test_std(joint_distribution) -> None:
    assert allclose(joint_distribution.standard_deviation, array([1.0, 1.0]))


def test_support(joint_distribution) -> None:
    expectation = array([-inf, inf])
    for element in joint_distribution.support:
        assert allclose(element, expectation)


def test_range(joint_distribution) -> None:
    expectation = array([-7.650628, 7.650628])
    for element in joint_distribution.range:
        assert allclose(element, expectation, 1e-3)


@pytest.mark.parametrize(
    ("n_marginals", "copula"),
    [
        pytest.param(1, NormalCopula(2), id="copula_dimension"),
        pytest.param(
            1, (((0, 1), NormalCopula(2)),), id="copula_dimension_with_blocks"
        ),
        pytest.param(1, (((0,), NormalCopula(2)),), id="block_dimension"),
        pytest.param(2, (((0, 2), NormalCopula(2)),), id="indices"),
        pytest.param(
            2,
            (((0,), NormalCopula(1)), ((0,), NormalCopula(1))),
            id="duplication",
        ),
    ],
)
def test_settings_error(snapshot, n_marginals, copula) -> None:
    """Check the errors raised when building invalid joint distribution settings.

    The cases are:
    a copula dimension not matching the number of marginals,
    with a copula or block copulas,
    a block copula dimension not matching the number of its components,
    a block copula component out of range,
    and two block copulas on the same component.
    """
    with assert_exception(ValueError, snapshot):
        OTJointDistribution_Settings(
            marginal_settings=[OTNormalDistribution_Settings()] * n_marginals,
            copula=copula,
        )


@pytest.fixture
def four_marginal_settings() -> list[BaseOTMarginalDistributionSettings]:
    """The settings of four marginal distributions."""
    return [
        OTNormalDistribution_Settings(),
        OTUniformDistribution_Settings(),
        OTNormalDistribution_Settings(mu=2.0, sigma=3.0),
        OTUniformDistribution_Settings(minimum=-1.0, maximum=4.0),
    ]


@pytest.fixture
def five_marginal_settings(
    four_marginal_settings,
) -> list[BaseOTMarginalDistributionSettings]:
    """The settings of five marginal distributions."""
    return [*four_marginal_settings, OTNormalDistribution_Settings(mu=-1.0, sigma=0.5)]


def check_transformation(distribution: OTJointDistribution) -> None:
    """Check the Rosenblatt transformation of OpenTURNS against the GEMSEO one.

    Args:
        distribution: The joint distribution.
    """
    standard_point = array([0.3, -0.2, 1.1, 0.4, 0.5])[: distribution.dimension]
    inverse_transformation = (
        distribution.distribution.getInverseIsoProbabilisticTransformation()
    )
    physical_point = array(inverse_transformation(standard_point))
    assert_allclose(
        physical_point, distribution.map_from_uniform(norm.cdf(standard_point))
    )
    transformation = distribution.distribution.getIsoProbabilisticTransformation()
    assert_allclose(array(transformation(physical_point)), standard_point)


def test_block_copulas_normal(four_marginal_settings, correlation_3d) -> None:
    """Check that normal and independent block copulas are merged into a normal one.

    The 3D normal block copula is given out of order
    with distinct off-diagonal correlations,
    so that a wrong permutation of its correlation matrix would be caught,
    and its span contains the component of an independent block copula.
    """
    distribution = OTJointDistribution(
        OTJointDistribution_Settings(
            marginal_settings=four_marginal_settings,
            copula=(
                ((3, 0, 2), NormalCopula(correlation_3d)),
                ((1,), IndependentCopula(1)),
            ),
        )
    )
    copula = distribution.distribution.getCopula()
    assert copula.getImplementation().getClassName() == "NormalCopula"
    # The block maps the copula coordinates 0, 1, 2 to the components 3, 0, 2.
    expected = CorrelationMatrix(4)
    expected[0, 3] = 0.2
    expected[2, 3] = 0.5
    expected[0, 2] = -0.3
    assert_allclose(array(copula.getShapeMatrix()), array(expected))
    check_transformation(distribution)


@pytest.mark.parametrize("n_marginals", [3, 4])
def test_block_copulas_student_permutation(
    four_marginal_settings, correlation_3d, n_marginals
) -> None:
    """Check the permutation of a 3D Student block copula given out of order.

    Its correlation matrix is permuted
    so that OpenTURNS computes the Rosenblatt transformation analytically
    instead of using the permuted block-independent copula.
    """
    distribution = OTJointDistribution(
        OTJointDistribution_Settings(
            marginal_settings=four_marginal_settings[:n_marginals],
            copula=(((2, 0, 1), StudentCopula(4.0, correlation_3d)),),
        )
    )
    copula = distribution.distribution.getCopula().getImplementation()
    if n_marginals == 3:
        assert isinstance(copula, StudentCopula)
    else:
        assert isinstance(copula, BlockIndependentCopula)
        assert [
            piece.getImplementation().getClassName()
            for piece in copula.getCopulaCollection()
        ] == [
            "StudentCopula",
            "IndependentCopula",
        ]
        copula = copula.getCopulaCollection()[0].getImplementation()

    # The block maps the copula coordinates 0, 1, 2 to the components 2, 0, 1.
    expected = CorrelationMatrix(3)
    expected[0, 1] = -0.3
    expected[0, 2] = 0.2
    expected[1, 2] = 0.5
    assert_allclose(array(copula.getR()), array(expected))
    assert copula.getNu() == 4.0
    if n_marginals == 4:
        # OpenTURNS 1.27 overflows the stack when computing
        # the iso-probabilistic transformation of a joint distribution
        # whose copula is a StudentCopula.
        check_transformation(distribution)


@pytest.mark.parametrize(
    ("blocks", "expected_classes", "expected_dimensions"),
    [
        pytest.param(
            (
                ((0, 1), NormalCopula(_create_correlation_matrix(0.5))),
                ((3, 4), NormalCopula(_create_correlation_matrix(0.5))),
            ),
            ["NormalCopula", "IndependentCopula", "NormalCopula"],
            [2, 1, 2],
            id="non_overlapping",
        ),
        pytest.param(
            (
                ((0, 2), NormalCopula(_create_correlation_matrix(0.5))),
                ((1, 3), NormalCopula(_create_correlation_matrix(0.5))),
            ),
            ["NormalCopula", "IndependentCopula"],
            [4, 1],
            id="overlapping",
        ),
    ],
)
def test_block_copulas_normal_spans(
    five_marginal_settings, blocks, expected_classes, expected_dimensions
) -> None:
    """Check how normal block copula spans combine.

    Two normal block copulas whose spans do not overlap stay separate,
    the component in between being independent,
    while two overlapping ones merge into a normal copula spanning both.
    """
    distribution = OTJointDistribution(
        OTJointDistribution_Settings(
            marginal_settings=five_marginal_settings, copula=blocks
        )
    )
    copula = distribution.distribution.getCopula()
    assert isinstance(copula.getImplementation(), BlockIndependentCopula)
    pieces = copula.getImplementation().getCopulaCollection()
    assert [
        piece.getImplementation().getClassName() for piece in pieces
    ] == expected_classes
    assert [piece.getDimension() for piece in pieces] == expected_dimensions
    check_transformation(distribution)


@pytest.mark.parametrize(
    ("n_marginals", "blocks", "class_name"),
    [
        pytest.param(
            4, (((1, 2), IndependentCopula(2)),), "IndependentCopula", id="independent"
        ),
        pytest.param(
            2, (((0, 1), ClaytonCopula(2.0)),), "ClaytonCopula", id="whole_vector"
        ),
    ],
)
def test_block_copulas_single_copula(
    five_marginal_settings, n_marginals, blocks, class_name
) -> None:
    """Check that block copulas collapsing into a single copula give this copula.

    This happens when independent block copulas are merged
    with the free components into an independent copula,
    or when a single block copula covers all the components.
    """
    distribution = OTJointDistribution(
        OTJointDistribution_Settings(
            marginal_settings=five_marginal_settings[:n_marginals], copula=blocks
        )
    )
    copula = distribution.distribution.getCopula()
    assert copula.getImplementation().getClassName() == class_name
    assert copula.getDimension() == n_marginals
    check_transformation(distribution)


@pytest.mark.parametrize(
    ("n_marginals", "blocks", "expected"),
    [
        pytest.param(
            4,
            (((0, 1), ClaytonCopula(2.0)), ((2, 3), FrankCopula(3.0))),
            "BlockIndependentCopula(ClaytonCopula(theta = 2), FrankCopula(theta = 3))",
            id="consecutive",
        ),
        pytest.param(
            4,
            (((2, 3), FrankCopula(3.0)), ((0, 1), ClaytonCopula(2.0))),
            "BlockIndependentCopula(ClaytonCopula(theta = 2), FrankCopula(theta = 3))",
            id="consecutive_declared_out_of_order",
        ),
        pytest.param(
            4,
            (((1, 2), ClaytonCopula(2.0)),),
            (
                "BlockIndependentCopula("
                "IndependentCopula(dimension = 1), "
                "ClaytonCopula(theta = 2), "
                "IndependentCopula(dimension = 1))"
            ),
            id="consecutive_with_free_components_on_both_sides",
        ),
        pytest.param(
            4,
            (((2, 3), ClaytonCopula(2.0)),),
            (
                "BlockIndependentCopula("
                "IndependentCopula(dimension = 2), ClaytonCopula(theta = 2))"
            ),
            id="consecutive_with_free_components_before",
        ),
        pytest.param(
            5,
            (((0, 1), NormalCopula(_create_correlation_matrix(0.5))),),
            (
                "BlockIndependentCopula("
                "NormalCopula(R = [[ 1   0.5 ]\n [ 0.5 1   ]]), "
                "IndependentCopula(dimension = 3))"
            ),
            id="normal_span_with_gap",
        ),
        pytest.param(
            4,
            (((1, 0), ClaytonCopula(2.0)),),
            (
                "BlockIndependentCopula("
                "ClaytonCopula(theta = 2), IndependentCopula(dimension = 2))"
            ),
            id="symmetric_reorder",
        ),
        pytest.param(
            5,
            (
                ((2, 0), NormalCopula(_create_correlation_matrix(0.5))),
                ((3, 4), ClaytonCopula(2.0)),
            ),
            (
                "BlockIndependentCopula("
                "NormalCopula(R = [[ 1   0   0.5 ]\n"
                " [ 0   1   0   ]\n"
                " [ 0.5 0   1   ]]), "
                "ClaytonCopula(theta = 2))"
            ),
            id="mixed_layout",
        ),
    ],
)
def test_block_copulas_block_independent(
    five_marginal_settings, n_marginals, blocks, expected
) -> None:
    """Check that block copulas give a bare block-independent copula.

    Consecutive block copulas are laid out in the order of the components,
    whatever the order they are declared in.
    A non-normal block copula such as `ClaytonCopula`
    given out of order is sorted in ascending order of its components,
    replaced by the copula of its components in that order,
    instead of falling back to a permuted one.
    A normal block copula only spans its own components,
    the remaining ones being independent,
    even when its components are given out of order
    and its span surrounds a component without any copula.
    """
    distribution = OTJointDistribution(
        OTJointDistribution_Settings(
            marginal_settings=five_marginal_settings[:n_marginals], copula=blocks
        )
    )
    copula = distribution.distribution.getCopula()
    assert isinstance(copula.getImplementation(), BlockIndependentCopula)
    assert str(copula) == expected
    check_transformation(distribution)


@pytest.mark.parametrize(
    ("blocks", "expected"),
    [
        pytest.param(
            (
                ((0, 3), NormalCopula(_create_correlation_matrix(0.6))),
                ((1, 2), ClaytonCopula(2.0)),
            ),
            None,
            id="normal_spanning_non_normal",
        ),
        pytest.param(
            (((2, 0), ClaytonCopula(2.0)),),
            None,
            id="symmetric_non_consecutive",
        ),
        pytest.param(
            (((0, 2), ClaytonCopula(2.0)),),
            (
                "MarginalDistribution(distribution=BlockIndependentCopula("
                "ClaytonCopula(theta = 2), IndependentCopula(dimension = 2)), "
                "indices=[0,2,1,3])"
            ),
            id="non_consecutive",
        ),
    ],
)
def test_block_copulas_fallback(four_marginal_settings, blocks, expected) -> None:
    """Check the fallback to a permuted block-independent copula.

    This happens when the span of a normal block copula overlaps a non-normal one,
    or when the components of a non-normal block copula
    are not consecutive,
    whatever the order of its components:
    sorting the components of a block such as `ClaytonCopula`
    given out of order only fixes the ascending-order requirement,
    not the consecutiveness one.

    The Rosenblatt transformation of OpenTURNS is too slow to be checked here,
    but the GEMSEO one remains block-wise and fast.
    The joint PDF is compared with the product of the PDFs
    of the independent blocks and of the free components,
    as the round trip through the transformation holds for any distribution.
    """
    distribution = OTJointDistribution(
        OTJointDistribution_Settings(
            marginal_settings=four_marginal_settings, copula=blocks
        )
    )
    copula = distribution.distribution.getCopula()
    assert isinstance(copula.getImplementation(), MarginalDistribution)
    if expected is not None:
        assert str(copula) == expected

    marginals = [marginal.distribution for marginal in distribution.marginals]
    free_indices = set(range(4)).difference(*(indices for indices, _ in blocks))
    for point in ([0.2, 0.4, 1.5, 0.7], [-0.3, 0.8, 3.0, 2.5]):
        expected_pdf = 1.0
        for indices, block_copula in blocks:
            block = JointDistribution([marginals[i] for i in indices], block_copula)
            expected_pdf *= block.computePDF([point[i] for i in indices])
        for index in free_indices:
            expected_pdf *= marginals[index].computePDF(point[index])
        assert distribution.distribution.computePDF(point) == pytest.approx(
            expected_pdf
        )

    uniform_point = array([0.3, 0.6, 0.1, 0.9])
    physical_point = distribution.map_from_uniform(uniform_point)
    assert_allclose(distribution.map_to_uniform(physical_point), uniform_point)


def test_block_copulas_reordered_factorization(four_marginal_settings) -> None:
    """Check that reordering a block copula given out of order keeps its law.

    `MarshallOlkinCopula` is used because it is not exchangeable,
    so sorting its components in ascending order
    replaces it by an `openturns.MarginalDistribution`
    permuting its own two arguments,
    rather than leaving it unchanged as for an exchangeable copula.
    This is checked by comparing the joint CDF
    with the product of the CDFs of its independent blocks,
    unlike a round trip through the transformation,
    which would pass even with a block in the wrong order.
    The comparison uses the CDF rather than the PDF because
    `MarshallOlkinCopula.computePDF` is not implemented
    in the installed OpenTURNS version,
    consistent with the copula not being absolutely continuous.
    """
    correlation = _create_correlation_matrix(0.6)
    marshall_olkin = MarshallOlkinCopula(0.3, 0.7)
    distribution = OTJointDistribution(
        OTJointDistribution_Settings(
            marginal_settings=four_marginal_settings,
            copula=(
                ((2, 3), NormalCopula(correlation)),
                ((1, 0), marshall_olkin),
            ),
        )
    )
    copula = distribution.distribution.getCopula()
    assert isinstance(copula.getImplementation(), BlockIndependentCopula)

    marginals = [marginal.distribution for marginal in distribution.marginals]
    block_1 = JointDistribution([marginals[2], marginals[3]], NormalCopula(correlation))
    block_2 = JointDistribution([marginals[1], marginals[0]], marshall_olkin)
    for point in ([0.2, -0.5, 0.7, 1.2], [-0.3, 0.1, -0.4, 0.6]):
        expected = block_1.computeCDF([point[2], point[3]]) * block_2.computeCDF([
            point[1],
            point[0],
        ])
        assert distribution.distribution.computeCDF(point) == pytest.approx(expected)


def test_block_copulas_reordered_transformation() -> None:
    """Check the Rosenblatt transformation of a reordered copula against OpenTURNS.

    The block copula is non-exchangeable and absolutely continuous,
    and its components are not given in the order of the random vector,
    so that conditioning them in the declared order
    would give a map different from the OpenTURNS one.
    The reordered copula remains an `EmpiricalBernsteinCopula`,
    whose Rosenblatt transformation is analytical.
    """
    RandomGenerator.SetSeed(0)
    sample = Normal().getSample(200)
    noise = Normal().getSample(200)
    sample.stack(
        Sample([[x[0] ** 2 + 0.3 * e[0]] for x, e in zip(sample, noise, strict=True)])
    )
    # With more bins,
    # the sequential conditional CDF of EmpiricalBernsteinCopula
    # is only accurate to about 1e-3 in OpenTURNS 1.27,
    # which breaks its own iso-probabilistic round trip.
    copula = EmpiricalBernsteinCopula(sample, 5)
    assert copula.computeCDF([0.3, 0.7]) != pytest.approx(copula.computeCDF([0.7, 0.3]))
    distribution = OTJointDistribution(
        OTJointDistribution_Settings(
            marginal_settings=[
                OTNormalDistribution_Settings(),
                OTNormalDistribution_Settings(mu=1.0),
            ],
            copula=(((1, 0), copula),),
        )
    )
    assert (
        distribution.distribution.getCopula().getImplementation().getClassName()
        == "EmpiricalBernsteinCopula"
    )
    check_transformation(distribution)
