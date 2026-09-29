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
#    INITIAL AUTHORS - initial API and implementation and/or initial
#                           documentation
#        :author: Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""The OpenTURNS-based joint probability distribution."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import ClassVar

from openturns import BlockIndependentCopula
from openturns import CorrelationMatrix
from openturns import DistributionImplementation
from openturns import IndependentCopula
from openturns import JointDistribution
from openturns import MarginalDistribution
from openturns import NormalCopula
from openturns import StudentCopula

from gemseo.uncertainty.distribution.openturns.joint_settings import (
    OTJointDistribution_Settings,
)
from gemseo.util.string import pretty_repr

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Sequence
    from typing import TypeAlias

    from gemseo.util.typing import RealArray

    _Block: TypeAlias = tuple[tuple[int, ...], DistributionImplementation]
    """A block of components.

    The positions of its components in the random vector and their copula.
    """

    _Span: TypeAlias = tuple[int, int, list[_Block]]
    """A span of overlapping normal blocks.

    Its first position, the position after its last one
    and the normal blocks it covers.
    """

import operator
from itertools import starmap

from numpy import argsort
from numpy import array
from numpy import eye
from numpy import ix_

from gemseo.uncertainty.distribution.core.base_joint import BaseJointDistribution


class OTJointDistribution(BaseJointDistribution):
    """The OpenTURNS-based joint probability distribution.

    The dependency structure is defined
    either by a single copula over the whole random vector
    or by a collection of copulas over independent blocks of components,
    the components outside these blocks being independent.

    !!! warning
        [OT_FORM][gemseo.uncertainty.reliability.openturns.form.OT_FORM],
        [OT_SORM][gemseo.uncertainty.reliability.openturns.sorm.OT_SORM],
        [OT_IS_FORM][gemseo.uncertainty.reliability.openturns.is_form.OT_IS_FORM]
        and
        [ISFORMSobolAnalysis][gemseo.uncertainty.sensitivity.is_form_sobol.ISFORMSobolAnalysis]
        rely on the Rosenblatt transformation of the random vector,
        called iso-probabilistic transformation by OpenTURNS.
        With a collection of block copulas,
        OpenTURNS computes it analytically
        only when the components of each non-normal block are consecutive,
        whatever their order,
        and the span of no normal block,
        from its lowest to its highest component,
        overlaps a non-normal block.
        Otherwise,
        it inverts the conditional distributions numerically,
        which costs several orders of magnitude more per point.
        For a few copula families, e.g. `openturns.JoeCopula`,
        a block given out of order
        is numerically transformed too,
        but on this block only,
        which remains much cheaper.
        [map_to_uniform()][gemseo.uncertainty.distribution.core.base_joint.BaseJointDistribution.map_to_uniform]
        and
        [map_from_uniform()][gemseo.uncertainty.distribution.core.base_joint.BaseJointDistribution.map_from_uniform]
        remain fast in all cases
        as they transform the random vector block-wise.
    """

    settings_class: ClassVar[type[OTJointDistribution_Settings]] = (
        OTJointDistribution_Settings
    )

    __block_distributions: list[tuple[list[int], JointDistribution]]
    """The joint distributions of the dependent blocks of components.

    Each item is a pair `(indices, distribution)`
    where `indices` are the positions of the components of the block
    in the random vector,
    in ascending order,
    and `distribution` is the joint distribution of these components.
    When a single copula is given for the whole random vector,
    a single block covers all the components.

    The independent components,
    including those of an independent block copula,
    belong to no block:
    `map_to_uniform` and `map_from_uniform` transform them marginal-wise.

    The Rosenblatt transformation of the whole random vector
    is the concatenation of the Rosenblatt transformations of these blocks,
    which is much faster than relying on the conditional distributions
    of a block-independent copula wrapped in an `openturns.MarginalDistribution`.
    """

    def __repr__(self) -> str:
        if len(self._settings.marginal_settings) == 1:
            return super().__repr__()

        return (
            f"{self.__class__.__name__}"
            f"({pretty_repr(self.marginals, sort=False, use_and=False)}; "
            f"{self.distribution.getCopula()})"
        )

    def _create_distribution(self, settings: OTJointDistribution_Settings) -> None:
        marginals = [marginal.distribution for marginal in self.marginals]
        copula = settings.copula
        if copula == ():
            self.distribution = JointDistribution(
                marginals, IndependentCopula(len(marginals))
            )
            self.__block_distributions = []
        elif isinstance(copula, DistributionImplementation):
            self.distribution = JointDistribution(marginals, copula)
            self.__block_distributions = [
                (list(range(self.dimension)), self.distribution)
            ]
        else:
            blocks = list(starmap(self.__sort_block, copula))
            self.distribution = JointDistribution(
                marginals, self.__assemble_copula(blocks)
            )
            self.__block_distributions = self.__create_block_distributions(blocks)

        self._set_bounds(self.marginals)

    @staticmethod
    def __permute_student_copula(
        copula: StudentCopula, order: Iterable[int]
    ) -> StudentCopula:
        """Permute the components of a Student copula.

        Permuting the components of a Student copula
        gives a Student copula of the same number of degrees of freedom
        whose correlation matrix is permuted accordingly.

        Args:
            copula: The Student copula.
            order: The positions, in the copula, of the permuted components.

        Returns:
            The copula of the permuted components.
        """
        order = list(order)
        # StudentCopula.getShapeMatrix overflows the stack in OpenTURNS 1.27.
        correlation = array(copula.getR())[ix_(order, order)]
        return StudentCopula(copula.getNu(), CorrelationMatrix(correlation))

    @classmethod
    def __sort_block(
        cls, indices: tuple[int, ...], copula: DistributionImplementation
    ) -> _Block:
        """Sort the components of a block in ascending order of position.

        The copula of the block is replaced
        by the copula of its components in that order,
        given by `getMarginal`,
        analytical for most copula families.
        A Student copula is permuted through its correlation matrix instead,
        so as to remain a `StudentCopula`
        rather than become a `SklarCopula`.

        Args:
            indices: The positions of the components of the block
                in the random vector.
            copula: The copula of the block.

        Returns:
            The block with its components in ascending order.
        """
        if list(indices) == sorted(indices):
            return indices, copula

        order = [int(index) for index in argsort(indices)]
        if isinstance(copula, StudentCopula):
            copula = cls.__permute_student_copula(copula, order)
        else:
            copula = copula.getMarginal(order).getImplementation()

        return tuple(sorted(indices)), copula

    @staticmethod
    def __group_overlapping_normal_blocks(
        normal_blocks: Iterable[_Block],
    ) -> list[_Span]:
        """Group the normal blocks whose spans overlap.

        The span of a block is the interval
        from its lowest to its highest component position.
        Two spans overlap when they share at least one position,
        e.g. when one is nested in the other;
        merely adjacent spans are kept apart.

        Args:
            normal_blocks: The normal blocks.

        Returns:
            The spans of the groups of overlapping normal blocks,
            sorted by their first position.
        """
        spans = sorted(
            (
                (min(indices), max(indices) + 1, indices, copula)
                for indices, copula in normal_blocks
            ),
            key=operator.itemgetter(0),
        )
        merged_spans = []
        for start, stop, indices, copula in spans:
            if merged_spans and start < merged_spans[-1][1]:
                previous_start, previous_stop, previous_blocks = merged_spans[-1]
                merged_spans[-1] = (
                    previous_start,
                    max(previous_stop, stop),
                    [*previous_blocks, (indices, copula)],
                )
            else:
                merged_spans.append((start, stop, [(indices, copula)]))

        return merged_spans

    @staticmethod
    def __needs_permuted_copula(
        non_normal_blocks: Sequence[_Block], spans: Iterable[_Span]
    ) -> bool:
        """Check whether the copula must be a permuted block-independent copula.

        Args:
            non_normal_blocks: The non-normal blocks,
                with components in ascending order.
            spans: The spans of the groups of overlapping normal blocks.

        Returns:
            Whether the components of a non-normal block are not consecutive
            or a span overlaps a non-normal block.
        """
        for indices, _ in non_normal_blocks:
            first_index = indices[0]
            if list(indices) != list(range(first_index, first_index + len(indices))):
                return True

        non_normal_ranges = [
            (indices[0], indices[0] + len(indices)) for indices, _ in non_normal_blocks
        ]
        return any(
            start < nn_stop and nn_start < stop
            for start, stop, _ in spans
            for nn_start, nn_stop in non_normal_ranges
        )

    def __assemble_copula(self, blocks: Sequence[_Block]) -> DistributionImplementation:
        """Assemble the copula of the random vector from its independent blocks.

        The copula is chosen so that OpenTURNS can compute
        its Rosenblatt transformation analytically whenever possible.
        When it cannot,
        the copula is a block-independent copula
        wrapped in an `openturns.MarginalDistribution` permuting its components.
        Otherwise,
        the copula is laid out in component order from pieces:
        a normal copula per span of overlapping normal blocks,
        the copula of each non-normal block
        and an independent copula per gap of components covered by neither,
        which includes the components of the independent blocks.
        A single piece is returned as is,
        several ones are wrapped in an `openturns.BlockIndependentCopula`.

        Args:
            blocks: The blocks, with components in ascending order.

        Returns:
            The copula of the random vector.
        """
        non_normal_blocks = sorted(
            (
                block
                for block in blocks
                if not isinstance(block[1], (NormalCopula, IndependentCopula))
            ),
            key=lambda block: block[0][0],
        )
        normal_blocks = [
            (indices, copula)
            for indices, copula in blocks
            if isinstance(copula, NormalCopula)
        ]
        spans = self.__group_overlapping_normal_blocks(normal_blocks)

        if self.__needs_permuted_copula(non_normal_blocks, spans):
            return self.__create_permuted_block_independent_copula(
                sorted(blocks, key=lambda block: block[0][0])
            )

        dimension = self.dimension
        pieces_by_start = {
            start: (stop, self.__merge_normal_copulas(start, stop, span_blocks))
            for start, stop, span_blocks in spans
        }
        pieces_by_start.update({
            indices[0]: (indices[0] + len(indices), copula)
            for indices, copula in non_normal_blocks
        })

        pieces = []
        position = 0
        for start in sorted(pieces_by_start):
            if start > position:
                pieces.append(IndependentCopula(start - position))
            stop, piece = pieces_by_start[start]
            pieces.append(piece)
            position = stop
        if position < dimension:
            pieces.append(IndependentCopula(dimension - position))

        if len(pieces) == 1:
            return pieces[0]

        return BlockIndependentCopula(pieces)

    @staticmethod
    def __merge_normal_copulas(
        start: int, stop: int, normal_blocks: Iterable[_Block]
    ) -> NormalCopula:
        """Merge the normal copulas of the blocks covered by a span.

        Args:
            start: The first position of the span.
            stop: The position after the last one of the span.
            normal_blocks: The normal blocks covered by the span.

        Returns:
            The normal copula of the span,
            whose correlation matrix gathers those of the normal blocks,
            the other correlations being zero.
        """
        correlation = eye(stop - start)
        for indices, copula in normal_blocks:
            local = [index - start for index in indices]
            correlation[ix_(local, local)] = array(copula.getShapeMatrix())

        return NormalCopula(CorrelationMatrix(correlation))

    def __create_block_distributions(
        self, blocks: Iterable[_Block]
    ) -> list[tuple[list[int], JointDistribution]]:
        """Create the joint distributions of the dependent blocks.

        The components of each block being in ascending order,
        the block is conditioned in the order of the random vector,
        as OpenTURNS does to compute the Rosenblatt transformation
        of the whole random vector.
        An independent block carries no dependence and is skipped.

        Args:
            blocks: The blocks, with components in ascending order.

        Returns:
            The positions of the components of each dependent block,
            in ascending order,
            and the joint distribution of these components.
        """
        block_distributions = []
        for indices, copula in blocks:
            if isinstance(copula, IndependentCopula):
                continue

            indices = list(indices)
            block_distributions.append((
                indices,
                JointDistribution(
                    [self.marginals[index].distribution for index in indices],
                    copula,
                ),
            ))

        return block_distributions

    def __create_permuted_block_independent_copula(
        self, blocks: Sequence[_Block]
    ) -> MarginalDistribution:
        """Create a block-independent copula whose components are permuted.

        The components belonging to no block are gathered
        into a last block with an independent copula.
        The block-independent copula orders its components block after block;
        the `openturns.MarginalDistribution` maps them back
        to their positions in the random vector.

        Args:
            blocks: The blocks.

        Returns:
            The block-independent copula
            wrapped in an `openturns.MarginalDistribution` permuting its components.
        """
        dimension = self.dimension
        remaining_indices = tuple(
            sorted(
                set(range(dimension))
                - {index for indices, _ in blocks for index in indices}
            )
        )
        extended_blocks = list(blocks)
        if remaining_indices:
            extended_blocks.append((
                remaining_indices,
                IndependentCopula(len(remaining_indices)),
            ))

        permutations = []
        for indices, _ in extended_blocks:
            permutations.extend(indices)

        inverse_permutations = [0] * dimension
        for index, original_index in enumerate(permutations):
            inverse_permutations[original_index] = index

        copula = BlockIndependentCopula([copula for _, copula in extended_blocks])
        return MarginalDistribution(copula, inverse_permutations)

    def compute_samples(  # noqa: D102
        self,
        n_samples: int = 1,
    ) -> RealArray:
        # We cast the value to int
        # because getSample does not support numpy.int_.
        return array(self.distribution.getSample(int(n_samples)))

    def compute_cdf(  # noqa: D102
        self,
        value: Iterable[float],
    ) -> RealArray:
        # We cast the values to float
        # because computeCDF does not support numpy.int32.
        return array([
            marginal.distribution.computeCDF(float(value_))
            for value_, marginal in zip(value, self.marginals, strict=False)
        ])

    def compute_inverse_cdf(  # noqa: D102
        self,
        value: Iterable[float],
    ) -> RealArray:
        return array([
            marginal.distribution.computeQuantile(value_)[0]
            for value_, marginal in zip(value, self.marginals, strict=False)
        ])

    def map_to_uniform(  # noqa: D102
        self,
        value: Iterable[float],
    ) -> RealArray:
        # The sequential conditional CDF is the Rosenblatt transformation.
        # We cast the values to float because OpenTURNS does not support numpy.int32.
        value = [float(value_) for value_ in value]
        # The independent components reduce to their marginal CDFs.
        result = self.compute_cdf(value)
        # The dependent components are transformed block-wise.
        for indices, distribution in self.__block_distributions:
            block = distribution.computeSequentialConditionalCDF([
                value[index] for index in indices
            ])
            for position, index in enumerate(indices):
                result[index] = block[position]

        return result

    def map_from_uniform(  # noqa: D102
        self,
        value: Iterable[float],
    ) -> RealArray:
        # We cast the values to float because OpenTURNS does not support numpy.int32.
        value = [float(value_) for value_ in value]
        # The independent components reduce to their inverse marginal CDFs.
        result = self.compute_inverse_cdf(value)
        # The dependent components are transformed block-wise.
        for indices, distribution in self.__block_distributions:
            block = distribution.computeSequentialConditionalQuantile([
                value[index] for index in indices
            ])
            for position, index in enumerate(indices):
                result[index] = block[position]

        return result
