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
#                           documentation
#        :author: Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Importance-sampling estimators of the Sobol' indices of a failure indicator."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from typing import TYPE_CHECKING
from typing import Final

from numpy import arange
from numpy import argsort
from numpy import array
from numpy import asarray
from numpy import atleast_2d
from numpy import column_stack
from numpy import cov
from numpy import empty
from numpy import exp
from numpy import full
from numpy import isnan
from numpy import ix_
from numpy import nan
from numpy import nanquantile
from numpy import roll
from numpy import sqrt
from numpy import zeros
from numpy.random import default_rng
from scipy.stats import norm

from gemseo.uncertainty.sensitivity._sobol_indices_estimator import (
    SobolIndicesEstimatorMixin,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Sequence

    from gemseo.util.typing import IntegerArray
    from gemseo.util.typing import RealArray

Algorithm = SobolIndicesEstimatorMixin.Algorithm

rank_bootstrap_sample_ratio: Final[float] = 0.8
"""The share of the rows drawn by the bootstrap of the rank-based estimator.

The rows are drawn without replacement,
as a row resampled twice would be ranked next to its own duplicate
and paired with it,
which would pull every replicate towards one.
This is the default of the OpenTURNS rank-based estimator,
whose `RankSobolSensitivityAlgorithm-DefaultBootstrapSampleRatio` it mirrors.
"""

complex_step: Final[float] = 1e-30
"""The step of the complex-step derivatives of the statistics.

The statistics are analytic functions of the means of the terms,
so the derivative is free of subtractive cancellation
and the step can be taken arbitrarily small.
"""


@dataclass(frozen=True)
class _Moments:
    """The named per-row terms whose means the statistics are functions of."""

    terms: RealArray
    """The per-row terms, shaped as `(n_rows, n_terms)`."""

    names: tuple[str, ...]
    """The names of the terms, in the order of the columns of `terms`."""

    def compute_means(self, rows: IntegerArray | None = None) -> RealArray:
        """Compute the means of the terms.

        Args:
            rows: The indices of the rows to average;
                if `None`, use all the rows.

        Returns:
            The means of the terms, shaped as `(n_terms,)`.
        """
        terms = self.terms if rows is None else self.terms[rows]
        return terms.mean(axis=0)

    def compute_covariance(self) -> RealArray:
        """Compute the covariance matrix of the terms.

        Returns:
            The covariance matrix, shaped as `(n_terms, n_terms)`.
        """
        return atleast_2d(cov(self.terms, rowvar=False, ddof=1))

    def get_reader(self, means: RealArray) -> Callable[[str], float]:
        """Return a function reading the mean of a term from its name.

        Args:
            means: The means of the terms, real or complex.

        Returns:
            The function mapping the name of a term to its mean.
        """
        name_to_index = {name: index for index, name in enumerate(self.names)}
        return lambda name: means[name_to_index[name]]


@dataclass(frozen=True)
class _Statistic:
    """A statistic that is a smooth function of the means of independent terms."""

    function: Callable[[Callable[[str], float]], float]
    """The function mapping a reader of the means of the terms to the statistic.

    It must only use additions, subtractions, multiplications, divisions
    and square roots, so that it can be evaluated with complex means.
    """

    moments: _Moments
    """The terms whose means the function reads."""

    def evaluate(self, means: RealArray) -> float:
        """Evaluate the statistic.

        Args:
            means: The means of the terms, shaped as `(n_terms,)`.

        Returns:
            The value of the statistic.
        """
        return float(self.function(self.moments.get_reader(means)))

    def compute_asymptotic_variance(
        self, means: RealArray, covariance: RealArray
    ) -> float:
        """Compute the asymptotic variance of the statistic by the delta method.

        The gradient of the function is computed by complex-step differentiation.

        Args:
            means: The means of the terms, shaped as `(n_terms,)`.
            covariance: The covariance matrix of the terms,
                shaped as `(n_terms, n_terms)`.

        Returns:
            The asymptotic variance.
        """
        gradient = empty(len(means))
        shifted_means = means.astype(complex)
        for index in range(len(means)):
            shifted_means[index] += 1j * complex_step
            gradient[index] = (
                self.function(self.moments.get_reader(shifted_means)).imag
                / complex_step
            )
            shifted_means[index] = means[index]

        return float(gradient @ covariance @ gradient / len(self.moments.terms))


class ISSobolIndicesEstimator:
    r"""An importance-sampling estimator of the Sobol' indices of a failure indicator.

    The pick-and-freeze algorithms
    (`Saltelli`, `Jansen`, `MauntzKucherenko` and `Martinez`)
    expect the design generated by
    [OTSobolDOE][gemseo.doe.openturns._algorithm.ot_sobol_doe.OTSobolDOE],
    i.e. the rows $[A; B; E^1; \ldots; E^d]$
    where $E^i$ is $A$ with its $i$-th column replaced by that of $B$,
    followed by $[C^1; \ldots; C^d]$
    where $C^i$ is $B$ with its $i$-th column replaced by that of $A$
    when the second-order indices are requested and $d\neq 2$.
    When $d=2$, the blocks $C^1$ and $C^2$ are the blocks $E^2$ and $E^1$.

    They are the OpenTURNS estimators
    in which every sample moment is replaced by its importance-sampling estimate:
    a mean of products of the indicators of two rows
    is weighted by the weight of the pair,
    a mean involving a single row by the weight of this row
    and a constant is left as is.
    With a design point at the origin, all the weights are equal to one
    and the estimators are the OpenTURNS ones.
    In particular, the indices are divided by the variance
    $\hat{V}=\frac{N}{N-1}\hat{p}_A(1-\hat{p}_A)$
    of the indicator over the block $A$ only,
    the first-order index $S_i$ is estimated from the pairs $(B, E^i)$,
    which share the $i$-th coordinate,
    the total-order index $S_i^T$ from the pairs $(A, E^i)$,
    which share all the coordinates but the $i$-th one,
    and the second-order index $S_{ij}$ from the pairs $(E^j, C^i)$,
    which share the $i$-th and $j$-th coordinates.

    The `Rank` algorithm expects instead independent rows
    drawn from the auxiliary density.
    It sorts them along each coordinate $i$
    and pairs each row $r_k$ with its successor $r_{N(k)}$ in that order
    (the last one with the first one),
    so that the two rows of a pair have close $i$-th coordinates
    while their other coordinates remain independent;
    it only provides the first-order indices.

    The indices are divided by the variance of the indicator
    over the block $A$ of the design only,
    which `ROSobolAnalysis` checks to be positive.
    This does not make every index defined:
    the `Martinez` estimator also divides by the variances of the blocks
    $B$ (first order) and $A$ and $E^i$ (total order),
    so an index is NaN when the indicator is constant over such a block.
    Likewise, a bootstrap replicate
    whose resampled block $A$ has a constant indicator
    (or, for the `Rank` algorithm, whose rows have a constant indicator)
    has a zero reference variance:
    its indices are NaN,
    it is counted in `n_degenerate_replicates`
    and the percentile intervals are computed from the other replicates;
    the bounds of an index are NaN when all its replicates are degenerate.

    The importance-sampling weights and the confidence intervals
    are described in the *Rare events* part
    of the Sobol' analysis section of the user guide.
    """

    __algorithm_to_builder_name: Final[dict[Algorithm, str]] = {
        Algorithm.SALTELLI: "_build_saltelli_statistics",
        Algorithm.JANSEN: "_build_jansen_statistics",
        Algorithm.MAUNTZ_KUCHERENKO: "_build_mauntz_kucherenko_statistics",
        Algorithm.MARTINEZ: "_build_martinez_statistics",
    }
    """The map from a pick-and-freeze algorithm to its statistics builder."""

    first_order_indices: RealArray
    """The first-order Sobol' indices, shaped as `(dimension,)`."""

    first_order_interval: tuple[RealArray, RealArray]
    """The lower and upper bounds of the first-order indices,
    each shaped as `(dimension,)`."""

    second_order_indices: RealArray
    """The second-order Sobol' indices, shaped as `(dimension, dimension)`,
    with zeros on the diagonal; an empty array for the `Rank` algorithm."""

    total_order_indices: RealArray
    """The total-order Sobol' indices, shaped as `(dimension,)`;
    an empty array for the `Rank` algorithm."""

    total_order_interval: tuple[RealArray, RealArray]
    """The lower and upper bounds of the total-order indices,
    each shaped as `(dimension,)`; empty arrays for the `Rank` algorithm."""

    variance: float
    r"""The variance the indices are divided by.

    It is $\frac{N}{N-1}\hat{p}_A(1-\hat{p}_A)$ for the block $A$
    of a pick-and-freeze design
    and $\frac{n}{n-1}\hat{p}(1-\hat{p})$ for the $n$ rows of the `Rank` algorithm.
    """

    n_degenerate_replicates: int
    """The number of bootstrap replicates having at least one undefined index.

    An index of a replicate is undefined (NaN)
    when the indicator is constant over a block the estimator divides by.
    It is 0 when the asymptotic confidence intervals are used,
    which the `Rank` algorithm does not offer.
    """

    __compute_second_order: bool
    """Whether the second-order indices are estimated."""

    __confidence_level: float
    """The confidence level of the intervals."""

    __coordinate_log_weights: RealArray
    """The per-coordinate log-likelihood ratios, shaped as `(n_rows, dimension)`."""

    __dimension: int
    """The dimension of the standard space."""

    __indicator: RealArray
    """The failure indicator of every row, shaped as `(n_rows,)`."""

    __row_log_weights: RealArray
    """The log-likelihood ratios of the rows, shaped as `(n_rows,)`."""

    __row_weights: RealArray
    """The likelihood ratios of the rows, shaped as `(n_rows,)`."""

    __sample_size: int
    """The number of rows of each block of the design."""

    __standard_samples: RealArray
    """The input samples in the standard space, shaped as `(n_rows, dimension)`."""

    def __init__(
        self,
        algorithm: Algorithm,
        standard_samples: RealArray,
        indicator: RealArray,
        standard_design_point: RealArray,
        sample_size: int,
        compute_second_order: bool,
        confidence_level: float,
        use_asymptotic_distributions: bool,
        n_replicates: int,
        seed: int | None,
    ) -> None:
        """
        Args:
            algorithm: The Sobol' estimation algorithm.
            standard_samples: The input samples in the standard space,
                drawn from the auxiliary density
                and shaped as `(n_rows, dimension)`,
                following the pick-and-freeze layout of
                [OTSobolDOE][gemseo.doe.openturns._algorithm.ot_sobol_doe.OTSobolDOE]
                for the pick-and-freeze algorithms
                or independent for the `Rank` algorithm.
            indicator: The failure indicator of every row, shaped as `(n_rows,)`.
            standard_design_point: The center of the auxiliary density,
                i.e. the design point in the standard space.
            sample_size: The number of rows $N$ of each block of the design
                (the number of rows for the `Rank` algorithm).
            compute_second_order: Whether to estimate the second-order indices;
                ignored by the `Rank` algorithm.
            confidence_level: The confidence level of the intervals.
            use_asymptotic_distributions: Whether to compute the confidence intervals
                from the asymptotic distributions of the estimators;
                otherwise, or for the `Rank` algorithm, use the bootstrap.
            n_replicates: The number of bootstrap replicates.
            seed: The seed of the bootstrap;
                if `None`, the bootstrap is not reproducible.

        Raises:
            ValueError: If the bootstrap is used with less than one replicate.
        """  # noqa: D205, D212, D415
        use_rank = algorithm == Algorithm.RANK
        if n_replicates < 1 and (use_rank or not use_asymptotic_distributions):
            msg = (
                "The bootstrap confidence intervals "
                f"require at least one replicate; got {n_replicates}."
            )
            raise ValueError(msg)

        self.n_degenerate_replicates = 0
        self.__sample_size = sample_size
        self.__confidence_level = confidence_level
        standard_samples = asarray(standard_samples, dtype=float)
        standard_design_point = asarray(standard_design_point, dtype=float).ravel()
        self.__standard_samples = standard_samples
        self.__dimension = standard_samples.shape[1]
        self.__indicator = asarray(indicator, dtype=float).ravel()
        self.__coordinate_log_weights = (
            -standard_design_point * standard_samples + standard_design_point**2 / 2
        )
        self.__row_log_weights = self.__coordinate_log_weights.sum(axis=1)
        self.__row_weights = exp(self.__row_log_weights)
        self.__compute_second_order = compute_second_order and not use_rank

        if use_rank:
            probability = self.__compute_weighted_indicator(arange(sample_size)).mean()
            self.variance = self.__compute_rank_variance(probability, sample_size)
            self.__estimate_rank_indices(n_replicates, seed)
            return

        moments = self.__create_moments()
        means = moments.compute_means()
        read = moments.get_reader(means)
        self.variance = self.__compute_reference_variance(read)
        builder = getattr(self, self.__algorithm_to_builder_name[algorithm])
        first_statistics, total_statistics = builder(moments)
        self.first_order_indices = array([s.evaluate(means) for s in first_statistics])
        self.total_order_indices = array([s.evaluate(means) for s in total_statistics])
        self.second_order_indices = self.__compute_second_order_indices(read)

        if use_asymptotic_distributions:
            covariance = moments.compute_covariance()
            self.first_order_interval = self.__compute_asymptotic_bounds(
                first_statistics, means, covariance, self.first_order_indices
            )
            self.total_order_interval = self.__compute_asymptotic_bounds(
                total_statistics, means, covariance, self.total_order_indices
            )
            return

        rng = default_rng(seed)
        first_replicates = empty((n_replicates, self.__dimension))
        total_replicates = empty((n_replicates, self.__dimension))
        for replicate in range(n_replicates):
            rows = rng.integers(0, sample_size, sample_size)
            replicate_means = moments.compute_means(rows)
            reference_variance = self.__compute_reference_variance(
                moments.get_reader(replicate_means)
            )
            if reference_variance <= 0.0:
                first_replicates[replicate] = nan
                total_replicates[replicate] = nan
                continue

            first_replicates[replicate] = [
                s.evaluate(replicate_means) for s in first_statistics
            ]
            total_replicates[replicate] = [
                s.evaluate(replicate_means) for s in total_statistics
            ]

        self.n_degenerate_replicates = int(
            (
                isnan(first_replicates).any(axis=1)
                | isnan(total_replicates).any(axis=1)
            ).sum()
        )
        self.first_order_interval = self.__compute_percentiles(first_replicates)
        self.total_order_interval = self.__compute_percentiles(total_replicates)

    # Confidence intervals

    def __compute_asymptotic_bounds(
        self,
        statistics: Sequence[_Statistic],
        means: RealArray,
        covariance: RealArray,
        estimates: RealArray,
    ) -> tuple[RealArray, RealArray]:
        """Compute the bounds of the asymptotic confidence intervals.

        Args:
            statistics: The statistics estimating the indices.
            means: The means of the terms of the statistics.
            covariance: The covariance matrix of the terms of the statistics.
            estimates: The estimates of the indices, shaped as `(dimension,)`.

        Returns:
            The lower and upper bounds of the intervals, each shaped as `(dimension,)`.
        """
        standard_deviations = sqrt(
            array([
                s.compute_asymptotic_variance(means, covariance) for s in statistics
            ])
        )
        half_width = norm.ppf((1.0 + self.__confidence_level) / 2) * standard_deviations
        return estimates - half_width, estimates + half_width

    def __compute_percentiles(
        self, replicates: RealArray
    ) -> tuple[RealArray, RealArray]:
        """Compute the bounds of the percentile bootstrap intervals.

        The undefined (NaN) replicates are ignored;
        the bounds of an index are NaN when all its replicates are undefined.

        Args:
            replicates: The bootstrap replicates of the indices,
                shaped as `(n_replicates, dimension)`.

        Returns:
            The lower and upper bounds of the intervals, each shaped as `(dimension,)`.
        """
        half_alpha = (1.0 - self.__confidence_level) / 2
        lower_bounds = full(replicates.shape[1], nan)
        upper_bounds = full(replicates.shape[1], nan)
        defined = ~isnan(replicates).all(axis=0)
        if defined.any():
            lower_bounds[defined] = nanquantile(
                replicates[:, defined], half_alpha, axis=0
            )
            upper_bounds[defined] = nanquantile(
                replicates[:, defined], 1.0 - half_alpha, axis=0
            )

        return lower_bounds, upper_bounds

    # Blocks of the pick-and-freeze design

    def __get_block(self, block: int) -> IntegerArray:
        """Return the indices of the rows of a block of the pick-and-freeze design.

        For $d=2$, the blocks $C^1$ and $C^2$ are not part of the design;
        as in OpenTURNS, they are the blocks $E^2$ and $E^1$.

        Args:
            block: The position of the block,
                i.e. `0` for $A$, `1` for $B$, `2+i` for $E^i$ and `2+d+i` for $C^i$.

        Returns:
            The indices of the rows in the whole design.
        """
        dimension = self.__dimension
        if dimension == 2 and block >= 2 + dimension:
            block = 2 + (dimension - 1) - (block - 2 - dimension)
        return block * self.__sample_size + arange(self.__sample_size)

    def __compute_pair_weights(
        self,
        first_rows: IntegerArray,
        second_rows: IntegerArray,
        shared_coordinates: Sequence[int] | IntegerArray,
    ) -> RealArray:
        r"""Compute the importance-sampling weights of pairs of rows.

        The pairs share the coordinates `shared_coordinates`,
        which are counted once:
        $\omega=w(r)w(r')/\prod_{k\in K} w_k(r_k)$.
        Weighting each row separately, i.e. using $w(r)w(r')$,
        would estimate the Sobol' indices
        of the weighted indicator under the auxiliary density
        instead of those of the indicator under the true density,
        with a bias that does not vanish with the sample size.

        The two rows of a pick-and-freeze pair share these coordinates exactly.
        The two rows of a rank-based pair only have *close* $i$-th coordinates,
        so the weight is that of the second row for this coordinate;
        it is consistent but not exactly unbiased at finite sample size,
        by a factor that vanishes with the spacing of the sorted coordinates.

        Args:
            first_rows: The indices of the first rows of the pairs.
            second_rows: The indices of the second rows of the pairs.
            shared_coordinates: The coordinates shared by the two rows of a pair.

        Returns:
            The weights of the pairs.
        """
        log_weights = (
            self.__row_log_weights[first_rows] + self.__row_log_weights[second_rows]
        )
        if len(shared_coordinates):
            log_weights = log_weights - self.__coordinate_log_weights[
                ix_(first_rows, shared_coordinates)
            ].sum(axis=1)
        return exp(log_weights)

    def __compute_weighted_indicator(self, rows: IntegerArray) -> RealArray:
        """Compute the importance-sampling weighted indicator of some rows.

        Args:
            rows: The indices of the rows.

        Returns:
            The weighted indicator of every row.
        """
        return self.__row_weights[rows] * self.__indicator[rows]

    def __compute_pair_products(
        self,
        first_rows: IntegerArray,
        second_rows: IntegerArray,
        shared_coordinates: Sequence[int] | IntegerArray,
    ) -> RealArray:
        """Compute the weighted products of the indicators of pairs of rows.

        Args:
            first_rows: The indices of the first rows of the pairs.
            second_rows: The indices of the second rows of the pairs.
            shared_coordinates: The coordinates shared by the two rows of a pair.

        Returns:
            The weighted product of the indicators of every pair.
        """
        return (
            self.__compute_pair_weights(first_rows, second_rows, shared_coordinates)
            * self.__indicator[first_rows]
            * self.__indicator[second_rows]
        )

    def __get_other_coordinates(self, *coordinates: int) -> IntegerArray:
        """Return the coordinates other than the given ones.

        Args:
            *coordinates: The coordinates to exclude.

        Returns:
            The other coordinates.
        """
        mask = zeros(self.__dimension, dtype=bool)
        mask[list(coordinates)] = True
        return arange(self.__dimension)[~mask]

    # Terms of the pick-and-freeze estimators

    def __create_moments(self) -> _Moments:
        r"""Create the per-row terms of the pick-and-freeze statistics.

        Row $k$ of every block of the design forms one row of independent terms,
        named as follows,
        with $y_X$ the indicator of the block $X$,
        $w_X$ its row weights,
        $\omega_K$ the weights of the pairs sharing the coordinates $K$
        and $L$ the blocks of the full design
        (those of $[A; B; E^1; \ldots; E^d]$
        plus $[C^1; \ldots; C^d]$ when the second-order indices are requested
        or when $d=2$, as in OpenTURNS, where $C^1=E^2$ and $C^2=E^1$):

        - `P_X`: $w_Xy_X$ for $X\in\{A, B, E^1,\ldots,E^d\}$,
        - `mu`: $\frac{1}{|L|}\sum_{X\in L} w_Xy_X$,
          i.e. the estimate of the mean by which OpenTURNS centers the outputs,
        - `Q_B_E{i}`: $\omega_{\{i\}}y_By_{E^i}$,
        - `Q_A_E{i}`: $\omega_{-i}y_Ay_{E^i}$,
        - `Q_A_B`: $\omega_{\emptyset}y_Ay_B$.

        Returns:
            The moments.
        """
        dimension = self.__dimension
        # As OpenTURNS, count the blocks C^1 = E^2 and C^2 = E^1 for d=2,
        # even when the second-order indices are not requested.
        has_c_blocks = self.__compute_second_order or dimension == 2
        n_blocks = 2 + dimension + (dimension if has_c_blocks else 0)
        rows_a = self.__get_block(0)
        rows_b = self.__get_block(1)
        terms = {
            "mu": sum(
                self.__compute_weighted_indicator(self.__get_block(block))
                for block in range(n_blocks)
            )
            / n_blocks,
            "P_A": self.__compute_weighted_indicator(rows_a),
            "P_B": self.__compute_weighted_indicator(rows_b),
            "Q_A_B": self.__compute_pair_products(rows_a, rows_b, []),
        }
        for i in range(dimension):
            rows_e = self.__get_block(2 + i)
            terms[f"P_E{i}"] = self.__compute_weighted_indicator(rows_e)
            terms[f"Q_B_E{i}"] = self.__compute_pair_products(rows_b, rows_e, [i])
            terms[f"Q_A_E{i}"] = self.__compute_pair_products(
                rows_a, rows_e, self.__get_other_coordinates(i)
            )

        return _Moments(column_stack(list(terms.values())), tuple(terms))

    def __compute_reference_variance(self, read: Callable[[str], float]) -> float:
        r"""Compute the variance the pick-and-freeze indices are divided by.

        It is that of the block $A$:
        $\hat{V}=\frac{N}{N-1}\hat{p}_A(1-\hat{p}_A)$.

        Args:
            read: The function reading the mean of a term from its name.

        Returns:
            The variance.
        """
        probability = read("P_A")
        return (
            self.__sample_size
            / (self.__sample_size - 1)
            * probability
            * (1.0 - probability)
        )

    @staticmethod
    def __compute_centered_product(
        product: float, first_mean: float, second_mean: float, mean: float
    ) -> float:
        r"""Compute the estimate of the mean of a product of centered indicators.

        With $\hat{q}$ the estimate of $\mathbb{E}[y_Xy_Y]$,
        $\hat{p}_X$ and $\hat{p}_Y$ those of $\mathbb{E}[y_X]$ and $\mathbb{E}[y_Y]$
        and $\hat{\mu}$ the centering mean,
        it is $\hat{q}-\hat{\mu}(\hat{p}_X+\hat{p}_Y)+\hat{\mu}^2$.

        Args:
            product: The estimate of the mean of the product of the indicators.
            first_mean: The estimate of the mean of the first indicator.
            second_mean: The estimate of the mean of the second indicator.
            mean: The centering mean.

        Returns:
            The estimate of the mean of the product of the centered indicators.
        """
        return product - mean * (first_mean + second_mean) + mean**2

    def __compute_centered_moment(
        self,
        read: Callable[[str], float],
        product: str,
        first_block: str,
        second_block: str,
    ) -> float:
        """Compute the centered product of the indicators of two blocks.

        Args:
            read: The function reading the mean of a term from its name.
            product: The name of the term of the weighted product of the indicators.
            first_block: The label of the first block, e.g. `"A"` or `"E0"`.
            second_block: The label of the second block.

        Returns:
            The estimate of the mean of the product of the centered indicators.
        """
        return self.__compute_centered_product(
            read(product),
            read(f"P_{first_block}"),
            read(f"P_{second_block}"),
            read("mu"),
        )

    def __compute_centered_mean(
        self, read: Callable[[str], float], block: str
    ) -> float:
        """Compute the mean of the centered indicator of a block.

        Args:
            read: The function reading the mean of a term from its name.
            block: The label of the block, e.g. `"A"` or `"E0"`.

        Returns:
            The estimate of the mean of the centered indicator.
        """
        return read(f"P_{block}") - read("mu")

    # Statistics of the pick-and-freeze estimators

    def _build_saltelli_statistics(
        self, moments: _Moments
    ) -> tuple[list[_Statistic], list[_Statistic]]:
        r"""Build the statistics of the weighted Saltelli estimator.

        With $\hat{Z}(X,Y)$ the centered product estimate
        and $\hat{M}_X$ the centered mean,
        $\hat{S}_i=\frac{\frac{N}{N-1}\hat{Z}(B,E^i)-\hat{M}_A\hat{M}_B}{\hat{V}}$
        and
        $\hat{S}_i^T=1-\frac{\frac{N}{N-1}\hat{Z}(A,E^i)-\hat{M}_A^2}{\hat{V}}$.

        Args:
            moments: The moments the statistics are functions of.

        Returns:
            The statistics of the first- and total-order indices of every input.
        """
        return (
            [
                _Statistic(partial(self.__compute_saltelli_first, i), moments)
                for i in range(self.__dimension)
            ],
            [
                _Statistic(partial(self.__compute_saltelli_total, i), moments)
                for i in range(self.__dimension)
            ],
        )

    def __compute_saltelli_first(self, i: int, read: Callable[[str], float]) -> float:
        """Compute the first-order index with the weighted Saltelli estimator.

        Args:
            i: The index of the input.
            read: The function reading the mean of a term from its name.

        Returns:
            The first-order index.
        """
        ratio = self.__sample_size / (self.__sample_size - 1)
        return (
            ratio * self.__compute_centered_moment(read, f"Q_B_E{i}", "B", f"E{i}")
            - self.__compute_centered_mean(read, "A")
            * self.__compute_centered_mean(read, "B")
        ) / self.__compute_reference_variance(read)

    def __compute_saltelli_total(self, i: int, read: Callable[[str], float]) -> float:
        """Compute the total-order index with the weighted Saltelli estimator.

        Args:
            i: The index of the input.
            read: The function reading the mean of a term from its name.

        Returns:
            The total-order index.
        """
        ratio = self.__sample_size / (self.__sample_size - 1)
        return 1.0 - (
            ratio * self.__compute_centered_moment(read, f"Q_A_E{i}", "A", f"E{i}")
            - self.__compute_centered_mean(read, "A") ** 2
        ) / self.__compute_reference_variance(read)

    def _build_jansen_statistics(
        self, moments: _Moments
    ) -> tuple[list[_Statistic], list[_Statistic]]:
        r"""Build the statistics of the weighted Jansen estimator.

        As $y^2=y$ for an indicator,
        the mean of the squared difference of two blocks is
        $\hat{D}(X,Y)=\hat{p}_X+\hat{p}_Y-2\hat{q}_{XY}$
        and
        $\hat{S}_i=1-\frac{N}{2N-1}\frac{\hat{D}(B,E^i)}{\hat{V}}$
        and
        $\hat{S}_i^T=\frac{N}{2N-1}\frac{\hat{D}(A,E^i)}{\hat{V}}$.

        Args:
            moments: The moments the statistics are functions of.

        Returns:
            The statistics of the first- and total-order indices of every input.
        """
        return (
            [
                _Statistic(partial(self.__compute_jansen_first, i), moments)
                for i in range(self.__dimension)
            ],
            [
                _Statistic(partial(self.__compute_jansen_total, i), moments)
                for i in range(self.__dimension)
            ],
        )

    def __compute_squared_difference(
        self,
        read: Callable[[str], float],
        product: str,
        first_block: str,
        second_block: str,
    ) -> float:
        """Compute the mean of the squared difference of the indicators of two blocks.

        Args:
            read: The function reading the mean of a term from its name.
            product: The name of the term of the weighted product of the indicators.
            first_block: The label of the first block, e.g. `"A"` or `"E0"`.
            second_block: The label of the second block.

        Returns:
            The estimate of the mean of the squared difference of the indicators,
            scaled by $N/(2N-1)$ and divided by the reference variance.
        """
        n = self.__sample_size
        return (
            n
            / (2 * n - 1)
            * (read(f"P_{first_block}") + read(f"P_{second_block}") - 2 * read(product))
            / self.__compute_reference_variance(read)
        )

    def __compute_jansen_first(self, i: int, read: Callable[[str], float]) -> float:
        """Compute the first-order index with the weighted Jansen estimator.

        Args:
            i: The index of the input.
            read: The function reading the mean of a term from its name.

        Returns:
            The first-order index.
        """
        return 1.0 - self.__compute_squared_difference(read, f"Q_B_E{i}", "B", f"E{i}")

    def __compute_jansen_total(self, i: int, read: Callable[[str], float]) -> float:
        """Compute the total-order index with the weighted Jansen estimator.

        Args:
            i: The index of the input.
            read: The function reading the mean of a term from its name.

        Returns:
            The total-order index.
        """
        return self.__compute_squared_difference(read, f"Q_A_E{i}", "A", f"E{i}")

    def _build_mauntz_kucherenko_statistics(
        self, moments: _Moments
    ) -> tuple[list[_Statistic], list[_Statistic]]:
        r"""Build the statistics of the weighted Mauntz-Kucherenko estimator.

        With $\hat{Z}(X,Y)$ the centered product estimate,
        $\hat{S}_i=\frac{N}{N-1}\frac{\hat{Z}(B,E^i)-\hat{Z}(B,A)}{\hat{V}}$
        and
        $\hat{S}_i^T=\frac{N}{N-1}\frac{\hat{Z}(A,A)-\hat{Z}(A,E^i)}{\hat{V}}$
        where $\hat{q}_{AA}=\hat{p}_A$.

        Args:
            moments: The moments the statistics are functions of.

        Returns:
            The statistics of the first- and total-order indices of every input.
        """
        return (
            [
                _Statistic(partial(self.__compute_mauntz_kucherenko_first, i), moments)
                for i in range(self.__dimension)
            ],
            [
                _Statistic(partial(self.__compute_mauntz_kucherenko_total, i), moments)
                for i in range(self.__dimension)
            ],
        )

    def __compute_mauntz_kucherenko_first(
        self, i: int, read: Callable[[str], float]
    ) -> float:
        """Compute the first-order index with the weighted Mauntz-Kucherenko estimator.

        Args:
            i: The index of the input.
            read: The function reading the mean of a term from its name.

        Returns:
            The first-order index.
        """
        ratio = self.__sample_size / (self.__sample_size - 1)
        return (
            ratio
            * (
                self.__compute_centered_moment(read, f"Q_B_E{i}", "B", f"E{i}")
                - self.__compute_centered_moment(read, "Q_A_B", "B", "A")
            )
            / self.__compute_reference_variance(read)
        )

    def __compute_mauntz_kucherenko_total(
        self, i: int, read: Callable[[str], float]
    ) -> float:
        """Compute the total-order index with the weighted Mauntz-Kucherenko estimator.

        Args:
            i: The index of the input.
            read: The function reading the mean of a term from its name.

        Returns:
            The total-order index.
        """
        ratio = self.__sample_size / (self.__sample_size - 1)
        return (
            ratio
            * (
                self.__compute_centered_moment(read, "P_A", "A", "A")
                - self.__compute_centered_moment(read, f"Q_A_E{i}", "A", f"E{i}")
            )
            / self.__compute_reference_variance(read)
        )

    def _build_martinez_statistics(
        self, moments: _Moments
    ) -> tuple[list[_Statistic], list[_Statistic]]:
        r"""Build the statistics of the weighted Martinez estimator.

        The indices are weighted correlation coefficients
        between the indicators of the two blocks of a pair,
        each block being centered and scaled
        with its own importance-sampling estimate of the failure probability:
        $\hat{S}_i=\hat{\rho}_{\omega_{\{i\}}}(y_B, y_{E^i})$
        and
        $\hat{S}_i^T=1-\hat{\rho}_{\omega_{-i}}(y_A, y_{E^i})$
        where
        $\hat{\rho}(y_X, y_Y)=(\hat{q}_{XY}-\hat{p}_X\hat{p}_Y)
        /\sqrt{\hat{p}_X(1-\hat{p}_X)\hat{p}_Y(1-\hat{p}_Y)}$.

        Args:
            moments: The moments the statistics are functions of.

        Returns:
            The statistics of the first- and total-order indices of every input.
        """
        return (
            [
                _Statistic(partial(self.__compute_martinez_first, i), moments)
                for i in range(self.__dimension)
            ],
            [
                _Statistic(partial(self.__compute_martinez_total, i), moments)
                for i in range(self.__dimension)
            ],
        )

    @staticmethod
    def __compute_correlation(
        read: Callable[[str], float],
        product: str,
        first_block: str,
        second_block: str,
    ) -> float:
        """Compute the weighted correlation coefficient of the indicators of two blocks.

        The coefficient is NaN when the estimated variance of a block is not positive,
        e.g. when the indicator is constant over this block.

        Args:
            read: The function reading the mean of a term from its name.
            product: The name of the term of the weighted product of the indicators.
            first_block: The label of the first block, e.g. `"A"` or `"E0"`.
            second_block: The label of the second block.

        Returns:
            The correlation coefficient.
        """
        first_mean = read(f"P_{first_block}")
        second_mean = read(f"P_{second_block}")
        variance_product = (
            first_mean * (1.0 - first_mean) * second_mean * (1.0 - second_mean)
        )
        # The means can be complex (complex-step derivative): test the real part.
        if variance_product.real <= 0.0:
            return nan

        return (read(product) - first_mean * second_mean) / sqrt(variance_product)

    def __compute_martinez_first(self, i: int, read: Callable[[str], float]) -> float:
        """Compute the first-order index with the weighted Martinez estimator.

        Args:
            i: The index of the input.
            read: The function reading the mean of a term from its name.

        Returns:
            The first-order index.
        """
        return self.__compute_correlation(read, f"Q_B_E{i}", "B", f"E{i}")

    def __compute_martinez_total(self, i: int, read: Callable[[str], float]) -> float:
        """Compute the total-order index with the weighted Martinez estimator.

        Args:
            i: The index of the input.
            read: The function reading the mean of a term from its name.

        Returns:
            The total-order index.
        """
        return 1.0 - self.__compute_correlation(read, f"Q_A_E{i}", "A", f"E{i}")

    # Second-order indices

    def __compute_second_order_indices(self, read: Callable[[str], float]) -> RealArray:
        r"""Estimate the second-order indices from the pick-and-freeze design.

        The closed second-order variance $V_{ij}^c$ is estimated
        from the pairs $(E^j, C^i)$ with $i<j$,
        which share the $i$-th and $j$-th coordinates:
        $\hat{S}_{ij}=\frac{\frac{N}{N-1}\hat{Z}(E^j,C^i)-\hat{Z}(A,B)}{\hat{V}}
        -\hat{S}_i-\hat{S}_j$,
        with the first-order indices of the algorithm.
        For $d=2$, $C^1=E^2$ and $C^2=E^1$, as in OpenTURNS.

        Args:
            read: The function reading the mean of a term from its name.

        Returns:
            The second-order indices, shaped as `(dimension, dimension)`;
            zeros if they are not requested.
        """
        dimension = self.__dimension
        first = self.first_order_indices
        second = zeros((dimension, dimension))
        if not self.__compute_second_order:
            return second

        ratio = self.__sample_size / (self.__sample_size - 1)
        mean = read("mu")
        reference = self.__compute_centered_moment(read, "Q_A_B", "A", "B")
        for i in range(dimension):
            rows_c_i = self.__get_block(2 + dimension + i)
            mean_c_i = self.__compute_weighted_indicator(rows_c_i).mean()
            for j in range(i + 1, dimension):
                rows_e_j = self.__get_block(2 + j)
                closed_moment = self.__compute_centered_product(
                    self.__compute_pair_products(rows_e_j, rows_c_i, [i, j]).mean(),
                    read(f"P_E{j}"),
                    mean_c_i,
                    mean,
                )
                second[i, j] = second[j, i] = (
                    (ratio * closed_moment - reference) / self.variance
                    - first[i]
                    - first[j]
                )

        return second

    # Rank-based estimator

    def __estimate_rank_indices(self, n_replicates: int, seed: int | None) -> None:
        """Estimate the first-order indices and their bootstrap intervals by ranks.

        Each replicate draws a share `rank_bootstrap_sample_ratio` of the rows
        without replacement.

        Args:
            n_replicates: The number of bootstrap replicates.
            seed: The seed of the bootstrap.
        """
        rows = arange(self.__sample_size)
        self.first_order_indices = self.__compute_rank_indices(rows)
        self.total_order_indices = empty(0)
        self.second_order_indices = empty((0, 0))
        self.total_order_interval = (empty(0), empty(0))
        rng = default_rng(seed)
        replicates = empty((n_replicates, self.__dimension))
        bootstrap_size = round(rank_bootstrap_sample_ratio * self.__sample_size)
        for replicate in range(n_replicates):
            replicates[replicate] = self.__compute_rank_indices(
                rng.choice(self.__sample_size, bootstrap_size, replace=False)
            )

        self.n_degenerate_replicates = int(isnan(replicates).any(axis=1).sum())
        self.first_order_interval = self.__compute_percentiles(replicates)

    @staticmethod
    def __compute_rank_variance(probability: float, n_rows: int) -> float:
        r"""Compute the variance the rank-based indices are divided by.

        Args:
            probability: The estimate of the failure probability.
            n_rows: The number of rows.

        Returns:
            The variance $\frac{n}{n-1}\hat{p}(1-\hat{p})$.
        """
        return n_rows / (n_rows - 1) * probability * (1.0 - probability)

    def __compute_rank_indices(self, rows: IntegerArray) -> RealArray:
        r"""Estimate the first-order indices with the weighted rank-based estimator.

        For each coordinate $i$,
        the rows are sorted along $u_i$
        and each row $r_k$ is paired with its successor $r_{N(k)}$
        (the last one with the first one):
        $\hat{S}_i=\frac{\frac{1}{n}\sum_k \omega_{\{i\}}(r_k, r_{N(k)}) y_k y_{N(k)}-\hat{p}^2}{\frac{n}{n-1}\hat{p}(1-\hat{p})}$
        where $n$ is the number of rows.

        Args:
            rows: The indices of the rows.

        Returns:
            The first-order indices.
        """  # noqa: E501
        probability = self.__compute_weighted_indicator(rows).mean()
        variance = self.__compute_rank_variance(probability, len(rows))
        if variance <= 0.0:
            return full(self.__dimension, nan)

        first = empty(self.__dimension)
        for i in range(self.__dimension):
            sorted_rows = rows[argsort(self.__standard_samples[rows, i])]
            successors = roll(sorted_rows, -1)
            first[i] = (
                self.__compute_pair_products(sorted_rows, successors, [i]).mean()
                - probability**2
            ) / variance

        return first
