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
#                         documentation
#        :author: Olivier Sapin
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
r"""Sensitivity analysis using control variates (CV) for Sobol' indices computation.

Introduction to CV
------------------

The CV estimator of a statistic :math:`\theta` is defined as

.. math::
    \hat{\theta}^{\textnormal{CV}}(\boldsymbol{\alpha}) = \hat{\theta} -
    \boldsymbol{\alpha^\intercal} (\hat{\boldsymbol{\tau}} - \boldsymbol{\tau}),

where

- :math:`\hat{\theta}` is the Monte Carlo estimator of :math:`\theta`,
- :math:`\boldsymbol{\tau} = (\tau_1,\dots,\tau_M)` with :math:`\tau_i` the statistic
  of the random variable :math:`Z_i`,
- :math:`\hat{\boldsymbol{\tau}} = (\hat{\tau_1},\dots,\hat{\tau_M})` with
  :math:`\hat{\tau_i}` the Monte Carlo estimator of the statistic :math:`\tau_i`,
- :math:`\boldsymbol{\alpha}  \in \mathbb{R}^M` is a control parameter.

The statistics :math:`\tau_1,\dots,\tau_M` corresponds to :math:`\theta`
and the random variables :math:`Z_1,\ldots,Z_M` are used as control variates.

The control parameter :math:`\boldsymbol{\alpha}` minimizing the variance of the CV
estimator :math:`\hat{\theta}^{\textnormal{CV}}(\boldsymbol{\alpha})` is given by

.. math::
    \boldsymbol{\alpha}^* = \boldsymbol{\Sigma}^{-1} \mathbf{c},

with :math:`\mathbf{c} = \mathbb{C}[\hat{\boldsymbol{\tau}},\hat{\theta}]
\in \mathbb{R}^{M}` and :math:`\boldsymbol{\Sigma} =
\mathbb{C}[\hat{\boldsymbol{\tau}},\hat{\boldsymbol{\tau}}] \in \mathbb{R}^{M \times M}`
where :math:`\mathbb{C}` is the covariance operator.

CV estimation of Sobol' indices
-------------------------------

Given the function of interest :math:`f` from :math:`\mathbb{R}^d` to
:math:`\mathbb{R}`, the :math:`d` independent input random variables
:math:`X_1,\ldots,X_d` and the output random variable :math:`Y=f(X_1,\ldots,X_d)`,
the numerator of the first-order index of :math:`X_i` is
:math:`\theta := V_{i} = \mathbb{V}[\mathbb{E}[Y|X_i]]`.

Its Monte Carlo estimator proposed by Saltelli in :cite:`saltelli2010` is

.. math::
   \hat{\theta} := \hat{V}_{i} = \hat{E}[Y(Y^{(i)}-Y')]

with :math:`Y'=f(X'_1,\ldots,X'_d)` and
:math:`Y^{(i)}=f(X'_1,\ldots,X'_{i-1},X_{i},X'_{i+1},\ldots,X'_{d})`
where the random vectors
:math:`(X'_1,\ldots,X'_d)`
and
:math:`(X_1,\ldots,X_d)`
are independent and identically distributed.

Given :math:`M` surrogate models :math:`f_1,\ldots,f_M` of :math:`f`,
the elements of the optimal control parameter :math:`\boldsymbol{\alpha}^*`
are given by

.. math::
   \mathbf{c} = \frac{1}{n} \mathbb{C}[Y (Y^{(i)}-Y'),
   \mathbf{Z} \odot (\mathbf{Z}^{(i)}-\mathbf{Z}')], \\
   \boldsymbol{\Sigma} = \frac{1}{n} \mathbb{V}[\mathbf{Z}
   \odot (\mathbf{Z}^{(i)}-\mathbf{Z}')].

where :math:`\odot` denotes the element-wise multiplication,
:math:`Z_j=f_j(X_1,\ldots,X_d)`,
:math:`Z_j'=f_j(X_1',\ldots,X_d')` and
:math:`Z_j^{(i)}=f_j(X'_1,\ldots,X'_{i-1},X_{i},X'_{i+1},\ldots,X'_{d})`.

For the numerator of the total-order index
:math:`T_{i} = \mathbb{V}[Y] - \mathbb{V}[
\mathbb{E}[Y|X_1,...,X_{i-1},X_{i+1},...,X_{d}]]`,
a Monte Carlo estimator is

.. math::
   \hat{T}_{i}  = \frac{1}{2} \hat{E}[(Y' - Y^{(i)})^2]

and the elements of the optimal control parameter :math:`\boldsymbol{\alpha}^*`
are given by

.. math::
   \mathbf{c} = \frac{1}{n} \mathbb{C}[(Y' - Y^{(i)})^2,
   (\mathbf{Z}' - \mathbf{Z}^{(i)})^{\odot 2}], \\
   \boldsymbol{\Sigma} = \frac{1}{n} \mathbb{V}[
   (\mathbf{Z}' - \mathbf{Z}^{(i)})^{\odot 2}].

where :math:`\mathbf{A}^{\odot 2}` is the element-wise square of :math:`\mathbf{A}`.
"""

from __future__ import annotations

import contextlib
from itertools import starmap
from typing import TYPE_CHECKING
from typing import Callable

from numpy import array
from numpy import cov
from numpy import diag
from numpy import newaxis
from numpy import quantile
from numpy import vstack
from numpy import zeros
from numpy.linalg import LinAlgError
from scipy.linalg import solve

from gemseo.utils.data_conversion import concatenate_dict_of_arrays_to_array

if TYPE_CHECKING:
    from collections.abc import Iterable

    from gemseo.typing import IntegerArray
    from gemseo.typing import RealArray


class CVSobolAlgorithm:
    """Algorithm to estimate the Sobol' indices using control variates.

    This algorithm is based on the pick-and-freeze (PF) technique.
    """

    __confidence_level: float
    """The level of the confidence intervals."""

    __cv_indices: tuple[dict[str, dict[str, RealArray]]]
    """The output Sobol' indices of the ``n_control_variates`` control variates."""

    __cv_variance: tuple[float]
    """The output variance of the ``n_control_variates`` control variates."""

    __f_a: RealArray
    """The discipline output data for the samples ``1`` to ``sample_size``.

    Shape: ``(sample_size,)``.
    """

    __f_b: RealArray
    """The discipline output data for the samples ``sample_size`` to ``2*sample_size``.

    Shape: ``(sample_size,)``.
    """

    __f_ab: RealArray
    """The centered discipline output data for the samples ``1`` to ``2*sample_size``.

    Shape: ``(2*sample_size,)``.
    """

    __f_mix: RealArray
    """The discipline output data for the PF-based samples.

    Shape: ``(n_inputs, sample_size)``.
    """

    __first_indices_interval: RealArray
    """The bootstrap confidence intervals for the first-order Sobol' indices.

    Shape: ``(2, n_inputs)``.
    """

    __g_a: RealArray
    """The CV output data for the samples ``1`` to ``sample_size``.

    Shape: ``(n_control_variates, sample_size)``.
    """

    __g_b: RealArray
    """The CV output data for the samples ``sample_size`` to ``2*sample_size``.

    Shape: ``(n_control_variates, sample_size)``.
    """

    __g_ab: RealArray
    """The centered CV output data for the samples ``1`` to ``2*sample_size``.

    Shape: ``(n_control_variates, 2*sample_size)``.
    """

    __g_mix: RealArray
    """The CV output data for the PF-based samples.

    Shape: ``(n_inputs, n_control_variates, sample_size)``.
    """

    __sample_size: int
    """The number of independent samples composing each of the two independent input
    datasets used for Sobol' analysis."""

    __total_indices_interval: RealArray
    """The bootstrap confidence intervals for the total-order Sobol' indices.

    Shape: ``(2, n_inputs)``.
    """

    variance: float
    """The output variance estimated with control variates."""

    def __init__(
        self,
        n_inputs: int,
        output_data: RealArray,
        cv_output_data: RealArray,
        cv_statistics: list[tuple[float, dict[str, dict[str, RealArray]]]],
        bootstrap_samples: Iterable[tuple[IntegerArray, IntegerArray]],
        confidence_level: float = 0.95,
    ) -> None:
        """
        Args:
            n_inputs: The dimension of the input data to estimate the sensitivity
                indices.
            output_data: The discipline output data shaped as ``(n_samples,)``.
            cv_output_data: The output data of the control variates to estimate the
                sensitivity indices shaped as ``(n_control_variates, n_samples)``.
            cv_statistics: For each control variate, the variance of the output
                and the Sobol' indices of the form ``{order: {input_name: numerator}``.
            bootstrap_samples: The bootstrap samples used for the computation of the
                confidence intervals.
            confidence_level: The level of the confidence intervals.
        """  # noqa: D205, D212
        self.__n_inputs = n_inputs
        self.__sample_size = sample_size = len(output_data) // (2 + n_inputs)

        samples_a = range(sample_size)
        samples_b = range(sample_size, 2 * sample_size)
        samples_ab = range(2 * sample_size)
        samples_mix = [
            range((2 + i) * sample_size, (3 + i) * sample_size) for i in range(n_inputs)
        ]

        self.__f_a = output_data[samples_a]
        self.__f_b = output_data[samples_b]
        f_ab = output_data[samples_ab]
        self.__f_ab = f_ab - f_ab.mean()
        self.__f_mix = array([output_data[samples] for samples in samples_mix])

        self.__g_a = cv_output_data[:, samples_a]
        self.__g_b = cv_output_data[:, samples_b]
        g_ab = cv_output_data[:, samples_ab]
        self.__g_ab = g_ab - g_ab.mean(axis=-1)[:, newaxis]
        self.__g_mix = array([
            [output_data[samples] for output_data in cv_output_data]
            for samples in samples_mix
        ])

        self.__cv_variance, self.__cv_indices = zip(*cv_statistics)
        self.variance = self.__compute_variance()
        self.__confidence_level = confidence_level
        self.__bootstrap_samples = bootstrap_samples

    @staticmethod
    def __compute_statistic(
        mc_stats: float | RealArray,
        mc_cv_stats: RealArray,
        cv_stats: RealArray,
        covariances: list[RealArray],
    ) -> RealArray:
        """Compute statistics using control variance.

        Args:
            mc_stats: The Monte Carlo estimation of the statistics.
            mc_cv_stats: The Monte Carlo estimations of the CV statistics,
                shaped as ``(n_statistics, n_control_variates)``.
            cv_stats: The CV statistics
                shaped as ``(n_statistics, n_control_variates)``.
            covariances: The covariance matrices
                of the estimator output and the control variates;
                one matrix per control variate.

        Returns:
            The statistics estimated using control variates shaped as
            ``(n_statistics,)``.
        """
        alpha_star = zeros([len(covariances), mc_cv_stats.shape[1]])
        for i, covariance in enumerate(covariances):
            with contextlib.suppress(LinAlgError):
                alpha_star[i, :] = solve(
                    covariance[1:, 1:],
                    covariance[0, 1:],
                    assume_a="sym",
                    overwrite_a=True,
                    overwrite_b=True,
                )
        return mc_stats - (alpha_star * (mc_cv_stats - cv_stats)).sum(axis=1)

    def __compute_intervals(
        self, f_s: Iterable[RealArray], g_s: Iterable[RealArray], cv_stats: RealArray
    ) -> RealArray:
        """Compute the confidence intervals via bootstrap.

        Args:
            f_s: The statistics output data;
                one matrix shaped as ``(n_samples,)`` per control variate.
            g_s: The statistics output data of the control variates;
                one matrix shaped as ``(n_control_variates, n_samples)`` per control
                variate.
            cv_stats: The CV statistics
                shaped as ``(n_statistics, n_control_variates)``.

        Returns:
            The confidence intervals shaped as ``(2, n_inputs,)``.
        """
        n_statistics = cv_stats.shape[0]
        stats = zeros([len(list(self.__bootstrap_samples)), n_statistics])
        n = self.__sample_size
        for k, (samples_a, samples_ab) in enumerate(self.__bootstrap_samples):
            cov_f_g_b = cov(self.__f_ab[samples_ab], self.__g_ab[:, samples_ab])
            cov_f2_cg2_b = cov(
                self.__f_ab[samples_ab] ** 2, self.__g_ab[:, samples_ab] ** 2
            )
            var_b = self.__compute_statistic(
                cov_f_g_b[0, 0],
                diag(cov_f_g_b[1:, 1:])[newaxis, :],
                array(self.__cv_variance, ndmin=2),
                [cov_f2_cg2_b + 2 / (2 * n - 1) * cov_f_g_b**2],
            )[0]

            f_s_b = [f_i[samples_a] for f_i in f_s]
            g_s_b = [g_i[:, samples_a] for g_i in g_s]
            stats[k, :] = (
                self.__compute_statistic(
                    array([f_s_b_i.mean(axis=-1) for f_s_b_i in f_s_b]),
                    array([g_s_b_i.mean(axis=-1) for g_s_b_i in g_s_b]),
                    cv_stats / self.variance * var_b,
                    list(starmap(cov, zip(f_s_b, g_s_b))),
                )
                / var_b
            )
        prob = (1.0 - self.__confidence_level) / 2
        return vstack([
            quantile(stats, prob, axis=0),
            quantile(stats, 1 - prob, axis=0),
        ])

    def __compute_variance(self) -> float:
        """Compute the variance using control variates.

        Returns:
            The variance estimated using control variates shaped as ``(1,)``.
        """
        n = 2 * self.__sample_size
        cov_f_g = cov(self.__f_ab, self.__g_ab)
        cov_f2_cg2 = cov(self.__f_ab**2, self.__g_ab**2)
        return self.__compute_statistic(
            cov_f_g[0, 0],
            diag(cov_f_g[1:, 1:])[newaxis, :],
            array(self.__cv_variance, ndmin=2),
            [cov_f2_cg2 + 2 / (n - 1) * cov_f_g**2],
        )[0]

    def __compute_indices(
        self, order: str, func: Callable[[RealArray, RealArray, RealArray], RealArray]
    ) -> tuple[RealArray, RealArray]:
        """Compute the Sobol' indices and their confidence intervals.

        Args:
            order: The order of the Sobol' indices.
            func: The function allowing to compute the output of the random variable
            associated to the estimator of the Sobol' indices.

        Returns:
            The Sobol' indices of the given order shaped as ``(n_inputs,)`` and their
            confidence intervals shaped as ``(2, n_inputs)``.
        """
        cv_indices_numerator = (
            self.variance
            * array([
                concatenate_dict_of_arrays_to_array(
                    indices[order], indices[order].keys()
                )
                for indices in self.__cv_indices
            ]).T
        )

        f_s = [func(self.__f_a, self.__f_b, f_mix) for f_mix in self.__f_mix]
        g_s = [func(self.__g_a, self.__g_b, g_mix) for g_mix in self.__g_mix]

        return (
            self.__compute_statistic(
                array([f_s_i.mean(axis=-1) for f_s_i in f_s]),
                array([g_s_i.mean(axis=-1) for g_s_i in g_s]),
                cv_indices_numerator,
                list(starmap(cov, zip(f_s, g_s))),
            )
            / self.variance,
            self.__compute_intervals(f_s, g_s, cv_indices_numerator),
        )

    def compute_first_indices(self) -> RealArray:
        """Compute the first-order Sobol' indices.

        Returns:
            The first-order Sobol' indices shaped as ``(n_inputs,)``.
        """
        first_indices, self.__first_indices_interval = self.__compute_indices(
            "first", lambda f_a, f_b, f_mix: f_b * (f_mix - f_a)
        )
        return first_indices

    def compute_total_indices(self) -> RealArray:
        """Compute the total-order Sobol' indices.

        Returns:
            The total-order Sobol' indices shaped as ``(n_inputs,)``.
        """
        total_indices, self.__total_indices_interval = self.__compute_indices(
            "total", lambda f_a, f_b, f_mix: (f_a - f_mix) ** 2 / 2
        )
        return total_indices

    @property
    def first_indices_interval(self) -> RealArray:
        """The confidence interval of the first-order Sobol' indices.

        Warnings:
            You must first call :meth:`.compute_first_indices`.

        Returns:
            The confidence intervals shaped as ``(2, n_inputs)``.
        """
        return self.__first_indices_interval

    @property
    def total_indices_interval(self) -> RealArray:
        """The confidence intervals of the total-order Sobol' indices.

        Warnings:
            You must first call :meth:`.compute_total_indices`.

        Returns:
            The confidence intervals shaped as ``(2, n_inputs)``.
        """
        return self.__total_indices_interval
