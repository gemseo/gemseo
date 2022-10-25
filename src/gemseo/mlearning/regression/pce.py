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
#        :author: Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
r"""Polynomial chaos expansion model.

The polynomial chaos expansion (PCE) model expresses an output variable
as a weighted sum of polynomial functions which are orthonormal
in the stochastic input space spanned by the random input variables:

.. math::

    Y = w_0 + w_1\phi_1(X) + w_2\phi_2(X) + ... + w_K\phi_K(X)

where :math:`\phi_i(x)=\psi_{\tau_1(i),1}(x_1)\times\ldots\times
\psi_{\tau_d(i),d}(x_d)`.

Enumerating strategy
--------------------

The choice of the function :math:`\tau=(\tau_1,\ldots,\tau_d)` is an
enumerating strategy and :math:`\tau_j(i)` is the polynomial degree of
:math:`\psi_{\tau_j(i),j}`.

Distributions
-------------

PCE models depend on random input variable
and are often used to deal with uncertainty quantification problems.

If :math:`X_j` is a Gaussian random variable,
:math:`(\psi_{ij})_{i\geq 0}` is the Legendre basis.
If :math:`X_j` is an uniform random variable,
:math:`(\psi_{ij})_{i\geq 0}` is the Hermite basis.

When the problem is deterministic,
we can still use PCE models under the assumptions that
the random variables are independent uniform random variables.
Then,
the orthonormal function basis is the Hermite one.

Degree
------

The degree :math:`P` of a PCE model is defined
in such a way that :math:`\text{degree}(\phi_i)=\sum_{j=1}^d\tau_j(i)\leq P`.

Estimation
----------

The coefficients :math:`(w_1, w_2, ..., w_K)` and the intercept :math:`w_0`
are estimated either by least squares regression,
sparse least squares regression or quadrature.

Dependence
----------
The PCE model relies on the FunctionalChaosAlgorithm class
of the `openturns library <https://openturns.github.io/openturns/latest/user_manual/
response_surface/_generated/openturns.FunctionalChaosAlgorithm.html>`_.
"""
from __future__ import annotations

import logging
from typing import ClassVar
from typing import Iterable
from typing import Mapping

import openturns
from numpy import all as np_all
from numpy import array
from numpy import concatenate
from numpy import isin
from numpy import ndarray
from numpy import zeros
from numpy import zeros_like
from openturns import AdaptiveStieltjesAlgorithm
from openturns import CleaningStrategy
from openturns import ComposedDistribution
from openturns import CorrectedLeaveOneOut
from openturns import FixedStrategy
from openturns import Function
from openturns import FunctionalChaosAlgorithm
from openturns import FunctionalChaosRandomVector
from openturns import FunctionalChaosSobolIndices
from openturns import GaussProductExperiment
from openturns import HyperbolicAnisotropicEnumerateFunction
from openturns import IntegrationStrategy
from openturns import LARS
from openturns import LeastSquaresMetaModelSelectionFactory
from openturns import LeastSquaresStrategy
from openturns import MarginalTransformationEvaluation
from openturns import OrthogonalBasis
from openturns import OrthogonalProductPolynomialFactory
from openturns import Point
from openturns import StandardDistributionPolynomialFactory

from gemseo.algos.parameter_space import ParameterSpace
from gemseo.core.dataset import Dataset
from gemseo.core.discipline import MDODiscipline
from gemseo.mlearning.core.ml_algo import TransformerType
from gemseo.mlearning.regression.regression import MLRegressionAlgo
from gemseo.uncertainty.distributions.openturns.distribution import OTDistribution
from gemseo.utils.python_compatibility import Final

LOGGER = logging.getLogger(__name__)


class PCERegressor(MLRegressionAlgo):
    """Polynomial chaos expansion model."""

    SHORT_ALGO_NAME: ClassVar[str] = "PCE"
    LIBRARY: Final[str] = "OpenTURNS"
    LS_STRATEGY: Final[str] = "LS"
    QUAD_STRATEGY: Final[str] = "Quad"
    SPARSE_STRATEGY: Final[str] = "SparseLS"
    AVAILABLE_STRATEGIES: list[str] = [LS_STRATEGY, QUAD_STRATEGY, SPARSE_STRATEGY]

    def __init__(
        self,
        data: Dataset,
        probability_space: ParameterSpace,
        discipline: MDODiscipline | None = None,
        transformer: Mapping[str, TransformerType] | None = None,
        input_names: Iterable[str] | None = None,
        output_names: Iterable[str] | None = None,
        strategy: str = LS_STRATEGY,
        degree: int = 2,
        n_quad: int | None = None,
        stieltjes: bool = True,
        sparse_param: Mapping[str, int | float] | None = None,
    ) -> None:
        """
        Args:
            probability_space: The set of random input variables
                defined by :class:`.OTDistribution` instances.
            discipline: The discipline to evaluate with the quadrature strategy
                if the learning set does not have output data.
                If None, use the output data from the learning set.
            strategy: The strategy to compute the parameters of the PCE,
                either 'LS' for *least-square*, 'Quad' for *quadrature*
                or 'SparseLS' for *sparse least-square*.
            degree: The polynomial degree of the PCE.
            n_quad: The total number of quadrature points
                used by the quadrature strategy
                to compute the marginal number of points by input dimension.
                If None, this degree will be set equal
                to the polynomial degree of the PCE plus one.
            stieltjes: Whether to use the Stieltjes method.
            sparse_param: The parameters for the Sparse Cleaning Truncation
                Strategy and/or hyperbolic truncation of the initial basis:

                - **max_considered_terms** (int) -- The maximum considered terms
                  (default: 120),
                - **most_significant** (int) -- The most Significant number to retain
                  (default: 30),
                - **significance_factor** (float) -- Significance Factor
                  (default: 1e-3),
                - **hyper_factor** (float) -- The factor for the hyperbolic truncation
                  strategy (default: 1.0).

                If None, use default values.

        Raises:
            ValueError: Either if the variables of the probability space
                and the input variables of the dataset are different,
                if transformers are specified for the inputs,
                if the strategy to compute the parameters of the PCE is unknown
                or if a probability distribution is not an :class:`.OTDistribution`.
        """
        if any(
            not isinstance(distribution, OTDistribution)
            for distribution in probability_space.distributions.values()
        ):
            raise ValueError(
                "The probability distributions must be instances of OTDistribution."
            )

        super().__init__(
            data,
            transformer=transformer,
            input_names=input_names,
            output_names=output_names,
            probability_space=probability_space,
            strategy=strategy,
            degree=degree,
            n_quad=n_quad,
            stieltjes=stieltjes,
            sparse_param=sparse_param,
        )
        self._prob_space = probability_space
        self._discipline = discipline
        if data:
            if set(self.input_names) != set(probability_space.variables_names):
                raise ValueError(
                    f"The input names {self.input_names} "
                    "and the names of the variables of the probability space "
                    f"{probability_space.variables_names} "
                    "are not all the same."
                )
        else:
            self.input_names = probability_space.variables_names

        forbidden_names = set(self.input_names).union({Dataset.INPUT_GROUP})
        if set(list(self.transformer.keys())).intersection(forbidden_names):
            raise ValueError("PCERegressor does not support input transformers.")

        self._distributions = probability_space.distributions
        self._sparse_param = {
            "max_considered_terms": 120,
            "most_significant": 30,
            "significance_factor": 1e-3,
            "hyper_factor": 1.0,
        }
        if sparse_param:
            self._sparse_param.update(sparse_param)
        self._input_dim = sum(dist.dimension for _, dist in self._distributions.items())
        self._strategy = strategy
        if self._strategy not in self.AVAILABLE_STRATEGIES:
            strategies = ", ".join(self.AVAILABLE_STRATEGIES)
            raise ValueError(
                f"The strategy {self._strategy} is not available; "
                f"available ones are: {strategies}."
            )
        self._stieltjes = stieltjes
        self._degree = degree
        self._n_quad = n_quad
        self._ot_distributions = [
            self._distributions[name].distribution for name in self.input_names
        ]
        self._dist = ComposedDistribution(self._ot_distributions)
        if self._strategy == self.QUAD_STRATEGY:
            (
                self._sample,
                self._weights,
                self._proj_strategy,
            ) = self._get_quadrature_points()
        else:
            self._proj_strategy = None
            self._sample = None
            self._weights = None

        self.__mean = array([])
        self.__covariance = array([])
        self.__variance = array([])
        self.__standard_deviation = array([])

    def _fit(
        self,
        input_data: ndarray,
        output_data: ndarray,
    ) -> None:
        proj_strategy = self._proj_strategy or self._get_proj_strategy(
            input_data, output_data
        )
        algo = FunctionalChaosAlgorithm(
            input_data,
            self._get_weights(input_data, proj_strategy),
            output_data,
            self._dist,
            self._get_trunc_strategy(),
        )
        algo.setProjectionStrategy(proj_strategy)
        algo.run()
        self.algo = algo.getResult()
        random_vector = FunctionalChaosRandomVector(self.algo)
        self.__mean = array(random_vector.getMean())
        self.__covariance = array(random_vector.getCovariance())
        self.__variance = self.__covariance.diagonal()
        self.__standard_deviation = self.__variance**0.5

    def _predict(
        self,
        input_data: ndarray,
    ) -> ndarray:
        return array(self.algo.getMetaModel()(input_data))

    @property
    def first_sobol_indices(self) -> dict[str, ndarray]:
        """The first-order Sobol' indices."""
        self._check_is_trained()
        sensitivity_analysis = FunctionalChaosSobolIndices(self.algo)
        LOGGER.info(str(sensitivity_analysis))
        first_order = {
            name: sensitivity_analysis.getSobolIndex(index)
            for index, name in enumerate(self.input_names)
        }
        return first_order

    @property
    def total_sobol_indices(self) -> dict[str, ndarray]:
        """The total Sobol' indices."""
        self._check_is_trained()
        sensitivity_analysis = FunctionalChaosSobolIndices(self.algo)
        LOGGER.info(str(sensitivity_analysis))
        return {
            name: sensitivity_analysis.getSobolTotalIndex(index)
            for index, name in enumerate(self.input_names)
        }

    def _get_basis(self) -> openturns.OrthogonalProductPolynomialFactory:
        """Return the orthogonal product polynomial factory for PCE construction.

        Returns:
            An orthogonal product polynomial factory computed by OpenTURNS.
        """
        enumerate_function = HyperbolicAnisotropicEnumerateFunction(
            self._input_dim, self._sparse_param["hyper_factor"]
        )
        if self._stieltjes:
            # Tend to result in performance issue
            return OrthogonalProductPolynomialFactory(
                [
                    StandardDistributionPolynomialFactory(
                        AdaptiveStieltjesAlgorithm(marginal)
                    )
                    for marginal in self._ot_distributions
                ],
                enumerate_function,
            )

        return OrthogonalProductPolynomialFactory(
            [
                StandardDistributionPolynomialFactory(margin)
                for margin in self._ot_distributions
            ],
            enumerate_function,
        )

    def _get_quadrature_points(
        self,
    ) -> tuple[ndarray, ndarray, openturns.IntegrationStrategy]:
        """Return the quadrature points for PCE construction.

        Returns:
            The quadrature points.
            The weights associated with the quadrature points.
            The projection strategy.
        """
        measure = self._get_basis().getMeasure()
        if self._n_quad is not None:
            degree_by_dim = int(self._n_quad ** (1.0 / self._input_dim))
        else:
            degree_by_dim = self._degree + 1
        degrees = [degree_by_dim] * self._input_dim

        proj_strategy = IntegrationStrategy(GaussProductExperiment(measure, degrees))
        sample, weights = proj_strategy.getExperiment().generateWithWeights()

        if not self._stieltjes:
            transformation = Function(
                MarginalTransformationEvaluation(
                    [measure.getMarginal(i) for i in range(self._input_dim)],
                    self._ot_distributions,
                    False,
                )
            )
            sample = transformation(sample)

        sample = array(sample)
        if not self.learning_set:
            inputs_names = self._prob_space.variables_names
            input_group = self.learning_set.INPUT_GROUP
            self.learning_set.set_from_array(
                sample,
                self._prob_space.variables_names,
                self._prob_space.variables_sizes,
                {name: input_group for name in inputs_names},
            )

        if self._discipline is not None:
            n_samples = len(self.learning_set)
            outputs = {name: [] for name in self._discipline.get_output_data_names()}
            for data in self.learning_set:
                output_data = self._discipline.execute(data)
                for name in self._discipline.get_output_data_names():
                    outputs[name] += list(output_data[name])

            for name in self._discipline.get_output_data_names():
                outputs[name] = array(outputs[name]).reshape((n_samples, -1))

            outputs_names = list(self._discipline.get_output_data_names())
            self.learning_set.add_group(
                self.learning_set.OUTPUT_GROUP,
                concatenate([outputs[name] for name in outputs_names], axis=1),
                outputs_names,
                {k: v.size for k, v in self._discipline.get_output_data().items()},
                cache_as_input=False,
            )
            self.output_names = outputs_names

        return sample, weights, proj_strategy

    def _get_trunc_strategy(
        self,
    ) -> openturns.AdaptiveStrategyImplementation:
        """Return the truncation strategy for PCE construction.

        Returns:
            The OpenTURNS truncation strategy.
        """
        enumerate_function = HyperbolicAnisotropicEnumerateFunction(
            self._input_dim, self._sparse_param["hyper_factor"]
        )
        basis = self._get_basis()
        if self._strategy != self.SPARSE_STRATEGY:
            return FixedStrategy(
                basis,
                enumerate_function.getStrataCumulatedCardinal(self._degree),
            )

        return CleaningStrategy(
            OrthogonalBasis(basis),
            self._sparse_param["max_considered_terms"],
            self._sparse_param["most_significant"],
            self._sparse_param["significance_factor"],
            True,
        )

    def _get_proj_strategy(
        self,
        input_data: ndarray,
        output_data: ndarray,
    ) -> openturns.LeastSquaresStrategy:
        """Return the projection strategy for PCE construction.

        Args:
            input_data: The input data with shape (n_samples, n_inputs).
            output_data: The output data with shape (n_samples, n_outputs).

        Returns:
            A projection strategy.
        """
        if self._strategy == self.QUAD_STRATEGY:
            return self._proj_strategy

        if self._strategy == self.LS_STRATEGY:
            return LeastSquaresStrategy(input_data, output_data)

        return LeastSquaresStrategy(
            input_data,
            output_data,
            LeastSquaresMetaModelSelectionFactory(LARS(), CorrectedLeaveOneOut()),
        )

    def _get_weights(self, input_data: ndarray, proj_strategy) -> ndarray:
        """Return the weights for PCE construction.

        Args:
            input_data: The input data with shape (n_samples, n_inputs).
        """
        if self._strategy == self.QUAD_STRATEGY:
            sample = zeros_like(self._sample)
            sample[: len(input_data)] = input_data
            sample[0] = self._sample[0]
            return array(self._weights)[np_all(isin(sample, self._sample), axis=1)]

        return proj_strategy.getExperiment().generateWithWeights()[1]

    def _predict_jacobian(
        self,
        input_data: ndarray,
    ) -> ndarray:
        gradient = self.algo.getMetaModel().gradient
        input_size, output_size = self._reduced_dimensions
        jac = zeros((input_data.shape[0], output_size, input_size))
        for index, data in enumerate(input_data):
            jac[index] = array(gradient(Point(data))).T

        return jac

    @property
    def mean(self) -> ndarray:
        """The mean vector of the PCE model output."""
        self._check_is_trained()
        return self.__mean

    @property
    def covariance(self) -> ndarray:
        """The covariance matrix of the PCE model output."""
        self._check_is_trained()
        return self.__covariance

    @property
    def variance(self) -> ndarray:
        """The variance vector of the PCE model output."""
        self._check_is_trained()
        return self.__variance

    @property
    def standard_deviation(self) -> ndarray:
        """The standard deviation vector of the PCE model output."""
        self._check_is_trained()
        return self.__standard_deviation
