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

.. _FunctionalChaosAlgorithm: https://openturns.github.io/openturns/latest/user_manual/response_surface/_generated/openturns.FunctionalChaosAlgorithm.html
.. _CleaningStrategy: https://openturns.github.io/openturns/latest/user_manual/response_surface/_generated/openturns.CleaningStrategy.html
.. _FixedStrategy: http://openturns.github.io/openturns/latest/user_manual/response_surface/_generated/openturns.FixedStrategy.html
.. _LARS: https://openturns.github.io/openturns/latest/theory/meta_modeling/polynomial_sparse_least_squares.html#polynomial-sparse-least-squares
.. _hyperbolic and anisotropic enumerate function: https://openturns.github.io/openturns/latest/user_manual/_generated/openturns.HyperbolicAnisotropicEnumerateFunction.html

The polynomial chaos expansion (PCE) model expresses an output variable
as a weighted sum of polynomial functions which are orthonormal
in the stochastic input space spanned by the random input variables:

.. math::

    Y = w_0 + w_1\phi_1(X) + w_2\phi_2(X) + ... + w_K\phi_K(X)

where :math:`\phi_i(x)=\psi_{\tau_1(i),1}(x_1)\times\ldots\times
\psi_{\tau_d(i),d}(x_d)`
and :math:`d` is the number of input variables.

Enumeration strategy
--------------------

The choice of the function :math:`\tau=(\tau_1,\ldots,\tau_d)` is an
enumeration strategy and :math:`\tau_j(i)` is the polynomial degree of
:math:`\psi_{\tau_j(i),j}`.

Distributions
-------------

PCE models depend on random input variables
and are often used to deal with uncertainty quantification problems.

If :math:`X_j` is a Gaussian random variable,
:math:`(\psi_{ij})_{i\geq 0}` is the Legendre basis.
If :math:`X_j` is a uniform random variable,
:math:`(\psi_{ij})_{i\geq 0}` is the Hermite basis.

When the problem is deterministic,
we can still use PCE models under the assumption that
the input variables are independent uniform random variables.
Then,
the orthonormal function basis is the Hermite one.

Degree
------

The degree :math:`P` of a PCE model is defined
in such a way that :math:`\max_i \text{degree}(\phi_i)=\sum_{j=1}^d\tau_j(i)=P`.

Estimation
----------

The coefficients :math:`(w_1, w_2, ..., w_K)` and the intercept :math:`w_0`
are estimated either by least-squares regression or a quadrature rule.
In the case of least-squares regression,
a sparse strategy can be considered with the `LARS`_ algorithm
and in both cases,
the `CleaningStrategy` can also remove the non-significant coefficients.

Dependence
----------
The PCE model relies on the OpenTURNS class `FunctionalChaosAlgorithm`_.
"""  # noqa: E501

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing import ClassVar
from typing import Final

from numpy import array
from numpy import atleast_1d
from numpy import concatenate
from numpy import ndarray
from numpy import vstack
from numpy import zeros
from openturns import LARS
from openturns import CleaningStrategy
from openturns import ComposedDistribution
from openturns import CorrectedLeaveOneOut
from openturns import FixedStrategy
from openturns import FunctionalChaosAlgorithm
from openturns import FunctionalChaosRandomVector
from openturns import FunctionalChaosSobolIndices
from openturns import GaussProductExperiment
from openturns import HyperbolicAnisotropicEnumerateFunction
from openturns import IntegrationStrategy
from openturns import LeastSquaresMetaModelSelectionFactory
from openturns import LeastSquaresStrategy
from openturns import OrthogonalBasis
from openturns import OrthogonalProductPolynomialFactory
from openturns import Point
from openturns import StandardDistributionPolynomialFactory

from gemseo.datasets.io_dataset import IODataset
from gemseo.mlearning.regression.regression import MLRegressionAlgo
from gemseo.uncertainty.distributions.openturns.composed import OTComposedDistribution
from gemseo.utils.string_tools import pretty_str

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Mapping

    from gemseo.algos.parameter_space import ParameterSpace
    from gemseo.core.discipline import MDODiscipline
    from gemseo.mlearning.core.ml_algo import TransformerType
    from gemseo.mlearning.core.supervised import SavedObjectType

LOGGER = logging.getLogger(__name__)


@dataclass
class CleaningOptions:
    """The options of the `CleaningStrategy`_."""

    max_considered_terms: int = 100
    """The maximum number of coefficients of the polynomial basis to be considered."""

    most_significant: int = 20
    """The maximum number of efficient coefficients of the polynomial basis to be
    kept."""

    significance_factor: float = 1e-4
    """The threshold to select the efficient coefficients of the polynomial basis."""


class PCERegressor(MLRegressionAlgo):
    """Polynomial chaos expansion model.

    See Also: API documentation of the OpenTURNS class `FunctionalChaosAlgorithm`_.
    """

    SHORT_ALGO_NAME: ClassVar[str] = "PCE"
    LIBRARY: Final[str] = "OpenTURNS"
    __WEIGHT: Final[str] = "weight"

    def __init__(
        self,
        data: IODataset | None,
        probability_space: ParameterSpace,
        transformer: TransformerType = MLRegressionAlgo.IDENTITY,
        input_names: Iterable[str] | None = None,
        output_names: Iterable[str] | None = None,
        degree: int = 2,
        discipline: MDODiscipline | None = None,
        use_quadrature: bool = False,
        use_lars: bool = False,
        use_cleaning: bool = False,
        hyperbolic_parameter: float = 1.0,
        n_quadrature_points: int = 0,
        cleaning_options: CleaningOptions | None = None,
    ) -> None:
        """
        Args:
            data: The learning dataset required
                in the case of the least-squares regression
                or when ``discipline`` is ``None`` in the case of quadrature.
            probability_space: The set of random input variables
                defined by :class:`.OTDistribution` instances.
            degree: The polynomial degree of the PCE.
            discipline: The discipline to be sampled
                if ``use_quadrature`` is ``True`` and ``data`` is ``None``.
            use_quadrature: Whether to estimate the coefficients of the PCE
                by a quadrature rule;
                if so, use the quadrature points stored in ``data``
                or sample ``discipline``.
                otherwise, estimate the coefficients by least-squares regression.
            use_cleaning: Whether
                to use the `CleaningStrategy`_ algorithm.
                Otherwise,
                use a fixed truncation strategy (`FixedStrategy`_).
            use_lars: Whether to use the `LARS`_ algorithm
                in the case of the least-squares regression.
            n_quadrature_points: The total number of quadrature points
                used by the quadrature strategy
                to compute the marginal number of points by input dimension
                when ``discipline`` is not ``None``.
                If ``0``, use :math:`(1+P)^d` points,
                where :math:`d` is the dimension of the input space
                and :math:`P` is the polynomial degree of the PCE.
            hyperbolic_parameter: The :math:`q`-quasi norm parameter
                of the `hyperbolic and anisotropic enumerate function`_,
                defined over the interval :math:`]0,1]`.
            cleaning_options: The options of the `CleaningStrategy`_.
                If ``None``, use :attr:`.DEFAULT_CLEANING_OPTIONS`.

        Raises:
            ValueError: When both data and discipline are missing,
                when both data and discipline are provided,
                when discipline is provided in the case of least-squares regression,
                when data is missing in the case of least-squares regression,
                when the probability space does not contain the distribution
                of an input variable,
                when an input variable has a data transformer
                or when a probability distribution is not an :class:`.OTDistribution`.
        """  # noqa: D205 D212
        if cleaning_options is None:
            cleaning_options = CleaningOptions()

        if use_quadrature:
            if discipline is None and data is None:
                raise ValueError(
                    "The quadrature rule requires either data or discipline."
                )

            if discipline is not None and data is not None and len(data):
                raise ValueError(
                    "The quadrature rule requires data or discipline but not both."
                )

            if use_lars:
                raise ValueError("LARS is not applicable with the quadrature rule.")

            if data is None:
                data = IODataset()

        else:
            if data is None:
                raise ValueError("The least-squares regression requires data.")

            if discipline is not None:
                raise ValueError(
                    "The least-squares regression does not require a discipline."
                )

        super().__init__(
            data,
            transformer=transformer,
            input_names=input_names,
            output_names=output_names,
            probability_space=probability_space,
            degree=degree,
            n_quadrature_points=n_quadrature_points,
            use_lars=use_lars,
            use_cleaning=use_cleaning,
            hyperbolic_parameter=hyperbolic_parameter,
            cleaning_options=cleaning_options,
        )

        if use_quadrature and data.empty:
            self.input_names = probability_space.variable_names

        if not data.empty:
            missing = set(self.input_names) - set(probability_space.uncertain_variables)
            if missing:
                raise ValueError(
                    "The probability space does not contain "
                    "the probability distributions "
                    f"of the random input variables: {pretty_str(missing)}."
                )

        if [
            key
            for key in self.transformer
            if key in self.input_names or key == IODataset.INPUT_GROUP
        ]:
            raise ValueError("PCERegressor does not support input transformers.")

        distributions = probability_space.distributions
        wrongly_distributed_random_variable_names = [
            input_name
            for input_name in self.input_names
            if not isinstance(
                distributions.get(input_name, None), OTComposedDistribution
            )
        ]
        if wrongly_distributed_random_variable_names:
            raise ValueError(
                "The probability distributions of the random variables "
                f"{pretty_str(wrongly_distributed_random_variable_names)} "
                "are not instances of OTComposedDistribution."
            )

        self.__variable_sizes = probability_space.variable_sizes
        self.__input_dimension = sum(
            self.__variable_sizes[name] for name in self.input_names
        )
        self.__use_quadrature = use_quadrature
        self.__use_lars_algorithm = use_lars
        self.__use_cleaning_truncation_algorithm = use_cleaning
        self.__cleaning = cleaning_options
        self.__hyperbolic_parameter = hyperbolic_parameter
        self.__degree = degree
        self.__composed_distribution = ComposedDistribution([
            marginal
            for input_name in self.input_names
            for distribution in distributions[input_name].marginals
            for marginal in distribution.marginals
        ])

        if use_quadrature:
            if discipline is not None:
                self.__quadrature_points_with_weights = self._get_quadrature_points(
                    n_quadrature_points, discipline
                )
            else:
                self.__quadrature_points_with_weights = (
                    self.learning_set.get_view(
                        group_names=self.learning_set.INPUT_GROUP
                    ).to_numpy(),
                    self.learning_set.get_view(variable_names=self.__WEIGHT)
                    .to_numpy()
                    .ravel(),
                )
        else:
            self.__quadrature_points_with_weights = None

        self._mean = array([])
        self._covariance = array([])
        self._variance = array([])
        self._standard_deviation = array([])
        self._first_order_sobol_indices = []
        self._second_order_sobol_indices = []
        self._total_order_sobol_indices = []
        self._prediction_function = None

    def __instantiate_functional_chaos_algorithm(
        self, input_data: ndarray, output_data: ndarray
    ) -> FunctionalChaosAlgorithm:
        """Instantiate the :class:`FunctionalChaosAlgorithm`.

        Args:
            input_data: The learning input data.
            output_data: The learning output data.

        Returns:
            A functional chaos algorithm fitted from learning data.
        """
        # Create the polynomial basis and the associated enumeration function.
        enumerate_function = HyperbolicAnisotropicEnumerateFunction(
            self.__input_dimension, self.__hyperbolic_parameter
        )
        polynomial_basis = OrthogonalProductPolynomialFactory(
            [
                StandardDistributionPolynomialFactory(marginal)
                for marginal in self.__composed_distribution.getDistributionCollection()
            ],
            enumerate_function,
        )

        # Create the strategy to compute the coefficients of the PCE.
        if self.__use_quadrature:
            evaluation_strategy = IntegrationStrategy()
        elif self.__use_lars_algorithm:
            evaluation_strategy = LeastSquaresStrategy(
                input_data,
                output_data,
                LeastSquaresMetaModelSelectionFactory(LARS(), CorrectedLeaveOneOut()),
            )
        else:
            evaluation_strategy = LeastSquaresStrategy(input_data, output_data)

        # Apply the cleaning strategy if desired;
        # otherwise use a standard fixed strategy.
        if self.__use_cleaning_truncation_algorithm:
            max_terms = enumerate_function.getMaximumDegreeCardinal(self.__degree)
            if self.__cleaning.max_considered_terms > max_terms:
                LOGGER.warning(
                    "max_considered_terms is too important; set it to max_terms."
                )
                self.__cleaning.max_considered_terms = max_terms

            if self.__cleaning.most_significant > self.__cleaning.max_considered_terms:
                LOGGER.warning(
                    "most_significant is too important; set it to max_considered_terms."
                )
                self.__cleaning.most_significant = self.__cleaning.max_considered_terms

            truncation_strategy = CleaningStrategy(
                OrthogonalBasis(polynomial_basis),
                self.__cleaning.max_considered_terms,
                self.__cleaning.most_significant,
                self.__cleaning.significance_factor,
                True,
            )
        else:
            truncation_strategy = FixedStrategy(
                polynomial_basis,
                enumerate_function.getStrataCumulatedCardinal(self.__degree),
            )

        # Return the function chaos algorithm.
        if self.__use_quadrature:
            return FunctionalChaosAlgorithm(
                input_data,
                self.__quadrature_points_with_weights[1],
                output_data,
                self.__composed_distribution,
                truncation_strategy,
                IntegrationStrategy(),
            )

        return FunctionalChaosAlgorithm(
            input_data,
            output_data,
            self.__composed_distribution,
            truncation_strategy,
            evaluation_strategy,
        )

    @staticmethod
    def __simplify_sobol_indices(
        x: list[float] | list[list[float]],
    ) -> float | list[float] | list[list[float]]:
        """Simplify Sobol' indices into a unique Sobol' index if possible.

        Args:
            x: The Sobol' indices.

        Returns:
            Either several Sobol' indices or a unique Sobol' index.
        """
        if isinstance(x[0], float):
            return x[0] if len(x) == 1 else x

        if len(x) == 1 and len(x[0]) == 1:
            return x[0][0]

        return x

    def _fit(
        self,
        input_data: ndarray,
        output_data: ndarray,
    ) -> None:
        # Create and train the PCE.
        algo = self.__instantiate_functional_chaos_algorithm(input_data, output_data)
        algo.run()
        self.algo = algo.getResult()
        self._prediction_function = self.algo.getMetaModel()

        # Compute some statistics.
        random_vector = FunctionalChaosRandomVector(self.algo)
        self._mean = array(random_vector.getMean())
        self._covariance = array(random_vector.getCovariance())
        self._variance = self._covariance.diagonal()
        self._standard_deviation = self._variance**0.5

        # Compute some sensitivity indices.
        names_to_positions = {}
        start = 0
        for name in self.input_names:
            stop = start + self.learning_set.variable_names_to_n_components[name]
            names_to_positions[name] = range(start, stop)
            start = stop

        ot_sobol_indices = FunctionalChaosSobolIndices(self.algo)
        self.__compute_first_or_total_order_indices(
            names_to_positions, ot_sobol_indices, True
        )
        self.__compute_second_order_indices(names_to_positions, ot_sobol_indices)
        self.__compute_first_or_total_order_indices(
            names_to_positions, ot_sobol_indices, False
        )

    def __compute_second_order_indices(
        self,
        names_to_positions: Mapping[str, Iterable[int]],
        ot_sobol_indices: FunctionalChaosSobolIndices,
    ) -> None:
        """Compute the second-order Sobol' indices.

        Args:
            names_to_positions: The input names
                bound to the positions in the uncertain input vector.
            ot_sobol_indices: The Sobol' indices.
        """
        self._second_order_sobol_indices = [
            {
                first_name: {
                    second_name: self.__simplify_sobol_indices([
                        [
                            (
                                ot_sobol_indices.getSobolGroupedIndex(
                                    [first_index, second_index], output_index
                                )
                                - ot_sobol_indices.getSobolIndex(
                                    first_index, output_index
                                )
                                - ot_sobol_indices.getSobolIndex(
                                    second_index, output_index
                                )
                            )
                            if first_index != second_index
                            else 0
                            for second_index in names_to_positions[second_name]
                        ]
                        for first_index in names_to_positions[first_name]
                    ])
                    for second_name in self.input_names
                }
                for first_name in self.input_names
            }
            for output_index in range(self.output_dimension)
        ]
        for names_to_names_to_indices in self._second_order_sobol_indices:
            for input_name, names_to_indices in names_to_names_to_indices.items():
                if not isinstance(names_to_indices[input_name], list):
                    names_to_indices.pop(input_name)

    def __compute_first_or_total_order_indices(
        self,
        names_to_positions: Mapping[str, Iterable[int]],
        ot_sobol_indices: FunctionalChaosSobolIndices,
        use_first: True,
    ) -> None:
        """Compute either the first- or total-order Sobol' indices.

        Args:
            names_to_positions: The input names
                bound to the positions in the uncertain input vector.
            ot_sobol_indices: The Sobol' indices.
            use_first: Whether to compute the first-order Sobol' indices.
        """
        method = (
            ot_sobol_indices.getSobolIndex
            if use_first
            else ot_sobol_indices.getSobolTotalIndex
        )
        indices = [
            {
                input_name: self.__simplify_sobol_indices([
                    method(input_index, output_index)
                    for input_index in names_to_positions[input_name]
                ])
                for input_name in self.input_names
            }
            for output_index in range(self.output_dimension)
        ]
        if use_first:
            self._first_order_sobol_indices = indices
        else:
            self._total_order_sobol_indices = indices

    def _predict(self, input_data: ndarray) -> ndarray:
        return array(self._prediction_function(input_data))

    def _get_quadrature_points(
        self, n_quadrature_points: int, discipline: MDODiscipline
    ) -> tuple[ndarray, ndarray]:
        """Return the quadrature points for PCE construction.

        Args:
            n_quadrature_points: The number of quadrature points
            discipline: The discipline to sample.

        Returns:
            The quadrature points with their associated weights.
        """
        if n_quadrature_points:
            degree_by_dim = int(n_quadrature_points ** (1.0 / self.__input_dimension))
        else:
            degree_by_dim = self.__degree + 1

        experiment = GaussProductExperiment(
            self.__composed_distribution, [degree_by_dim] * self.__input_dimension
        )
        quadrature_points, weights = experiment.generateWithWeights()
        quadrature_points, weights = array(quadrature_points), array(weights)
        input_group = self.learning_set.INPUT_GROUP
        self.learning_set.add_group(
            input_group,
            quadrature_points,
            self.input_names,
            self.__variable_sizes,
        )
        self.learning_set.add_variable(self.__WEIGHT, weights[:, None])

        output_names = list(discipline.get_output_data_names())
        input_names = list(discipline.get_input_data_names())
        outputs = [[] for _ in output_names]
        for input_data in self.learning_set.get_view(
            group_names=self.learning_set.INPUT_GROUP, variable_names=input_names
        ).to_numpy():
            input_data = {
                input_names[i]: atleast_1d(input_data[i])
                for i in range(len(input_data))
            }
            output_data = discipline.execute(input_data)
            for index, name in enumerate(output_names):
                outputs[index].append(output_data[name])

        self.learning_set.add_group(
            self.learning_set.OUTPUT_GROUP,
            concatenate(
                [vstack(outputs[index]) for index, _ in enumerate(output_names)], axis=1
            ),
            output_names,
            {k: v.size for k, v in discipline.get_output_data().items()},
        )
        self.output_names = output_names
        return quadrature_points, weights

    def _predict_jacobian(
        self,
        input_data: ndarray,
    ) -> ndarray:
        gradient = self._prediction_function.gradient
        input_size, output_size = self._reduced_dimensions
        jac = zeros((input_data.shape[0], output_size, input_size))
        for index, data in enumerate(input_data):
            jac[index] = array(gradient(Point(data))).T

        return jac

    @property
    def mean(self) -> ndarray:
        """The mean vector of the PCE model output."""
        self._check_is_trained()
        return self._mean

    @property
    def covariance(self) -> ndarray:
        """The covariance matrix of the PCE model output."""
        self._check_is_trained()
        return self._covariance

    @property
    def variance(self) -> ndarray:
        """The variance vector of the PCE model output."""
        self._check_is_trained()
        return self._variance

    @property
    def standard_deviation(self) -> ndarray:
        """The standard deviation vector of the PCE model output."""
        self._check_is_trained()
        return self._standard_deviation

    @property
    def first_sobol_indices(self) -> list[dict[str, float]]:
        """The first-order Sobol' indices for the different output dimensions."""
        self._check_is_trained()
        return self._first_order_sobol_indices

    @property
    def second_sobol_indices(self) -> list[dict[str, dict[str, float]]]:
        """The second-order Sobol' indices for the different output dimensions."""
        self._check_is_trained()
        return self._second_order_sobol_indices

    @property
    def total_sobol_indices(self) -> list[dict[str, float]]:
        """The total Sobol' indices for the different output dimensions."""
        self._check_is_trained()
        return self._total_order_sobol_indices

    def _get_objects_to_save(self) -> dict[str, SavedObjectType]:
        objects = super()._get_objects_to_save()
        objects["_prediction_function"] = self._prediction_function
        objects["_mean"] = self._mean
        objects["_covariance"] = self._covariance
        objects["_variance"] = self._variance
        objects["_standard_deviation"] = self._standard_deviation
        objects["_first_order_sobol_indices"] = self._first_order_sobol_indices
        objects["_second_order_sobol_indices"] = self._second_order_sobol_indices
        objects["_total_order_sobol_indices"] = self._total_order_sobol_indices
        return objects
