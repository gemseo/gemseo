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

.. _FunctionalChaosAlgorithm: https://openturns.github.io/
openturns/latest/user_manual/response_surface/response_surface.html
.. _CleaningStrategy: https://openturns.github.io/
openturns/latest/user_manual/response_surface/_generated/openturns.CleaningStrategy.html
.. _LARS: https://openturns.github.io/
openturns/latest/theory/meta_modeling/polynomial_sparse_least_squares.html
.. _hyperbolic and anisotropic enumerate function: https://openturns.github.io/
openturns/latest/user_manual/_generated/openturns.HyperbolicAnisotropicEnumerateFunction.html

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
the `CleaningStrategy`_ can also remove the non-significant coefficients.

Dependence
----------
The PCE model relies on the OpenTURNS class `FunctionalChaosAlgorithm`_.
"""  # noqa: E501

from __future__ import annotations

import logging
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar
from typing import Final

from numpy import array
from numpy import atleast_1d
from numpy import concatenate
from numpy import hstack
from numpy import newaxis
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
from scipy.linalg import solve

from gemseo.datasets.io_dataset import IODataset
from gemseo.mlearning.regression.algos.base_regressor import BaseRegressor
from gemseo.mlearning.regression.algos.pce_settings import CleaningOptions
from gemseo.mlearning.regression.algos.pce_settings import PCERegressor_Settings
from gemseo.uncertainty.distributions.openturns.joint import OTJointDistribution
from gemseo.utils.pydantic import create_model
from gemseo.utils.string_tools import pretty_str

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Mapping

    from gemseo.core.discipline import Discipline
    from gemseo.typing import RealArray

LOGGER = logging.getLogger(__name__)


class PCERegressor(BaseRegressor):
    """Polynomial chaos expansion model.

    See Also: API documentation of the OpenTURNS class `FunctionalChaosAlgorithm`_.
    """

    SHORT_ALGO_NAME: ClassVar[str] = "PCE"
    LIBRARY: ClassVar[str] = "OpenTURNS"
    __WEIGHT: Final[str] = "weight"

    Settings: ClassVar[type[PCERegressor_Settings]] = PCERegressor_Settings

    _coefficients: RealArray | None
    """The coefficients to differentiate with respect to the special variables.

    Shaped as ``(output_dimension, special_variable_dimension, n_basis_functions)``.
    """

    _mean_jacobian_wrt_special_variables: RealArray | None
    """The gradient of the mean with respect to the special variables.

    Shaped as ``(output_dimension, special_variable_dimension)``.
    """

    _standard_deviation_jacobian_wrt_special_variables: RealArray | None
    """The gradient of the standard deviation with respect the special variables.

    Shaped as ``(output_dimension, special_variable_dimension)``.
    """

    _variance_jacobian_wrt_special_variables: RealArray | None
    """The gradient of the variance with respect the special variables.

    Shaped as ``(output_dimension, special_variable_dimension)``.
    """

    def __init__(
        self,
        data: IODataset,
        settings_model: PCERegressor_Settings | None = None,
        **settings: Any,
    ) -> None:
        """
        Args:
            data: The training dataset
                whose input space ``data.misc["input_space"]``
                is expected to be a :class:`.ParameterSpace`
                defining the random input variables.
                The training dataset can be empty
                in the case of quadrature when ``discipline`` is not ``None``.

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
        settings_ = create_model(
            self.Settings, settings_model=settings_model, **settings
        )
        cleaning_options = settings_.cleaning_options
        if cleaning_options is None:
            cleaning_options = CleaningOptions()

        # TODO: API: remove backward compatibility wrt data in gemseo v7.
        there_are_data = data is not None and len(data) > 0
        there_is_a_discipline = settings_.discipline is not None
        if settings_.use_quadrature:
            if not there_is_a_discipline and not there_are_data:
                msg = "The quadrature rule requires either data or discipline."
                raise ValueError(msg)

            if there_is_a_discipline and there_are_data:
                msg = "The quadrature rule requires data or discipline but not both."
                raise ValueError(msg)

            if settings_.use_lars:
                msg = "LARS is not applicable with the quadrature rule."
                raise ValueError(msg)

            if data is None:
                data = IODataset()

        else:
            if not there_are_data:
                msg = "The least-squares regression requires data."
                raise ValueError(msg)

            if there_is_a_discipline:
                msg = "The least-squares regression does not require a discipline."
                raise ValueError(msg)

        if settings_.probability_space is not None:
            data.misc["input_space"] = settings_.probability_space

        super().__init__(data, settings_model=settings_)

        probability_space = data.misc["input_space"]
        if self._settings.use_quadrature and data.empty:
            self.input_names = probability_space.variable_names

        if not data.empty:
            missing = set(self.input_names) - set(probability_space.uncertain_variables)
            if missing:
                msg = (
                    "The probability space does not contain "
                    "the probability distributions "
                    f"of the random input variables: {pretty_str(missing)}."
                )
                raise ValueError(msg)

        if [
            key
            for key in self.transformer
            if key in self.input_names or key == IODataset.INPUT_GROUP
        ]:
            msg = "PCERegressor does not support input transformers."
            raise ValueError(msg)

        distributions = probability_space.distributions
        wrongly_distributed_random_variable_names = [
            input_name
            for input_name in self.input_names
            if not isinstance(distributions.get(input_name, None), OTJointDistribution)
        ]
        if wrongly_distributed_random_variable_names:
            msg = (
                "The probability distributions of the random variables "
                f"{pretty_str(wrongly_distributed_random_variable_names)} "
                "are not instances of OTJointDistribution."
            )
            raise ValueError(msg)

        self.__variable_sizes = probability_space.variable_sizes
        self.__input_dimension = sum(
            self.__variable_sizes[name] for name in self.input_names
        )
        self.__use_quadrature = self._settings.use_quadrature
        self.__use_lars_algorithm = self._settings.use_lars
        self.__use_cleaning_truncation_algorithm = self._settings.use_cleaning
        self.__cleaning = cleaning_options
        self.__hyperbolic_parameter = self._settings.hyperbolic_parameter
        self.__degree = self._settings.degree
        self.__composed_distribution = ComposedDistribution([
            marginal.distribution
            for input_name in self.input_names
            for marginal in distributions[input_name].marginals
        ])

        if self._settings.use_quadrature:
            if self._settings.discipline is not None:
                self.__quadrature_points_with_weights = self._get_quadrature_points(
                    self._settings.n_quadrature_points, self._settings.discipline
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
        self._coefficients = None
        self._mean_jacobian_wrt_special_variables = None
        self._standard_deviation_jacobian_wrt_special_variables = None
        self._variance_jacobian_wrt_special_variables = None

    def __instantiate_functional_chaos_algorithm(
        self, input_data: RealArray, output_data: RealArray
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
        input_data: RealArray,
        output_data: RealArray,
    ) -> None:
        # Create and train the PCE.
        algo = self.__instantiate_functional_chaos_algorithm(input_data, output_data)
        algo.run()
        self.algo = pce_result = algo.getResult()
        self._prediction_function = pce_result.getMetaModel()

        # Compute some statistics.
        random_vector = FunctionalChaosRandomVector(pce_result)
        self._mean = array(random_vector.getMean())
        self._covariance = array(random_vector.getCovariance())
        self._variance = self._covariance.diagonal()
        self._standard_deviation = self._variance**0.5

        # Compute some sensitivity indices.
        names_to_positions = {}
        start = 0
        names_to_sizes = self.learning_set.variable_names_to_n_components
        for name in self.input_names:
            stop = start + names_to_sizes[name]
            names_to_positions[name] = range(start, stop)
            start = stop

        ot_sobol_indices = FunctionalChaosSobolIndices(pce_result)
        self.__compute_first_or_total_order_indices(
            names_to_positions, ot_sobol_indices, True
        )
        self.__compute_second_order_indices(names_to_positions, ot_sobol_indices)
        self.__compute_first_or_total_order_indices(
            names_to_positions, ot_sobol_indices, False
        )

        # Compute the derivatives of the coefficients wrt the special variables.
        # as well as those of the mean and variance.
        if self._jacobian_data is not None:
            basis_functions = pce_result.getReducedBasis()
            input_sample = pce_result.getInputSample()
            transformation = self.algo.getTransformation()
            phi = hstack([
                array(basis_function(transformation(input_sample)))
                for basis_function in basis_functions
            ])
            jacobian_data = self._jacobian_data[self._learning_samples_indices]
            coefficients = (
                jacobian_data.T
                @ solve(
                    phi.T @ phi,
                    phi.T,
                    overwrite_a=True,
                    overwrite_b=True,
                    assume_a="sym",
                ).T
            )
            shape = (self._reduced_output_dimension, -1, len(basis_functions))
            self._coefficients = coefficients.reshape(shape)
            self._variance_jacobian_wrt_special_variables = vstack([
                2 * ci_jac @ ci_out
                for ci_jac, ci_out in zip(
                    self._coefficients[..., 1:],
                    array(pce_result.getCoefficients()).T[..., 1:],
                )
            ])
            # _variance_jacobian_wrt_special_variables: (n_out, n_in)
            # _standard_deviation: (n_out,)
            self._standard_deviation_jacobian_wrt_special_variables = (
                self._variance_jacobian_wrt_special_variables
                / 2
                / self._standard_deviation[:, newaxis]
            )

            first_basis_function = basis_functions[0]
            phi = array(first_basis_function(transformation(input_sample)))
            coefficients = (
                solve(
                    phi.T @ phi,
                    phi.T,
                    overwrite_a=True,
                    overwrite_b=True,
                    assume_a="sym",
                )
                @ jacobian_data
            )
            self._mean_jacobian_wrt_special_variables = coefficients.sum(0).reshape(
                self._reduced_output_dimension, -1
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
            for output_index in range(self._reduced_output_dimension)
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
            for output_index in range(self._reduced_output_dimension)
        ]
        if use_first:
            self._first_order_sobol_indices = indices
        else:
            self._total_order_sobol_indices = indices

    def _predict(self, input_data: RealArray) -> RealArray:
        return array(self._prediction_function(input_data))

    def _get_quadrature_points(
        self, n_quadrature_points: int, discipline: Discipline
    ) -> tuple[RealArray, RealArray]:
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

        output_names = list(discipline.io.output_grammar)
        input_names = self.input_names
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
            {k: v.size for k, v in discipline.io.get_output_data().items()},
        )
        self.output_names = output_names
        return quadrature_points, weights

    def _predict_jacobian(
        self,
        input_data: RealArray,
    ) -> RealArray:
        gradient = self._prediction_function.gradient
        jac = zeros((
            len(input_data),
            self._reduced_output_dimension,
            self._reduced_input_dimension,
        ))
        for index, data in enumerate(input_data):
            jac[index] = array(gradient(Point(data))).T

        return jac

    def _predict_jacobian_wrt_special_variables(  # noqa: D102
        self, input_data: RealArray
    ) -> RealArray:
        basis_functions = self.algo.getReducedBasis()
        transformation = self.algo.getTransformation()
        y = array([
            basis_function(transformation(input_data))[0]
            for basis_function in basis_functions
        ])
        return self._coefficients @ y

    @property
    def mean_jacobian_wrt_special_variables(self) -> RealArray:
        """The gradient of the mean with respect to the special variables.

        See :meth:`.predict_jacobian_wrt_special_variables`
        for more information about the notion of special variables.

        Raises:
            ValueError: When the training dataset does not include gradient information.
        """
        self._check_is_trained()
        self._check_jacobian_learning_data("mean_jacobian_wrt_special_variables")
        return self._mean_jacobian_wrt_special_variables

    @property
    def standard_deviation_jacobian_wrt_special_variables(self) -> RealArray:
        """The gradient of the standard deviation with respect to the special variables.

        See :meth:`.predict_jacobian_wrt_special_variables`
        for more information about the notion of special variables.

        Raises:
            ValueError: When the training dataset does not include gradient information.
        """
        self._check_is_trained()
        self._check_jacobian_learning_data(
            "standard_deviation_jacobian_wrt_special_variables"
        )
        return self._standard_deviation_jacobian_wrt_special_variables

    @property
    def variance_jacobian_wrt_special_variables(self) -> RealArray:
        """The gradient of the variance with respect to the special variables.

        See :meth:`.predict_jacobian_wrt_special_variables`
        for more information about the notion of special variables.

        Raises:
            ValueError: When the training dataset does not include gradient information.
        """
        self._check_is_trained()
        self._check_jacobian_learning_data("variance_jacobian_wrt_special_variables")
        return self._variance_jacobian_wrt_special_variables

    @property
    def mean(self) -> RealArray:
        """The mean vector of the PCE model output.

        .. warning::

           This statistic is expressed in relation to the transformed output space.
           You can sample the :meth:`.predict` method
           to estimate it in relation to the original output space
           if it is different from the transformed output space.
        """
        self._check_is_trained()
        return self._mean

    @property
    def covariance(self) -> RealArray:
        """The covariance matrix of the PCE model output.

        .. warning::

           This statistic is expressed in relation to the transformed output space.
           You can sample the :meth:`.predict` method
           to estimate it in relation to the original output space
           if it is different from the transformed output space.
        """
        self._check_is_trained()
        return self._covariance

    @property
    def variance(self) -> RealArray:
        """The variance vector of the PCE model output.

        .. warning::

           This statistic is expressed in relation to the transformed output space.
           You can sample the :meth:`.predict` method
           to estimate it in relation to the original output space
           if it is different from the transformed output space.
        """
        self._check_is_trained()
        return self._variance

    @property
    def standard_deviation(self) -> RealArray:
        """The standard deviation vector of the PCE model output.

        .. warning::

           This statistic is expressed in relation to the transformed output space.
           You can sample the :meth:`.predict` method
           to estimate it in relation to the original output space
           if it is different from the transformed output space.
        """
        self._check_is_trained()
        return self._standard_deviation

    @property
    def first_sobol_indices(self) -> list[dict[str, float]]:
        """The first-order Sobol' indices for the different output components.

        .. warning::

           These statistics are expressed in relation to the transformed output space.
           You can use a :class:`.SobolAnalysis`
           to estimate them in relation to the original output space
           if it is different from the transformed output space.
        """
        self._check_is_trained()
        return self._first_order_sobol_indices

    @property
    def second_sobol_indices(self) -> list[dict[str, dict[str, float]]]:
        """The second-order Sobol' indices for the different output components.

        .. warning::

           These statistics are expressed in relation to the transformed output space.
           You can use a :class:`.SobolAnalysis`
           to estimate them in relation to the original output space
           if it is different from the transformed output space.
        """
        self._check_is_trained()
        return self._second_order_sobol_indices

    @property
    def total_sobol_indices(self) -> list[dict[str, float]]:
        """The total Sobol' indices for the different output components.

        .. warning::

           These statistics are expressed in relation to the transformed output space.
           You can use a :class:`.SobolAnalysis`
           to estimate them in relation to the original output space
           if it is different from the transformed output space.
        """
        self._check_is_trained()
        return self._total_order_sobol_indices
