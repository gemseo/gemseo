# -*- coding: utf-8 -*-
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
r"""The polynomial chaos expansion algorithm for regression.

The polynomial chaos expansion (PCE) model expresses the model output
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

PCE are stochastic models whose inputs are random variables
and are often used to deal with uncertainty quantification problems.

If :math:`X_j` is a Gaussian random variable,
:math:`(\psi_{ij})_{i\geq 0}` is the Legendre basis.
If :math:`X_j` is an uniform random variable,
:math:`(\psi_{ij})_{i\geq 0}` is the Hermite basis.

When the problem is deterministic,
we can still use PCE under the assumptions that
the random variables are independent uniform random variables.
Then,
the orthonormal basis function is the Hermite basis.

Degree
------

The degree :math:`P` of a PCE is defined
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
from __future__ import division, unicode_literals

import logging
from typing import Dict, Iterable, Mapping, Optional, Tuple, Union

import openturns
from numpy import all as np_all
from numpy import array, concatenate, isin, ndarray, zeros, zeros_like
from openturns import (
    LARS,
    AdaptiveStieltjesAlgorithm,
    CleaningStrategy,
    ComposedDistribution,
    CorrectedLeaveOneOut,
    FixedStrategy,
    Function,
    FunctionalChaosAlgorithm,
    FunctionalChaosSobolIndices,
    GaussProductExperiment,
    HyperbolicAnisotropicEnumerateFunction,
    IntegrationStrategy,
    LeastSquaresMetaModelSelectionFactory,
    LeastSquaresStrategy,
    MarginalTransformationEvaluation,
    OrthogonalBasis,
    OrthogonalProductPolynomialFactory,
    Point,
    StandardDistributionPolynomialFactory,
)

from gemseo.algos.parameter_space import ParameterSpace
from gemseo.core.dataset import Dataset
from gemseo.core.discipline import MDODiscipline
from gemseo.mlearning.core.ml_algo import TransformerType
from gemseo.mlearning.regression.regression import MLRegressionAlgo

LOGGER = logging.getLogger(__name__)


class PCERegression(MLRegressionAlgo):
    """Polynomial chaos expansion."""

    LIBRARY = "openturns"
    ABBR = "PCE"
    LS_STRATEGY = "LS"
    QUAD_STRATEGY = "Quad"
    SPARSE_STRATEGY = "SparseLS"
    AVAILABLE_STRATEGIES = [LS_STRATEGY, QUAD_STRATEGY, SPARSE_STRATEGY]

    def __init__(
        self,
        data,  # type: Dataset
        probability_space,  # type: ParameterSpace
        discipline=None,  # type: Optional[MDODiscipline]
        transformer=None,  # type: Optional[TransformerType]
        input_names=None,  # type: Optional[Iterable[str]]
        output_names=None,  # type: Optional[Iterable[str]]
        strategy=LS_STRATEGY,  # type: str
        degree=2,  # type: int
        n_quad=None,  # type: Optional[int]
        stieltjes=True,  # type: bool
        sparse_param=None,  # type: Optional[Mapping[str,Union[int,float]]]
    ):  # type: (...) -> None
        """
        Args:
            probability_space: The probability space
                defining the probability distributions of the model inputs.
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
                or if the strategy to compute the parameters of the PCE is unknown.
        """
        prob_space = probability_space
        super(PCERegression, self).__init__(
            data,
            transformer=transformer,
            input_names=input_names,
            output_names=output_names,
            probability_space=prob_space,
            strategy=strategy,
            degree=degree,
            n_quad=n_quad,
            stieltjes=stieltjes,
            sparse_param=sparse_param,
        )
        self._prob_space = prob_space
        self._discipline = discipline
        try:
            if data:
                u_names = set(prob_space.variables_names)
                assert set(self.input_names) == u_names
            else:
                self.input_names = prob_space.variables_names
        except Exception:
            raise ValueError(
                "Data inputs names are %s "
                "while probability distributions are defined "
                "%s." % (self.input_names, list(prob_space.distributions.keys()))
            )
        forbidden_names = set(self.input_names).union({Dataset.INPUT_GROUP})
        if set(list(self.transformer.keys())).intersection(forbidden_names):
            raise ValueError("PCERegression does not support input transformers.")

        self._distributions = prob_space.distributions
        self._sparse_param = sparse_param or {}
        self._input_dim = sum(
            [dist.dimension for _, dist in self._distributions.items()]
        )
        self._strategy = strategy
        if self._strategy not in self.AVAILABLE_STRATEGIES:
            strategies = " ".join(self.AVAILABLE_STRATEGIES)
            raise ValueError(
                "{} is a wrong strategy."
                " Available ones are: {}.".format(self._strategy, strategies)
            )
        self._stieltjes = stieltjes
        self._degree = degree
        self._n_quad = n_quad
        self._ot_distributions = [
            self._distributions[name].distribution for name in self.input_names
        ]
        self._dist = ComposedDistribution(self._ot_distributions)
        hyper_factor = self._sparse_param.get("hyper_factor", 1.0)
        self._enumerate_function = HyperbolicAnisotropicEnumerateFunction(
            self._input_dim, hyper_factor
        )
        self._basis = self._get_basis()
        self._n_basis = self._get_basis_size()
        self._trunc_strategy = self._get_trunc_strategy()
        if self._strategy == self.QUAD_STRATEGY:
            (
                self._sample,
                self._weights,
                self._proj_strategy,
            ) = self._get_quadrature_points()
        else:
            self._sample = None
            self._weights = None

    def _fit(
        self,
        input_data,  # type: ndarray
        output_data,  # type: ndarray
    ):  # type: (...) -> None
        self._proj_strategy = self._get_proj_strategy(input_data, output_data)
        weights = self._get_weights(input_data)
        self.algo = self._build_pce(input_data, weights, output_data)

    def _predict(
        self,
        input_data,  # type: ndarray
    ):  # type: (...) -> ndarray
        return array(self.algo.getMetaModel()(input_data))

    @property
    def first_sobol_indices(self):  # type: (...) -> Dict[str,ndarray]
        """The first Sobol' indices."""
        sensitivity_analysis = FunctionalChaosSobolIndices(self.algo)
        LOGGER.info(str(sensitivity_analysis))
        first_order = {
            name: sensitivity_analysis.getSobolIndex(index)
            for index, name in enumerate(self.input_names)
        }
        return first_order

    @property
    def total_sobol_indices(self):  # type: (...) -> Dict[str,ndarray]
        """The total Sobol' indices."""
        sensitivity_analysis = FunctionalChaosSobolIndices(self.algo)
        LOGGER.info(str(sensitivity_analysis))
        total_order = {
            name: sensitivity_analysis.getSobolTotalIndex(index)
            for index, name in enumerate(self.input_names)
        }
        return total_order

    def _build_pce(
        self,
        input_data,  # type: ndarray
        weights,  # type: ndarray
        output_data,  # type: ndarray
    ):  # type: (...) -> openturns.FunctionalChaosResult
        """Build the PCE with OpenTURNS.

        Args:
            input_data: The input data with shape (n_samples, n_inputs).
            weights: The data weights.
            output_data: The output data with shape (n_samples, n_outputs).

        Returns:
            An OpenTURNS PCE.
        """
        pce_algo = FunctionalChaosAlgorithm(
            input_data, weights, output_data, self._dist, self._trunc_strategy
        )
        pce_algo.setProjectionStrategy(self._proj_strategy)
        pce_algo.run()
        return pce_algo.getResult()

    def _get_basis(self):  # type: (...) -> openturns.OrthogonalProductPolynomialFactory
        """Return the orthogonal product polynomial factory for PCE construction.

        Returns:
            An orthogonal product polynomial factory computed by OpenTURNS.
        """
        if self._stieltjes:
            # Tend to result in performance issue
            basis = OrthogonalProductPolynomialFactory(
                [
                    StandardDistributionPolynomialFactory(
                        AdaptiveStieltjesAlgorithm(marginal)
                    )
                    for marginal in self._ot_distributions
                ],
                self._enumerate_function,
            )
        else:
            basis = OrthogonalProductPolynomialFactory(
                [
                    StandardDistributionPolynomialFactory(margin)
                    for margin in self._ot_distributions
                ],
                self._enumerate_function,
            )
        return basis

    def _get_basis_size(self):  # type: (...) -> int
        """Return the basis size for PCE construction.

        Returns:
            The number of basis functions.
        """
        return self._enumerate_function.getStrataCumulatedCardinal(self._degree)

    def _get_quadrature_points(
        self,
    ):  # type: (...) -> Tuple[ndarray,ndarray,openturns.IntegrationStrategy]
        """Return the quadrature points for PCE construction.

        Returns:
            The quadrature points.
            The weights associated with the quadrature points.
            The projection strategy.
        """
        measure = self._basis.getMeasure()
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
            in_grp = self.learning_set.INPUT_GROUP
            self.learning_set.set_from_array(
                sample,
                self._prob_space.variables_names,
                self._prob_space.variables_sizes,
                {name: in_grp for name in inputs_names},
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
            outputs = [outputs[name] for name in outputs_names]
            data = concatenate(outputs, axis=1)
            out_grp = self.learning_set.OUTPUT_GROUP
            sizes = {
                name: len(self._discipline.local_data[name]) for name in outputs_names
            }
            self.learning_set.add_group(
                out_grp, data, outputs_names, sizes, cache_as_input=False
            )
            self.output_names = outputs_names

        return sample, weights, proj_strategy

    def _get_trunc_strategy(
        self,
    ):  # type: (...) -> openturns.AdaptiveStrategyImplementation
        """Return the truncation strategy for PCE construction.

        Returns:
            The OpenTURNS truncation strategy.

        Raises:
            ValueError: If the truncation strategy is invalid.
        """
        if self._strategy == self.SPARSE_STRATEGY:
            sparse_param = self._sparse_param
            sparse_param = {} if sparse_param is None else sparse_param
            max_considered_terms = sparse_param.get("max_considered_terms", 120)
            most_significant = sparse_param.get("most_significant", 30)
            significance_factor = sparse_param.get("significance_factor", 1e-3)

            trunc_strategy = CleaningStrategy(
                OrthogonalBasis(self._basis),
                max_considered_terms,
                most_significant,
                significance_factor,
                True,
            )
        else:
            trunc_strategy = FixedStrategy(self._basis, self._n_basis)
        return trunc_strategy

    def _get_proj_strategy(
        self,
        input_data,  # type: ndarray
        output_data,  # type: ndarray
    ):  # type: (...) -> openturns.LeastSquaresStrategy
        """Return the projection strategy for PCE construction.

        Args:
            input_data: The input data with shape (n_samples, n_inputs).
            output_data: The output data with shape (n_samples, n_outputs).

        Returns:
            A projection strategy.
        """
        if self._strategy == self.QUAD_STRATEGY:
            proj_strategy = self._proj_strategy
        elif self._strategy == self.LS_STRATEGY:
            proj_strategy = LeastSquaresStrategy(input_data, output_data)
        else:
            app = LeastSquaresMetaModelSelectionFactory(LARS(), CorrectedLeaveOneOut())
            proj_strategy = LeastSquaresStrategy(input_data, output_data, app)
        return proj_strategy

    def _get_ls_weights(self):  # type: (...) -> openturns.Point
        """Return LS weights for PCE construction.

        Returns:
            The least-squares weights.
        """
        _, weights = self._proj_strategy.getExperiment().generateWithWeights()
        return weights

    def _get_quad_weights(
        self,
        input_data,  # type: ndarray
    ):  # type: (...) -> ndarray
        """Return quadrature weights for PCE construction.

        Args:
            input_data: The input data with shape (n_samples, n_inputs).

        Returns:
            The weights.
        """
        sample = zeros_like(self._sample)
        common_len = len(input_data)
        sample[:common_len] = input_data
        sample[0] = self._sample[0]
        sample_arg = np_all(isin(sample, self._sample), axis=1)
        new_weights = array(self._weights)[sample_arg]
        return new_weights

    def _get_weights(
        self,
        input_data,  # type: ndarray
    ):  # type: (...) -> ndarray
        """Return the weights for PCE construction.

        Args:
            input_data: The input data with shape (n_samples, n_inputs).
        """
        if self._strategy == self.QUAD_STRATEGY:
            weights = self._get_quad_weights(input_data)
        else:
            weights = self._get_ls_weights()
        return weights

    def _predict_jacobian(
        self,
        input_data,  # type: ndarray
    ):  # type: (...) -> ndarray
        input_shape, output_shape = self._get_raw_shapes()
        gradient = self.algo.getMetaModel().gradient
        jac = zeros((input_data.shape[0], int(output_shape), int(input_shape)))
        for index, data in enumerate(input_data):
            jac[index] = array(gradient(Point(data))).T
        return jac
