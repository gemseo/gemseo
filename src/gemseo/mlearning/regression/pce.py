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
r"""
Polynomial chaos expansion
==========================

The polynomial chaos expansion (PCE) discipline expresses the model output
as a weighted sum of polynomial functions which are orthonormal
in the stochastic input space spanned by the random input variables:

.. math::

    Y = w_0 + w_1\phi_1(X) + w_2\phi_2(X) + ... + w_K\phi_K(X)

where :math:`\phi_i(x)=\phi_{\tau_1(i),1}(x_1)\times\ldots\times
\phi_{\tau_d(i),d}(x_d)`.

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

When the problem is deterministic, we can still use PCE
under the assumptions that
the random variables are independent uniform random variables.
Then, the orthonormal basis function is the Hermite basis.

Degree
------

The degree :math:`P` of a PCE is defined
in such a way that :math:`\text{degree}(\psi_i)=\sum_{j=1}^d\tau_j(i)\leq P`.

Estimation
----------

The coefficients :math:`(w_1, w_2, ..., w_K)` and the intercept
:math:`w_0` are estimated either by least square regression,
sparse least square regression or quadrature.

Dependence
----------
The PCE model relies on the FunctionalChaosAlgorithm class
of the `openturns library <http://openturns.github.io/openturns/1.9/user_manual
/response_surface/_generated/openturns.FunctionalChaosAlgorithm.html>`_.
"""
from __future__ import absolute_import, division, unicode_literals

from future import standard_library
from numpy import all as np_all
from numpy import array, concatenate, isin, zeros, zeros_like
from openturns import (
    LARS,
    AdaptiveStieltjesAlgorithm,
    CleaningStrategy,
    ComposedDistribution,
    CorrectedLeaveOneOut,
    EnumerateFunction,
    FixedStrategy,
    Function,
    FunctionalChaosAlgorithm,
    FunctionalChaosSobolIndices,
    GaussProductExperiment,
    IntegrationStrategy,
    LeastSquaresMetaModelSelectionFactory,
    LeastSquaresStrategy,
    MarginalTransformationEvaluation,
    OrthogonalBasis,
    OrthogonalProductPolynomialFactory,
    Point,
    StandardDistributionPolynomialFactory,
)

from gemseo.mlearning.regression.regression import MLRegressionAlgo

standard_library.install_aliases()


from gemseo import LOGGER


class PCERegression(MLRegressionAlgo):
    """ Polynomial chaos expansion. """

    LIBRARY = "openturns"
    ABBR = "PCE"
    LS_STRATEGY = "LS"
    QUAD_STRATEGY = "Quad"
    SPARSE_STRATEGY = "SparseLS"
    AVAILABLE_STRATEGIES = [LS_STRATEGY, QUAD_STRATEGY, SPARSE_STRATEGY]

    def __init__(
        self,
        data,
        probability_space,
        discipline=None,
        transformer=None,
        input_names=None,
        output_names=None,
        strategy=LS_STRATEGY,
        degree=2,
        n_quad=None,
        stieltjes=True,
        sparse_param=None,
    ):
        """Constructor.

        :param data: learning dataset
        :type data: Dataset
        :param probability_space: probability space.
        :type probability_space: ParameterSpace
        :param discipline: discipline to evaluate if strategy='Quad' and
            data is empty.
        :type discipline: MDODiscipline
        :param transformer: transformation strategy for data groups.
            If None, do not transform data. Default: None.
        :type transformer: dict(str)
        :param input_names: names of the input variables.
        :type input_names: list(str)
        :param output_names: names of the output variables.
        :type output_names: list(str)
        :param strategy: strategy to compute the parameters of the PCE,
            either 'LS', 'Quad' or 'SparseLS'. Default: 'LS'.
        :type strategy: str
        :param degree: polynomial degree of the PCE
        :type degree: int
        :param n_quad: number of quadrature points
        :type n_quad: int
        :param stieltjes: stieltjes
        :type stieltjes: bool
        :param sparse_param: Parameters for the Sparse Cleaning Truncation
            Strategy and/or hyperbolic truncation of the initial basis:

            - **max_considered_terms** (int) -- Maximum Considered Terms,
            - **most_significant** (int), Most Significant number to retain,
            - **significance_factor** (float), Significance Factor,
            - **hyper_factor** (float), factor for hyperbolic truncation
              strategy.
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
        self.prob_space = prob_space
        self.discipline = discipline
        if transformer is not None and transformer != {}:
            raise ValueError("PCERegression does not support transformers.")
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
                "%s." % (self.input_names, list(prob_space.marginals.keys()))
            )

        self.distributions = prob_space.marginals
        self.sparse_param = sparse_param or {}
        self.input_dim = sum([dist.dimension for _, dist in self.distributions.items()])
        self.strategy = strategy
        if self.strategy not in self.AVAILABLE_STRATEGIES:
            strategies = " ".join(self.AVAILABLE_STRATEGIES)
            raise ValueError(
                "{} is a wrong strategy."
                " Available ones are: {}".format(self.strategy, strategies)
            )
        self.stieltjes = stieltjes
        self.degree = degree
        self.n_quad = n_quad
        self.ot_distributions = [
            self.distributions[name].distribution for name in self.input_names
        ]
        self.dist = ComposedDistribution(self.ot_distributions)
        hyper_factor = self.sparse_param.get("hyper_factor", 1.0)
        self.enumerate_function = EnumerateFunction(self.input_dim, hyper_factor)
        self.basis = self._get_basis()
        self.n_basis = self._get_basis_size()
        self.trunc_strategy = self._get_trunc_strategy()
        if self.strategy == self.QUAD_STRATEGY:
            (
                self.sample,
                self.weights,
                self.proj_strategy,
            ) = self._get_quadrature_points()
        else:
            self.sample = None
            self.weights = None

    def _fit(self, input_data, output_data):
        """Fit the regression model.

        :param ndarray input_data: input data (2D)
        :param ndarray output_data: output data (2D)
        """
        self.proj_strategy = self._get_proj_strategy(input_data, output_data)
        weights = self._get_weights(input_data)
        self.algo = self._build_pce(input_data, weights, output_data)

    def _predict(self, input_data):
        """Predict output.

        :param ndarray input_data: input data (2D).
        :return: output prediction (2D).
        :rtype: ndarray
        """
        return array(self.algo.getMetaModel()(input_data))

    def compute_sobol(self):
        """Compute first and total Sobol' indices

        :returns: first and total SObol' indices
        :rtype: list, list
        """
        sensitivity_analysis = FunctionalChaosSobolIndices(self.algo)
        LOGGER.info(str(sensitivity_analysis))
        dimension = len(self.input_names)
        first_order = [sensitivity_analysis.getSobolIndex(i) for i in range(dimension)]
        total_order = [
            sensitivity_analysis.getSobolTotalIndex(i) for i in range(dimension)
        ]
        return first_order, total_order

    def _build_pce(self, x_learn, weights, y_learn):
        """Build PCE"""
        pce_algo = FunctionalChaosAlgorithm(
            x_learn, weights, y_learn, self.dist, self.trunc_strategy
        )
        pce_algo.setProjectionStrategy(self.proj_strategy)
        pce_algo.run()
        return pce_algo.getResult()

    def _get_basis(self):
        """Get basis function for PCE construction"""
        if self.stieltjes:
            # Tend to result in performance issue
            basis = OrthogonalProductPolynomialFactory(
                [
                    StandardDistributionPolynomialFactory(
                        AdaptiveStieltjesAlgorithm(marginal)
                    )
                    for marginal in self.ot_distributions
                ],
                self.enumerate_function,
            )
        else:
            basis = OrthogonalProductPolynomialFactory(
                [
                    StandardDistributionPolynomialFactory(margin)
                    for margin in self.ot_distributions
                ],
                self.enumerate_function,
            )
        return basis

    def _get_basis_size(self):
        """Get basis size for PCE construction"""
        return self.enumerate_function.getStrataCumulatedCardinal(self.degree)

    def _get_quadrature_points(self):
        """Get quadrature points for PCE construction"""
        measure = self.basis.getMeasure()
        if self.n_quad is not None:
            degree_by_dim = int(self.n_quad ** (1.0 / self.input_dim))
        else:
            degree_by_dim = self.degree + 1
        degrees = [degree_by_dim] * self.input_dim

        proj_strategy = IntegrationStrategy(GaussProductExperiment(measure, degrees))
        sample, weights = proj_strategy.getExperiment().generateWithWeights()

        if not self.stieltjes:
            transformation = Function(
                MarginalTransformationEvaluation(
                    [measure.getMarginal(i) for i in range(self.input_dim)],
                    self.ot_distributions,
                    False,
                )
            )
            sample = transformation(sample)
        sample = array(sample)
        if not self.learning_set:
            inputs_names = self.prob_space.variables_names
            in_grp = self.learning_set.INPUT_GROUP
            self.learning_set.set_from_array(
                sample,
                self.prob_space.variables_names,
                self.prob_space.variables_sizes,
                {name: in_grp for name in inputs_names},
            )
        if self.discipline is not None:
            n_samples = len(self.learning_set)
            outputs = {name: [] for name in self.discipline.get_output_data_names()}
            for data in self.learning_set:
                output_data = self.discipline.execute(data)
                for name in self.discipline.get_output_data_names():
                    outputs[name] += list(output_data[name])
            for name in self.discipline.get_output_data_names():
                outputs[name] = array(outputs[name]).reshape((n_samples, -1))
            outputs_names = list(self.discipline.get_output_data_names())
            outputs = [outputs[name] for name in outputs_names]
            data = concatenate(outputs, axis=1)
            out_grp = self.learning_set.OUTPUT_GROUP
            sizes = {
                name: len(self.discipline.local_data[name]) for name in outputs_names
            }
            self.learning_set.add_group(
                out_grp, data, outputs_names, sizes, cache_as_input=False
            )
            self.output_names = outputs_names

        return sample, weights, proj_strategy

    def _get_trunc_strategy(self):
        """Get truncation strategy for PCE construction"""
        if self.strategy in [self.LS_STRATEGY, self.QUAD_STRATEGY]:
            trunc_strategy = FixedStrategy(self.basis, self.n_basis)
        elif self.strategy == self.SPARSE_STRATEGY:
            sparse_param = self.sparse_param
            sparse_param = {} if sparse_param is None else sparse_param
            max_considered_terms = sparse_param.get("max_considered_terms", 120)
            most_significant = sparse_param.get("most_significant", 30)
            significance_factor = sparse_param.get("significance_factor", 1e-3)

            trunc_strategy = CleaningStrategy(
                OrthogonalBasis(self.basis),
                max_considered_terms,
                most_significant,
                significance_factor,
                True,
            )
        return trunc_strategy

    def _get_proj_strategy(self, x_learn, y_learn):
        """Get projection strategy for PCE construction

        :param x_learn: input data
        :type x_learn: array
        :param y_learn: output data
        :type y_learn: array
        """
        if self.strategy == self.QUAD_STRATEGY:
            proj_strategy = self.proj_strategy
        elif self.strategy == self.LS_STRATEGY:
            proj_strategy = LeastSquaresStrategy(x_learn, y_learn)
        else:
            app = LeastSquaresMetaModelSelectionFactory(LARS(), CorrectedLeaveOneOut())
            proj_strategy = LeastSquaresStrategy(x_learn, y_learn, app)
        return proj_strategy

    def _get_ls_weights(self):
        """Get LS weights for PCE construction"""
        _, weights = self.proj_strategy.getExperiment().generateWithWeights()
        return weights

    def _get_quad_weights(self, x_learn):
        """Get quadrature weights for PCE construction"""
        sample = zeros_like(self.sample)
        common_len = len(x_learn)
        sample[:common_len] = x_learn
        sample[0] = self.sample[0]
        sample_arg = np_all(isin(sample, self.sample), axis=1)
        new_weights = array(self.weights)[sample_arg]
        return new_weights

    def _get_weights(self, x_learn):
        """Get weights for PCE construction)"""
        if self.strategy == self.QUAD_STRATEGY:
            weights = self._get_quad_weights(x_learn)
        else:
            weights = self._get_ls_weights()
        return weights

    def _predict_jacobian(self, input_data):
        """Predict Jacobian of the regression model for the given input data.

        :param ndarray input_data: input_data (2D).
        :return: Jacobian matrices (3D, one for each sample).
        :rtype: ndarray
        """
        input_shape, output_shape = self._get_raw_shapes()
        gradient = self.algo.getMetaModel().gradient
        jac = zeros((input_data.shape[0], int(output_shape), int(input_shape)))
        for index, data in enumerate(input_data):
            jac[index] = array(gradient(Point(data))).T
        return jac
