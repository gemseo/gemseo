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
#        :author: Francois Gallard, Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
r"""
Gaussian process regression
===========================

Overview
--------

The Gaussian process regression (GPR) surrogate discipline
expresses the model output as a weighted sum of kernel functions
centered on the learning input data:

.. math::

    y = \mu
        + w_1\kappa(\|x-x_1\|;\epsilon)
        + w_2\kappa(\|x-x_2\|;\epsilon)
        + ...
        + w_N\kappa(\|x-x_N\|;\epsilon)

Details
-------

The GPR model relies on the assumption
that the original model :math:`f` to replace
is an instance of a Gaussian process (GP) with mean :math:`\mu`
and covariance :math:`\sigma^2\kappa(\|x-x'|;\epsilon)`.

Then, the GP conditioned by the learning set
:math:`(x_i,y_i)_{1\leq i \leq N}`
is entirely defined by its expectation:

.. math::

    \hat{f}(x) = \hat{\mu} + w^T k(x)

and its covariance:

.. math::

    \hat{c}(x,x') = \hat{\sigma}^2 - k(x)^T K^{-1} k(x')

where :math:`[\hat{\mu};\hat{w}]=([1_N~K]^T[1_N~K])^{-1}[1_N~K]^TY` with
:math:`K_{ij}=\kappa(\|x_i-x_j\|;\hat{\epsilon})`,
:math:`k_i(x)=\kappa(\|x-x_i\|;\hat{\epsilon})`
and :math:`Y_i=y_i`.

The correlation length vector :math:`\epsilon`
is estimated by numerical non-linear optimization.

Surrogate model
---------------

The expectation :math:`\hat{f}` is the GPR surrogate model of :math:`f`.

Error measure
-------------

The standard deviation :math:`\hat{s}` is a local error measure
of :math:`\hat{f}`:

.. math::

    \hat{s}(x):=\sqrt{c(x,x)}

Interpolation or regression
---------------------------

The GPR surrogate discipline can be regressive or interpolative
according to the value of the nugget effect :math:`\\alpha\geq 0`
which is a regularization term
applied to the correlation matrix :math:`K`.
When :math:`\\alpha = 0`,
the surrogate model interpolates the learning data.

Dependence
----------
The GPR model relies on the GaussianProcessRegressor class
of the `scikit-learn library <https://scikit-learn.org/stable/modules/
generated/sklearn.gaussian_process.GaussianProcessRegressor.html>`_.
"""
from __future__ import absolute_import, division, unicode_literals

from future import standard_library
from numpy import atleast_2d
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

from gemseo.mlearning.regression.regression import MLRegressionAlgo
from gemseo.utils.data_conversion import DataConversion

standard_library.install_aliases()


from gemseo import LOGGER


class GaussianProcessRegression(MLRegressionAlgo):
    """ Gaussian process regression """

    LIBRARY = "scikit-learn"
    ABBR = "GPR"

    def __init__(
        self,
        data,
        transformer=None,
        input_names=None,
        output_names=None,
        kernel=None,
        alpha=1e-10,
        optimizer="fmin_l_bfgs_b",
        n_restarts_optimizer=10,
        random_state=None,
    ):
        """Constructor.

        :param data: learning dataset
        :type data: Dataset
        :param transformer: transformation strategy for data groups.
            If None, do not transform data. Default: None.
        :type transformer: dict(str)
        :param input_names: names of the input variables.
        :type input_names: list(str)
        :param output_names: names of the output variables.
        :type output_names: list(str)
        :param kernel: kernel function. If None, use a Matern(2.5).
            Default: None.
        :type kernel: openturns.Kernel
        :param alpha: nugget effect. Default: 1e-10.
        :type alpha: float or array
        :param optimizer: optimization algorithm. Default: 'fmin_l_bfgs_b'.
        :type optimizer: str or callable
        :param n_restarts_optimizer: number of restarts of the optimizer.
            Default: 10.
        :type n_restarts_optimizer: int
        :param random_state: the seed used to initialize the centers.
            If None, the random number generator is the RandomState instance
            used by `np.random`
            Default: None.
        :type random_state: int
        """
        super(GaussianProcessRegression, self).__init__(
            data,
            transformer=transformer,
            input_names=input_names,
            output_names=output_names,
            kernel=kernel,
            alpha=alpha,
            optimizer=optimizer,
            n_restarts_optimizer=n_restarts_optimizer,
            random_state=random_state,
        )

        if kernel is None:
            raw_input_shape, _ = self._get_raw_shapes()
            self.kernel = Matern(
                (1.0,) * raw_input_shape, [(0.01, 100)] * raw_input_shape, nu=2.5
            )
        else:
            self.kernel = kernel

        nro = n_restarts_optimizer
        self.algo = GaussianProcessRegressor(
            normalize_y=False,
            kernel=self.kernel,
            copy_X_train=True,
            alpha=alpha,
            optimizer=optimizer,
            n_restarts_optimizer=nro,
            random_state=random_state,
        )
        self.parameters["kernel"] = self.kernel.__class__.__name__

    def _fit(self, input_data, output_data):
        """Fit the regression model.

        :param ndarray input_data: input data (2D)
        :param ndarray output_data: output data (2D)
        """
        self.algo.fit(input_data, output_data)

    def _predict(self, input_data):
        """Predict output.

        :param ndarray input_data: input data (2D).
        :return: output prediction (2D).
        :rtype: ndarray
        """
        output_pred = self.algo.predict(input_data, False)
        return output_pred

    def predict_std(self, input_data):
        """Predict standard deviation value for given input data.

        :param dict(ndarray) input_data: input data (1D or 2D).
        :return: output data (1D or 2D, same as input_data).
        :rtype: dict(ndarray)
        """
        as_dict = isinstance(input_data, dict)
        if as_dict:
            input_data = DataConversion.dict_to_array(input_data, self.input_names)
        input_data = atleast_2d(input_data)
        inputs = self.learning_set.INPUT_GROUP
        if inputs in self.transformer:
            input_data = self.transformer[inputs].transform(input_data)
        _, output_std = self.algo.predict(input_data, True)
        return sum(output_std) / len(output_std)
