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
r"""The Gaussian process algorithm for regression.

Overview
--------

The Gaussian process regression (GPR) surrogate model
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
and covariance :math:`\sigma^2\kappa(\|x-x'\|;\epsilon)`.

Then, the GP conditioned by the learning set
:math:`(x_i,y_i)_{1\leq i \leq N}`
is entirely defined by its expectation:

.. math::

    \hat{f}(x) = \hat{\mu} + \hat{w}^T k(x)

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

    \hat{s}(x):=\sqrt{\hat{c}(x,x)}

Interpolation or regression
---------------------------

The GPR surrogate model can be regressive or interpolative
according to the value of the nugget effect :math:`\\alpha\geq 0`
which is a regularization term
applied to the correlation matrix :math:`K`.
When :math:`\alpha = 0`,
the surrogate model interpolates the learning data.

Dependence
----------
The GPR model relies on the GaussianProcessRegressor class
of the `scikit-learn library <https://scikit-learn.org/stable/modules/
generated/sklearn.gaussian_process.GaussianProcessRegressor.html>`_.
"""
from __future__ import division, unicode_literals

import logging
from typing import Callable, Iterable, Optional, Union

import openturns
from numpy import atleast_2d, ndarray
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

from gemseo.core.dataset import Dataset
from gemseo.mlearning.core.ml_algo import DataType, TransformerType
from gemseo.mlearning.regression.regression import MLRegressionAlgo
from gemseo.utils.data_conversion import DataConversion

LOGGER = logging.getLogger(__name__)


class GaussianProcessRegression(MLRegressionAlgo):
    """Gaussian process regression."""

    LIBRARY = "scikit-learn"
    ABBR = "GPR"

    def __init__(
        self,
        data,  # type: Dataset
        transformer=None,  # type: Optional[TransformerType]
        input_names=None,  # type: Optional[Iterable[str]]
        output_names=None,  # type: Optional[Iterable[str]]
        kernel=None,  # type: Optional[openturns.CovarianceModel]
        alpha=1e-10,  # type: Union[float,ndarray]
        optimizer="fmin_l_bfgs_b",  # type: Union[str,Callable]
        n_restarts_optimizer=10,  # type: int
        random_state=None,  # type: Optional[int]
    ):  # type: (...) -> None
        """
        Args:
            kernel: The kernel function. If None, use a ``Matern(2.5)``.
            alpha: The nugget effect to regularize the model.
            optimizer: The optimization algorithm to find the hyperparameters.
            n_restarts_optimizer: The number of restarts of the optimizer.
            random_state: The seed used to initialize the centers.
                If None, the random number generator is the RandomState instance
                used by `numpy.random`.
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

    def _fit(
        self,
        input_data,  # type: ndarray
        output_data,  # type: ndarray
    ):  # type: (...) -> None
        self.algo.fit(input_data, output_data)

    def _predict(
        self,
        input_data,  # type: ndarray
    ):  # type: (...) -> ndarray
        output_pred = self.algo.predict(input_data, False)
        return output_pred

    def predict_std(
        self,
        input_data,  # type: DataType
    ):  # type: (...) -> ndarray
        """Predict the standard deviation from input data.

        The user can specify these input data either as a NumPy array,
        e.g. :code:`array([1., 2., 3.])`
        or as a dictionary,
        e.g.  :code:`{'a': array([1.]), 'b': array([2., 3.])}`.

        If the NumPy arrays are of dimension 2,
        their i-th rows represent the input data of the i-th sample;
        while if the NumPy arrays are of dimension 1,
        there is a single sample.

        Args:
            input_data: The input data.

        Returns:
            The standard deviation at the query points.

        Warning:
            The standard deviation at a query point is defined as a positive scalar,
            whatever the output dimension.
            By the way,
            if the output variables are transformed before the training stage,
            then the standard deviation is related to this transformed output space
            unlike :meth:`.predict` which returns values in the original output space.
        """
        as_dict = isinstance(input_data, dict)
        if as_dict:
            input_data = DataConversion.dict_to_array(input_data, self.input_names)
        input_data = atleast_2d(input_data)
        inputs = self.learning_set.INPUT_GROUP
        if inputs in self.transformer:
            input_data = self.transformer[inputs].transform(input_data)
        _, output_std = self.algo.predict(input_data, True)
        return output_std
