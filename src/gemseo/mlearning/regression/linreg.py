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
Linear regression
=================

The linear regression surrogate discipline expresses the model output
as a weighted sum of the model inputs:

.. math::

    y = w_0 + w_1x_1 + w_2x_2 + ... + w_dx_d
    + \alpha \left( \lambda \|w\|_2 + (1-\lambda) \|w\|_1 \right),

where the coefficients :math:`(w_1, w_2, ..., w_d)` and the intercept
:math:`w_0` are estimated by least square regression. They are are easily
accessible via the arguments *coefficients* and *intercept*.

The penalty level :math:`\alpha` is a non-negative parameter intended to
prevent overfitting, while the penalty ratio :math:`\lambda\in [0, 1]`
expresses the ratio between :math:`\ell_2`- and :math:`\ell_1`-regularization.
When :math:`\lambda=1`, there is no :math:`\ell_1`-regularization, and a Ridge
regression is performed. When :math:`\lambda=0`, there is no
:math:`\ell_2`-regularization, and a Lasso regression is performed. For
:math:`\lambda` between 0 and 1, an elastic net regression is performed.

One may also choose not to penalize the regression at all, by setting
:math:`\alpha=0`. In this case, a simple least squares regression is performed.

This concept is implemented through the :class:`.LinearRegression` class which
inherits from the :class:`.MLRegressionAlgo` class.

Dependence
----------
The linear model relies on the LinearRegression, Ridge, Lasso and ElasticNet
classes of the `scikit-learn library <https://scikit-learn.org/stable/modules/
linear_model.html>`_.
"""
from __future__ import absolute_import, division, unicode_literals

from future import standard_library
from numpy import array, repeat, zeros
from sklearn.linear_model import ElasticNet, Lasso
from sklearn.linear_model import LinearRegression as LinReg
from sklearn.linear_model import Ridge

from gemseo.core.dataset import Dataset
from gemseo.mlearning.regression.regression import MLRegressionAlgo
from gemseo.mlearning.transform.dimension_reduction.dimension_reduction import (
    DimensionReduction,
)
from gemseo.utils.data_conversion import DataConversion

standard_library.install_aliases()


from gemseo import LOGGER


class LinearRegression(MLRegressionAlgo):
    """ Linear regression """

    LIBRARY = "scikit-learn"
    ABBR = "LinReg"

    def __init__(
        self,
        data,
        transformer=None,
        input_names=None,
        output_names=None,
        fit_intercept=True,
        penalty_level=0.0,
        l2_penalty_ratio=1.0,
        **parameters
    ):
        """Constructor.

        :param data: learning dataset.
        :type data: Dataset
        :param transformer: transformation strategy for data groups.
            If None, do not transform data. Default: None.
        :type transformer: dict(str)
        :param input_names: names of the input variables.
        :type input_names: list(str)
        :param output_names: names of the output variables.
        :type output_names: list(str)
        :param fit_intercept: if True, fit intercept. Default: True.
        :type fit_intercept: bool
        :param penalty_level: penalty level greater or equal to 0.
            If 0, there is no penalty. Default: 0.
        :type penalty_level: float
        :param l2_penalty_ratio: penalty ratio related to the l2
            regularization. If 1, the penalty is the Ridge penalty. If 0,
            this is the Lasso penalty. Between 0 and 1, the penalty is the
            ElasticNet penalty. Default: None.
        :type l2_penalty_ratio: float
        """
        super(LinearRegression, self).__init__(
            data,
            transformer=transformer,
            input_names=input_names,
            output_names=output_names,
            fit_intercept=fit_intercept,
            penalty_level=penalty_level,
            l2_penalty_ratio=l2_penalty_ratio,
            **parameters
        )

        if penalty_level == 0.0:
            self.algo = LinReg(
                normalize=False, copy_X=False, fit_intercept=fit_intercept, **parameters
            )
        else:
            if l2_penalty_ratio == 1.0:
                self.algo = Ridge(
                    normalize=False,
                    copy_X=False,
                    fit_intercept=fit_intercept,
                    alpha=penalty_level,
                    **parameters
                )
            elif l2_penalty_ratio == 0.0:
                self.algo = Lasso(
                    normalize=False,
                    copy_X=False,
                    fit_intercept=fit_intercept,
                    alpha=penalty_level,
                    **parameters
                )
            else:
                self.algo = ElasticNet(
                    normalize=False,
                    copy_X=False,
                    fit_intercept=fit_intercept,
                    alpha=penalty_level,
                    l1_ratio=1 - l2_penalty_ratio,
                    **parameters
                )

    def _fit(self, input_data, output_data):
        """Fit the regression model.

        :param ndarray input_data: input data (2D).
        :param ndarray output_data: output data (2D).
        """
        self.algo.fit(input_data, output_data)

    def _predict(self, input_data):
        """Predict output for given input data.

        :param ndarray input_data: input data (2D).
        :return: output prediction (2D).
        :rtype: ndarray.
        """
        return self.algo.predict(input_data)

    def _predict_jacobian(self, input_data):
        """Predict Jacobian of the regression model for the given input data.

        :param ndarray input_data: input_data (2D).
        :return: Jacobian matrices (3D, one for each sample).
        :rtype: ndarray
        """
        n_samples = input_data.shape[0]
        return repeat(self.algo.coef_[None], n_samples, axis=0)

    @property
    def coefficients(self):
        """ Return the regression coefficients of the linear fit. """
        return self.algo.coef_

    @property
    def intercept(self):
        """ Return the regression intercepts of the linear fit. """
        if self.parameters["fit_intercept"]:
            intercept = self.algo.intercept_
        else:
            intercept = zeros(self.algo.coef_.shape[0])
        return intercept

    def get_coefficients(self, as_dict=True):
        """Return the regression coefficients of the linear fit
        as a numpy array or as a dict.

        :param bool as_dict: if True, returns coefficients as a dictionary.
            Default: True.
        """
        coefficients = self.coefficients
        if as_dict:
            if any(
                [
                    isinstance(transformer, DimensionReduction)
                    for _, transformer in self.transformer.items()
                ]
            ):
                raise ValueError(
                    "Coefficients are only representable in dict "
                    "form if the transformers do not change the "
                    "dimensions of the variables."
                )
            coefficients = self.__convert_array_to_dict(coefficients)
        return coefficients

    def get_intercept(self, as_dict=True):
        """Returns the regression intercept of the linear fit
        as a numpy array or as a dict.

        :param bool as_dict: if True, returns intercept as a dictionary.
            Default: True.
        """
        intercept = self.intercept
        if as_dict:
            if Dataset.OUTPUT_GROUP in self.transformer:
                raise ValueError(
                    "Intercept is only representable in dict "
                    "form if the transformers do not change the "
                    "dimensions of the output variables."
                )
            varsizes = self.learning_set.sizes
            intercept = DataConversion.array_to_dict(
                intercept, self.output_names, varsizes
            )
            intercept = {key: list(val) for key, val in intercept.items()}
        return intercept

    def __convert_array_to_dict(self, data):
        """Convert a data array into a dictionary.

        :param ndarray data: data.
        """
        varsizes = self.learning_set.sizes
        data = [
            DataConversion.array_to_dict(row, self.input_names, varsizes)
            for row in data
        ]
        data = [{key: list(val) for key, val in element.items()} for element in data]
        data = DataConversion.array_to_dict(array(data), self.output_names, varsizes)
        data = {key: list(val) for key, val in data.items()}
        return data
