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
#        :author: Syver Doving Agdestein
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
r"""
Polynomial regression
=====================

Polynomial regression class is a particular case of the linear regression,
where the input data is transformed before the regression is applied. This
transform consists of creating a matrix of monomials (Vandermonde) by raising
the input data to different powers up to a certain degree :math:`D`. In the
case where there is only one input variable, the input data
:math:`(x_i)_{i=1, \dots, n}\in\mathbb{R}^n` is transformed into the
Vandermonde matrix

.. math::

    \begin{pmatrix}
        x_1^1  & x_1^2  & \cdots & x_1^D\\
        x_2^1  & x_2^2  & \cdots & x_2^D\\
        \vdots & \vdots & \ddots & \vdots\\
        x_n^1  & x_n^2  & \cdots & x_n^D\\
    \end{pmatrix}
    = (x_i^d)_{i=1, \dots, n;\ d=1, \dots, D}.

The output is expressed as a weighted sum of monomials:

.. math::

     y = w_0 + w_1 x^1 + w_2 x^2 + ... + w_D x^D,

where the coefficients :math:`(w_1, w_2, ..., w_d)` and the intercept
:math:`w_0` are estimated by least square regression.

In the case of a multidimensional input, i.e.
:math:`X = (x_{ij})_{i=1,\dots,n; j=1,\dots,m}`, where :math:`n` is the number
of samples and :math:`m` is the number of input variables, the Vandermonde
matrix is expressed through different combinations of monomials of degree
:math:`d, (1 \leq d \leq D)`; e.g. for three variables
:math:`(x, y, z)` and degree :math:`D=3`, the different terms are
:math:`x`, :math:`y`, :math:`z`, :math:`x^2`, :math:`xy`, :math:`xz`,
:math:`y^2`, :math:`yz`, :math:`z^2`, :math:`x^3`, :math:`x^2y` etc. More
generally, for m input variables, the total number of monomials of degree
:math:`1 \leq d \leq D` is given by
:math:`P = \binom{m+D}{m} = \frac{(m+D)!}{m!D!}`. In the case of 3 input
variables given above, the total number of monomial combinations of degree
lesser than or equal to three is thus :math:`P = \binom{6}{3} = 20`. The linear
regression has to identify the coefficients :math:`(w_1, \dots, w_P)`, in
addition to the intercept :math:`w_0`.

This concept is implemented through the :class:`.PolynomialRegression` class
which inherits from the :class:`.MLRegressionAlgo` class.

Dependence
----------
The polynomial regression model relies on the LinearRegression class
of the `LinearRegression <https://scikit-learn.org/stable/modules/
linear_model.html>`_ and  `PolynomialFeatures <https://scikit-learn.org/stable/
modules/generated/sklearn.preprocessing.PolynomialFeatures.html>`_ classes of
the `scikit-learn library <https://scikit-learn.org/stable/modules/
linear_model.html>`_.
"""
from __future__ import absolute_import, division, unicode_literals

import pickle
from os.path import join

from future import standard_library
from numpy import concatenate, where, zeros
from sklearn.preprocessing import PolynomialFeatures

from gemseo.mlearning.regression.linreg import LinearRegression

standard_library.install_aliases()


from gemseo import LOGGER


class PolynomialRegression(LinearRegression):
    """ Polynomial regression. """

    LIBRARY = "scikit-learn"
    ABBR = "PolyReg"

    def __init__(
        self,
        data,
        degree,
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
        :param degree: Degree of polynomial. Default: 2.
        :type degree: int
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
        :type penalty_leve: float
        :param l2_penalty_ratio: penalty ratio related to the l2
            regularization. If 1, the penalty is the Ridge penalty. If 0,
            this is the Lasso penalty. Between 0 and 1, the penalty is the
            ElasticNet penalty. Default: None.
        :type l2_penalty_ratio: float
        """
        super(PolynomialRegression, self).__init__(
            data,
            transformer=transformer,
            input_names=input_names,
            output_names=output_names,
            fit_intercept=fit_intercept,
            penalty_level=penalty_level,
            l2_penalty_ratio=l2_penalty_ratio,
            **parameters
        )
        self.poly = PolynomialFeatures(degree=degree, include_bias=False)
        self.parameters["degree"] = degree
        if degree < 1:
            raise ValueError("Degree must be >= 1.")

    def _fit(self, input_data, output_data):
        """Fit the regression model.

        :param ndarray input_data: input data (2D).
        :param ndarray output_data: output data (2D).
        """
        input_data = self.poly.fit_transform(input_data)
        super(PolynomialRegression, self)._fit(input_data, output_data)

    def _predict(self, input_data):
        """Predict output for given input data.

        :param ndarray input_data: input data (2D).
        :return: output prediction (2D).
        :rtype: ndarray.
        """
        input_data = self.poly.transform(input_data)
        return super(PolynomialRegression, self)._predict(input_data)

    def _predict_jacobian(self, input_data):
        """Predict Jacobian of the regression model for the given input data.

        :param ndarray input_data: input_data (2D).
        :return: Jacobian matrices (3D, one for each sample).
        :rtype: ndarray
        """
        # Dimensions:
        # powers:        (           ,            ,  n_powers ,  n_inputs )
        # coefs:         (           ,  n_outputs ,  n_powers ,           )
        # jac_coefs:     (           ,  n_outputs ,  n_powers ,  n_inputs )
        # vandermonde:   ( n_samples ,            ,  n_powers ,           )
        # contributions: ( n_samples ,  n_outputs ,  n_powers ,  n_inputs )
        # jacobians:     ( n_samples ,  n_outputs ,           ,  n_inputs )
        #
        # n_powers is given by the formula
        # n_powers = binom(n_inputs+degree, n_inputs)+1

        vandermonde = self.poly.transform(input_data)

        powers = self.poly.powers_
        n_inputs = self.poly.n_input_features_
        n_powers = self.poly.n_output_features_
        n_outputs = self.algo.coef_.shape[0]
        coefs = self.get_coefficients(False)

        jac_intercept = zeros((n_outputs, n_inputs))
        jac_coefs = zeros((n_outputs, n_powers, n_inputs))

        # Compute partial derivatives with respect to each input separately
        for index in range(n_inputs):

            # Coefficients of monomial derivatives
            dcoefs = powers[None, :, index] * coefs

            # Powers of monomial derivatives
            dpowers = powers.copy()
            dpowers[:, index] -= 1

            # Keep indices of remaining monomials only
            mask_zero = (dpowers == 0).prod(axis=1) == 1
            mask_keep = dpowers[:, index] >= 0
            mask_keep[mask_zero == 1] = False

            # Extract intercept for Jacobian (0th order term)
            dintercept = dcoefs[:, mask_zero].flatten()

            # Filter kept terms
            dcoefs = dcoefs[:, mask_keep]  # Coefficients of kept terms
            dpowers = dpowers[mask_keep]  # Power keys of kept terms

            # Find indices for the given powers
            inds_keep = [
                where((powers == dpowers[i]).prod(axis=1) == 1)
                for i in range(dpowers.shape[0])
            ]
            if len(inds_keep) > 0:
                inds_keep = concatenate(inds_keep).flatten()

            # Coefficients of partial derivatives in terms of original powers
            jac_intercept[:, index] = dintercept
            jac_coefs[:, inds_keep, index] = dcoefs

        # Assemble polynomial (sum of weighted monomials)
        contributions = jac_coefs[None] * vandermonde[:, None, :, None]
        jacobians = jac_intercept + contributions.sum(axis=2)

        return jacobians

    def get_coefficients(self, as_dict=True):
        """Return the regression coefficients of the linear fit
        as a numpy array or as a dict.

        :param bool as_dict: if True, returns coefficients as a dictionary.
            Default: True.
        """
        coefficients = self.coefficients
        if as_dict:
            raise NotImplementedError
        return coefficients

    def _save_algo(self, directory):
        """Save external machine learning algorithm.

        :param str directory: algorithm directory.
        """
        super(PolynomialRegression, self)._save_algo(directory)
        filename = join(directory, "poly.pkl")
        with open(filename, "wb") as handle:
            pickle.dump(self.poly, handle)

    def load_algo(self, directory):
        """Load external machine learning algorithm.

        :param str directory: algorithm directory.
        """
        super(PolynomialRegression, self).load_algo(directory)
        filename = join(directory, "poly.pkl")
        with open(filename, "rb") as handle:
            poly = pickle.load(handle)
        self.poly = poly
