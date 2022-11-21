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
r"""Polynomial regression model.

Polynomial regression is a particular case of the linear regression,
where the input data is transformed before the regression is applied.
This transform consists of creating a matrix of monomials
by raising the input data to different powers up to a certain degree :math:`D`.
In the case where there is only one input variable,
the input data :math:`(x_i)_{i=1, \dots, n}\in\mathbb{R}^n` is transformed
into the Vandermonde matrix:

.. math::

    \begin{pmatrix}
        x_1^1  & x_1^2  & \cdots & x_1^D\\
        x_2^1  & x_2^2  & \cdots & x_2^D\\
        \vdots & \vdots & \ddots & \vdots\\
        x_n^1  & x_n^2  & \cdots & x_n^D\\
    \end{pmatrix}
    = (x_i^d)_{i=1, \dots, n;\ d=1, \dots, D}.

The output variable is expressed as a weighted sum of monomials:

.. math::

     y = w_0 + w_1 x^1 + w_2 x^2 + ... + w_D x^D,

where the coefficients :math:`w_1, w_2, ..., w_d` and the intercept :math:`w_0`
are estimated by least square regression.

In the case of a multidimensional input,
i.e. :math:`X = (x_{ij})_{i=1,\dots,n; j=1,\dots,m}`,
where :math:`n` is the number of samples and :math:`m` is the number of input variables,
the Vandermonde matrix is expressed
through different combinations of monomials of degree :math:`d, (1 \leq d \leq D)`;
e.g. for three variables :math:`(x, y, z)` and degree :math:`D=3`,
the different terms are
:math:`x`, :math:`y`, :math:`z`, :math:`x^2`, :math:`xy`, :math:`xz`,
:math:`y^2`, :math:`yz`, :math:`z^2`, :math:`x^3`, :math:`x^2y` etc.
More generally,
for :math:`m` input variables,
the total number of monomials of degree :math:`1 \leq d \leq D` is given
by :math:`P = \binom{m+D}{m} = \frac{(m+D)!}{m!D!}`.
In the case of 3 input variables given above,
the total number of monomial combinations of degree lesser than or equal to three
is thus :math:`P = \binom{6}{3} = 20`.
The linear regression has to identify the coefficients :math:`w_1, \dots, w_P`,
in addition to the intercept :math:`w_0`.

Dependence
----------
The polynomial regression model relies
on the `LinearRegression <https://scikit-learn.org/stable/modules/
linear_model.html>`_ and  `PolynomialFeatures <https://scikit-learn.org/stable/
modules/generated/sklearn.preprocessing.PolynomialFeatures.html>`_ classes of
the `scikit-learn library <https://scikit-learn.org/stable/modules/
linear_model.html>`_.
"""
from __future__ import annotations

import pickle
from pathlib import Path
from typing import ClassVar
from typing import Iterable

from numpy import concatenate
from numpy import ndarray
from numpy import where
from numpy import zeros
from sklearn.preprocessing import PolynomialFeatures

from gemseo.core.dataset import Dataset
from gemseo.mlearning.core.ml_algo import DataType
from gemseo.mlearning.core.ml_algo import TransformerType
from gemseo.mlearning.regression.linreg import LinearRegressor


class PolynomialRegressor(LinearRegressor):
    """Polynomial regression model."""

    SHORT_ALGO_NAME: ClassVar[str] = "PolyReg"

    def __init__(
        self,
        data: Dataset,
        degree: int,
        transformer: TransformerType = LinearRegressor.IDENTITY,
        input_names: Iterable[str] | None = None,
        output_names: Iterable[str] | None = None,
        fit_intercept: bool = True,
        penalty_level: float = 0.0,
        l2_penalty_ratio: float = 1.0,
        **parameters: float | int | str | bool | None,
    ) -> None:
        """
        Args:
            degree: The polynomial degree.
            fit_intercept: Whether to fit the intercept.
            penalty_level: The penalty level greater or equal to 0.
                If 0, there is no penalty.
            l2_penalty_ratio: The penalty ratio
                related to the l2 regularization.
                If 1, the penalty is the Ridge penalty.
                If 0, this is the Lasso penalty.
                Between 0 and 1, the penalty is the ElasticNet penalty.

        Raises:
            ValueError: If the degree is lower than one.
        """
        super().__init__(
            data,
            degree=degree,
            transformer=transformer,
            input_names=input_names,
            output_names=output_names,
            fit_intercept=fit_intercept,
            penalty_level=penalty_level,
            l2_penalty_ratio=l2_penalty_ratio,
            **parameters,
        )
        self._poly = PolynomialFeatures(degree=degree, include_bias=False)
        self.parameters["degree"] = degree
        if degree < 1:
            raise ValueError("Degree must be >= 1.")

    def _fit(
        self,
        input_data: ndarray,
        output_data: ndarray,
    ) -> None:
        input_data = self._poly.fit_transform(input_data)
        super()._fit(input_data, output_data)

    def _predict(
        self,
        input_data: ndarray,
    ) -> ndarray:
        input_data = self._poly.transform(input_data)
        return super()._predict(input_data)

    def _predict_jacobian(
        self,
        input_data: ndarray,
    ) -> ndarray:
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

        vandermonde = self._poly.transform(input_data)

        powers = self._poly.powers_
        n_inputs = self._poly.n_input_features_
        n_powers = self._poly.n_output_features_
        n_outputs = self.algo.coef_.shape[0]
        coefs = self.get_coefficients()

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

    def get_coefficients(
        self,
        as_dict: bool = False,
    ) -> DataType:
        """Return the regression coefficients of the linear model.

        Args:
            as_dict: If True, return the coefficients as a dictionary of Numpy arrays
                indexed by the names of the coefficients.
                Otherwise, return the coefficients as a Numpy array.
                For now the only valid value is False.

        Returns:
            The regression coefficients of the linear model.

        Raises:
            NotImplementedError: If the coefficients are required as a dictionary.
        """
        coefficients = self.coefficients
        if as_dict:
            raise NotImplementedError(
                "For now the coefficients can only be obtained "
                "in the form of a NumPy array"
            )
        return coefficients

    def _save_algo(
        self,
        directory: Path,
    ) -> None:
        super()._save_algo(directory)
        with (directory / "poly.pkl").open("wb") as handle:
            pickle.dump(self._poly, handle)

    def load_algo(
        self,
        directory: str | Path,
    ) -> None:
        directory = Path(directory)
        super().load_algo(directory)
        with (directory / "poly.pkl").open("rb") as handle:
            poly = pickle.load(handle)
        self._poly = poly
