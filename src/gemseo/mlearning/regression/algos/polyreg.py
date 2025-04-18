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

from typing import TYPE_CHECKING
from typing import ClassVar

from numpy import concatenate
from numpy import newaxis
from numpy import zeros
from sklearn.preprocessing import PolynomialFeatures

from gemseo.mlearning.regression.algos.linreg import LinearRegressor
from gemseo.mlearning.regression.algos.polyreg_settings import (
    PolynomialRegressor_Settings,
)
from gemseo.utils.compatibility.sklearn import get_n_input_features_

if TYPE_CHECKING:
    from gemseo.mlearning.core.algos.ml_algo import DataType
    from gemseo.typing import RealArray


class PolynomialRegressor(LinearRegressor):
    """Polynomial regression model."""

    SHORT_ALGO_NAME: ClassVar[str] = "PolyReg"

    Settings: ClassVar[type[PolynomialRegressor_Settings]] = (
        PolynomialRegressor_Settings
    )

    def _post_init(self):
        """
        Raises:
            ValueError: If the degree is lower than one.
        """  # noqa: D205 D212
        super()._post_init()
        self._poly = PolynomialFeatures(
            degree=self._settings.degree, include_bias=False
        )

    def _fit(
        self,
        input_data: RealArray,
        output_data: RealArray,
    ) -> None:
        super()._fit(self._poly.fit_transform(input_data), output_data)

    def _predict(
        self,
        input_data: RealArray,
    ) -> RealArray:
        return super()._predict(self._poly.transform(input_data))

    def _predict_jacobian(
        self,
        input_data: RealArray,
    ) -> RealArray:
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
        powers = self._poly.powers_
        n_inputs = get_n_input_features_(self._poly)
        n_outputs = self.algo.coef_.shape[0]
        coefs = self.get_coefficients()
        jac_intercept = zeros((n_outputs, n_inputs))
        jac_coefs = zeros((n_outputs, self._poly.n_output_features_, n_inputs))

        # Compute partial derivatives with respect to each input separately
        for index in range(n_inputs):
            # Coefficients of monomial derivatives
            dcoefs = powers[newaxis, :, index] * coefs

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
                ((powers == dpowers[i]).prod(axis=1) == 1).nonzero()[0]
                for i in range(dpowers.shape[0])
            ]
            if len(inds_keep) > 0:
                inds_keep = concatenate(inds_keep).flatten()

            # Coefficients of partial derivatives in terms of original powers
            jac_intercept[:, index] = dintercept
            jac_coefs[:, inds_keep, index] = dcoefs

        # Assemble polynomial (sum of weighted monomials)
        vandermonde = self._poly.transform(input_data)
        contributions = jac_coefs[newaxis] * vandermonde[:, newaxis, :, newaxis]
        return jac_intercept + contributions.sum(axis=2)

    def get_coefficients(
        self,
        as_dict: bool = False,
    ) -> DataType:
        """Return the regression coefficients of the linear model.

        Args:
            as_dict: If ``True``,
                return the coefficients as a dictionary of Numpy arrays
                indexed by the names of the coefficients.
                Otherwise, return the coefficients as a Numpy array.
                For now the only valid value is False.

        Returns:
            The regression coefficients of the linear model.

        Raises:
            NotImplementedError: If the coefficients are required as a dictionary.
        """
        if as_dict:
            msg = (
                "For now the coefficients can only be obtained "
                "in the form of a NumPy array"
            )
            raise NotImplementedError(msg)
        return self.coefficients
