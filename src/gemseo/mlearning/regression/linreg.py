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
r"""Linear regression model.

The linear regression model expresses the output variables
as a weighted sum of the input ones:

.. math::

    y = w_0 + w_1x_1 + w_2x_2 + ... + w_dx_d
    + \alpha \left( \lambda \|w\|_2 + (1-\lambda) \|w\|_1 \right),

where the coefficients :math:`(w_1, w_2, ..., w_d)` and the intercept
:math:`w_0` are estimated by least square regression. They are easily
accessible via the arguments :attr:`.coefficients` and :attr:`.intercept`.

The penalty level :math:`\alpha` is a non-negative parameter intended to
prevent overfitting, while the penalty ratio :math:`\lambda\in [0, 1]`
expresses the ratio between :math:`\ell_2`- and :math:`\ell_1`-regularization.
When :math:`\lambda=1`, there is no :math:`\ell_1`-regularization, and a Ridge
regression is performed. When :math:`\lambda=0`, there is no
:math:`\ell_2`-regularization, and a Lasso regression is performed. For
:math:`\lambda` between 0 and 1, an Elastic Net regression is performed.

One may also choose not to penalize the regression at all, by setting
:math:`\alpha=0`. In this case, a simple least squares regression is performed.

Dependence
----------
The linear model relies on the ``LinearRegression``,
``Ridge``, ``Lasso`` and ``ElasticNet``
classes of the `scikit-learn library <https://scikit-learn.org/stable/modules/
linear_model.html>`_.
"""
from __future__ import annotations

from typing import ClassVar
from typing import Iterable

from numpy import array
from numpy import ndarray
from numpy import repeat
from numpy import zeros
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression as LinReg
from sklearn.linear_model import Ridge

from gemseo.core.dataset import Dataset
from gemseo.mlearning.core.ml_algo import DataType
from gemseo.mlearning.core.ml_algo import TransformerType
from gemseo.mlearning.regression.regression import MLRegressionAlgo
from gemseo.mlearning.transform.dimension_reduction.dimension_reduction import (
    DimensionReduction,
)
from gemseo.utils.data_conversion import split_array_to_dict_of_arrays
from gemseo.utils.python_compatibility import Final


class LinearRegressor(MLRegressionAlgo):
    """Linear regression model."""

    SHORT_ALGO_NAME: ClassVar[str] = "LinReg"
    LIBRARY: Final[str] = "scikit-learn"

    def __init__(
        self,
        data: Dataset,
        transformer: TransformerType = MLRegressionAlgo.IDENTITY,
        input_names: Iterable[str] | None = None,
        output_names: Iterable[str] | None = None,
        fit_intercept: bool = True,
        penalty_level: float = 0.0,
        l2_penalty_ratio: float = 1.0,
        **parameters: float | int | str | bool | None,
    ) -> None:
        """
        Args:
            fit_intercept: Whether to fit the intercept.
            penalty_level: The penalty level greater or equal to 0.
                If 0, there is no penalty.
            l2_penalty_ratio: The penalty ratio related to the l2 regularization.
                If 1, use the Ridge penalty.
                If 0, use the Lasso penalty.
                Between 0 and 1, use the ElasticNet penalty.
            **parameters: The parameters of the machine learning algorithm.
        """
        super().__init__(
            data,
            transformer=transformer,
            input_names=input_names,
            output_names=output_names,
            fit_intercept=fit_intercept,
            penalty_level=penalty_level,
            l2_penalty_ratio=l2_penalty_ratio,
            **parameters,
        )
        if "degree" in parameters:
            del parameters["degree"]

        if penalty_level == 0.0:
            self.algo = LinReg(copy_X=False, fit_intercept=fit_intercept, **parameters)
        else:
            if l2_penalty_ratio == 1.0:
                self.algo = Ridge(
                    copy_X=False,
                    fit_intercept=fit_intercept,
                    alpha=penalty_level,
                    **parameters,
                )
            elif l2_penalty_ratio == 0.0:
                self.algo = Lasso(
                    copy_X=False,
                    fit_intercept=fit_intercept,
                    alpha=penalty_level,
                    **parameters,
                )
            else:
                self.algo = ElasticNet(
                    copy_X=False,
                    fit_intercept=fit_intercept,
                    alpha=penalty_level,
                    l1_ratio=1 - l2_penalty_ratio,
                    **parameters,
                )

    def _fit(
        self,
        input_data: ndarray,
        output_data: ndarray,
    ) -> None:
        self.algo.fit(input_data, output_data)

    def _predict(
        self,
        input_data: ndarray,
    ) -> ndarray:
        return self.algo.predict(input_data)

    def _predict_jacobian(
        self,
        input_data: ndarray,
    ) -> ndarray:
        n_samples = input_data.shape[0]
        return repeat(self.algo.coef_[None], n_samples, axis=0)

    @property
    def coefficients(self) -> ndarray:
        """The regression coefficients of the linear model."""
        return self.algo.coef_

    @property
    def intercept(self) -> ndarray:
        """The regression intercepts of the linear model."""
        if self.parameters["fit_intercept"]:
            intercept = self.algo.intercept_
        else:
            intercept = zeros(self.algo.coef_.shape[0])
        return intercept

    def get_coefficients(
        self,
        as_dict: bool = True,
    ) -> DataType:
        """Return the regression coefficients of the linear model.

        Args:
            as_dict: If True, return the coefficients as a dictionary.
                Otherwise, return the coefficients as a numpy.array

        Returns:
            The regression coefficients of the linear model.

        Raises:
            ValueError: If the coefficients are required as a dictionary
                even though the transformers change the variables dimensions.
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
                    "Coefficients are only representable in dictionary "
                    "form if the transformers do not change the "
                    "dimensions of the variables."
                )
            coefficients = self.__convert_array_to_dict(coefficients)
        return coefficients

    def get_intercept(
        self,
        as_dict: bool = True,
    ) -> DataType:
        """Return the regression intercepts of the linear model.

        Args:
            as_dict: If True, return the intercepts as a dictionary.
                Otherwise, return the intercepts as a numpy.array

        Returns:
            The regression intercepts of the linear model.

        Raises:
            ValueError: If the coefficients are required as a dictionary
                even though the transformers change the variables dimensions.
        """
        intercept = self.intercept
        if as_dict:
            if Dataset.OUTPUT_GROUP in self.transformer:
                raise ValueError(
                    "Intercept is only representable in dictionary "
                    "form if the transformers do not change the "
                    "dimensions of the output variables."
                )
            varsizes = self.learning_set.sizes
            intercept = split_array_to_dict_of_arrays(
                intercept, varsizes, self.output_names
            )
            intercept = {key: list(val) for key, val in intercept.items()}
        return intercept

    def __convert_array_to_dict(
        self,
        data: ndarray,
    ) -> dict[str, ndarray]:
        """Convert a data array into a dictionary.

        Args:
            data: The data to be converted.

        Returns:
            The converted data.
        """
        varsizes = self.learning_set.sizes
        data = [
            split_array_to_dict_of_arrays(row, varsizes, self.input_names)
            for row in data
        ]
        data = [{key: list(val) for key, val in element.items()} for element in data]
        data = split_array_to_dict_of_arrays(array(data), varsizes, self.output_names)
        data = {key: list(val) for key, val in data.items()}
        return data
