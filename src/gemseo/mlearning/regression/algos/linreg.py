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

from typing import TYPE_CHECKING
from typing import ClassVar

from numpy import array
from numpy import repeat
from numpy import zeros
from sklearn.linear_model import ElasticNet
from sklearn.linear_model import Lasso
from sklearn.linear_model import LinearRegression as LinReg
from sklearn.linear_model import Ridge

from gemseo.datasets.io_dataset import IODataset
from gemseo.mlearning.regression.algos.base_regressor import BaseRegressor
from gemseo.mlearning.regression.algos.linreg_settings import LinearRegressor_Settings
from gemseo.mlearning.transformers.dimension_reduction.base_dimension_reduction import (
    BaseDimensionReduction,
)
from gemseo.utils.data_conversion import split_array_to_dict_of_arrays

if TYPE_CHECKING:
    from gemseo.typing import RealArray


class LinearRegressor(BaseRegressor):
    """Linear regression model."""

    SHORT_ALGO_NAME: ClassVar[str] = "LinReg"
    LIBRARY: ClassVar[str] = "scikit-learn"

    Settings: ClassVar[type[LinearRegressor_Settings]] = LinearRegressor_Settings

    def _post_init(self):
        super()._post_init()
        penalty_level = self._settings.penalty_level
        if penalty_level == 0.0:
            self.algo = LinReg(
                copy_X=False,
                fit_intercept=self._settings.fit_intercept,
                **self._settings.parameters,
            )
        elif self._settings.l2_penalty_ratio == 1.0:
            self.algo = Ridge(
                copy_X=False,
                fit_intercept=self._settings.fit_intercept,
                alpha=penalty_level,
                random_state=self._settings.random_state,
                **self._settings.parameters,
            )
        elif self._settings.l2_penalty_ratio == 0.0:
            self.algo = Lasso(
                copy_X=False,
                fit_intercept=self._settings.fit_intercept,
                alpha=penalty_level,
                random_state=self._settings.random_state,
                **self._settings.parameters,
            )
        else:
            self.algo = ElasticNet(
                copy_X=False,
                fit_intercept=self._settings.fit_intercept,
                alpha=penalty_level,
                l1_ratio=1 - self._settings.l2_penalty_ratio,
                random_state=self._settings.random_state,
                **self._settings.parameters,
            )

    def _fit(
        self,
        input_data: RealArray,
        output_data: RealArray,
    ) -> None:
        self.algo.fit(input_data, output_data)

    def _predict(
        self,
        input_data: RealArray,
    ) -> RealArray:
        return self.algo.predict(input_data).reshape((len(input_data), -1))

    def _predict_jacobian(
        self,
        input_data: RealArray,
    ) -> RealArray:
        return repeat(self.algo.coef_[None], len(input_data), axis=0)

    @property
    def coefficients(self) -> RealArray:
        """The regression coefficients of the linear model."""
        return self.algo.coef_

    @property
    def intercept(self) -> RealArray:
        """The regression intercepts of the linear model."""
        if self._settings.fit_intercept:
            return self.algo.intercept_

        return zeros(self.algo.coef_.shape[0])

    def get_coefficients(
        self,
        as_dict: bool = True,
    ) -> RealArray | dict[str, list[dict[str, list[float]]]]:
        """Return the regression coefficients of the linear model.

        Args:
            as_dict: If ``True``, return the coefficients as a dictionary.
                Otherwise, return the coefficients as a numpy.array

        Returns:
            The regression coefficients of the linear model.

        Raises:
            ValueError: If the coefficients are required as a dictionary
                even though the transformers change the variables dimensions.
        """
        coefficients = self.coefficients
        if not as_dict:
            return coefficients

        if any(
            isinstance(transformer, BaseDimensionReduction)
            for transformer in self.transformer.values()
        ):
            msg = (
                "Coefficients are only representable in dictionary "
                "form if the transformers do not change the "
                "dimensions of the variables."
            )
            raise ValueError(msg)
        return self.__convert_array_to_dict(coefficients)

    def get_intercept(self, as_dict: bool = True) -> RealArray | dict[str, list[float]]:
        """Return the regression intercepts of the linear model.

        Args:
            as_dict: If ``True``, return the intercepts as a dictionary.
                Otherwise, return the intercepts as a numpy.array

        Returns:
            The regression intercepts of the linear model.

        Raises:
            ValueError: If the coefficients are required as a dictionary
                even though the transformers change the variables dimensions.
        """
        intercept = self.intercept
        if not as_dict:
            return intercept

        if IODataset.OUTPUT_GROUP in self.transformer:
            msg = (
                "Intercept is only representable in dictionary "
                "form if the transformers do not change the "
                "dimensions of the output variables."
            )
            raise ValueError(msg)
        intercept = split_array_to_dict_of_arrays(
            intercept,
            self.learning_set.variable_names_to_n_components,
            self.output_names,
        )
        return {key: list(val) for key, val in intercept.items()}

    def __convert_array_to_dict(
        self,
        data: RealArray,
    ) -> dict[str, list[dict[str, list[float]]]]:
        """Convert a data array into a dictionary.

        Args:
            data: The data to be converted.

        Returns:
            The converted data.
        """
        varsizes = self.learning_set.variable_names_to_n_components
        data = [
            split_array_to_dict_of_arrays(row, varsizes, self.input_names)
            for row in data
        ]
        data = [{key: list(val) for key, val in element.items()} for element in data]
        data = split_array_to_dict_of_arrays(array(data), varsizes, self.output_names)
        return {key: list(val) for key, val in data.items()}
