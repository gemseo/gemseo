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
#        :author: Syver Doving Agdestein
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""The base class for regression algorithms."""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING
from typing import ClassVar
from typing import NoReturn

from gemseo.datasets.io_dataset import IODataset
from gemseo.mlearning.core.algos.supervised import BaseMLSupervisedAlgo
from gemseo.mlearning.data_formatters.regression_data_formatters import (
    RegressionDataFormatters,
)
from gemseo.mlearning.regression.algos.base_regressor_settings import (
    BaseRegressorSettings,
)
from gemseo.mlearning.transformers.scaler.min_max_scaler import MinMaxScaler

if TYPE_CHECKING:
    from gemseo.mlearning.core.algos.ml_algo import DataType
    from gemseo.mlearning.core.algos.ml_algo import DefaultTransformerType
    from gemseo.typing import RealArray


class BaseRegressor(BaseMLSupervisedAlgo):
    """The base class for regression algorithms."""

    DEFAULT_TRANSFORMER: DefaultTransformerType = MappingProxyType({
        IODataset.INPUT_GROUP: MinMaxScaler(),
        IODataset.OUTPUT_GROUP: MinMaxScaler(),
    })

    DataFormatters = RegressionDataFormatters

    Settings: ClassVar[type[BaseRegressorSettings]] = BaseRegressorSettings

    def predict_raw(
        self,
        input_data: RealArray,
    ) -> RealArray:
        """Predict output data from input data.

        Args:
            input_data: The input data with shape (n_samples, n_inputs).

        Returns:
            The predicted output data with shape (n_samples, n_outputs).
        """
        return self._predict(input_data)

    @DataFormatters.format_dict_jacobian
    @DataFormatters.format_samples()
    @DataFormatters.transform_jacobian
    def predict_jacobian(
        self,
        input_data: DataType,
    ) -> DataType:
        """Predict the Jacobians of the regression model at input_data.

        The user can specify these input data either as a NumPy array,
        e.g. ``array([1., 2., 3.])``
        or as a dictionary,
        e.g.  ``{'a': array([1.]), 'b': array([2., 3.])}``.

        If the NumPy arrays are of dimension 2,
        their i-th rows represent the input data of the i-th sample;
        while if the NumPy arrays are of dimension 1,
        there is a single sample.

        The type of the output data and the dimension of the output arrays
        will be consistent
        with the type of the input data and the size of the input arrays.

        Args:
            input_data: The input data.

        Returns:
            The predicted Jacobian data.
        """
        return self._predict_jacobian(input_data)

    def _predict_jacobian(
        self,
        input_data: RealArray,
    ) -> NoReturn:
        """Predict the Jacobian matrices of the regression model at input_data.

        Args:
            input_data: The input data with shape (n_samples, n_inputs).

        Returns:
            The predicted Jacobian data with shape (n_samples, n_outputs, n_inputs).

        Raises:
            NotImplementedError: When the method is called.
        """
        msg = f"Derivatives are not available for {self.__class__.__name__}."
        raise NotImplementedError(msg)
