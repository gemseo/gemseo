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

from gemseo.algos.database import Database
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

    _jacobian_data: RealArray | None
    """The Jacobian data if any."""

    def _post_init(self):
        super()._post_init()
        if self.learning_set.GRADIENT_GROUP in self.learning_set.group_names:
            self._jacobian_data = self.learning_set.get_view(
                group_names=self.learning_set.GRADIENT_GROUP,
                variable_names=[
                    Database.get_gradient_name(output_name)
                    for output_name in self.output_names
                ],
                indices=self._learning_samples_indices,
            ).to_numpy()
        else:
            self._jacobian_data = None

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
        r"""Predict the Jacobian with respect to the input variables.

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

    def predict_jacobian_wrt_special_variables(
        self,
        input_data: RealArray,
    ) -> RealArray:
        r"""Predict the Jacobian with respect to special variables.

        The method :meth:`.predict_jacobian` predicts the standard Jacobian,
        which is the matrix of partial derivatives with respect to the input variables.

        In some cases,
        the regressor :math:`\hat{f}(x)` is used
        to approximate a model :math:`f(x,p)` at point :math:`p`
        given a training dataset
        :math:`\left(x_i,f(x_i,p),\partial_p f(x_i,p)\right)_{1\leq i \leq N}`
        including not only the input and output samples
        :math:`\left(x_i,f(x_i,p)\right)_{1\leq i \leq N}`
        but also the samples of the partial derivatives of the outputs
        with respect to a special variable :math:`p`
        that is not an input variable of the regressor.
        Then,
        as the regressor :math:`\hat{f}(x)` is a function of
        :math:`\left(f(x_i,p)\right)_{1\leq i \leq N}`,
        it is also a function of :math:`p`.
        Consequently,
        it can be differentiated with respect to :math:`p`
        using the chain rule principle if the regressor implements this mechanism.

        Args:
            input_data: The input data
                with shape ``(n_samples, special_variable_dimension)``.

        Returns:
            The predicted Jacobian data
            with shape ``(n_samples, n_outputs, special_variable_dimension)``.

        Raises:
            ValueError: When the training dataset does not include gradient information.
        """
        self._check_jacobian_learning_data("predict_jacobian_wrt_special_variables")
        return self._predict_jacobian_wrt_special_variables(input_data)

    def _check_jacobian_learning_data(self, attribute_name: str) -> None:
        """Check whether the attribute can be used.

        Raises:
            ValueError: When the training dataset does not include gradient information.
        """
        if self._jacobian_data is None:
            msg = (
                f"You cannot use {attribute_name} "
                "because the training dataset does not include gradient information."
            )
            raise ValueError(msg)

    def _predict_jacobian_wrt_special_variables(
        self,
        input_data: RealArray,
    ) -> NoReturn:
        r"""Predict the Jacobian matrices of the regression model.

        Args:
            input_data: The input data
                with shape ``(n_samples, special_variable_dimension)``.

        Returns:
            The predicted Jacobian data
            with shape ``(n_samples, n_outputs, special_variable_dimension)``.

        Raises:
            NotImplementedError: When the derivatives are not available.
        """
        msg = (
            f"{self.__class__.__name__} does not implement "
            f"differentiation with respect to special variables."
        )
        raise NotImplementedError(msg)

    def _predict_jacobian(
        self,
        input_data: RealArray,
    ) -> NoReturn:
        """Predict the Jacobian matrices of the regression model at input_data.

        These Jacobian matrices includes the partial derivatives of the outputs
        with respect to the differentiated inputs.

        Args:
            input_data: The input data with shape ``(n_samples, n_inputs)``.

        Returns:
            The predicted Jacobian data with shape ``(n_samples, n_outputs, n_inputs)``.

        Raises:
            NotImplementedError: When the derivatives are not available.
        """
        msg = f"Derivatives are not available for {self.__class__.__name__}."
        raise NotImplementedError(msg)
