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

"""Functional chaos expansion model."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar

from numpy import array
from numpy import swapaxes
from numpy.linalg import norm

from gemseo.mlearning._basis.factory import BasisFactory
from gemseo.mlearning.linear_model_fitting.factory import LinearModelFitterFactory
from gemseo.mlearning.regression.algos.base_fce import BaseFCERegressor
from gemseo.mlearning.regression.algos.fce_settings import FCERegressor_Settings
from gemseo.utils.pydantic import create_model

if TYPE_CHECKING:
    from gemseo.datasets.io_dataset import IODataset
    from gemseo.mlearning._basis.base_basis import BaseBasis
    from gemseo.typing import RealArray
    from gemseo.typing import StrKeyMapping


class FCERegressor(BaseFCERegressor):
    r"""Functional chaos expansion model.

    Given a training dataset
    whose input samples are generated from OpenTURNS probability distributions,
    this regression algorithm can use any linear model fitting algorithm,
    including sparse techniques,
    to fit a functional chaos expansion (FCE) model of the form

    .. math::

       y = \sum_{i\in\mathcal{I}\subset\mathbb{N}^d} w_i\Psi_i(x)

    where :math:`\Psi_i(x)=\prod_{j=1}^d\psi_{i,j}(x_j)`
    and :math:`\mathbb{E}[\Psi_i(x)\Psi_j(x)]=\delta_{ij}`
    with :math:`\delta` the Kronecker delta.
    """

    SHORT_ALGO_NAME: ClassVar[str] = "FCE"
    LIBRARY: ClassVar[str] = "GEMSEO"

    Settings: ClassVar[type[FCERegressor_Settings]] = FCERegressor_Settings

    __basis: BaseBasis | None
    """The orthonormal multivariate basis after training, ``None`` before."""

    _ATTR_NOT_TO_SERIALIZE: ClassVar[set[str]] = (
        BaseFCERegressor._ATTR_NOT_TO_SERIALIZE.union({
            "_FCERegressor__basis",
            "_basis_functions",
            "_isoprobabilistic_transformation",
        })
    )

    def __init__(
        self,
        data: IODataset,
        settings_model: FCERegressor_Settings | None = None,
        **settings: Any,
    ) -> None:
        """
        Args:
            data: The training dataset
                whose input space ``data.misc["input_space"]``
                is expected to be a :class:`.ParameterSpace`
                defining the random input variables as :class:`.OTDistribution` objects.
        """  # noqa: D205 D212
        super().__init__(
            data,
            settings_model=create_model(
                self.Settings, settings_model=settings_model, **settings
            ),
        )
        self.__basis = None

    def _get_features_for_special_jacobian_data_use(
        self, features: RealArray
    ) -> RealArray:
        """
        Args:
            features: The features matrix.
        """  # noqa: D205, D212
        return features

    def _compute_sobol_indices(self, features: RealArray) -> None:
        """
        Args:
            features: The features matrix.
        """  # noqa: D205, D212
        convert = self.learning_set.misc["input_space"].convert_array_to_dict
        degree = self._settings.degree
        input_dimension = self._reduced_input_dimension
        get_index = self.__basis.get_index
        indices = [
            [
                get_index([n if j == i else 0 for j in range(input_dimension)])
                for n in range(1, degree + 1)
            ]
            for i in range(input_dimension)
        ]
        self._first_order_sobol_indices = [
            convert(
                array([
                    sum(coefficients[j] ** 2 for j in indices[i])
                    for i in range(input_dimension)
                ])
                / variance
            )
            for coefficients, variance in zip(
                self._coefficients.T, self._variance, strict=False
            )
        ]
        get_multi_index = self.__basis.get_multi_index
        multi_indices = [
            get_multi_index(i) for i in range(len(self.__basis.basis_functions))
        ]
        indices = [
            [
                j
                for j, multi_index in enumerate(multi_indices)
                if (d := multi_index[i]) > 0 and sum(multi_index) >= d
            ]
            for i in range(input_dimension)
        ]
        self._total_order_sobol_indices = [
            convert(
                array([
                    sum(coefficients[j] ** 2 for j in indices[i])
                    for i in range(input_dimension)
                ])
                / variance
            )
            for coefficients, variance in zip(
                self._coefficients.T, self._variance, strict=False
            )
        ]

    def _compute_statistics(self, features: RealArray) -> None:
        """
        Args:
            features: The features matrix.
        """  # noqa: D205, D212
        self._mean = self._coefficients[0, :]
        self._variance = (self._coefficients[1:, :] ** 2).sum(axis=0)
        self._standard_deviation = self._variance**0.5

    def _create_predictor(
        self, input_data: RealArray, output_data: RealArray
    ) -> tuple[RealArray]:
        """
        Returns:
            The features matrix.
        """  # noqa: D205, D212
        self.__set_basis()
        features, jac_features = self._evaluate_basis_functions(input_data)
        linear_model_fitter_settings = self._settings.linear_model_fitter_settings
        linear_model_fitter_settings.fit_intercept = False
        linear_model = LinearModelFitterFactory().create(
            linear_model_fitter_settings._TARGET_CLASS_NAME,
            linear_model_fitter_settings,
        )
        output_scale = norm(output_data, axis=0)
        output_data /= output_scale
        if jac_features is None:
            extra_data = ()
        else:
            jac_data = self._create_jacobian_for_linear_model_fitting(input_data)
            jac_data /= output_scale
            extra_data = ((jac_features, jac_data),)

        unscaled_coefficients = linear_model.fit(features, output_data, *extra_data).T
        self._coefficients = unscaled_coefficients * output_scale
        return (features,)

    def _evaluate_basis_functions(  # noqa: D102
        self, input_data: RealArray
    ) -> tuple[RealArray, RealArray | None]:
        features = self.__basis.compute_output_data(input_data)
        if self._settings.learn_jacobian_data:
            jac_features = self.__basis.compute_jacobian_data(input_data)
            # The shape of jac_features is (n_samples, input_dimension, n_features).
            jac_features = jac_features.reshape((-1, len(self.__basis.basis_functions)))
        else:
            jac_features = None

        return features, jac_features

    def __set_basis(self) -> None:
        """Set the private attribute ``__basis``."""
        self.__basis = BasisFactory().create(
            self._settings.basis,
            self.learning_set.misc["input_space"].distribution,
            self._settings.degree,
        )
        self._basis_functions = self.__basis.basis_functions
        self._isoprobabilistic_transformation = self.__basis._evaluate_transformation

    def _predict(
        self,
        input_data: RealArray,
    ) -> RealArray:
        features = self.__basis.compute_output_data(input_data)
        return features @ self._coefficients

    def _predict_jacobian(
        self,
        input_data: RealArray,
    ) -> RealArray:
        features = self.__basis.compute_jacobian_data(input_data)
        return swapaxes(features @ self._coefficients, 1, 2)

    def _predict_jacobian_wrt_special_variables(  # noqa: D102
        self, input_data: RealArray
    ) -> RealArray:
        transformed_input_data = self.__basis._evaluate_transformation(input_data)
        y = array([
            basis_function(transformed_input_data)[0]
            for basis_function in self.__basis.basis_functions
        ])
        return self._jac_coefficients @ y

    def __setstate__(
        self,
        state: StrKeyMapping,
    ) -> None:
        super().__setstate__(state)
        self.__set_basis()
