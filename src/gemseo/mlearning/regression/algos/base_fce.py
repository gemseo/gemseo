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
"""Base class for functional chaos expansion models."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar

from numpy import array
from numpy import newaxis
from numpy import swapaxes
from numpy import vstack
from scipy.linalg import solve

from gemseo.mlearning.regression.algos.base_fce_settings import (
    BaseFCERegressor_Settings,
)
from gemseo.mlearning.regression.algos.base_regressor import BaseRegressor
from gemseo.utils.pydantic import create_model

if TYPE_CHECKING:
    from collections.abc import Callable
    from collections.abc import Sequence

    from numpy.typing import ArrayLike

    from gemseo.datasets.io_dataset import IODataset
    from gemseo.typing import RealArray


class BaseFCERegressor(BaseRegressor):
    """Base class for functional chaos expansion models."""

    Settings: ClassVar[type[BaseFCERegressor_Settings]] = BaseFCERegressor_Settings

    _mean_jacobian_wrt_special_variables: RealArray | None
    """The gradient of the mean with respect to the special variables.

    Shaped as ``(output_dimension, special_variable_dimension)``.
    """

    _standard_deviation_jacobian_wrt_special_variables: RealArray | None
    """The gradient of the standard deviation with respect the special variables.

    Shaped as ``(output_dimension, special_variable_dimension)``.
    """

    _variance_jacobian_wrt_special_variables: RealArray | None
    """The gradient of the variance with respect the special variables.

    Shaped as ``(output_dimension, special_variable_dimension)``.
    """

    _mean: RealArray
    """The output mean vector of the model output."""

    _standard_deviation: RealArray
    """The standard deviation vector of the model output."""

    _variance: RealArray
    """The variance vector of the model output."""

    _first_order_sobol_indices: list[dict[str, RealArray]]
    """The first-order Sobol' indices for the different output components."""

    _total_order_sobol_indices: list[dict[str, RealArray]]
    """The total-order Sobol' indices for the different output components."""

    _coefficients: RealArray
    """The coefficients of the PCE model, as (output_dimension, n_basis_functions)."""

    _jac_coefficients: RealArray | None
    """The coefficients to differentiate with respect to the special variables.

    Shaped as ``(output_dimension, special_variable_dimension, n_basis_functions)``.
    """

    _basis_functions: Sequence[Callable[[RealArray], ArrayLike]]
    """The basis functions."""

    _isoprobabilistic_transformation: Callable[[RealArray], RealArray]
    """The isoprobabilistic transformation after training, ``None`` before."""

    def __init__(
        self,
        data: IODataset,
        settings_model: BaseFCERegressor_Settings | None = None,
        **settings: Any,
    ) -> None:
        """
        Args:
            data: The training dataset
                whose input space ``data.misc["input_space"]``
                is expected to be a :class:`.ParameterSpace`
                defining the random input variables.

        Raises:
            ValueError: When ``learn_jacobian_data`` or ``use_special_jacobian_data``
                is ``True`` but the training dataset does not contain Jacobian data.
        """  # noqa: D205 D212
        settings_ = create_model(
            self.Settings, settings_model=settings_model, **settings
        )
        super().__init__(data, settings_model=settings_)
        self._mean = array([])
        self._variance = array([])
        self._standard_deviation = array([])
        self._first_order_sobol_indices = []
        self._total_order_sobol_indices = []
        self._mean_jacobian_wrt_special_variables = None
        self._standard_deviation_jacobian_wrt_special_variables = None
        self._variance_jacobian_wrt_special_variables = None
        self._coefficients = array([])
        self._jac_coefficients = None
        self._basis_functions = ()
        self._isoprobabilistic_transformation = None
        if self._settings.learn_jacobian_data and self._jacobian_data is None:
            msg = (
                "Option learn_jacobian_data is True "
                "but the training dataset does not contain Jacobian data."
            )
            raise ValueError(msg)

        if self._settings.use_special_jacobian_data and self._jacobian_data is None:
            msg = (
                "Option use_special_jacobian_data is True "
                "but the training dataset does not contain Jacobian data."
            )
            raise ValueError(msg)

    def _fit(
        self,
        input_data: RealArray,
        output_data: RealArray,
    ) -> None:
        args = self._create_predictor(input_data, output_data)
        self._compute_statistics(*args)
        self._compute_sobol_indices(*args)

        if self._settings.use_special_jacobian_data:
            self.__use_special_jacobian_data(*args)

    @abstractmethod
    def _create_predictor(
        self,
        input_data: RealArray,
        output_data: RealArray,
    ) -> tuple[Any, ...]:
        """Create the elements to predict output and Jacobian data.

        Args:
            input_data: The input data.
            output_data: The output data.

        Returns:
            Objects to be used in the :meth:`._fit` method, if any.
        """

    @abstractmethod
    def _compute_statistics(self, *args: Any) -> None:
        """Compute the statistics of the model output, e.g. mean and variance."""

    @abstractmethod
    def _compute_sobol_indices(self, *args: Any) -> None:
        """Compute the Sobol' indices of the model output."""

    def __use_special_jacobian_data(self, *args: Any) -> None:
        """Manage the special Jacobian data available in the training dataset.

        Special Jacobian data are samples of partial derivatives
        with respect to variables that are not inputs of the FCE.

        Args:
            *args: Positional arguments.
        """
        features = self._get_features_for_special_jacobian_data_use(*args)
        jacobian_data = self._jacobian_data[self._learning_samples_indices]
        jac_coefficients = (
            solve(
                features.T @ features,
                features.T,
                overwrite_a=True,
                overwrite_b=False,
                assume_a="sym",
            )
            @ jacobian_data
        )
        shape = (self._reduced_output_dimension, -1, features.shape[1])
        self._jac_coefficients = jac_coefficients.T.reshape(shape)
        self._mean_jacobian_wrt_special_variables = self._jac_coefficients[..., 0]
        self._variance_jacobian_wrt_special_variables = vstack([
            2 * ci_jac @ ci_out
            for ci_jac, ci_out in zip(
                self._jac_coefficients[..., 1:],
                self._coefficients.T[:, 1:],
                strict=False,
            )
        ])
        # _variance_jacobian_wrt_special_variables: (n_out, n_in)
        # _standard_deviation: (n_out,)
        self._standard_deviation_jacobian_wrt_special_variables = (
            self._variance_jacobian_wrt_special_variables
            / 2
            / self._standard_deviation[:, newaxis]
        )

    @abstractmethod
    def _get_features_for_special_jacobian_data_use(self, *args: Any) -> RealArray:
        """Return the features matrix for using the special Jacobian data.

        Args:
            *args: Positional arguments.

        Returns:
            The features matrix for using the special Jacobian data.
        """

    @abstractmethod
    def _evaluate_basis_functions(
        self, input_data: RealArray
    ) -> tuple[RealArray, RealArray | None]:
        r"""Evaluate the :math:`p` basis functions and their partial derivatives.

        Args:
            input_data: The input data, shaped as :math:`(n, d)`,
                where :math:`n` is the number of samples
                and :math:`d` is the input dimension.

        Returns:
            The evaluations of the basis functions, shaped as :math:`(n, p)`,
            and the evaluations of their partial derivatives, shaped as :math:`(nd, p)`,
            if these derivative functions are implemented (otherwise ``None``).
            In the evaluations of their partial derivatives,
            the :math:`d` first rows correspond to the first sample,
            the following :math:`d` rows correspond to the second sample,
            and so on.
        """

    def _create_jacobian_for_linear_model_fitting(
        self, input_data: RealArray
    ) -> RealArray:
        """Create the Jacobian data for the linear model fitting problem.

        Args:
            input_data: The input data.

        Returns:
            The Jacobian matrix related to the training dataset.
        """
        jacobian_data = self._jacobian_data[self._learning_samples_indices]
        # The shape of jacobian_data is (n_samples, output_dimension * input_dimension).
        jacobian_data = jacobian_data.reshape((
            -1,
            self._reduced_output_dimension,
            self._reduced_input_dimension,
        ))
        jacobian_data = swapaxes(jacobian_data, 1, 2)
        return jacobian_data.reshape((-1, self._reduced_output_dimension))

    @property
    def mean_jacobian_wrt_special_variables(self) -> RealArray:
        """The gradient of the mean with respect to the special variables.

        See :meth:`.predict_jacobian_wrt_special_variables`
        for more information about the notion of special variables.

        Raises:
            ValueError: When the training dataset does not include gradient information.
        """
        self._check_is_trained()
        self._check_jacobian_learning_data("mean_jacobian_wrt_special_variables")
        return self._mean_jacobian_wrt_special_variables

    @property
    def standard_deviation_jacobian_wrt_special_variables(self) -> RealArray:
        """The gradient of the standard deviation with respect to the special variables.

        See :meth:`.predict_jacobian_wrt_special_variables`
        for more information about the notion of special variables.

        Raises:
            ValueError: When the training dataset does not include gradient information.
        """
        self._check_is_trained()
        self._check_jacobian_learning_data(
            "standard_deviation_jacobian_wrt_special_variables"
        )
        return self._standard_deviation_jacobian_wrt_special_variables

    @property
    def variance_jacobian_wrt_special_variables(self) -> RealArray:
        """The gradient of the variance with respect to the special variables.

        See :meth:`.predict_jacobian_wrt_special_variables`
        for more information about the notion of special variables.

        Raises:
            ValueError: When the training dataset does not include gradient information.
        """
        self._check_is_trained()
        self._check_jacobian_learning_data("variance_jacobian_wrt_special_variables")
        return self._variance_jacobian_wrt_special_variables

    @property
    def mean(self) -> RealArray:
        """The mean vector of the model output.

        .. warning::

           This statistic is expressed in relation to the transformed output space.
           You can sample the :meth:`.predict` method
           to estimate it in relation to the original output space
           if it is different from the transformed output space.
        """
        self._check_is_trained()
        return self._mean

    @property
    def variance(self) -> RealArray:
        """The variance vector of the model output.

        .. warning::

           This statistic is expressed in relation to the transformed output space.
           You can sample the :meth:`.predict` method
           to estimate it in relation to the original output space
           if it is different from the transformed output space.
        """
        self._check_is_trained()
        return self._variance

    @property
    def standard_deviation(self) -> RealArray:
        """The standard deviation vector of the model output.

        .. warning::

           This statistic is expressed in relation to the transformed output space.
           You can sample the :meth:`.predict` method
           to estimate it in relation to the original output space
           if it is different from the transformed output space.
        """
        self._check_is_trained()
        return self._standard_deviation

    @property
    def first_sobol_indices(self) -> list[dict[str, RealArray]]:
        """The first-order Sobol' indices for the different output components.

        .. warning::

           These statistics are expressed in relation to the transformed output space.
           You can use a :class:`.SobolAnalysis`
           to estimate them in relation to the original output space
           if it is different from the transformed output space.
        """
        self._check_is_trained()
        return self._first_order_sobol_indices

    @property
    def total_sobol_indices(self) -> list[dict[str, RealArray]]:
        """The total Sobol' indices for the different output components.

        .. warning::

           These statistics are expressed in relation to the transformed output space.
           You can use a :class:`.SobolAnalysis`
           to estimate them in relation to the original output space
           if it is different from the transformed output space.
        """
        self._check_is_trained()
        return self._total_order_sobol_indices

    def _check_jacobian_learning_data(self, attribute_name: str) -> None:
        if not self._settings.use_special_jacobian_data:
            msg = (
                f"You cannot use {attribute_name} "
                "because use_special_jacobian_data is False."
            )
            raise ValueError(msg)
