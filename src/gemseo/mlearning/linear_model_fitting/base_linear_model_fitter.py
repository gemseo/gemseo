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

"""Base class for linear model fitting algorithms."""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar
from typing import Generic
from typing import TypeVar

from numpy import hstack
from numpy import ones
from numpy import vstack

from gemseo import READ_ONLY_EMPTY_DICT
from gemseo.mlearning.linear_model_fitting.base_linear_model_fitter_settings import (
    BaseLinearModelFitter_Settings,
)
from gemseo.utils.metaclasses import ABCGoogleDocstringInheritanceMeta

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Mapping

    from gemseo.typing import RealArray
    from gemseo.typing import StrKeyMapping

FitterType = TypeVar("FitterType", bound=Any)
SettingsType = TypeVar("SettingsType", bound=BaseLinearModelFitter_Settings)


class BaseLinearModelFitter(
    Generic[FitterType, SettingsType], metaclass=ABCGoogleDocstringInheritanceMeta
):
    """Base class for linear model fitting algorithms."""

    # TODO: API: rename to settings_class.
    Settings: ClassVar[type[BaseLinearModelFitter_Settings]]
    """The class for defining the settings of the linear model fitting algorithm."""

    _fitter: FitterType
    """The wrapped linear model fitting algorithm."""

    _settings: SettingsType
    """The settings of the linear model fitting algorithm."""

    _PRIORITARY_FITTER_KWARGS: ClassVar[Mapping[str, Any]] = READ_ONLY_EMPTY_DICT
    """Some keyword arguments of the wrapped linear model fitting algorithm."""

    @property
    @abstractmethod
    def _FITTER_CLASS(self) -> type[FitterType]:  # noqa: N802
        """The wrapped linear model fitting algorithm."""

    def __init__(self, settings: SettingsType | None = None) -> None:
        """
        Args:
            settings: The settings of the linear model fitting algorithm.
                If ``None``, use a default instance of :attr:`.Settings`.
        """  # noqa: D205, D212
        if settings is None:
            settings = self.Settings()

        fitter_kwargs = {
            key: value
            for key, value in settings.model_dump().items()
            if key not in BaseLinearModelFitter_Settings.model_fields
        }
        fitter_kwargs.update(self._PRIORITARY_FITTER_KWARGS)
        self._fitter = self._FITTER_CLASS(**fitter_kwargs)
        self._settings = settings

    def fit(
        self,
        input_data: RealArray,
        output_data: RealArray,
        *extra_data: tuple[RealArray, RealArray],
    ) -> RealArray:
        """Fit a linear model to data.

        Args:
            input_data: The features matrix,
                shaped as ``(n_samples, n_features)``.
            output_data: The observations matrix,
                shaped as ``(n_samples, n_targets)``.
            *extra_data: Additional pairs "(features matrix, observations matrix)".
                where a features matrix has ``n_features`` columns,
                an observations matrix as ``n_targets`` columns
                and the matrices of a pair have the same number of rows.
                E.g., Jacobian observations.
                This argument cannot be used
                when the option ``fit_intercept`` is ``True``.

        Returns:
            The coefficients of the linear model, shaped as ``(n_targets, n_features)``.
        """
        if self._settings.fit_intercept and not extra_data:
            input_data = hstack((ones((len(input_data), 1)), input_data))

        return self._fit(input_data, output_data, *extra_data)

    @abstractmethod
    def _fit(
        self,
        input_data: RealArray,
        output_data: RealArray,
        *extra_data: tuple[RealArray, RealArray],
    ) -> RealArray:
        """Fit a linear model to data.

        Args:
            input_data: The features matrix,
                shaped as ``(n_samples, n_features)``.
            output_data: The observations matrix,
                shaped as ``(n_samples, n_targets)``.
            *extra_data: Additional pairs "(features matrix, observations matrix)".
                where a features matrix has ``n_features`` columns,
                an observations matrix as ``n_targets`` columns
                and the matrices of a pair have the same number of rows.

        Returns:
            The coefficients of the linear model, shaped as ``(n_targets, n_features)``.
        """

    @staticmethod
    def _stack_data(
        features: RealArray,
        observations: RealArray,
        extra_data: Iterable[tuple[RealArray, RealArray]],
    ) -> tuple[RealArray, RealArray]:
        """Stack the features and observations matrices vertically.

        Args:
            features: The features matrix,
                shaped as ``(n_samples, n_features)``.
            observations: The observations matrix,
                shaped as ``(n_samples, n_targets)``.
            extra_data: Additional pairs "(features matrix, observations matrix)".
                where a features matrix has ``n_features`` columns,
                an observations matrix as ``n_targets`` columns
                and the matrices of a pair have the same number of rows.

        Returns:
            The features matrices stacked vertically,
            and the observations matrices stacked vertically.
        """
        if not extra_data:
            return features, observations

        stacked_features = vstack([features, *[x[0] for x in extra_data]])
        stacked_observations = vstack([observations, *[x[1] for x in extra_data]])
        return stacked_features, stacked_observations


class _WrappedFittingFunction(metaclass=ABCGoogleDocstringInheritanceMeta):
    """Base class for wrapped linear model fitting algorithms defined as functions."""

    _kwargs: StrKeyMapping
    """The options of the wrapped fitter."""

    def __init__(self, **kwargs: Any) -> None:
        """
        Args:
            **kwargs: The options of the wrapped fitter.
        """  # noqa: D205, D212
        self._kwargs = kwargs

    @abstractmethod
    def fit(
        self,
        input_data: RealArray,
        output_data: RealArray,
        *extra_data: tuple[RealArray, RealArray],
    ) -> RealArray:
        """Fit a linear model to data.

        Args:
            input_data: The features matrix,
                shaped as ``(n_samples, n_features)``.
            output_data: The observations matrix,
                shaped as ``(n_samples, n_targets)``.
            *extra_data: Additional pairs "(features matrix, observations matrix)".
                where a features matrix has ``n_features`` columns,
                an observations matrix as ``n_targets`` columns
                and the matrices of a pair have the same number of rows.

        Returns:
            The coefficients of the linear model, shaped as ``(n_targets, n_features)``.
        """
