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
"""A transformer to apply operations on NumPy arrays.

The abstract :class:`.BaseTransformer` class implements
the concept of a data transformer.
Inheriting classes shall implement the :meth:`.BaseTransformer.fit`,
:meth:`.BaseTransformer.transform`
and possibly :meth:`.BaseTransformer.inverse_transform` methods.

.. seealso::

   :mod:`~gemseo.mlearning.transformers.scaler.scaler`
   :mod:`~gemseo.mlearning.transformers.dimension_reduction.dimension_reduction`
"""

from __future__ import annotations

from abc import abstractmethod
from copy import deepcopy
from typing import TYPE_CHECKING
from typing import Any
from typing import ClassVar
from typing import NoReturn
from typing import Union

from numpy import atleast_2d
from numpy import ndarray
from numpy import newaxis

from gemseo.core.base_factory import BaseFactory
from gemseo.utils.metaclasses import ABCGoogleDocstringInheritanceMeta

if TYPE_CHECKING:
    from collections.abc import Callable

    from typing_extensions import ParamSpecArgs
    from typing_extensions import ParamSpecKwargs

    from gemseo.typing import RealArray

ParameterType = Union[bool, int, float, ndarray, str, None]
TransformerFitOptionType = Union[float, int, str]


class BaseTransformer(metaclass=ABCGoogleDocstringInheritanceMeta):
    """A data transformer fitted from some samples."""

    name: str
    """The name of the transformer."""

    CROSSED: ClassVar[bool] = False
    """Whether the :meth:`.fit` method requires two data arrays."""

    def __init__(self, name: str = "", **parameters: ParameterType) -> None:
        """
        Args:
            name: A name for this transformer.
            **parameters: The parameters of the transformer.
        """  # noqa: D205 D212
        self.name = name or self.__class__.__name__
        self.__parameters = parameters
        self.__is_fitted = False

    @property
    def is_fitted(self) -> bool:
        """Whether the transformer has been fitted from some data."""
        return self.__is_fitted

    @property
    def parameters(self) -> dict[str, ParameterType]:
        """The parameters of the transformer."""
        return self.__parameters

    def duplicate(self) -> BaseTransformer:
        """Duplicate the current object.

        Returns:
            A deepcopy of the current instance.
        """
        return deepcopy(self)

    def fit(self, data: ndarray, *args: TransformerFitOptionType) -> None:
        """Fit the transformer to the data.

        Args:
            data: The data to be fitted,
                shaped as ``(n_observations, n_features)`` or ``(n_observations, )``.
        """
        if data.ndim == 1:
            data = data[:, newaxis]

        self._fit(data, *args)
        self.__is_fitted = True

    @abstractmethod
    def _fit(self, data: ndarray, *args: TransformerFitOptionType) -> None:
        """Fit the transformer to the data.

        Args:
            data: The data to be fitted, shaped as ``(n_observations, n_features)``.
            *args: The options for the transformer.
        """

    @abstractmethod
    def transform(self, data: ndarray) -> ndarray:
        """Transform the data.

        Args:
            data: The data to be transformed,
                shaped as ``(n_observations, n_features)`` or ``(n_features, )``.

        Returns:
            The transformed data, shaped as ``data``.
        """

    def inverse_transform(self, data: ndarray) -> NoReturn:
        """Perform an inverse transform on the data.

        Args:
            data: The data to be inverse transformed,
                shaped as ``(n_observations, n_features)`` or ``(n_features, )``.

        Returns:
            The inverse transformed data, shaped as ``data``.
        """
        raise NotImplementedError

    def fit_transform(self, data: ndarray, *args: TransformerFitOptionType) -> ndarray:
        """Fit the transformer to the data and transform the data.

        Args:
            data: The data to be transformed,
                shaped as ``(n_observations, n_features)`` or ``(n_observations, )``.

        Returns:
            The transformed data, shaped as ``data``.
        """
        if data.ndim == 1:
            data = data[:, newaxis]

        self.fit(data, *args)
        return self.transform(data)

    def compute_jacobian(self, data: RealArray) -> NoReturn:
        """Compute the Jacobian of :meth:`.transform`.

        Args:
            data: The data where the Jacobian is to be computed,
                shaped as ``(n_observations, n_features)`` or ``(n_features, )``.

        Returns:
            The Jacobian matrix, shaped according to ``data``.
        """
        raise NotImplementedError

    def compute_jacobian_inverse(self, data: RealArray) -> NoReturn:
        """Compute the Jacobian of the :meth:`.inverse_transform`.

        Args:
            data: The data where the Jacobian is to be computed,
                shaped as ``(n_observations, n_features)`` or ``(n_features, )``.

        Returns:
            The Jacobian matrix, shaped according to ``data``..
        """
        raise NotImplementedError

    def __str__(self) -> str:
        return self.__class__.__name__

    @staticmethod
    def _use_2d_array(
        f: Callable[[ndarray, ParamSpecArgs, ParamSpecKwargs], Any],
    ) -> Callable[[ndarray, ParamSpecArgs, ParamSpecKwargs], Any]:
        """Force the NumPy array passed to a function as 1st argument to be at least 2D.

        Args:
            f: The function.
        """

        def g(self, data: ndarray, *args: Any, **kwargs: Any) -> Any:
            """Force a NumPy array to be at least 2D and evaluate the function ``f``.

            ``f`` expects a 2D array shaped as ``(n_points, input_dimension)``
            and returns a nD arrays shaped as ``(..., n_points, output_dimension)``
            or ``(..., n_points, output_dimension, input_dimension)``.

            If the original ``data`` is a 1D array shaped as ``(input_dimension,)``,
            then
            this wrapper returns a (n-1)D array shaped as ``(..., output_dimension)``
            or ``(..., output_dimension, intput_dimension)``.

            Args:
                data: A NumPy array.
                *args: The positional arguments.
                **kwargs: The optional arguments.

            Returns:
                Any kind of output;
                if a NumPy array,
                its dimension is made consistent with the shape of ``data``.
            """
            if data.ndim >= 2:
                # data has already at least 2 dimensions.
                return f(self, data, *args, **kwargs)

            # Force data to have at least 2 dimensions.
            out = f(self, atleast_2d(data), *args, **kwargs)
            if not isinstance(out, ndarray):
                return out

            if f.__name__ in ["compute_jacobian", "compute_jacobian_inverse"]:
                # Case (..., n_points, output_dimension, input_dimension)
                return out[..., 0, :, :]

            # Case (..., n_points, output_dimension, input_dimension)
            return out[..., 0, :]

        return g


class TransformerFactory(BaseFactory):
    """A factory of transformers."""

    _CLASS = BaseTransformer
    _PACKAGE_NAMES = ("gemseo.mlearning.transformers",)
