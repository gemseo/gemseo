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

The abstract :class:`.Transformer` class implements the concept of a data transformer.
Inheriting classes shall implement the :meth:`.Transformer.fit`,
:meth:`.Transformer.transform`
and possibly :meth:`.Transformer.inverse_transform` methods.

.. seealso::

   :mod:`~gemseo.mlearning.transform.scaler.scaler`
   :mod:`~gemseo.mlearning.transform.dimension_reduction.dimension_reduction`
"""
from __future__ import annotations

from abc import abstractmethod
from typing import ClassVar
from typing import NoReturn
from typing import Union

from numpy import ndarray

from gemseo.core.factory import Factory
from gemseo.utils.metaclasses import ABCGoogleDocstringInheritanceMeta

ParameterType = Union[bool, int, float, ndarray, str, None]
TransformerFitOptionType = Union[float, int, str]


class Transformer(metaclass=ABCGoogleDocstringInheritanceMeta):
    """A data transformer fitted from some samples."""

    name: str
    """The name of the transformer."""

    CROSSED: ClassVar[bool] = False
    """Whether the :meth:`.fit` method requires two data arrays."""

    def __init__(self, name: str = "Transformer", **parameters: ParameterType) -> None:
        """
        Args:
            name: A name for this transformer.
            **parameters: The parameters of the transformer.
        """
        self.name = name
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

    def duplicate(self) -> Transformer:
        """Duplicate the current object.

        Returns:
            A deepcopy of the current instance.
        """
        return self.__class__(self.name, **self.parameters)

    def fit(self, data: ndarray, *args: TransformerFitOptionType) -> None:
        """Fit the transformer to the data.

        Args:
            data: The data to be fitted.
        """
        self._fit(data, *args)
        self.__is_fitted = True

    @abstractmethod
    def _fit(self, data: ndarray, *args: TransformerFitOptionType) -> None:
        """Fit the transformer to the data.

        Args:
            data: The data to be fitted.
        """

    @abstractmethod
    def transform(self, data: ndarray) -> ndarray:
        """Transform the data.

        Args:
            data: The data to be transformed.

        Returns:
            The transformed data.
        """

    def inverse_transform(self, data: ndarray) -> NoReturn:
        """Perform an inverse transform on the data.

        Args:
            data: The data to be inverse transformed.

        Returns:
            The inverse transformed data.
        """
        raise NotImplementedError

    def fit_transform(self, data: ndarray, *args: TransformerFitOptionType) -> ndarray:
        """Fit the transformer to the data and transform the data.

        Args:
            data: The data to be transformed.

        Returns:
            The transformed data.
        """
        self.fit(data, *args)
        return self.transform(data)

    def compute_jacobian(self, data: ndarray) -> NoReturn:
        """Compute Jacobian of transformer.transform().

        Args:
            data: The data where the Jacobian is to be computed.

        Returns:
            The Jacobian matrix.
        """
        raise NotImplementedError

    def compute_jacobian_inverse(self, data: ndarray) -> NoReturn:
        """Compute Jacobian of the transformer.inverse_transform().

        Args:
            data: The data where the Jacobian is to be computed.

        Returns:
            The Jacobian matrix.
        """
        raise NotImplementedError

    def __str__(self) -> str:
        return self.__class__.__name__


class TransformerFactory(Factory):
    """A factory of :class:`.Transformer`."""

    def __init__(self) -> None:
        super().__init__(Transformer, ("gemseo.mlearning.transform",))
