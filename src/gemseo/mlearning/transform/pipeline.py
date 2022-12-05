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
"""A pipeline to chain transformers.

The :class:`.Pipeline` class chains a sequence of tranformers, and provides global fit(),
transform(), fit_transform() and inverse_transform() methods.
"""
from __future__ import annotations

from typing import Sequence

from numpy import eye
from numpy import matmul
from numpy import ndarray

from gemseo.mlearning.transform.transformer import Transformer
from gemseo.mlearning.transform.transformer import TransformerFitOptionType


class Pipeline(Transformer):
    """Transformer pipeline."""

    transformers: Sequence[Transformer]
    """The sequence of transformers."""

    def __init__(
        self,
        name: str = "Pipeline",
        transformers: Sequence[Transformer] | None = None,
    ) -> None:
        """
        Args:
            name: A name for this pipeline.
            transformers: A sequence of transformers to be
                chained. The transformers are chained in the order of appearance in
                the list, i.e. the first transformer is applied first. If
                transformers is an empty list or None, then the pipeline
                transformer behaves like an identity transformer.
        """
        super().__init__(name)
        self.transformers = transformers or []

    def duplicate(self) -> Pipeline:
        """Duplicate the current object.

        Returns:
            A deepcopy of the current instance.
        """
        transformers = [trans.duplicate() for trans in self.transformers]
        return self.__class__(self.name, transformers)

    def _fit(
        self,
        data: ndarray,
        *args: TransformerFitOptionType,
    ) -> None:
        """Fit the transformer pipeline to the data.

        All the transformers are fitted, transforming the data in place.

        Args:
            data: The data to be fitted.
        """
        for transformer in self.transformers:
            data = transformer.fit_transform(data, *args)

    def transform(
        self,
        data: ndarray,
    ) -> ndarray:
        """Transform the data.

        The data is transformed sequentially,
        where the output of one transformer is the input of the next.

        Args:
            data: The data to be transformed.

        Returns:
            The transformed data.
        """
        for transformer in self.transformers:
            data = transformer.transform(data)
        return data

    def inverse_transform(
        self,
        data: ndarray,
    ) -> ndarray:
        """Perform an inverse transform on the data.

        The data is inverse transformed sequentially,
        starting with the last transformer in the list.

        Args:
            data: The data to be inverse transformed.

        Returns:
            The inverse transformed data.
        """
        for transformer in self.transformers[::-1]:
            data = transformer.inverse_transform(data)
        return data

    def compute_jacobian(
        self,
        data: ndarray,
    ) -> ndarray:
        """Compute the Jacobian of the ``pipeline.transform()``.

        Args:
            data: The data where the Jacobian is to be computed.

        Returns:
            The Jacobian matrix.
        """
        jacobian = eye(data.shape[-1])
        for transformer in self.transformers:
            jacobian = matmul(transformer.compute_jacobian(data), jacobian)
            data = transformer.transform(data)
        return jacobian

    def compute_jacobian_inverse(
        self,
        data: ndarray,
    ) -> ndarray:
        """Compute the Jacobian of the ``pipeline.inverse_transform()``.

        Args:
            data: The data where the Jacobian is to be computed.

        Returns:
            The Jacobian matrix.
        """
        jacobian = eye(data.shape[-1])
        for transformer in self.transformers[::-1]:
            jacobian = matmul(transformer.compute_jacobian_inverse(data), jacobian)
            data = transformer.inverse_transform(data)
        return jacobian
