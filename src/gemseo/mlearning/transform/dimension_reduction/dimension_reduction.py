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
#        :author: Matthias De Lozzo, Syver Doving Agdestein
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Dimension reduction as a generic transformer.

The :class:`.DimensionReduction` class implements the concept of dimension reduction.

.. seealso::

   :mod:`~gemseo.mlearning.transform.dimension_reduction.pca`
"""
from __future__ import annotations

from gemseo.mlearning.transform.transformer import Transformer


class DimensionReduction(Transformer):
    """Dimension reduction."""

    def __init__(
        self,
        name: str = "DimensionReduction",
        n_components: int | None = None,
        **parameters: bool | int | float | str | None,
    ) -> None:
        """
        Args:
            name: A name for this transformer.
            n_components: The number of components of the latent space.
                If ``None``,
                use the maximum number allowed by the technique,
                typically ``min(n_samples, n_features)``.
            **parameters: The parameters of the transformer.
        """
        super().__init__(name, n_components=n_components, **parameters)

    @property
    def n_components(self) -> int:
        """The number of components."""
        return self.parameters["n_components"]
