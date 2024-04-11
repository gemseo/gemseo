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
#    INITIAL AUTHORS - initial API and implementation and/or
#                  initial documentation
#        :author:  Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""A factory of scalable models."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

from gemseo.core.base_factory import BaseFactory
from gemseo.problems.mdo.scalable.data_driven.model import ScalableModel

if TYPE_CHECKING:
    from collections.abc import Mapping

    from gemseo.datasets.io_dataset import IODataset


class ScalableModelFactory(BaseFactory):
    """A factory of scalable models."""

    _CLASS = ScalableModel
    _MODULE_NAMES = ("gemseo.problems.mdo.scalable",)

    def create(
        self,
        model_name: str,
        data: IODataset,
        sizes: Mapping[str, int] | None = None,
        **parameters: Any,
    ) -> ScalableModel:
        """Create a scalable model.

        Args:
            model_name: The name of the scalable model (its class name).
            data: The input-output dataset.
            sizes: The sizes of the inputs and outputs.
                If ``None``, use the original sizes.
            **parameters: model parameters

        Returns:
            The scalable model.
        """
        return super().create(model_name, data=data, sizes=sizes, **parameters)

    @property
    def scalable_models(self) -> list[str]:
        """The available scalable models."""
        return self.class_names
