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
#                           documentation
#        :author: Matthias De Lozzo, Syver Doving Agdestein
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""A factory of machine learning algorithms."""

from __future__ import annotations

import pickle
from pathlib import Path

from gemseo.core.base_factory import BaseFactory
from gemseo.mlearning.core.algos.ml_algo import BaseMLAlgo


class MLAlgoFactory(BaseFactory):
    """A factory of machine learning algorithms."""

    _CLASS = BaseMLAlgo
    _MODULE_NAMES = ("gemseo.mlearning",)

    def load(
        self,
        directory: str | Path,
    ) -> BaseMLAlgo:
        """Load an instance of machine learning algorithm from the disk.

        Args:
            directory: The name of the directory
                containing an instance of a machine learning algorithm.

        Returns:
            The instance of the machine learning algorithm.
        """
        directory = Path(directory)
        with (directory / BaseMLAlgo.FILENAME).open("rb") as handle:
            objects = pickle.load(handle)
        algo_name = objects.pop("_algo_name")
        model = super().create(
            algo_name,
            data=objects.pop("data"),
            **objects.pop("parameters"),
        )
        for key, value in objects.items():
            setattr(model, key, value)
        model.load_algo(directory)
        return model
