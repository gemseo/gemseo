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
"""The factory to create the machine learning algorithms.

This module contains a factory to instantiate a :class:`.MLAlgo` from its class name.
This factory also provides a list of available machine learning algorithms and allows
testing if a machine learning algorithm is available.
"""
from __future__ import annotations

import pickle
from pathlib import Path

from gemseo.core.dataset import Dataset
from gemseo.core.factory import Factory
from gemseo.mlearning.core.ml_algo import MLAlgo
from gemseo.mlearning.core.ml_algo import MLAlgoParameterType
from gemseo.mlearning.core.ml_algo import TransformerType
from gemseo.utils.python_compatibility import Final


class MLAlgoFactory:
    """This factory instantiates a :class:`.MLAlgo` from its class name.

    The class can be either internal or external. In this second case, it can be either
    implemented in a module referenced in the ``GEMSEO_PATH`` or in a module The class
    can be either internal or external. In the second case, it can be either implemented
    in a module referenced in the ``GEMSEO_PATH`` environment variable or in a module
    starting with ``gemseo_`` and referenced in the ``PYTHONPATH`` environment variable.
    """

    # GEMSEO 4.0 renamed several algorithms with the format "{Prefix}Regressor".
    # This mapping allows to import algorithms using the old naming.
    __OLD_TO_NEW_NAMES: Final = {
        "GaussianProcessRegression": "GaussianProcessRegressor",
        "LinearRegression": "LinearRegressor",
        "MixtureOfExperts": "MOERegressor",
        "PCERegression": "PCERegressor",
        "PolynomialRegression": "PolynomialRegressor",
        "RBFRegression": "RBFRegressor",
    }

    def __init__(self) -> None:
        self.factory = Factory(MLAlgo, ("gemseo.mlearning",))

    def create(
        self,
        ml_algo: str,
        **options: Dataset | TransformerType | MLAlgoParameterType | None,
    ) -> MLAlgo:
        """Create an instance of a machine learning algorithm.

        Args:
            ml_algo: The name of a machine learning algorithm
                (its class name).
            **options: The options of the machine learning algorithm.

        Returns:
            The instance of the machine learning algorithm.
        """
        return self.factory.create(ml_algo, **options)

    @property
    def models(self) -> list[str]:
        """The available machine learning algorithms."""
        return self.factory.classes

    def is_available(
        self,
        ml_algo: str,
    ) -> bool:
        """Check the availability of a machine learning algorithm.

        Args:
            ml_algo: The name of a machine learning algorithm (its class name).

        Returns:
            Whether the machine learning algorithm is available.
        """
        return self.factory.is_available(ml_algo)

    def load(
        self,
        directory: str | Path,
    ) -> MLAlgo:
        """Load an instance of machine learning algorithm from the disk.

        Args:
            directory: The name of the directory
                containing an instance of a machine learning algorithm.

        Returns:
            The instance of the machine learning algorithm.
        """
        directory = Path(directory)
        with (directory / MLAlgo.FILENAME).open("rb") as handle:
            objects = pickle.load(handle)
        algo_name = objects.pop("algo_name")
        model = self.factory.create(
            self.__OLD_TO_NEW_NAMES.get(algo_name, algo_name),
            data=objects.pop("data"),
            **objects.pop("parameters"),
        )
        for key, value in objects.items():
            setattr(model, key, value)
        model.load_algo(directory)
        return model
