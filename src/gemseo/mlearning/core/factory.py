# -*- coding: utf-8 -*-
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
This factory also provides a list of available machine learning algorithms and allows to
test if a machine learning algorithm is available.
"""
from __future__ import division, unicode_literals

import logging
import pickle
from typing import List, Optional, Union

from gemseo.core.dataset import Dataset
from gemseo.core.factory import Factory
from gemseo.mlearning.core.ml_algo import MLAlgo, MLAlgoParameterType, TransformerType
from gemseo.utils.py23_compat import Path

LOGGER = logging.getLogger(__name__)


class MLAlgoFactory(object):
    """This factory instantiates a :class:`.MLAlgo` from its class name.

    The class can be either internal to |g| or external. In this second case, it can be
    either implemented in a module referenced in the "GEMSEO_PATH" or in a module The
    class can be either internal to |g| or external. In the second case, it can be
    either implemented in a module referenced in the GEMSEO_PATH environment variable or
    in a module starting with "gemseo_" and referenced in the PYTHONPATH environment
    variable.
    """

    def __init__(self):  # type: (...) -> None
        self.factory = Factory(MLAlgo, ("gemseo.mlearning",))

    def create(
        self,
        ml_algo,  # type: str
        **options  # type: Optional[Union[Dataset,TransformerType,MLAlgoParameterType]]
    ):  # type: (...) -> MLAlgo
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
    def models(self):  # type: (...) -> List[str]
        """The available machine learning algorithms."""
        return self.factory.classes

    def is_available(
        self,
        ml_algo,  # type: str
    ):  # type: (...) -> bool
        """Check the availability of a machine learning algorithm.

        Args:
            ml_algo: The name of a machine learning algorithm (its class name).

        Returns:
            Whether the machine learning algorithm is available.
        """
        return self.factory.is_available(ml_algo)

    def load(
        self,
        directory,  # type:Union[str,Path]
    ):  # type: (...) -> MLAlgo
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
        model = self.factory.create(
            objects.pop("algo_name"),
            data=objects.pop("data"),
            **objects.pop("parameters")
        )
        for key, value in objects.items():
            setattr(model, key, value)
        model.load_algo(directory)
        return model
