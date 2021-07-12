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
#                         documentation
#        :author: Syver Doving Agdestein
#        :author: Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""This module contains the base class for machine learning algorithms.

Machine learning is the art of building models from data,
the latter being samples of properties of interest
that can sometimes be sorted by group, such as inputs, outputs, categories, ...

In the absence of such groups,
the data can be analyzed through a study of commonalities,
leading to plausible clusters.
This is referred to as clustering,
a branch of unsupervised learning
dedicated to the detection of patterns in unlabeled data.

.. seealso::

   :mod:`~gemseo.mlearning.core.unsupervised`,
   :mod:`~gemseo.mlearning.cluster.cluster`

When data can be separated into at least two categories by a human,
supervised learning can start with classification
whose purpose is to model the relations
between these categories and the properties of interest.
Once trained,
a classification model can predict the category
corresponding to new property values.

.. seealso::

   :mod:`~gemseo.mlearning.core.supervised`,
   :mod:`~gemseo.mlearning.classification.classification`

When the distinction between inputs and outputs can be made among the data properties,
another branch of supervised learning can be considered: regression modeling.
Once trained,
a regression model can predict the outputs corresponding to new inputs values.

.. seealso::

   :mod:`~gemseo.mlearning.core.supervised`,
   :mod:`~gemseo.mlearning.regression.regression`

The quality of a machine learning algorithm can be measured
using a :class:`.MLQualityMeasure`
either with respect to the learning dataset
or to a test dataset or using resampling methods,
such as K-folds or leave-one-out cross-validation techniques.
The challenge is to avoid over-learning the learning data
leading to a loss of generality.
We often want to build models that are not too dataset-dependent.
For that,
we want to maximize both a learning quality and a generalization quality.
In unsupervised learning,
a quality measure can represent the robustness of clusters definition
while in supervised learning, a quality measure can be interpreted as an error,
whether it is a misclassification in the case of the classification algorithms
or a prediction one in the case of the regression algorithms.
This quality can often be improved
by building machine learning models from standardized data
in such a way that the data properties have the same order of magnitude.


.. seealso::

   :mod:`~gemseo.mlearning.qual_measure.quality_measure`,
   :mod:`~gemseo.mlearning.transform.transformer`

Lastly,
a machine learning algorithm often depends on hyperparameters
to be carefully tuned in order to maximize the generalization power of the model.

.. seealso::

   :mod:`~gemseo.mlearning.core.calibration`
   :mod:`~gemseo.mlearning.core.selection`
"""
from __future__ import division, unicode_literals

import logging
import pickle
import re
from os import makedirs
from os.path import exists, join
from typing import Any, Dict, List, Optional, Union

import six
from custom_inherit import DocInheritMeta
from numpy import ndarray

from gemseo.core.dataset import Dataset
from gemseo.mlearning.transform.transformer import Transformer
from gemseo.utils.string_tools import MultiLineString, pretty_repr

LOGGER = logging.getLogger(__name__)

TransformerType = Dict[str, Transformer]
SavedObjectType = Union[Dataset, TransformerType, str, bool]
DataType = Union[ndarray, Dict[str, ndarray]]
MLAlgoParameterType = Optional[Any]


@six.add_metaclass(
    DocInheritMeta(
        abstract_base_class=True,
        style="google_with_merge",
    )
)
class MLAlgo(object):
    """An abstract machine learning algorithm.

    Such a model is built from a training dataset,
    data transformation options and parameters. This abstract class defines the
    :meth:`.MLAlgo.learn`, :meth:`.MLAlgo.save` methods and the boolean
    property, :attr:`!MLAlgo.is_trained`. It also offers a string
    representation for end users.
    Derived classes shall overload the :meth:`.MLAlgo.learn`,
    :meth:`!MLAlgo._save_algo` and :meth:`!MLAlgo._load_algo` methods.

    Attributes:
        learning_set (Dataset): The learning dataset.
        parameters (Dict[str,MLAlgoParameterType]): The parameters
            of the machine learning algorithm.
        transformer (Dict[str,Transformer]): The strategies to transform the variables.
            The values are instances of :class:`.Transformer`
            while the keys are the names of
            either the variables
            or the groups of variables,
            e.g. "inputs" or "outputs" in the case of the regression algorithms.
            If a group is specified,
            the :class:`.Transformer` will be applied
            to all the variables of this group.
            If None, do not transform the variables.
        algo (Any): The interfaced machine learning algorithm.
    """

    LIBRARY = None
    ABBR = "MLAlgo"
    FILENAME = "ml_algo.pkl"

    def __init__(
        self,
        data,  # type:Dataset
        transformer=None,  # type: Optional[TransformerType]
        **parameters  # type: MLAlgoParameterType
    ):  # type: (...) -> None
        """
        Args:
            data: The learning dataset.
            transformer: The strategies to transform the variables.
                The values are instances of :class:`.Transformer`
                while the keys are the names of
                either the variables
                or the groups of variables,
                e.g. "inputs" or "outputs" in the case of the regression algorithms.
                If a group is specified,
                the :class:`.Transformer` will be applied
                to all the variables of this group.
                If None, do not transform the variables.
            **parameters: The parameters of the machine learning algorithm.
        """
        self.learning_set = data
        self.parameters = parameters
        self.transformer = {}
        if transformer:
            self.transformer = {
                group: transf.duplicate() for group, transf in transformer.items()
            }

        self.algo = None
        self._trained = False

    class DataFormatters(object):
        """Decorators for the internal MLAlgo methods."""

    @property
    def is_trained(self):  # type: (...) -> bool
        """Return whether the algorithm is trained."""
        return self._trained

    def learn(
        self,
        samples=None,  # type: List[int]
    ):  # type: (...) -> None
        """Train the machine learning algorithm from the learning dataset.

        Args:
            samples: The indices of the learning samples.
                If None, use the whole learning dataset.
        """
        raise NotImplementedError

    def __str__(self):  # type: (...) -> str
        msg = MultiLineString()
        msg.add("{}({})", self.__class__.__name__, pretty_repr(self.parameters))
        msg.indent()
        if self.LIBRARY is not None:
            msg.add("based on the {} library", self.LIBRARY)
        msg.add("built from {} learning samples", self.learning_set.length)
        return str(msg)

    def save(
        self,
        directory=None,  # type: Optional[str]
        path=".",  # type: str
        save_learning_set=False,  # type: bool
    ):  # type: (...) -> str
        """Save the machine learning algorithm.

        Args:
            directory: The name of the directory to save the algorithm.
            path: The path to parent directory where to create the directory.
            save_learning_set: If False, do not save the learning set
                to lighten the saved files.

        Returns:
            The path to the directory where the algorithm is saved.
        """
        if not save_learning_set:
            self.learning_set.data = {}
            self.learning_set.length = 0
        splitted_class_name = re.findall("[A-Z][a-z]*", self.__class__.__name__)
        splitted_class_name = [word.lower() for word in splitted_class_name]
        algo_name = "_".join(splitted_class_name)
        algo_name += "_" + self.learning_set.name
        directory = directory or algo_name
        directory = join(path, directory)
        if not exists(directory):
            makedirs(directory)

        filename = join(directory, self.FILENAME)
        objects = self._get_objects_to_save()
        with open(filename, "wb") as handle:
            pickle.dump(objects, handle)

        self._save_algo(directory)

        return directory

    def _save_algo(
        self,
        directory,  # type: str
    ):  # type: (...) -> None
        """Save the interfaced machine learning algorithm.

        Args:
            directory: The path to the directory
                where to save the interfaced machine learning algorithm.
        """
        filename = join(directory, "algo.pkl")
        with open(filename, "wb") as handle:
            pickle.dump(self.algo, handle)

    def load_algo(
        self,
        directory,  # type: str
    ):  # type: (...) -> None
        """Load a machine learning algorithm from a directory.

        Args:
            directory: The path to the directory
                where the machine learning algorithm is saved.
        """

        filename = join(directory, "algo.pkl")
        with open(filename, "rb") as handle:
            algo = pickle.load(handle)
        self.algo = algo

    def _get_objects_to_save(self):  # type: (...) -> Dict[str,SavedObjectType]
        """Return the objects to save.

        Returns:
            The objects to save.
        """
        objects = {
            "data": self.learning_set,
            "transformer": self.transformer,
            "parameters": self.parameters,
            "algo_name": self.__class__.__name__,
            "_trained": self._trained,
        }
        return objects
