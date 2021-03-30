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
"""
Machine learning algorithm baseclass
====================================

Machine learning is the art of building models from data,
the latter being samples of properties of interest
that can sometimes be sorted by group, such as inputs, outputs, categories, ...

In the absence of such groups, the data can be analyzed through a study of
commonalities, leading to plausible clusters. This is referred to as
clustering, a branch of unsupervised learning dedicated to the detection
of patterns in unlabeled data.

.. seealso::

   :mod:`~gemseo.mlearning.core.unsupervised`,
   :mod:`~gemseo.mlearning.cluster.cluster`

When data can be separated into at least two categories by a human, supervised
learning can start with classification whose purpose is to model the relation
between these categories and the properties of interest. Once trained,
a classification model can predict the category corresponding to new
property values.

.. seealso::

   :mod:`~gemseo.mlearning.core.supervised`,
   :mod:`~gemseo.mlearning.classification.classification`

When the distinction between inputs and outputs can be made among the data
properties, another branch of supervised learning can be called: regression
modeling. Once trained, a regression model can predict the outputs
corresponding to new inputs values.

.. seealso::

   :mod:`~gemseo.mlearning.core.supervised`,
   :mod:`~gemseo.mlearning.regression.regression`

The quality of a machine learning algorithm can be measured
using a :class:`.MLQualityMeasure` either with respect to the learning dataset
or to a test dataset or using resampling methods, such as K-folds or
leave-one-out cross-validation techniques. The challenge is to avoid
over-learning the learning data leading to a loss of generality.
We often want to build models that are not too dataset-dependent. For that,
we want to maximize both a learning quality and a generalization quality.
In unsupervised learning, a quality measure can represent the robustness of
clusters definition while in supervised learning, a quality measure can be
interpreted as an error, whether it is a misclassification in the case of the
classification algorithms or a prediction one in the case of the regression
algorithms. This quality can often be improved by building machine learning
models from standardized data in such a way that the data properties have the
same order of magnitude.


.. seealso::

   :mod:`~gemseo.mlearning.qual_measure.quality_measure`,
   :mod:`~gemseo.mlearning.transform.transformer`

Lastly, a machine learning algorithm often depends on hyperparameters to
carefully tune in order to maximize the generalization power of the model.

.. seealso::

   :mod:`~gemseo.mlearning.core.calibration`

"""
from __future__ import absolute_import, division, unicode_literals

import pickle
import re
from os import makedirs
from os.path import exists, join

from future import standard_library

standard_library.install_aliases()


class MLAlgo(object):
    """The :class:`.MLAlgo` abstract class implements the concept of machine
    learning algorithm. Such a model is built from a training dataset,
    data transformation options and parameters. This abstract class defines the
    :meth:`.MLAlgo.learn`, :meth:`.MLAlgo.save` methods and the boolean
    property, :attr:`!MLAlgo.is_trained`. It also offers a string
    representation for end user.
    Inheriting classes should overload the :meth:`.MLAlgo.learn`,
    :meth:`!MLAlgo._save_algo` and :meth:`!MLAlgo._load_algo` methods.
    """

    LIBRARY = None
    ABBR = "MLAlgo"
    FILENAME = "ml_algo.pkl"

    def __init__(self, data, transformer=None, **parameters):
        """Constructor.

        :param Dataset data: learning dataset
        :param transformer: transformation strategy for data groups. If None,
            do not transform data. The dictionary keys are the groups to
            transform. The dictionary items are the transformers, either
            referenced by their names as strings, or provided directly.
            Default: None.
        :type transformer: dict(str) or dict(Transformer)
        :param parameters: algorithm parameters
        """
        self.learning_set = data
        self.parameters = parameters
        self.transformer = transformer or {}
        self.algo = None
        self._trained = False

    class DataFormatters(object):
        """ Decorators for internal MLAlgo methods. """

    @property
    def is_trained(self):
        """Check if algorithm is trained.
        :return: bool
        """
        return self._trained

    def learn(self, samples=None):
        """Train machine learning algorithm on learning set, possibly filtered
        using the given parameters.
        :param list(int) samples: indices of training samples.
        """
        raise NotImplementedError

    def __str__(self):
        """ String representation for end user. """
        string = self.__class__.__name__ + "("
        parameters = [key + "=" + str(val) for key, val in self.parameters.items()]
        string += ", ".join(parameters)
        string += ")\n"
        if self.LIBRARY is not None:
            string += "| based on the " + self.LIBRARY + " library\n"
        string += "| built from " + str(self.learning_set.length)
        string += " learning samples"
        return string

    def save(self, directory=None, path=".", save_learning_set=False):
        """Save the machine learning algorithm.

        :param str directory: directory name
        :param str path: path name
        :return: location of saved file
        :rtype: str
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

    def _save_algo(self, directory):
        """Save external machine learning algorithm.

        :param str directory: algorithm directory.
        """
        filename = join(directory, "algo.pkl")
        with open(filename, "wb") as handle:
            pickle.dump(self.algo, handle)

    def load_algo(self, directory):
        """Load external machine learning algorithm.

        :param str directory: algorithm directory.
        """

        filename = join(directory, "algo.pkl")
        with open(filename, "rb") as handle:
            algo = pickle.load(handle)
        self.algo = algo

    def _get_objects_to_save(self):
        """Get objects to save.

        :return: objects to save.
        :rtype: dict
        """
        objects = {
            "data": self.learning_set,
            "transformer": self.transformer,
            "parameters": self.parameters,
            "algo_name": self.__class__.__name__,
            "_trained": self._trained,
        }
        return objects
