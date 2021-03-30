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
#        :author: Francois Gallard, Matthias De Lozzo, Syver Doving Agdestein
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""
K-nearest neighbors classification model
========================================

The k-nearest neighbor classification algorithm is an approach to predict the
output class of a new input point by selecting the majority class
among the k nearest neighbors in a training set through voting. The algorithm
may also predict the probabilties of belonging to each class by counting the
number of occurences of the class withing the k nearest neighbors.

Let :math:`(x_i)_{i=1,\\cdots,n_{\\text{samples}}}\\in
\\mathbb{R}^{n_{\\text{samples}}\\times n_{\\text{inputs}}}` and
:math:`(y_i)_{i=1,\\cdots,n_{\\text{samples}}}\\in
\\{1,\\cdots,n_{\\text{classes}}\\}^{n_{\\text{samples}}}` denote the input and
output training data respectively.

The procedure for predicting the class of a new input point :math:`x\\in
\\mathbb{R}^{n_{\\text{inputs}}}` is the following:

Let :math:`i_1(x), \\cdots, i_{n_{\\text{samples}}}(x)` be the indices
of the input training points sorted by distance to the prediction point
:math:`x`, i.e.

.. math::

    \\|x-x_{i_1(x)}\\| \\leq \\cdots \\leq
    \\|x-x_{i_{n_{\\text{samples}}}(x)}\\|.

The ordered indices may be formally determined through the inductive formula

.. math::

    i_p(x) = \\underset{i\\in I_p(x)}{\\operatorname{argmin}}\\|x-x_i\\|,\\quad
    p=1,\\cdots,n_{\\text{samples}}

where

.. math::

    I_1(x) = \\{1,\\cdots,n_{\\text{samples}}\\}\\\\
    I_{p+1} = I_p(x)\\setminus \\{i_p(x)\\},\\quad
    p=1,\\cdots,n_{\\text{samples}}-1,

that is

.. math::

    I_p(x) = \\{1,\\cdots,n_{\\text{samples}}\\}\\setminus
    \\{i_1(x),\\cdots,i_{p-1}(x)\\}.

Then, by denoting :math:`\\operatorname{mode}(\\cdot)` the mode operator, i.e.
the operator that extracts the element with the highest occurence,
we may define the prediction operator as the mode of the set of output classes
associated to the :math:`k` first indices (classes of the :math:`k`-nearest
neighbors of :math:`x`):

.. math::

    f(x) = \\operatorname{mode}(y_{i_1(x)}, \\cdots, y_{i_k(x)})


This concept is implemented through the :class:`.KNNClassifier` class which
inherits from the :class:`.MLClassificationAlgo` class.

Dependence
----------
The classifier relies on the KNeighborsClassifier class
of the `scikit-learn library <https://scikit-learn.org/stable/modules/
generated/sklearn.neighbors.KNeighborsClassifier.html>`_.
"""
from __future__ import absolute_import, division, unicode_literals

from future import standard_library
from numpy import stack
from sklearn.neighbors import KNeighborsClassifier as SKLKNN

from gemseo.mlearning.classification.classification import MLClassificationAlgo

standard_library.install_aliases()


from gemseo import LOGGER


class KNNClassifier(MLClassificationAlgo):
    """ K nearest neighbors classification algorithm. """

    LIBRARY = "scikit-learn"
    ABBR = "KNN"

    def __init__(
        self,
        data,
        transformer=None,
        input_names=None,
        output_names=None,
        n_neighbors=5,
        **parameters
    ):
        """Constructor.

        :param data: learning dataset.
        :type data: Dataset
        :param transformer: transformation strategy for data groups.
            If None, do not transform data. Default: None.
        :type transformer: dict(str)
        :param input_names: names of the input variables.
        :type input_names: list(str)
        :param output_names: names of the output variables.
        :type output_names: list(str)
        :param n_neighbors: number of neighbors.
        :type n_neighbords: int
        :param parameters: other keyword arguments for sklearn KNN.
        """
        super(KNNClassifier, self).__init__(
            data,
            transformer=transformer,
            input_names=input_names,
            output_names=output_names,
            n_neighbors=n_neighbors,
            **parameters
        )
        self.algo = SKLKNN(n_neighbors, **parameters)

    def _fit(self, input_data, output_data):
        """Fit the classification model.

        :param ndarray input_data: input data (2D).
        :param ndarray(int) output_data: output data.
        """
        if output_data.shape[1] == 1:
            output_data = output_data.ravel()
        self.algo.fit(input_data, output_data)

    def _predict(self, input_data):
        """Predict output data from input data.

        :param ndarray input_data: input data (n_samples, n_inputs).
        :return: output data (n_samples, n_outputs).
        :rtype: ndarray(int)
        """
        output_data = self.algo.predict(input_data).astype(int)
        if len(output_data.shape) == 1:
            output_data = output_data[:, None]
        return output_data

    def _predict_proba_soft(self, input_data):
        """Predict probability of belonging to each class.

        :param ndarray input_data: input data (n_samples, n_inputs).
        :return: probabilities of belonging to each class
            (n_samples, n_outputs, n_classes). For a given sample and output
            variable, the sum of probabilities is one.
        :rtype: ndarray
        """
        probas = self.algo.predict_proba(input_data)
        if len(probas[0].shape) == 1:
            probas = probas[..., None]
        else:
            probas = stack(probas, axis=-1)
        return probas
