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
"""The k-nearest neighbors for classification.

The k-nearest neighbor classification algorithm is an approach
to predict the output class of a new input point
by selecting the majority class among the k nearest neighbors in a training set
through voting.
The algorithm may also predict the probabilities of belonging to each class
by counting the number of occurrences of the class withing the k nearest neighbors.

Let :math:`(x_i)_{i=1,\\cdots,n_{\\text{samples}}}\\in
\\mathbb{R}^{n_{\\text{samples}}\\times n_{\\text{inputs}}}`
and :math:`(y_i)_{i=1,\\cdots,n_{\\text{samples}}}\\in
\\{1,\\cdots,n_{\\text{classes}}\\}^{n_{\\text{samples}}}`
denote the input and output training data respectively.

The procedure for predicting the class of a new input point :math:`x\\in
\\mathbb{R}^{n_{\\text{inputs}}}` is the following:

Let :math:`i_1(x), \\cdots, i_{n_{\\text{samples}}}(x)` be
the indices of the input training points
sorted by distance to the prediction point :math:`x`,
i.e.

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

Then,
by denoting :math:`\\operatorname{mode}(\\cdot)` the mode operator,
i.e. the operator that extracts the element with the highest occurrence,
we may define the prediction operator as the mode of the set of output classes
associated to the :math:`k` first indices
(classes of the :math:`k`-nearest neighbors of :math:`x`):

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
from __future__ import annotations

from typing import ClassVar
from typing import Iterable

from numpy import ndarray
from numpy import newaxis
from numpy import stack
from sklearn.neighbors import KNeighborsClassifier

from gemseo.core.dataset import Dataset
from gemseo.mlearning.classification.classification import MLClassificationAlgo
from gemseo.mlearning.core.ml_algo import TransformerType
from gemseo.utils.python_compatibility import Final


class KNNClassifier(MLClassificationAlgo):
    """The k-nearest neighbors classification algorithm."""

    SHORT_ALGO_NAME: ClassVar[str] = "KNN"
    LIBRARY: Final[str] = "scikit-learn"

    def __init__(
        self,
        data: Dataset,
        transformer: TransformerType = MLClassificationAlgo.IDENTITY,
        input_names: Iterable[str] | None = None,
        output_names: Iterable[str] | None = None,
        n_neighbors: int = 5,
        **parameters: int | str,
    ) -> None:
        """
        Args:
            n_neighbors: The number of neighbors.
        """
        super().__init__(
            data,
            transformer=transformer,
            input_names=input_names,
            output_names=output_names,
            n_neighbors=n_neighbors,
            **parameters,
        )
        self.algo = KNeighborsClassifier(n_neighbors, **parameters)

    def _fit(
        self,
        input_data: ndarray,
        output_data: ndarray,
    ) -> None:
        if output_data.shape[1] == 1:
            output_data = output_data.ravel()
        self.algo.fit(input_data, output_data)

    def _predict(
        self,
        input_data: ndarray,
    ) -> ndarray:
        output_data = self.algo.predict(input_data).astype(int)
        if len(output_data.shape) == 1:
            output_data = output_data[:, newaxis]
        return output_data

    def _predict_proba_soft(
        self,
        input_data: ndarray,
    ) -> ndarray:
        probas = self.algo.predict_proba(input_data)
        if len(probas[0].shape) == 1:
            probas = probas[..., None]
        else:
            probas = stack(probas, axis=-1)
        return probas
