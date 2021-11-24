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
"""The Support Vector Machine algorithm for classification.

This module implements the SVMClassifier class.
A support vector machine (SVM) passes the data through a kernel
in order to increase its dimension
and thereby make the classes linearly separable.

Dependence
----------
The classifier relies on the SVC class
of the `scikit-learn library <https://scikit-learn.org/stable/modules/
generated/sklearn.svm.SVC.html>`_.
"""
from __future__ import division, unicode_literals

import logging
from typing import Callable, Iterable, Optional, Union

from numpy import ndarray
from sklearn.svm import SVC

from gemseo.core.dataset import Dataset
from gemseo.mlearning.classification.classification import MLClassificationAlgo
from gemseo.mlearning.core.ml_algo import TransformerType

LOGGER = logging.getLogger(__name__)


class SVMClassifier(MLClassificationAlgo):
    """The Support Vector Machine algorithm for classification."""

    LIBRARY = "scikit-learn"
    ABBR = "SVM"

    def __init__(
        self,
        data,  # type: Dataset
        transformer=None,  # type: Optional[TransformerType]
        input_names=None,  # type: Optional[Iterable[str]]
        output_names=None,  # type: Optional[Iterable[str]]
        C=1.0,  # noqa: N803 # type: float
        kernel="rbf",  # type: Optional[str,Callable]
        probability=False,  # type: bool
        **parameters  # type: Optional[Union[int,float,bool,str]]
    ):  # type: (...) -> None
        # noqa: D205,D212,D415
        """
        Args:
            C: The inverse L2 regularization parameter.
                   Higher values give less regularization.
            kernel: The name of the kernel or a callable for the SVM.
                Examples: "linear", "poly", "rbf", "sigmoid", "precomputed"
                or a callable.
            probability: Whether to enable the probability estimates.
                The algorithm is faster if set to False.
        """
        super(SVMClassifier, self).__init__(
            data,
            transformer=transformer,
            input_names=input_names,
            output_names=output_names,
            C=C,
            kernel=kernel,
            probability=probability,
            **parameters
        )
        self.algo = SVC(C=C, kernel=kernel, probability=probability, **parameters)

    def _fit(
        self,
        input_data,  # type:ndarray
        output_data,  # type:ndarray
    ):  # type: (...) -> None
        self.algo.fit(input_data, output_data.ravel())

    def _predict(
        self,
        input_data,  # type:ndarray
    ):  # type: (...) -> ndarray
        return self.algo.predict(input_data)[:, None].astype(int)

    def _predict_proba_soft(
        self,
        input_data,  # type: ndarray
    ):  # type: (...)-> ndarray
        if not self.parameters["probability"]:
            raise NotImplementedError(
                "SVMClassifier soft probability prediction is only available if the "
                "parameter 'probability' is set to True."
            )
        return self.algo.predict_proba(input_data)[..., None]
