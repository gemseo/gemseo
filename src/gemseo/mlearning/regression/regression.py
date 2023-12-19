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
#        :author: Matthias De Lozzo
#        :author: Syver Doving Agdestein
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
r"""This module contains the baseclass for regression algorithms.

The :mod:`~gemseo.mlearning.regression.regression` module
implements regression algorithms,
where the goal is to find relationships
between continuous input and output variables.
After being fitted to a learning set,
the regression algorithms can predict output values of new input data.

A regression algorithm consists of identifying a function
:math:`f: \\mathbb{R}^{n_{\\textrm{inputs}}} \\to
\\mathbb{R}^{n_{\\textrm{outputs}}}`.
Given an input point
:math:`x \\in \\mathbb{R}^{n_{\\textrm{inputs}}}`,
the predict method of the regression algorithm will return
the output point :math:`y = f(x) \\in \\mathbb{R}^{n_{\\textrm{outputs}}}`.
See :mod:`~gemseo.mlearning.core.supervised` for more information.

Wherever possible,
the regression algorithms should also be able
to compute the Jacobian matrix of the function it has learned to represent.
Thus,
given an input point :math:`x \\in \\mathbb{R}^{n_{\\textrm{inputs}}}`,
the Jacobian prediction method of the regression algorithm should return the matrix

.. math::

    J_f(x) = \\frac{\\partial f}{\\partial x} =
    \\begin{pmatrix}
    \\frac{\\partial f_1}{\\partial x_1} & \\cdots & \\frac{\\partial f_1}
        {\\partial x_{n_{\\textrm{inputs}}}}\\\\
    \\vdots & \\ddots & \\vdots\\\\
    \\frac{\\partial f_{n_{\\textrm{outputs}}}}{\\partial x_1} & \\cdots &
        \\frac{\\partial f_{n_{\\textrm{outputs}}}}
        {\\partial x_{n_{\\textrm{inputs}}}}
    \\end{pmatrix}
    \\in \\mathbb{R}^{n_{\\textrm{outputs}}\\times n_{\\textrm{inputs}}}.

This concept is implemented through the :class:`.MLRegressionAlgo` class
which inherits from the :class:`.MLSupervisedAlgo` class.
"""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING
from typing import NoReturn

from gemseo.datasets.io_dataset import IODataset
from gemseo.mlearning.core.supervised import MLSupervisedAlgo
from gemseo.mlearning.data_formatters.regression_data_formatters import (
    RegressionDataFormatters,
)
from gemseo.mlearning.transformers.scaler.min_max_scaler import MinMaxScaler

if TYPE_CHECKING:
    from collections.abc import Iterable

    from numpy import ndarray

    from gemseo.mlearning.core.ml_algo import DataType
    from gemseo.mlearning.core.ml_algo import DefaultTransformerType
    from gemseo.mlearning.core.ml_algo import MLAlgoParameterType
    from gemseo.mlearning.core.ml_algo import TransformerType


class MLRegressionAlgo(MLSupervisedAlgo):
    """Machine Learning Regression Model Algorithm.

    Inheriting classes shall implement the :meth:`!MLSupervisedAlgo._fit` and
    :meth:`!MLSupervisedAlgo._predict` methods, and
    :meth:`!MLRegressionAlgo._predict_jacobian` method if possible.
    """

    DEFAULT_TRANSFORMER: DefaultTransformerType = MappingProxyType({
        IODataset.INPUT_GROUP: MinMaxScaler(),
        IODataset.OUTPUT_GROUP: MinMaxScaler(),
    })

    DataFormatters = RegressionDataFormatters

    def __init__(  # noqa: D107
        self,
        data: IODataset,
        transformer: TransformerType = MLSupervisedAlgo.IDENTITY,
        input_names: Iterable[str] | None = None,
        output_names: Iterable[str] | None = None,
        **parameters: MLAlgoParameterType,
    ) -> None:
        super().__init__(
            data,
            transformer=transformer,
            input_names=input_names,
            output_names=output_names,
            **parameters,
        )

    def predict_raw(
        self,
        input_data: ndarray,
    ) -> ndarray:
        """Predict output data from input data.

        Args:
            input_data: The input data with shape (n_samples, n_inputs).

        Returns:
            The predicted output data with shape (n_samples, n_outputs).
        """
        return self._predict(input_data)

    @DataFormatters.format_dict_jacobian
    @DataFormatters.format_samples
    @DataFormatters.transform_jacobian
    def predict_jacobian(
        self,
        input_data: DataType,
    ) -> NoReturn:
        """Predict the Jacobians of the regression model at input_data.

        The user can specify these input data either as a NumPy array,
        e.g. ``array([1., 2., 3.])``
        or as a dictionary,
        e.g.  ``{'a': array([1.]), 'b': array([2., 3.])}``.

        If the NumPy arrays are of dimension 2,
        their i-th rows represent the input data of the i-th sample;
        while if the NumPy arrays are of dimension 1,
        there is a single sample.

        The type of the output data and the dimension of the output arrays
        will be consistent
        with the type of the input data and the size of the input arrays.

        Args:
            input_data: The input data.

        Returns:
            The predicted Jacobian data.
        """
        return self._predict_jacobian(input_data)

    def _predict_jacobian(
        self,
        input_data: ndarray,
    ) -> NoReturn:
        """Predict the Jacobian matrices of the regression model at input_data.

        Args:
            input_data: The input data with shape (n_samples, n_inputs).

        Returns:
            The predicted Jacobian data with shape (n_samples, n_outputs, n_inputs).

        Raises:
            NotImplementedError: When the method is called.
        """
        raise NotImplementedError(
            f"Derivatives are not available for {self.__class__.__name__}."
        )
