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
"""This module contains the baseclass for regression algorithms.

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

import collections
from types import MappingProxyType
from typing import Callable
from typing import Iterable
from typing import NoReturn

from numpy import eye
from numpy import matmul
from numpy import ndarray

from gemseo.core.dataset import Dataset
from gemseo.mlearning.core.ml_algo import DataType
from gemseo.mlearning.core.ml_algo import DefaultTransformerType
from gemseo.mlearning.core.ml_algo import MLAlgoParameterType
from gemseo.mlearning.core.ml_algo import TransformerType
from gemseo.mlearning.core.supervised import MLSupervisedAlgo
from gemseo.mlearning.transform.scaler.min_max_scaler import MinMaxScaler
from gemseo.utils.data_conversion import concatenate_dict_of_arrays_to_array
from gemseo.utils.data_conversion import split_array_to_dict_of_arrays


class MLRegressionAlgo(MLSupervisedAlgo):
    """Machine Learning Regression Model Algorithm.

    Inheriting classes shall implement the :meth:`!MLSupervisedAlgo._fit` and
    :meth:`!MLSupervisedAlgo._predict` methods, and
    :meth:`!MLRegressionAlgo._predict_jacobian` method if possible.
    """

    DEFAULT_TRANSFORMER: DefaultTransformerType = MappingProxyType(
        {
            Dataset.INPUT_GROUP: MinMaxScaler(),
            Dataset.OUTPUT_GROUP: MinMaxScaler(),
        }
    )

    def __init__(
        self,
        data: Dataset,
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

    class DataFormatters(MLSupervisedAlgo.DataFormatters):
        """Machine learning regression model decorators."""

        @classmethod
        def format_dict_jacobian(
            cls,
            predict_jac: Callable[[ndarray], ndarray],
        ) -> Callable[[DataType], DataType]:
            """Wrap an array-based function to make it callable with a dictionary of
            NumPy arrays.

            Args:
                predict_jac: The function to be called;
                    it takes a NumPy array in input and returns a NumPy array.

            Returns:
                The wrapped 'predict_jac' function, callable with
                either a NumPy data array
                or a dictionary of numpy data arrays indexed by variables names.
                The return value will have the same type as the input data.
            """

            def wrapper(self, input_data, *args, **kwargs):
                """Evaluate 'predict_jac' with either array or dictionary-based data.

                Firstly,
                the pre-processing stage converts the input data to a NumPy data array,
                if these data are expressed as a dictionary of NumPy data arrays.

                Then,
                the processing evaluates the function 'predict_jac'
                from this NumPy input data array.

                Lastly,
                the post-processing transforms the output data
                to a dictionary of output NumPy data array
                if the input data were passed as a dictionary of NumPy data arrays.

                Args:
                    input_data: The input data.
                    *args: The positional arguments of the function 'predict_jac'.
                    **kwargs: The keyword arguments of the function 'predict_jac'.

                Returns:
                    The output data with the same type as the input one.
                """
                as_dict = isinstance(input_data, collections.abc.Mapping)
                if as_dict:
                    input_data = concatenate_dict_of_arrays_to_array(
                        input_data, self.input_names
                    )
                single_sample = len(input_data.shape) == 1
                jacobians = predict_jac(self, input_data, *args, **kwargs)
                if as_dict:
                    varsizes = self.learning_set.sizes
                    if single_sample:
                        jacobians = split_array_to_dict_of_arrays(
                            jacobians, varsizes, self.output_names, self.input_names
                        )
                    else:
                        jacobians = split_array_to_dict_of_arrays(
                            jacobians, varsizes, self.output_names, self.input_names
                        )
                return jacobians

            return wrapper

        @classmethod
        def transform_jacobian(
            cls,
            predict_jac: Callable[[ndarray], ndarray],
        ) -> Callable[[ndarray], ndarray]:
            """Apply transformation to inputs and inverse transformation to outputs.

            Args:
                predict_jac: The function of interest to be called.

            Returns:
                A function evaluating the function 'predict_jac',
                after transforming its input data
                and/or before transforming its output data.
            """

            def wrapper(self, input_data, *args, **kwargs):
                """Evaluate 'predict_jac' after or before data transformation.

                Firstly,
                the pre-processing stage transforms the input data if required.

                Then,
                the processing evaluates the function 'predict_jac'.

                Lastly,
                the post-processing stage transforms the output data if required.

                Args:
                    input_data: The input data.
                    *args: The positional arguments of the function.
                    **kwargs: The keyword arguments of the function.

                Returns:
                    Either the raw output data of 'predict_jac'
                    or a transformed version according to the requirements.

                Raises:
                    NotImplementedError: When the transformer is applied to a variable
                        rather than to a group of variables.
                """
                if (
                    self._input_variables_to_transform
                    or self._output_variables_to_transform
                ):
                    # TODO: implement this case
                    raise NotImplementedError(
                        "The Jacobian of regression models cannot be computed "
                        "when the transformed quantities are variables; "
                        "please transform the whole group 'inputs' or 'outputs' "
                        "or do not use data transformation."
                    )

                inputs = self.learning_set.INPUT_GROUP
                if inputs in self.transformer:
                    jac = self.transformer[inputs].compute_jacobian(input_data)
                    input_data = self.transformer[inputs].transform(input_data)
                else:
                    jac = eye(input_data.shape[1])

                jac = matmul(predict_jac(self, input_data, *args, **kwargs), jac)
                output_data = self.predict_raw(input_data)

                outputs = self.learning_set.OUTPUT_GROUP
                if outputs in self.transformer:
                    jac = matmul(
                        self.transformer[outputs].compute_jacobian_inverse(output_data),
                        jac,
                    )
                return jac

            return wrapper

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
        e.g. :code:`array([1., 2., 3.])`
        or as a dictionary,
        e.g.  :code:`{'a': array([1.]), 'b': array([2., 3.])}`.

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
        name = self.__class__.__name__
        raise NotImplementedError(f"Derivatives are not available for {name}.")
