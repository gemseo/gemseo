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
#        :author: Matthias De Lozzo
#        :author: Syver Doving Agdestein
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""
Regression model
================

The :mod:`~gemseo.mlearning.regression.regression` module
implements regression algorithms, where the goal is to find relationships
between continuous input and output variables. After being fitted to a learning
set, Regression algorithms can predict output values of new input data.

A regression algorithm consists of identifying a function
:math:`f: \\mathbb{R}^{n_{\\textrm{inputs}}} \\to
\\mathbb{R}^{n_{\\textrm{outputs}}}`. Given an input point
:math:`x \\in \\mathbb{R}^{n_{\\textrm{inputs}}}`, the predict method of the
regression algorithm will return the output point
:math:`y = f(x) \\in \\mathbb{R}^{n_{\\textrm{outputs}}}`. See
:mod:`~gemseo.mlearning.core.supervised` for more information.

Wherever possible, regression algorithms should also be able to compute the
Jacobian matrix of the function it has learned to represent. Given an input
point :math:`x \\in \\mathbb{R}^{n_{\\textrm{inputs}}}`, the Jacobian predict
method of the regression algorithm should thus return the matrix

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

This concept is implemented through
the :class:`.MLRegressionAlgo` class which
inherits from the :class:`.MLSupervisedAlgo` class.
"""
from __future__ import absolute_import, division, unicode_literals

from future import standard_library
from numpy import eye, matmul

from gemseo.mlearning.core.supervised import MLSupervisedAlgo
from gemseo.utils.data_conversion import DataConversion

standard_library.install_aliases()


class MLRegressionAlgo(MLSupervisedAlgo):
    """Machine Learning Regression Model Algorithm.

    Inheriting classes should implement the :meth:`!MLSupervisedAlgo._fit` and
    :meth:`!MLSupervisedAlgo._predict` methods, and
    :meth:`!MLRegressionAlgo._predict_jacobian` method if possible.
    """

    class DataFormatters(MLSupervisedAlgo.DataFormatters):
        """ Machine learning regression model decorators. """

        @classmethod
        def format_dict_jacobian(cls, predict):
            """If input_data is passed as a dictionary, then convert it to
            ndarray, and convert output_data to dictionary. Else, do nothing.

            :param predict: Method whose input_data and output_data are to be
                formatted.
            """

            def wrapper(self, input_data, *args, **kwargs):
                as_dict = isinstance(input_data, dict)
                if as_dict:
                    input_data = DataConversion.dict_to_array(
                        input_data, self.input_names
                    )
                single_sample = len(input_data.shape) == 1
                jacobians = predict(self, input_data, *args, **kwargs)
                if as_dict:
                    varsizes = self.learning_set.sizes
                    if single_sample:
                        jacobians = DataConversion.jac_2dmat_to_dict(
                            jacobians, self.output_names, self.input_names, varsizes
                        )
                    else:
                        jacobians = DataConversion.jac_3dmat_to_dict(
                            jacobians, self.output_names, self.input_names, varsizes
                        )
                return jacobians

            return wrapper

        @classmethod
        def transform_jacobian(cls, predict_jac):
            """Apply transform to inputs, and inverse transform to outputs.

            :param predict: Method whose input_data and output_data are to be
                formatted.
            """

            def wrapper(self, input_data, *args, **kwargs):
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

    def predict_raw(self, input_data):
        """Predict output data from input data, assuming both are 2D.

        :param ndarray input_data: input data (n_samples, n_inputs).
        :return: output data (n_samples, n_outputs).
        :rtype: ndarray(int)
        """
        return self._predict(input_data)

    @DataFormatters.format_dict_jacobian
    @DataFormatters.format_samples
    @DataFormatters.transform_jacobian
    def predict_jacobian(self, input_data):
        """Predict Jacobian of the regression model of input_data.

        :param input_data: 1D input data.
        :type input_data: dict(ndarray) or ndarray
        :return: Jacobian for given input data.
        :rtype: dict(dict(ndarray)) or ndarray
        """
        return self._predict_jacobian(input_data)

    def _predict_jacobian(self, input_data):
        """Predict Jacobian of the regression model for the given input data.

        :param ndarray input_data: input_data (2D).
        :return: Jacobian matrices (3D, one for each sample).
        :rtype: ndarray
        """
        raise NotImplementedError
