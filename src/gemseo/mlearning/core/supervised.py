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
Supervised machine learning algorithm
=====================================

Supervised machine learning is a task of learning relationships
between input and output variables based on an input-output dataset. One
usually distinguishes between to types of supervised machine learning
algorithms, based on the nature of the outputs. For a continuous output
variable, a regression is performed, while for a discrete output variable,
a classification is performed.

Given a set of input variables
:math:`x \\in \\mathbb{R}^{n_{\\text{samples}}\\times n_{\\text{inputs}}}` and
a set of output variables
:math:`y\\in \\mathbb{K}^{n_{\\text{samples}}\\times n_{\\text{outputs}}}`,
where :math:`n_{\\text{inputs}}` is the dimension of the input variable,
:math:`n_{\\text{outputs}}` is the dimension of the output variable,
:math:`n_{\\text{samples}}` is the number of training samples and
:math:`\\mathbb{K}` is either :math:`\\mathbb{R}` or :math:`\\mathbb{N}` for
regression and classification tasks respectively, a supervised learning
algorithm seeks to find a function
:math:`f: \\mathbb{R}^{n_{\\text{inputs}}} \\to
\\mathbb{K}^{n_{\\text{outputs}}}` such that :math:`y=f(x)`.

In addition, we often want to impose some additional constraints on the
function :math:`f`, mainly to ensure that it has a generalization capacity
beyond the training data, i.e. it is able to correctly predict output values of
new input values. This is called regularization. Assuming :math:`f` is
parametrized by a set of parameters :math:`\\theta`, and denoting
:math:`f_\\theta` the parametrized function, one typically seeks to minimize
a function of the form

.. math::

    \\mu(y, f_\\theta(x)) + \\Omega(\\theta),

where :math:`\\mu` is a distance-like measure, typically a mean squared error
or a cross entropy in the case of a regression, or a probability to be
maximized in the case of a classification, and :math:`\\Omega` is a
regularization term that limits the parameters from overfitting, typically some
norm of its argument.

The :mod:`~gemseo.mlearning.core.supervised` module implements this concept
through the :class:`.MLSupervisedAlgo` class based on a :class:`.Dataset`.
"""
from __future__ import absolute_import, division, unicode_literals

from future import standard_library
from numpy import atleast_2d

from gemseo.core.dataset import Dataset
from gemseo.mlearning.core.ml_algo import MLAlgo
from gemseo.mlearning.transform.dimension_reduction.dimension_reduction import (
    DimensionReduction,
)
from gemseo.utils.data_conversion import DataConversion

standard_library.install_aliases()


class MLSupervisedAlgo(MLAlgo):
    """Supervised machine learning algorithm.

    Inheriting classes should overload the :meth:`!MLSupervisedAlgo._fit` and
    :meth:`!MLSupervisedAlgo._predict` methods.
    """

    ABBR = "MLSupervisedAlgo"

    def __init__(
        self, data, transformer=None, input_names=None, output_names=None, **parameters
    ):
        """Constructor.

        :param Dataset data: learning dataset.
        :param transformer: transformation strategy for data groups.
            If None, do not scale data. Default: None.
        :type transformer: dict(str)
        :param input_names: names of the input variables.
        :type input_names: list(str)
        :param output_names: names of the output variables.
        :type output_names: list(str)
        :param parameters: algorithm parameters.
        """
        super(MLSupervisedAlgo, self).__init__(
            data, transformer=transformer, **parameters
        )
        self.input_names = input_names or data.get_names(data.INPUT_GROUP)
        self.output_names = output_names or data.get_names(data.OUTPUT_GROUP)

    class DataFormatters(MLAlgo.DataFormatters):
        """Decorators for supervised algorithms. """

        @staticmethod
        def _array_to_dict(data_array, data_names, data_sizes):
            """Convert an array into a dict

            :param data_array: the array
            :param data_names: list of names (keys of the resulting dict)
            :param data_sizes: dict of (name, size)
            :returns: a dict
            :rtype: dict
            """
            current_position = 0
            array_dict = {}
            for name in data_names:
                array_dict[name] = data_array[
                    ..., current_position : current_position + data_sizes[name]
                ]
                current_position += data_sizes[name]
            return array_dict

        @classmethod
        def format_dict(cls, predict):
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
                output_data = predict(self, input_data, *args, **kwargs)
                if as_dict:
                    varsizes = self.learning_set.sizes
                    output_data = cls._array_to_dict(
                        output_data, self.output_names, varsizes
                    )
                return output_data

            return wrapper

        @classmethod
        def format_samples(cls, predict):
            """If input_data has shape (n_inputs,), reshape input_data to
            (1, n_inputs), and then reshape output data from (1, n_outputs)
            to (n_outputs,).
            If input_data has shape (n_samples, n_inputs), then do nothing.

            :param predict: Method whose input_data and output_data are to be
                formatted.
            """

            def wrapper(self, input_data, *args, **kwargs):
                """Format data before and after applying predictor. """
                single_sample = input_data.ndim == 1
                input_data = atleast_2d(input_data)
                output_data = predict(self, input_data, *args, **kwargs)
                if single_sample:
                    output_data = output_data[0]
                return output_data

            return wrapper

        @classmethod
        def format_transform(cls, transform_inputs=True, transform_outputs=True):
            """Apply transform to inputs, and inverse transform to outputs.

            :param bool format_inputs: Indicates whether to transform inputs.
            :param bool format_outputs: Indicates whether to transform outputs.
            """

            def format_transform_(predict):
                """Apply transform to inputs, and inverse transform to
                outputs.

                :param predict: Method whose input_data and output_data are to
                    be formatted.
                """

                def wrapper(self, input_data, *args, **kwargs):
                    """Wrapped version of predict function. """
                    inputs = self.learning_set.INPUT_GROUP
                    if transform_inputs and inputs in self.transformer:
                        input_data = self.transformer[inputs].transform(input_data)
                    output_data = predict(self, input_data, *args, **kwargs)
                    outputs = self.learning_set.OUTPUT_GROUP
                    if transform_outputs and outputs in self.transformer:
                        output_data = self.transformer[outputs].inverse_transform(
                            output_data
                        )
                    return output_data

                return wrapper

            return format_transform_

        @classmethod
        def format_input_output(cls, predict):
            """Format dict, samples and transform successively.

            :param predict: Method whose input_data and output_data are to be
                formatted.
            """

            @cls.format_dict
            @cls.format_samples
            @cls.format_transform()
            def wrapper(self, input_data, *args, **kwargs):
                return predict(self, input_data, *args, **kwargs)

            return wrapper

    def learn(self, samples=None):
        """Train machine learning algorithm on learning set, possibly filtered
        using the given parameters.

        :param list(int) samples: indices of training samples.
        """
        input_grp = self.learning_set.INPUT_GROUP
        output_grp = self.learning_set.OUTPUT_GROUP
        input_data = self.learning_set.get_data_by_names(self.input_names, False)
        output_data = self.learning_set.get_data_by_names(self.output_names, False)

        if samples is not None:
            input_data = input_data[samples]
            output_data = output_data[samples]

        if input_grp in self.transformer:
            input_data = self.transformer[input_grp].fit_transform(input_data)

        if output_grp in self.transformer:
            output_data = self.transformer[output_grp].fit_transform(output_data)

        self._fit(input_data, output_data)
        self._trained = True

    def _fit(self, input_data, output_data):
        """Fit input-output relationship from data learning.

        :param ndarray input_data: input data (2D).
        :param ndarray output_data: output data (2D).
        """
        raise NotImplementedError

    @DataFormatters.format_input_output
    def predict(self, input_data):
        """Predict output data from input data.

        :param input_data: input data (n_inputs,) or (n_samples, n_inputs).
        :type input_data: dict(ndarray) or ndarray
        :return: predicted output data (n_outputs,) or (n_samples, n_outputs).
        :rtype: dict(ndarray) or ndarray(int)
        """
        return self._predict(input_data)

    def _predict(self, input_data):
        """Predict output data from input data.

        :param ndarray input_data: input data (n_samples, n_inputs).
        :return: output data (n_samples, n_outputs).
        :rtype: ndarray(int)
        """
        raise NotImplementedError

    def _get_raw_shapes(self):
        """Get raw input and output shapes.

        The raw dimensions are the shapes of input and output variables after
        applying transformers.

        return: raw input shape, raw output shape
        rtype: tuple(int)
        """
        reduce_inputs = Dataset.INPUT_GROUP in self.transformer and isinstance(
            self.transformer[Dataset.INPUT_GROUP], DimensionReduction
        )
        if reduce_inputs:
            input_shape = self.transformer[Dataset.INPUT_GROUP].n_components
        else:
            input_shape = self.input_shape

        reduce_outputs = Dataset.OUTPUT_GROUP in self.transformer and isinstance(
            self.transformer[Dataset.OUTPUT_GROUP], DimensionReduction
        )
        if reduce_outputs:
            output_shape = self.transformer[Dataset.OUTPUT_GROUP].n_components
        else:
            output_shape = self.output_shape

        return input_shape, output_shape

    @property
    def input_shape(self):
        """ Dimension of input variables before applying transformers. """
        sizes = [self.learning_set.sizes[name] for name in self.input_names]
        return sum(sizes)

    @property
    def output_shape(self):
        """ Dimension of output variables before applying transformers. """
        sizes = [self.learning_set.sizes[name] for name in self.output_names]
        return sum(sizes)

    def _get_objects_to_save(self):
        """Get objects to save.

        :return: objects to save.
        :rtype: dict
        """
        objects = super(MLSupervisedAlgo, self)._get_objects_to_save()
        objects["input_names"] = self.input_names
        objects["output_names"] = self.output_names
        return objects
