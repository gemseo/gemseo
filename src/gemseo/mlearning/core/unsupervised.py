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
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""
Unsupervised machine learning algorithm
=======================================

The :mod:`~gemseo.mlearning.core.unsupervised` module implements the concept of
unsupervised machine learning models, where the data has no notion of
input or output.

This concept is implemented through the :class:`.MLUnsupervisedAlgo` class,
which inherits from the :class:`.MLAlgo` class.
"""
from __future__ import absolute_import, division, unicode_literals

from future import standard_library
from numpy import hstack

from gemseo.mlearning.core.ml_algo import MLAlgo

standard_library.install_aliases()


class MLUnsupervisedAlgo(MLAlgo):
    """Unsupervised machine learning algorithm.

    Inheriting classes should overload the
    :meth:`!MLUnsupervisedAlgo._fit` method.
    """

    ABBR = "MLUnupervisedAlgo"

    def __init__(self, data, transformer=None, var_names=None, **parameters):
        """Constructor.

        :param Dataset data: learning dataset
        :param transformer: transformation strategy for data groups.
            If None, do not scale data. Default: None.
        :type transformer: dict(str)
        :param var_names: names of the variables to consider.
        :type var_names: list(str)
        :param parameters: algorithm parameters
        """
        super(MLUnsupervisedAlgo, self).__init__(
            data, transformer=transformer, var_names=var_names, **parameters
        )
        self.var_names = var_names or data.variables

    def learn(self, samples=None):
        """Train machine learning algorithm on learning set.

        :param list(str) names: learning variables. Default: None.
        :param list(int) samples: training samples (indices). Default: None.
        """
        if set(self.var_names) == set(self.learning_set.variables):
            data = []
            for group in self.learning_set.groups:
                sub_data = self.learning_set.get_data_by_group(group)
                if group in self.transformer:
                    sub_data = self.transformer[group].fit_transform(sub_data)
                data.append(sub_data)
            data = hstack(data)
        else:
            data = []
            for name in self.var_names:
                sub_data = self.learning_set.get_data_by_names([name], False)
                if name in self.transformer:
                    sub_data = self.transformer[name].fit_transform(sub_data)
                data.append(sub_data)
            data = hstack(data)

        if samples is not None:
            data = data[samples]

        self._fit(data)
        self._trained = True

    def _fit(self, data):
        """Fit model on data.

        :param ndarray data: training data (2D).
        """
        raise NotImplementedError
