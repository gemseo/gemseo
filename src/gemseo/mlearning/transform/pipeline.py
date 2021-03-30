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
Data transformer pipeline
=========================

The :class:`.Pipeline` class chains a sequence of tranformers, and provides
global fit(), transform(), fit_transform() and inverse_transform() methods.
"""
from __future__ import absolute_import, division, unicode_literals

from future import standard_library
from numpy import eye, matmul

from gemseo.mlearning.transform.transformer import Transformer

standard_library.install_aliases()


class Pipeline(Transformer):
    """ Transformer pipeline. """

    def __init__(self, name="Pipeline", transformers=None):
        """Constructor.

        :param str name: transformer pipeline name. Default: 'Pipeline'.
        :param list(Transformer) transformers: Sequence of transformers to be
            chained. The transformers are chained in the order of appearance in
            the list, i.e. the first transformer is applied first. If
            transformers is an empty list or None, then the pipeline
            transformer behaves like an identity transformer.
            Default: None.
        """
        super(Pipeline, self).__init__(name)
        self.transformers = transformers or []

    def duplicate(self):
        """ Duplicate the constructor. """
        transformers = [trans.duplicate() for trans in self.transformers]
        return Pipeline(self.name, transformers)

    def fit(self, data):
        """Fit transformer pipeline to data. All the transformers are fitted,
        transforming the data along the way.

        :param ndarray data: data to be fitted.
        """
        for transformer in self.transformers:
            data = transformer.fit_transform(data)

    def transform(self, data):
        """Transform data. The data is transformed sequentially, where the
        output of one transformer is the input of the next.

        :param ndarray data: data to be transformed.
        :return: transformed data.
        :rtype: ndarray
        """
        for transformer in self.transformers:
            data = transformer.transform(data)
        return data

    def inverse_transform(self, data):
        """Perform an inverse transform on the data. The data is inverse
        transformed sequentially, starting with the last tranformer in the
        list.

        :param ndarray data: data  to be inverse transformed.
        :return: inverse transformed data.
        :rtype: ndarray
        """
        for transformer in self.transformers[::-1]:
            data = transformer.inverse_transform(data)
        return data

    def compute_jacobian(self, data):
        """Compute Jacobian of the pipeline transform.

        :param ndarray data: data where the Jacobian is to be computed.
        :return: Jacobian matrix.
        :rtype: ndarray
        """
        jacobian = eye(data.shape[-1])
        for transformer in self.transformers:
            jacobian = matmul(transformer.compute_jacobian(data), jacobian)
            data = transformer.transform(data)
        return jacobian

    def compute_jacobian_inverse(self, data):
        """Compute Jacobian of the pipeline inverse_transform.

        :param ndarray data: data where the Jacobian is to be computed.
        :return: Jacobian matrix.
        :rtype: ndarray
        """
        jacobian = eye(data.shape[-1])
        for transformer in self.transformers[::-1]:
            jacobian = matmul(transformer.compute_jacobian_inverse(data), jacobian)
            data = transformer.inverse_transform(data)
        return jacobian
