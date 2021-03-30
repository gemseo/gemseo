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
Data transformer
================

The abstract :class:`.Transformer` class implements the concept of a data
transformer. Inheriting classes should implement the
:meth:`.Transformer.fit`, :meth:`.Transformer.transform` and
possibly :meth:`.Transformer.inverse_transform` methods.

.. seealso::

   :mod:`~gemseo.mlearning.transform.scaler.scaler`
   :mod:`~gemseo.mlearning.transform.dimension_reduction.dimension_reduction`
"""
from __future__ import absolute_import, division, unicode_literals

from future import standard_library

standard_library.install_aliases()


class Transformer(object):
    """ Transformer baseclass. """

    def __init__(self, name="Transformer", **parameters):
        """Constructor.

        :param str name: transformer name. Default: 'Transformer'.
        :param parameters: transformer parameters.
        """
        self.name = name
        self.parameters = parameters

    def duplicate(self):
        """ Duplicate the constructor. """
        return self.__class__(self.name, **self.parameters)

    def fit(self, data):
        """Fit transformer to data.

        :param ndarray data: data to be fitted.
        """
        raise NotImplementedError

    def transform(self, data):
        """Transform data.

        :param ndarray data: data  to be transformed.
        :return: transformed data.
        :rtype: ndarray
        """
        raise NotImplementedError

    def inverse_transform(self, data):
        """Perform an inverse transform on the data.

        :param ndarray data: data to be inverse transformed.
        :return: inverse transformed data.
        :rtype: ndarray
        """
        raise NotImplementedError

    def fit_transform(self, data):
        """Fit transformer to data and transform data.

        :param ndarray data: data to be fitted and transformed.
        :return: transformed data.
        :rtype: ndarray
        """
        self.fit(data)
        transformed_data = self.transform(data)
        return transformed_data

    def compute_jacobian(self, data):
        """Compute Jacobian of the transformer transform.

        :param ndarray data: data where the Jacobian is to be computed.
        :return: Jacobian matrix.
        :rtype: ndarray
        """
        raise NotImplementedError

    def compute_jacobian_inverse(self, data):
        """Compute Jacobian of the transformer inverse_transform.

        :param ndarray data: data where the Jacobian is to be computed.
        :return: Jacobian matrix.
        :rtype: ndarray
        """
        raise NotImplementedError

    def __str__(self):
        """ String representation for end user. """
        string = self.__class__.__name__
        return string
