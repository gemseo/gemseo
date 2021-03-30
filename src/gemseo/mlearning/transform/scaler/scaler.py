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
#        :author: Matthias De Lozzo, Syver Doving Agdestein
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""
Data scaler
===========

The :class:`.Scaler` class implements the default scaling method applying to
some parameter :math:`z`:

.. math::

    \\bar{z} := \\text{offset} + \\text{coefficient}\\times z

where :math:`\\bar{z}` is the scaled version of z. This scaling method is a
linear transformation parameterized by an offset and a coefficient.

In this default scaling method, the offset is equal to 0 and the coefficient is
equal to 1. Consequently, the scaling operation is the identity:
:math:`\\bar{z}=z`. This method has to be overloaded.

.. seealso::

   :mod:`~gemseo.mlearning.transform.scaler.min_max_scaler`
   :mod:`~gemseo.mlearning.transform.scaler.standard_scaler`
"""
from __future__ import absolute_import, division, unicode_literals

from future import standard_library
from numpy import diag, eye, ndarray

from gemseo.mlearning.transform.transformer import Transformer

standard_library.install_aliases()


class Scaler(Transformer):
    """ Data scaler. """

    def __init__(self, name="Scaler", offset=0.0, coefficient=1.0):
        """Constructor.

        :param str name: name of the scaler.
        :param float offset: offset of the linear transformation.
            Default: 0.
        :param float coefficient: coefficient of the linear transformation.
            Default: 1.
        """
        super(Scaler, self).__init__(name, offset=offset, coefficient=coefficient)

    @property
    def offset(self):
        """ Offset. """
        return self.parameters["offset"]

    @property
    def coefficient(self):
        """ Coefficient. """
        return self.parameters["coefficient"]

    @offset.setter
    def offset(self, value):
        """ Set offset. """
        self.parameters["offset"] = value

    @coefficient.setter
    def coefficient(self, value):
        """ Set coefficient. """
        self.parameters["coefficient"] = value

    def fit(self, data):
        """Fit scaler to data. Offset and coefficient terms are already
        defined in the constructor.

        :param ndarray data: data to be fitted.
        """
        return

    def transform(self, data):
        """Scale data using the offset and coefficient terms.

        :param ndarray data: data  to be transformed.
        :return: transformed data.
        :rtype: ndarray
        """
        scaled_data = self.offset + self.coefficient * data
        return scaled_data

    def inverse_transform(self, data):
        """Unscale data using the offset and coefficient terms.

        :param ndarray data: data to be inverse transformed.
        :return: inverse transformed data.
        :rtype: ndarray
        """
        unscaled_data = (data - self.offset) / self.coefficient
        return unscaled_data

    def compute_jacobian(self, data):
        """Compute Jacobian of the scaler transform.

        :param ndarray data: data where the Jacobian is to be computed.
        :return: Jacobian matrix.
        :rtype: ndarray
        """
        if not isinstance(self.coefficient, ndarray):
            jac = self.coefficient * eye(data.shape[-1])
        else:
            jac = diag(self.coefficient)
        return jac

    def compute_jacobian_inverse(self, data):
        """Compute Jacobian of the scaler inverse_transform.

        :param ndarray data: data where the Jacobian is to be computed.
        :return: Jacobian matrix.
        :rtype: ndarray
        """
        if not isinstance(self.coefficient, ndarray):
            jac_inv = 1 / self.coefficient * eye(data.shape[-1])
        else:
            jac_inv = diag(1 / self.coefficient)
        return jac_inv
