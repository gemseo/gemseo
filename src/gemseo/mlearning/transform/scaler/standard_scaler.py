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
Standard data scaler
====================

The :class:`.StandardScaler` class implements the Standard scaling method
applying to some parameter :math:`z`:

.. math::

    \\bar{z} := \\text{offset} + \\text{coefficient}\\times z
    = \\frac{z-\\text{mean}(z)}{\\text{std}(z)}

where :math:`\\text{offset}=-\\text{mean}(z)/\\text{std}(z)` and
:math:`\\text{coefficient}=1/\\text{std}(z)`.

In this Standard scaling method, the scaling operation linearly transforms the
original variable math:`z` such that in the scaled space, the original data
have zero mean and unit standard deviation.
"""
from __future__ import absolute_import, division, unicode_literals

from future import standard_library
from numpy import mean, std
from past.utils import old_div

from gemseo.mlearning.transform.scaler.scaler import Scaler

standard_library.install_aliases()


class StandardScaler(Scaler):
    """ Standard scaler. """

    def __init__(self, name="StandardScaler", offset=0.0, coefficient=1.0):
        """Constructor.

        :param str name: name of the scaler. Default: 'StandardScaler'.
        :param float offset: offset of the linear transformation. Default: 0.
        :param float coefficient: coefficient of the linear transformation.
            Default: 1.
        """
        super(StandardScaler, self).__init__(name, offset, coefficient)

    def fit(self, data):
        """Fit offset and coefficient terms from a data array. The mean and
        standard deviation are computed along the first axis of the data.

        :param array data: data to be fitted.
        """
        super(StandardScaler, self).fit(data)
        average = mean(data, 0)
        std_ = std(data, 0)
        self.offset = old_div(-average, std_)
        self.coefficient = 1.0 / std_
