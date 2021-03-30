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
Min-max data scaler
===================

The :class:`.MinMaxScaler` class implements the MinMax scaling method applying
to some parameter :math:`z`:

.. math::

    \\bar{z} := \\text{offset} + \\text{coefficient}\\times z
    = \\frac{z-\\text{min}(z)}{(\\text{max}(z)-\\text{min}(z))},

where :math:`\\text{offset}=-\\text{min}(z)/(\\text{max}(z)-\\text{min}(z))`
and :math:`\\text{coefficient}=1/(\\text{max}(z)-\\text{min}(z))`.

In the MinMax scaling method, the scaling operation linearly transforms the
original variable :math:`z` such that the minimum of the original data
corresponds to 0 and the maximum to 1.
"""
from __future__ import absolute_import, division, unicode_literals

from future import standard_library

from gemseo.mlearning.transform.scaler.scaler import Scaler

standard_library.install_aliases()


class MinMaxScaler(Scaler):
    """ Min-max scaler. """

    def __init__(self, name="MinMaxScaler", offset=0.0, coefficient=1.0):
        """Constructor.

        :param str name: name of the scaler. Default: 'MinMaxScaler'.
        :param float offset: offset of the linear transformation.
            Default: 0.
        :param float coefficient: coefficient of the linear transformation.
            Default: 1.
        """
        super(MinMaxScaler, self).__init__(name, offset, coefficient)

    def fit(self, data):
        """Fit offset and coefficient terms from a data array. The min and the
        max are computed along the first axis of the data.

        :param array data: data to be fitted.
        """
        super(MinMaxScaler, self).fit(data)
        l_b = data.min(0)
        u_b = data.max(0)
        self.offset = -l_b / (u_b - l_b)
        self.coefficient = 1 / (u_b - l_b)
