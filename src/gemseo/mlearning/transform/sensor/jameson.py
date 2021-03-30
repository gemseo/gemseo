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
Jameson sensor
==============
"""
from __future__ import absolute_import, division, unicode_literals

from future import standard_library
from numpy import abs as np_abs
from numpy import amax

from gemseo.mlearning.transform.transformer import Transformer

standard_library.install_aliases()


class JamesonSensor(Transformer):
    """ Jameson Sensor. """

    def __init__(
        self, name="JamesonSensor", threshold=0.3, removing_part=0.01, dimension=1
    ):
        """Constructor.

        :param str name: name of the sensor. Default: 'JamesonSensor'.
        :param float threshold: value to add to the denominator
            to avoid zero division. Default: 0.3.
        :param float removing_part: define the level of the signal to
            remove in order to avoid leading and trailing edge effects.
            ONLY FOR 1D MESH, to redefine for 2D mesh. Default: 0.01.
        :param int dimension: mesh dimension. Default: 1.
        """
        super(JamesonSensor, self).__init__(name)
        self.threshold = threshold
        self.removing_part = removing_part
        self.dimension = dimension

    def fit(self, data):
        """Fit sensor to data.

        :param array data: data to be fitted.
        """
        self.threshold = self.threshold * amax(data)

    def transform(self, data):
        """Transform data.

        :param ndarray data: data  to be transformed.
        :return: transformed data.
        :rtype: ndarray
        """
        if self.dimension == 1:
            transformed_data = self._jameson_1_d(data)
        else:
            raise NotImplementedError()

        return transformed_data

    def _jameson_1_d(self, data):
        """Transform data.

        :param ndarray data: data  to be transformed.
        :return: transformed data.
        :rtype: ndarray
        """
        mesh_size = data.shape[1] - 2
        min_mesh_size = int(mesh_size * self.removing_part)
        max_mesh_size = int(mesh_size * (1 - self.removing_part))
        norm = np_abs(data[:, :-2])
        norm += 2 * np_abs(data[:, 1:-1])
        norm += np_abs(data[:, 2:])
        norm = norm + self.threshold
        result = abs(data[:, :-2] - 2 * data[:, 1:-1] + data[:, 2:]) / norm
        result = result[:, min_mesh_size:max_mesh_size]
        return result
