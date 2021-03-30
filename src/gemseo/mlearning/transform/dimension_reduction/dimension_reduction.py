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
Dimension reduction
===================

The :class:`.DimensionReduction` class implements the concept of dimension
reduction.

.. seealso::

   :mod:`~gemseo.mlearning.transform.dimension_reduction.pca`
"""
from __future__ import absolute_import, division, unicode_literals

from future import standard_library

from gemseo.mlearning.transform.transformer import Transformer

standard_library.install_aliases()


class DimensionReduction(Transformer):
    """ Dimension reduction. """

    def __init__(self, name="DimensionReduction", n_components=5, **parameters):
        """Constructor.

        :param str name: name of the scaler.
        :param int n_components: number of components. Default: 5.
        :param parameters: parameters for the dimension reduction algorithm.
        """
        super(DimensionReduction, self).__init__(
            name, n_components=n_components, **parameters
        )

    def fit(self, data):
        """Fit dimension reduction algorithm to data.

        :param ndarray data: data to be fitted.
        """
        raise NotImplementedError

    @property
    def n_components(self):
        """ Number of components """
        return self.parameters["n_components"]
