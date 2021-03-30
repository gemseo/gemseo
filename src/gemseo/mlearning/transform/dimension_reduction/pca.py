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
Principal component dimension reduction algorithm
=================================================

The :class:`PCA` class wraps the PCA from Scikit-learn.

Dependence
----------
This dimension reduction algorithm relies on the PCA class
of the `scikit-learn library <https://scikit-learn.org/stable/modules/
generated/sklearn.decomposition.PCA.html>`_.
"""
from __future__ import absolute_import, division, unicode_literals

from future import standard_library
from numpy import sqrt
from sklearn.decomposition import PCA as SKLPCA

from gemseo.mlearning.transform.dimension_reduction.dimension_reduction import (
    DimensionReduction,
)

standard_library.install_aliases()


class PCA(DimensionReduction):
    """ Principal component dimension reduction algorithm. """

    def __init__(self, name="PCA", n_components=5, **parameters):
        """Constructor.

        :param str name: transformer name. Default: 'PCA'.
        :param int n_components: number of components. Default: 5.
        :param parameters: Optional parameters for sklearn PCA constructor.
        """
        super(PCA, self).__init__(name, n_components=n_components, **parameters)
        self.algo = SKLPCA(n_components, **parameters)

    def fit(self, data):
        """Fit transformer to data.

        :param ndarray data: data to be fitted.
        """
        self.algo.fit(data)

    def transform(self, data):
        """Transform data.

        :param ndarray data: data  to be transformed.
        :return: transformed data.
        :rtype: ndarray
        """
        return self.algo.transform(data)

    def inverse_transform(self, data):
        """Perform an inverse transform on the data.

        :param ndarray data: data to be inverse transformed.
        :return: inverse transformed data.
        :rtype: ndarray
        """
        return self.algo.inverse_transform(data)

    def compute_jacobian(self, data):
        """Compute Jacobian of the pca transform.

        :param ndarray data: data where the Jacobian is to be computed.
        :return: Jacobian matrix.
        :rtype: ndarray
        """
        return self.algo.components_

    def compute_jacobian_inverse(self, data):
        """Compute Jacobian of the pca inverse_transform.

        :param ndarray data: data where the Jacobian is to be computed.
        :return: Jacobian matrix.
        :rtype: ndarray
        """
        return self.algo.components_.T

    @property
    def components(self):
        """ Components """
        return sqrt(self.algo.singular_values_) * self.algo.components_.T
