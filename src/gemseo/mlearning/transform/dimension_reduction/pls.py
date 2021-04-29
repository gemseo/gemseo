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
Transformer - Partial least square regression
=============================================

The :class:`PLS` class wraps the PCA from Scikit-learn.

Dependence
----------
This dimension reduction algorithm relies on the PLSRegression class
of the `scikit-learn library <https://scikit-learn.org/stable/modules/
generated/sklearn.cross_decomposition.PLSRegression.html>`_.
"""
from __future__ import absolute_import, division, unicode_literals

from numpy import matmul
from sklearn.cross_decomposition import PLSRegression

from gemseo.mlearning.transform.dimension_reduction.dimension_reduction import (
    DimensionReduction,
)


class PLS(DimensionReduction):
    """Partial Least Square regression."""

    CROSSED = True

    def __init__(self, name="PLS", n_components=5, **parameters):
        """Constructor.

        :param str name: transformer name. Default: 'PLS'.
        :param int n_components: number of components. Default: 5.
        :param parameters: Optional parameters for sklearn PLSRegression constructor.
        """
        super(PLS, self).__init__(name, n_components=n_components, **parameters)
        self.algo = PLSRegression(n_components, **parameters)

    def fit(self, data, other_data):
        """Fit transformer to data w.r.t. second data group.

        :param ndarray data: data to be fitted.
        :param ndarray output_data: output data to be fitted.
        """
        self.algo.fit(data, other_data)

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
        inv_data = matmul(data, self.algo.x_loadings_.T)
        inv_data *= self.algo.x_std_
        inv_data += self.algo.x_mean_
        return inv_data

    def compute_jacobian(self, data):
        """Compute Jacobian of the pca transform.

        :param ndarray data: data where the Jacobian is to be computed.
        :return: Jacobian matrix.
        :rtype: ndarray
        """
        raise NotImplementedError

    def compute_jacobian_inverse(self, data):
        """Compute Jacobian of the pca inverse_transform.

        :param ndarray data: data where the Jacobian is to be computed.
        :return: Jacobian matrix.
        :rtype: ndarray
        """
        raise NotImplementedError

    @property
    def components(self):
        """Components."""
        raise NotImplementedError
