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
#        :author: Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""
Karhunen Loeve SVD Algorithm
============================

The :class:`KLSVD` class wraps the KarhunenLoeveSVDAlgorithm
`from OpenTURNS <https://openturns.github.io/openturns/latest/user_manual/
_generated/openturns.KarhunenLoeveSVDAlgorithm.html>`_.
"""
from __future__ import absolute_import, division, unicode_literals

from future import standard_library
from numpy import array
from openturns import (
    Basis,
    Field,
    FunctionCollection,
    KarhunenLoeveResult,
    KarhunenLoeveSVDAlgorithm,
    Matrix,
    Mesh,
    Point,
    ProcessSample,
    RankMCovarianceModel,
    Sample,
)

from gemseo.mlearning.transform.dimension_reduction.dimension_reduction import (
    DimensionReduction,
)

standard_library.install_aliases()


class KLSVD(DimensionReduction):
    """ Karhunen Loeve SVD Algorithm. """

    def __init__(self, mesh, n_components=5, name="KLSVD"):
        """Constructor.

        :param ndarray mesh: mesh passed a 2D array whose rows are nodes
            and columns are dimensions.
        :param int n_components: number of components. Default: 5.
        :param str name: transformer name. Default: 'KLSVD'.
        """
        super(KLSVD, self).__init__(name, mesh=mesh, n_components=n_components)
        self.algo = None
        self.ot_mesh = Mesh(Sample(mesh))

    def mesh(self):
        """ mesh """
        return self.parameters["mesh"]

    def fit(self, data):
        """Fit transformer to data.

        :param ndarray data: data to be fitted.
        """
        sample = self._get_process_sample(data)
        mesh_size = self.ot_mesh.getVerticesNumber()
        klsvd = KarhunenLoeveSVDAlgorithm(sample, [1] * mesh_size, 0.0, True)
        klsvd.run()
        result = klsvd.getResult()
        if self.n_components < data.shape[1]:
            result = self._truncate_kl_result(result)
        self.algo = result

    def transform(self, data):
        """Transform data.

        :param ndarray data: data  to be transformed.
        :return: transformed data.
        :rtype: ndarray
        """
        sample = self._get_process_sample(data)
        return array(self.algo.project(sample))

    def inverse_transform(self, data):
        """Perform an inverse transform on the data.

        :param ndarray data: data to be inverse transformed.
        :return: inverse transformed data.
        :rtype: ndarray
        """
        pred = []
        for coefficients in data:
            coeff = Point(list(coefficients))
            pred.append(list(self.algo.liftAsSample(coeff)))
        return array(pred)[:, :, 0]

    @property
    def output_dimension(self):
        """Number of output dimensions (reduced).

        :return: Number of output dimensions.
        :rtype: int
        """
        return len(self.algo.getModes())

    @property
    def components(self):
        """ Principal components """
        tmp = array(self.algo.getScaledModesAsProcessSample())[:, :, 0].T
        return tmp

    @property
    def eigenvalues(self):
        """ Eigen values """
        return array(self.algo.getEigenValues())

    def _get_process_sample(self, data):
        """Convert a ndarray data to openturns.ProcessSample.

        :param ndarray data: data to be fitted.
        """
        datum = [[x_i] for x_i in data[0, :]]
        sample = ProcessSample(1, Field(self.ot_mesh, datum))
        for datum in data[1:, :]:
            datum = [[x_i] for x_i in datum]
            sample.add(Field(self.ot_mesh, datum))
        return sample

    def _truncate_kl_result(self, result):
        """Truncate an openturns.KarhurenLoeveResult

        :param result: KL result
        """
        # These lines come from an issue opened by Michael Baudin
        # => https://github.com/openturns/openturns/issues/1470

        # Truncate eigenvalues
        eigenvalues = result.getEigenValues()
        full_n_modes = eigenvalues.getDimension()
        n_modes = min(self.n_components, full_n_modes)
        trunc_eigenvalues = eigenvalues[:n_modes]
        trunc_thresh = eigenvalues[n_modes - 1] / eigenvalues[0]
        # Truncate modes
        modes = result.getModes()
        trunc_modes = FunctionCollection()
        for i in range(n_modes):
            trunc_modes.add(modes[i])
        # Truncate process sample modes
        modes_as_proc_samp = result.getModesAsProcessSample()
        datum = list(modes_as_proc_samp[0])
        trunc_modes_as_proc_samp = ProcessSample(1, Field(self.ot_mesh, datum))
        for i in range(1, n_modes):
            trunc_modes_as_proc_samp.add(modes_as_proc_samp[i])
        # Truncate projection matrix
        proj_matrix = result.getProjectionMatrix()
        n_cols = proj_matrix.getNbColumns()
        trunc_proj_mat = Matrix(n_modes, n_cols)
        for i in range(n_modes):
            trunc_proj_mat[i, :] = proj_matrix[i, :]
        # Truncate covariance model
        trunc_cov_model = RankMCovarianceModel(trunc_eigenvalues, Basis(trunc_modes))
        result = KarhunenLoeveResult(
            trunc_cov_model,
            trunc_thresh,
            trunc_eigenvalues,
            trunc_modes,
            trunc_modes_as_proc_samp,
            trunc_proj_mat,
        )
        return result
