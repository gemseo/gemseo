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
"""The Karhunen-Loève SVD algorithm to reduce the dimension of a variable.

The :class:`KLSVD` class wraps the KarhunenLoeveSVDAlgorithm
`from OpenTURNS <https://openturns.github.io/openturns/latest/user_manual/
_generated/openturns.KarhunenLoeveSVDAlgorithm.html>`_.
"""
from __future__ import division, unicode_literals

import openturns
from numpy import array, ndarray
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
from gemseo.mlearning.transform.transformer import TransformerFitOptionType
from gemseo.utils.compatibility.openturns import get_eigenvalues


class KLSVD(DimensionReduction):
    """The Karhunen-Loève SVD Algorithm."""

    def __init__(
        self,
        mesh,  # type: ndarray
        n_components=5,  # type: int
        name="KLSVD",  # type: str
    ):  # type: (...) -> None
        """
        Args:
            mesh: A mesh passed a 2D array
                whose rows are nodes and columns are the dimensions of the nodes.
        """
        super(KLSVD, self).__init__(name, mesh=mesh, n_components=n_components)
        self.algo = None
        self.ot_mesh = Mesh(Sample(mesh))

    @property
    def mesh(self):  # type: (...) -> ndarray
        """The mesh."""
        return self.parameters["mesh"]

    def fit(
        self,
        data,  # type: ndarray
        *args  # type: TransformerFitOptionType
    ):  # type: (...) -> None
        sample = self._get_process_sample(data)
        mesh_size = self.ot_mesh.getVerticesNumber()
        klsvd = KarhunenLoeveSVDAlgorithm(sample, [1] * mesh_size, 0.0, True)
        klsvd.run()
        result = klsvd.getResult()
        if self.n_components < data.shape[1]:
            result = self._truncate_kl_result(result)
        self.algo = result

    def transform(
        self,
        data,  # type: ndarray
    ):  # type: (...) -> ndarray
        sample = self._get_process_sample(data)
        return array(self.algo.project(sample))

    def inverse_transform(
        self,
        data,  # type: ndarray
    ):  # type: (...) -> ndarray
        pred = []
        for coefficients in data:
            coeff = Point(list(coefficients))
            pred.append(list(self.algo.liftAsSample(coeff)))
        return array(pred)[:, :, 0]

    @property
    def output_dimension(self):  # type: (...) -> int
        """The dimension of the latent space."""
        return len(self.algo.getModes())

    @property
    def components(self):  # type: (...) -> ndarray
        """The principal components."""
        tmp = array(self.algo.getScaledModesAsProcessSample())[:, :, 0].T
        return tmp

    @property
    def eigenvalues(self):  # type: (...) -> ndarray
        """The eigen values."""
        return array(get_eigenvalues(self.algo))

    def _get_process_sample(
        self,
        data,  # type: ndarray
    ):  # type: (...) -> openturns.ProcessSample
        """Convert numpy.ndarray data to an openturns.ProcessSample.

        Args:
            data: The data to be fitted.

        Returns:
            A sample representing a process.
        """
        datum = [[x_i] for x_i in data[0, :]]
        sample = ProcessSample(1, Field(self.ot_mesh, datum))
        for datum in data[1:, :]:
            datum = [[x_i] for x_i in datum]
            sample.add(Field(self.ot_mesh, datum))
        return sample

    def _truncate_kl_result(
        self,
        result,  # type: openturns.KarhunenLoeveResult
    ):  # type:(...) -> openturns.KarhunenLoeveResult
        """Truncate an openturns.KarhunenLoeveResult.

        Args:
            result: The original KarhunenLoeveResult.

        Returns:
            The truncated KarhunenLoeveResult.
        """
        # These lines come from an issue opened by Michael Baudin
        # => https://github.com/openturns/openturns/issues/1470

        # Truncate eigenvalues
        eigenvalues = get_eigenvalues(result)
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
