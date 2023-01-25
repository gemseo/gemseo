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

The :class:`.KLSVD` class wraps the ``KarhunenLoeveSVDAlgorithm``
`from OpenTURNS <https://openturns.github.io/openturns/latest/user_manual/
_generated/openturns.KarhunenLoeveSVDAlgorithm.html>`_.
"""
from __future__ import annotations

import openturns
from numpy import array
from numpy import ndarray
from openturns import Field
from openturns import KarhunenLoeveSVDAlgorithm
from openturns import Mesh
from openturns import Point
from openturns import ProcessSample
from openturns import ResourceMap
from openturns import Sample

from gemseo.mlearning.transform.dimension_reduction.dimension_reduction import (
    DimensionReduction,
)
from gemseo.mlearning.transform.transformer import TransformerFitOptionType
from gemseo.utils.compatibility.openturns import get_eigenvalues


class KLSVD(DimensionReduction):
    """The Karhunen-Loève SVD algorithm.

    Based on OpenTURNS.

    Warnings:
        Under Python 3.7,
        |g| requires openturns 1.19 which removes the non-significant components,
        whatever the value of ``n_components`` passed at instantiation.
        Under Python 3.8 and above,
        the number of components is equal to ``n_components`` if not ``None``,
        otherwise to the mesh size.
    """

    # Remove the Warnings block once python 3.7 removed.

    __HALKO2010 = "halko2010"
    __HALKO2011 = "halko2011"
    __RANDOM_SVD_MAXIMUM_RANK = "KarhunenLoeveSVDAlgorithm-RandomSVDMaximumRank"
    __RANDOM_SVD_VARIANT = "KarhunenLoeveSVDAlgorithm-RandomSVDVariant"
    __USE_RANDOM_SVD = "KarhunenLoeveSVDAlgorithm-UseRandomSVD"

    def __init__(
        self,
        mesh: ndarray,
        n_components: int | None = None,
        name: str = "KLSVD",
        use_random_svd: bool = False,
        n_singular_values: int | None = None,
        use_halko2010: bool = True,
    ) -> None:
        """
        Args:
            mesh: A mesh passed as a 2D NumPy array
                whose rows are nodes and columns are the dimensions of the nodes.
            use_random_svd: Whether to use a stochastic algorithm
                to compute the SVD decomposition;
                if so,
                the number of singular values has to be fixed a priori.
            n_singular_values: The number of singular values to compute
                when ``use_random_svd`` is ``True``;
                if ``None``, use the default value implemented by OpenTURNS.
            use_halko2010: Whether to use the *halko2010* algorithm
                or the *halko2011* one.
        """
        super().__init__(
            name,
            mesh=mesh,
            n_components=n_components,
            use_random_svd=use_random_svd,
            n_singular_values=n_singular_values,
            use_halko2010=use_halko2010,
        )
        self.algo = None
        self.ot_mesh = Mesh(Sample(mesh))

    @property
    def mesh(self) -> ndarray:
        """The mesh."""
        return self.parameters["mesh"]

    def _fit(self, data: ndarray, *args: TransformerFitOptionType) -> None:
        self.__update_resource_map()
        klsvd = KarhunenLoeveSVDAlgorithm(
            self._get_process_sample(data),
            [1] * self.ot_mesh.getVerticesNumber(),
            0.0,
            True,
        )
        if self.n_components is not None:
            klsvd.setNbModes(self.n_components)

        klsvd.run()
        self.algo = klsvd.getResult()
        self.parameters["n_components"] = len(get_eigenvalues(self.algo))

    def __update_resource_map(self) -> None:
        """Update OpenTURNS constants by using its ResourceMap."""
        use_random_svd = self.parameters["use_random_svd"]
        ResourceMap.SetAsBool(self.__USE_RANDOM_SVD, use_random_svd)
        n_singular_values = self.parameters["n_singular_values"]
        if n_singular_values:
            ResourceMap.SetAsUnsignedInteger(
                self.__RANDOM_SVD_MAXIMUM_RANK, n_singular_values
            )
        ResourceMap.SetAsString(
            self.__RANDOM_SVD_VARIANT,
            self.__HALKO2010 if self.parameters["use_halko2010"] else self.__HALKO2011,
        )

    @DimensionReduction._use_2d_array
    def transform(self, data: ndarray) -> ndarray:
        return array(self.algo.project(self._get_process_sample(data)))

    @DimensionReduction._use_2d_array
    def inverse_transform(self, data: ndarray) -> ndarray:
        return array(
            [
                list(self.algo.liftAsSample(Point(list(coefficients))))
                for coefficients in data
            ]
        )[:, :, 0]

    @property
    def output_dimension(self) -> int:
        """The dimension of the latent space."""
        return len(self.algo.getModes())

    @property
    def components(self) -> ndarray:
        """The principal components."""
        return array(self.algo.getScaledModesAsProcessSample())[:, :, 0].T

    @property
    def eigenvalues(self) -> ndarray:
        """The eigen values."""
        return array(get_eigenvalues(self.algo))

    def _get_process_sample(self, data: ndarray) -> openturns.ProcessSample:
        """Convert numpy.ndarray data to an openturns.ProcessSample.

        Args:
            data: The data to be fitted.

        Returns:
            A sample representing a process.
        """
        datum = [[x_i] for x_i in data[0, :]]
        sample = ProcessSample(1, Field(self.ot_mesh, datum))
        for datum in data[1:, :]:
            sample.add(Field(self.ot_mesh, [[x_i] for x_i in datum]))
        return sample
