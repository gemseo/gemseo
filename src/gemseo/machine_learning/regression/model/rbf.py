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
#        :author: Francois Gallard, Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""The RBF network for regression."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import ClassVar
from typing import Final

from numpy import prod
from scipy.interpolate import RBFInterpolator

from gemseo.machine_learning.regression.core.base_regressor import BaseRegressor
from gemseo.machine_learning.regression.model._rbf_derivatives import RBFDerivatives
from gemseo.machine_learning.regression.model.rbf_settings import (
    BaseRBFRegressorSettings,
)
from gemseo.machine_learning.regression.model.rbf_settings import RBFRegressor_Settings

if TYPE_CHECKING:
    from gemseo.util.typing import RealArray

_SCALE_INVARIANT_KERNELS: Final[frozenset[str]] = frozenset((
    "linear",
    "thin_plate_spline",
    "cubic",
    "quintic",
))
"""The kernels for which the shape parameter has no effect."""


class RBFRegressor(BaseRegressor):
    r"""Radial basis function (RBF) regression.

    The output of an RBF regressor is
    a weighted sum of kernel functions centered on the learning input data
    completed by a low-degree polynomial:

    $$
        y = w_1 K(\epsilon\|x-x_1\|) + w_2 K(\epsilon\|x-x_2\|) + \ldots
            + w_n K(\epsilon\|x-x_n\|) + P(x)
    $$

    where the coefficients $(w_1, w_2, \ldots, w_n)$ and the polynomial $P$
    are estimated by solving a linear system.
    By default, the model interpolates the learning points exactly;
    increasing the setting `smoothing` relaxes this interpolation
    in favor of a smoother model.

    This class relies on the
    [RBFInterpolator](https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.RBFInterpolator.html)
    SciPy class.
    """

    SHORT_NAME: ClassVar[str] = "RBF"
    LIBRARY: ClassVar[str] = "SciPy"

    settings_class: ClassVar[type[BaseRBFRegressorSettings]] = RBFRegressor_Settings

    def _fit(self, input_data: RealArray, output_data: RealArray) -> None:
        epsilon = self._settings.epsilon_
        if epsilon is None and self._settings.kernel_ not in _SCALE_INVARIANT_KERNELS:
            extents = input_data.max(0) - input_data.min(0)
            extents = extents[extents.nonzero()]
            size = extents.size
            if size:
                epsilon = 1.0 / (prod(extents) / len(input_data)) ** (1.0 / size)
            else:
                # The data for all the input dimensions are constant.
                epsilon = 1.0

        self.algo = RBFInterpolator(
            input_data,
            output_data,
            neighbors=self._settings.neighbors,
            smoothing=self._settings.smoothing,
            kernel=self._settings.kernel_,
            epsilon=epsilon,
            degree=self._settings.degree,
        )
        # A local interpolant, i.e. when the setting 'neighbors' is set,
        # is discontinuous where the set of nearest learning points changes,
        # hence has no Jacobian.
        if self._settings.neighbors is None:
            self.__derivatives = RBFDerivatives(self.algo)

    def _predict(
        self,
        input_data: RealArray,
    ) -> RealArray:
        return self.algo(input_data)

    def _predict_jacobian(
        self,
        input_data: RealArray,
    ) -> RealArray:
        """
        Raises:
            NotImplementedError: When the model is a local interpolant,
                i.e. the setting `neighbors` is not `None`.
        """  # noqa: D205, D212
        if self._settings.neighbors is not None:
            msg = (
                "The Jacobian is not implemented "
                "when the setting 'neighbors' is set, "
                "because the model is then a local interpolant, "
                "discontinuous where the set of nearest learning points changes."
            )
            raise NotImplementedError(msg)

        return self.__derivatives.compute_jacobian(input_data)

    @property
    def kernel(self) -> str:
        """The name of the kernel function."""
        return self.algo.kernel
