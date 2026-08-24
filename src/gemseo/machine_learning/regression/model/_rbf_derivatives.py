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
r"""The derivatives of the RBF network for regression.

The kernels follow the conventions of
[RBFInterpolator][scipy.interpolate.RBFInterpolator],
i.e. a kernel is a function $\phi(\epsilon r)$ of the scaled radius $\epsilon r$,
where $r$ is a distance between two points
and $\epsilon$ is the shape parameter.
"""

from __future__ import annotations

from abc import abstractmethod
from types import MappingProxyType
from typing import TYPE_CHECKING
from typing import Final

from numpy import clip
from numpy import delete
from numpy import einsum
from numpy import empty
from numpy import exp
from numpy import log
from numpy import newaxis
from numpy import sqrt
from numpy.linalg import norm

from gemseo.machine_learning.regression.model.rbf_settings import RBF
from gemseo.util.constant import EPSILON
from gemseo.util.metaclass import ABCGoogleDocstringInheritanceMeta

if TYPE_CHECKING:
    from collections.abc import Mapping

    from scipy.interpolate import RBFInterpolator

    from gemseo.util.typing import IntegerArray
    from gemseo.util.typing import RealArray


class BaseKernelDerivative(metaclass=ABCGoogleDocstringInheritanceMeta):
    r"""The derivative of a radial basis function (RBF) kernel.

    For a kernel of the form $\phi(\epsilon r)$ with $r$ a scalar,
    the derivative function is defined by
    $\nabla_x \phi(\epsilon\|x\|)
    = \epsilon\phi'(\epsilon\|x\|)\frac{x}{\|x\|}$.
    """

    _epsilon: float
    """The shape parameter."""

    def __init__(self, epsilon: float) -> None:
        """
        Args:
            epsilon: The shape parameter.
        """  # noqa: D205, D212, D415
        self._epsilon = epsilon

    @abstractmethod
    def compute(self, input_data: RealArray, norm_input_data: RealArray) -> RealArray:
        """Compute the derivative of the kernel with respect to the input data.

        Args:
            input_data: The input data, shaped as `(..., n_inputs)`.
            norm_input_data: The Euclidean norms of the input data,
                shaped as `(..., 1)`.

        Returns:
            The derivative of the kernel, shaped as `(..., n_inputs)`.
        """


class LinearDerivative(BaseKernelDerivative):
    r"""The derivative of $\phi(\epsilon r)=-\epsilon r$.

    If $x=0$, the derivative is 0 (determined up to a tolerance).
    """

    def compute(self, input_data: RealArray, norm_input_data: RealArray) -> RealArray:  # noqa: D102
        return (
            (norm_input_data > EPSILON)
            * -self._epsilon
            * input_data
            / (norm_input_data + EPSILON)
        )


class ThinPlateSplineDerivative(BaseKernelDerivative):
    r"""The derivative of $\phi(\epsilon r)=(\epsilon r)^2\log(\epsilon r)$.

    If $x=0$, the derivative is 0 (determined up to a tolerance).
    """

    def compute(self, input_data: RealArray, norm_input_data: RealArray) -> RealArray:  # noqa: D102
        return (
            (norm_input_data > EPSILON)
            * self._epsilon**2
            * input_data
            * (2 * log(self._epsilon * norm_input_data + EPSILON) + 1)
        )


class CubicDerivative(BaseKernelDerivative):
    r"""The derivative of $\phi(\epsilon r)=(\epsilon r)^3$."""

    def compute(self, input_data: RealArray, norm_input_data: RealArray) -> RealArray:  # noqa: D102
        return 3 * self._epsilon**3 * norm_input_data * input_data


class QuinticDerivative(BaseKernelDerivative):
    r"""The derivative of $\phi(\epsilon r)=-(\epsilon r)^5$."""

    def compute(self, input_data: RealArray, norm_input_data: RealArray) -> RealArray:  # noqa: D102
        return -5 * self._epsilon**5 * norm_input_data**3 * input_data


class MultiquadricDerivative(BaseKernelDerivative):
    r"""The derivative of $\phi(\epsilon r)=-\sqrt{(\epsilon r)^2+1}$."""

    def compute(self, input_data: RealArray, norm_input_data: RealArray) -> RealArray:  # noqa: D102
        return (
            -(self._epsilon**2)
            * input_data
            / sqrt((self._epsilon * norm_input_data) ** 2 + 1)
        )


class InverseMultiquadricDerivative(BaseKernelDerivative):
    r"""The derivative of $\phi(\epsilon r)=1/\sqrt{(\epsilon r)^2+1}$."""

    def compute(self, input_data: RealArray, norm_input_data: RealArray) -> RealArray:  # noqa: D102
        return (
            -(self._epsilon**2)
            * input_data
            / ((self._epsilon * norm_input_data) ** 2 + 1) ** 1.5
        )


class InverseQuadraticDerivative(BaseKernelDerivative):
    r"""The derivative of $\phi(\epsilon r)=1/((\epsilon r)^2+1)$."""

    def compute(self, input_data: RealArray, norm_input_data: RealArray) -> RealArray:  # noqa: D102
        return (
            -2
            * self._epsilon**2
            * input_data
            / ((self._epsilon * norm_input_data) ** 2 + 1) ** 2
        )


class GaussianDerivative(BaseKernelDerivative):
    r"""The derivative of $\phi(\epsilon r)=\exp(-(\epsilon r)^2)$."""

    def compute(self, input_data: RealArray, norm_input_data: RealArray) -> RealArray:  # noqa: D102
        return (
            -2
            * self._epsilon**2
            * input_data
            * exp(-((self._epsilon * norm_input_data) ** 2))
        )


KERNEL_DERIVATIVES: Final[Mapping[RBF, type[BaseKernelDerivative]]] = MappingProxyType({
    RBF.LINEAR: LinearDerivative,
    RBF.THIN_PLATE_SPLINE: ThinPlateSplineDerivative,
    RBF.CUBIC: CubicDerivative,
    RBF.QUINTIC: QuinticDerivative,
    RBF.MULTIQUADRIC: MultiquadricDerivative,
    RBF.INVERSE_MULTIQUADRIC: InverseMultiquadricDerivative,
    RBF.INVERSE_QUADRATIC: InverseQuadraticDerivative,
    RBF.GAUSSIAN: GaussianDerivative,
})
"""The derivative of each radial basis function kernel."""


class RBFDerivatives:
    """The derivatives of a fitted SciPy RBF interpolator."""

    __centers: RealArray
    """The centers of the kernels, shaped as `(n_centers, n_inputs)`."""

    __kernel_coefficients: RealArray
    """The coefficients of the kernels, shaped as `(n_centers, n_outputs)`."""

    __kernel_derivative: BaseKernelDerivative
    """The derivative of the kernel."""

    __polynomial_coefficients: RealArray
    """The coefficients of the monomials, shaped as `(n_monomials, n_outputs)`."""

    __powers: IntegerArray
    """The exponents of the monomials, shaped as `(n_monomials, n_inputs)`."""

    __scale: RealArray
    """The scale normalizing the input points, shaped as `(n_inputs,)`."""

    __shift: RealArray
    """The shift normalizing the input points, shaped as `(n_inputs,)`."""

    def __init__(self, interpolator: RBFInterpolator) -> None:
        """
        Args:
            interpolator: The RBF interpolator fitted on all the learning points.
        """  # noqa: D205 D212
        self.__centers = interpolator.y
        self.__powers = interpolator.powers
        kernel_derivative_class = KERNEL_DERIVATIVES[interpolator.kernel]
        self.__kernel_derivative = kernel_derivative_class(interpolator.epsilon)
        # These SciPy attributes are private but stable since scipy 1.7;
        # the numerical Jacobian tests guard against upstream changes.
        n_centers = len(self.__centers)
        self.__kernel_coefficients = interpolator._coeffs[:n_centers]
        self.__polynomial_coefficients = interpolator._coeffs[n_centers:]
        self.__shift = interpolator._shift
        self.__scale = interpolator._scale

    def compute_jacobian(self, input_data: RealArray) -> RealArray:
        """Compute the Jacobian of the interpolator.

        Args:
            input_data: The input points, shaped as `(n_points, n_inputs)`.

        Returns:
            The Jacobian, shaped as `(n_points, n_outputs, n_inputs)`.
        """
        jacobian = self.__compute_kernel_jacobian(input_data)
        if self.__powers.size:
            jacobian += self.__compute_polynomial_jacobian(input_data)

        return jacobian

    def __compute_kernel_jacobian(self, input_data: RealArray) -> RealArray:
        """Compute the Jacobian of the kernel part of the interpolator.

        Args:
            input_data: The input points, shaped as `(n_points, n_inputs)`.

        Returns:
            The Jacobian, shaped as `(n_points, n_outputs, n_inputs)`.
        """
        # Dimensions: q: n_samples, p: n_learn_samples, n: n_inputs, s: n_outputs
        diffs = input_data[:, newaxis, :] - self.__centers[newaxis]
        dists = norm(diffs, axis=2)[..., newaxis]
        return einsum(
            "qpn,ps->qsn",
            self.__kernel_derivative.compute(diffs, dists),
            self.__kernel_coefficients,
        )

    def __compute_polynomial_jacobian(self, input_data: RealArray) -> RealArray:
        """Compute the Jacobian of the polynomial part of the interpolator.

        The monomials are evaluated at the shifted and scaled input points.

        Args:
            input_data: The input points, shaped as `(n_points, n_inputs)`.

        Returns:
            The Jacobian, shaped as `(n_points, n_outputs, n_inputs)`.
        """
        # Dimensions: q: n_samples, r: n_monomials, n: n_inputs, s: n_outputs
        powers = self.__powers
        scaled_input_data = (input_data - self.__shift) / self.__scale
        monomials = scaled_input_data[:, newaxis, :] ** powers[newaxis]
        n_inputs = input_data.shape[1]
        gradients = empty((len(input_data), len(powers), n_inputs))
        for index in range(n_inputs):
            exponents = powers[:, index]
            gradients[:, :, index] = (
                exponents
                * scaled_input_data[:, [index]] ** clip(exponents - 1, 0, None)
                * delete(monomials, index, axis=2).prod(axis=2)
                / self.__scale[index]
            )

        return einsum("qrn,rs->qsn", gradients, self.__polynomial_coefficients)
