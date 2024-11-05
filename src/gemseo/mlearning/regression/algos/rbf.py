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
r"""The RBF network for regression.

The radial basis function surrogate discipline expresses the model output
as a weighted sum of kernel functions centered on the learning input data:

.. math::

    y = w_1K(\|x-x_1\|;\epsilon) + w_2K(\|x-x_2\|;\epsilon) + \ldots
        + w_nK(\|x-x_n\|;\epsilon)

and the coefficients :math:`(w_1, w_2, \ldots, w_n)` are estimated
by least squares minimization.

Dependence
----------
The RBF model relies on the Rbf class of the
`scipy library
<https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.Rbf.html>`_.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Callable
from typing import ClassVar
from typing import Final
from typing import Union

from numpy import average
from numpy import exp
from numpy import finfo
from numpy import log
from numpy import newaxis
from numpy import sqrt
from numpy.linalg import norm
from scipy.interpolate import Rbf

from gemseo.mlearning.core.algos.supervised import SavedObjectType as _SavedObjectType
from gemseo.mlearning.regression.algos.base_regressor import BaseRegressor
from gemseo.mlearning.regression.algos.rbf_settings import RBFRegressor_Settings

if TYPE_CHECKING:
    from gemseo.typing import RealArray

SavedObjectType = Union[_SavedObjectType, float, Callable]


class RBFRegressor(BaseRegressor):
    r"""Regression based on radial basis functions (RBFs).

    This model relies on the SciPy class :class:`scipy.interpolate.Rbf`.
    """

    der_function: Callable[[RealArray], RealArray]
    """The derivative of the radial basis function."""

    y_average: RealArray
    """The mean of the learning output data."""

    SHORT_ALGO_NAME: ClassVar[str] = "RBF"
    LIBRARY: ClassVar[str] = "SciPy"

    EUCLIDEAN: Final[str] = "euclidean"

    Settings: ClassVar[type[RBFRegressor_Settings]] = RBFRegressor_Settings

    def _post_init(self):
        super()._post_init()
        self.y_average = 0.0
        self.der_function = self._settings.der_function

    class RBFDerivatives:
        r"""Derivatives of functions used in :class:`.RBFRegressor`.

        For an RBF of the form :math:`f(r)`, :math:`r` scalar,
        the derivative functions are defined by :math:`d(f(r))/dx`,
        with :math:`r=|x|/\epsilon`. The functions are thus defined
        by :math:`df/dx = \epsilon^{-1} x/|x| f'(|x|/\epsilon)`.
        This convention is chosen to avoid division by :math:`|x|` when
        the terms may be cancelled out, as :math:`f'(r)` often has a term
        in :math:`r`.
        """

        TOL = finfo(float).eps

        @classmethod
        def der_multiquadric(
            cls,
            input_data: RealArray,
            norm_input_data: float,
            eps: float,
        ) -> RealArray:
            r"""Compute derivative of :math:`f(r) = \sqrt{r^2 + 1}` w.r.t. :math:`x`.

            Args:
                input_data: The 1D input data.
                norm_input_data: The norm of the input variable.
                eps: The correlation length.

            Returns:
                The derivative of the function.
            """
            return input_data / eps**2 / sqrt((norm_input_data / eps) ** 2 + 1)

        @classmethod
        def der_inverse_multiquadric(
            cls,
            input_data: RealArray,
            norm_input_data: float,
            eps: float,
        ) -> RealArray:
            r"""Compute derivative of :math:`f(r)=1/\sqrt{r^2 + 1}` w.r.t. :math:`x`.

            Args:
                input_data: The 1D input data.
                norm_input_data: The norm of the input variable.
                eps: The correlation length.

            Returns:
                The derivative of the function.
            """
            return -input_data / eps**2 / ((norm_input_data / eps) ** 2 + 1) ** 1.5

        @classmethod
        def der_gaussian(
            cls,
            input_data: RealArray,
            norm_input_data: float,
            eps: float,
        ) -> RealArray:
            r"""Compute derivative of :math:`f(r)=\exp(-r^2)` w.r.t. :math:`x`.

            Args:
                input_data: The 1D input data.
                norm_input_data: The norm of the input variable.
                eps: The correlation length.

            Returns:
                The derivative of the function.
            """
            return -2 * input_data / eps**2 * exp(-((norm_input_data / eps) ** 2))

        @classmethod
        def der_linear(
            cls,
            input_data: RealArray,
            norm_input_data: float,
            eps: float,
        ) -> RealArray:
            """Compute derivative of :math:`f(r)=r` w.r.t. :math:`x`.

            If :math:`x=0`, return 0 (determined up to a tolerance).

            Args:
                input_data: The 1D input data.
                norm_input_data: The norm of the input variable.
                eps: The correlation length.

            Returns:
                The derivative of the function.
            """
            return (
                (norm_input_data > cls.TOL)
                * input_data
                / eps
                / (norm_input_data + cls.TOL)
            )

        @classmethod
        def der_cubic(
            cls,
            input_data: RealArray,
            norm_input_data: float,
            eps: float,
        ) -> RealArray:
            """Compute derivative w.r.t. :math:`x` of the function :math:`f(r) = r^3`.

            Args:
                input_data: The 1D input data.
                norm_input_data: The norm of the input variable.
                eps: The correlation length.

            Returns:
                The derivative of the function.
            """
            return 3 * norm_input_data * input_data / eps**3

        @classmethod
        def der_quintic(
            cls,
            input_data: RealArray,
            norm_input_data: float,
            eps: float,
        ) -> RealArray:
            """Compute derivative w.r.t. :math:`x` of the function :math:`f(r) = r^5`.

            Args:
                input_data: The 1D input data.
                norm_input_data : The norm of the input variable.
                eps: The correlation length.

            Returns:
                The derivative of the function.
            """
            return 5 * norm_input_data**3 * input_data / eps**5

        @classmethod
        def der_thin_plate(
            cls,
            input_data: RealArray,
            norm_input_data: float,
            eps: float,
        ) -> RealArray:
            r"""Compute derivative of :math:`f(r) = r^2\log(r)` w.r.t. :math:`x`.

            If :math:`x=0`, return 0 (determined up to a tolerance).

            Args:
                input_data: The 1D input data.
                norm_input_data: The norm of the input variable.
                eps: The correlation length.

            Returns:
                The derivative of the function.
            """
            return (
                (norm_input_data > cls.TOL)
                * input_data
                / eps**2
                * (1 + 2 * log(norm_input_data / eps + cls.TOL))
            )

    def _fit(self, input_data: RealArray, output_data: RealArray) -> None:
        self.y_average = average(output_data, axis=0)
        output_data -= self.y_average
        args = [*list(input_data.T), output_data]
        self.algo = Rbf(
            *args,
            mode="N-D",
            function=self._settings.function,
            epsilon=self._settings.epsilon,
            smooth=self._settings.smooth,
            norm=self._settings.norm,
        )

    def _predict(
        self,
        input_data: RealArray,
    ) -> RealArray:
        return self.algo(*input_data.T).reshape((len(input_data), -1)) + self.y_average

    def _predict_jacobian(
        self,
        input_data: RealArray,
    ) -> RealArray:
        self._check_available_jacobian()
        der_func = self.der_function or getattr(
            self.RBFDerivatives, f"der_{self.function}"
        )
        #             predict_samples                        learn_samples
        # Dimensions : ( n_samples , n_outputs , n_inputs , n_learn_samples )
        # input_data : ( n_samples ,           , n_inputs ,                 )
        # ref_points : (           ,           , n_inputs , n_learn_samples )
        # nodes      : (           , n_outputs ,          , n_learn_samples )
        # jacobians  : ( n_samples , n_outputs , n_inputs ,                 )
        ref_points = self.algo.xi[newaxis, newaxis]
        nodes = self.algo.nodes.T[newaxis, :, newaxis]
        input_data = input_data[:, newaxis, :, newaxis]
        diffs = input_data - ref_points
        dists = norm(diffs, axis=2)[:, :, newaxis]
        return (nodes * der_func(diffs, dists, eps=self.algo.epsilon)).sum(-1)

    def _check_available_jacobian(self) -> None:
        """Check if the Jacobian is available for the given setup.

        Raises:
            NotImplementedError: Either if the Jacobian computation is not implemented
                or if the derivative of the radial basis function is missing.
        """
        if self.algo.norm != self.EUCLIDEAN:
            msg = "Jacobian is only implemented for Euclidean norm."
            raise NotImplementedError(msg)

        if callable(self.function) and self.der_function is None:
            msg = (
                "No der_function is provided."
                "Add der_function in RBFRegressor constructor."
            )
            raise NotImplementedError(msg)

    @property
    def function(self) -> str:
        """The name of the kernel function.

        The name is possibly different from self.parameters['function'], as it is mapped
        (scipy). Examples:

        'inverse'              -> 'inverse_multiquadric' 'InverSE MULtiQuadRIC' ->
        'inverse_multiquadric'
        """
        return self.algo.function
