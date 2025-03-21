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
r"""Gaussian process regression model.

Overview
--------

The Gaussian process regression (GPR) model
expresses the model output as a weighted sum of kernel functions
centered on the learning input data:

.. math::

    y = \mu
        + w_1\kappa(\|x-x_1\|;\epsilon)
        + w_2\kappa(\|x-x_2\|;\epsilon)
        + ...
        + w_N\kappa(\|x-x_N\|;\epsilon)

Details
-------

The GPR model relies on the assumption
that the original model :math:`f` to replace
is an instance of a Gaussian process (GP) with mean :math:`\mu`
and covariance :math:`\sigma^2\kappa(\|x-x'\|;\epsilon)`.

Then, the GP conditioned by the training dataset
:math:`(x_i,y_i)_{1\leq i \leq N}`
is entirely defined by its expectation:

.. math::

    \hat{f}(x) = \hat{\mu} + \hat{w}^T k(x)

and its covariance:

.. math::

    \hat{c}(x,x') = \hat{\sigma}^2 - k(x)^T K^{-1} k(x')

where :math:`[\hat{\mu};\hat{w}]=([1_N~K]^T[1_N~K])^{-1}[1_N~K]^TY` with
:math:`K_{ij}=\kappa(\|x_i-x_j\|;\hat{\epsilon})`,
:math:`k_i(x)=\kappa(\|x-x_i\|;\hat{\epsilon})`
and :math:`Y_i=y_i`.

The correlation length vector :math:`\epsilon`
is estimated by numerical non-linear optimization.

Surrogate model
---------------

The expectation :math:`\hat{f}` is the surrogate model of :math:`f`.

Error measure
-------------

The standard deviation :math:`\hat{s}` is a local error measure
of :math:`\hat{f}`:

.. math::

    \hat{s}(x):=\sqrt{\hat{c}(x,x)}

Interpolation or regression
---------------------------

The GPR model can be regressive or interpolative
according to the value of the nugget effect :math:`\alpha\geq 0`
which is a regularization term
applied to the correlation matrix :math:`K`.
When :math:`\alpha = 0`,
the surrogate model interpolates the learning data.

Dependence
----------
The GPR model relies on the GaussianProcessRegressor class
of the `scikit-learn library <https://scikit-learn.org/stable/modules/
generated/sklearn.gaussian_process.GaussianProcessRegressor.html>`_.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING
from typing import ClassVar
from typing import Final

import sklearn.gaussian_process
from numpy import atleast_2d
from numpy import newaxis
from numpy import repeat
from numpy import swapaxes
from sklearn.gaussian_process.kernels import Matern

from gemseo.mlearning.data_formatters.regression_data_formatters import (
    RegressionDataFormatters,
)
from gemseo.mlearning.regression.algos.base_random_process_regressor import (
    BaseRandomProcessRegressor,
)
from gemseo.mlearning.regression.algos.gpr_settings import (
    GaussianProcessRegressor_Settings,
)
from gemseo.utils.data_conversion import concatenate_dict_of_arrays_to_array

if TYPE_CHECKING:
    from gemseo.mlearning.core.algos.ml_algo import DataType
    from gemseo.typing import RealArray

__Bounds = tuple[float, float]


class GaussianProcessRegressor(BaseRandomProcessRegressor):
    """Gaussian process regression model."""

    SHORT_ALGO_NAME: ClassVar[str] = "GPR"
    LIBRARY: ClassVar[str] = "scikit-learn"
    __DEFAULT_BOUNDS: Final[tuple[float, float]] = (0.01, 100.0)

    Settings: ClassVar[type[GaussianProcessRegressor_Settings]] = (
        GaussianProcessRegressor_Settings
    )

    def _post_init(self):
        super()._post_init()
        kernel = self._settings.kernel
        if kernel is None:
            kernel = Matern(
                length_scale=(1.0,) * self._reduced_input_dimension,
                length_scale_bounds=self.__compute_parameter_length_scale_bounds(
                    self._settings.bounds
                ),
                nu=2.5,
            )
        self.algo = sklearn.gaussian_process.GaussianProcessRegressor(
            alpha=self._settings.alpha,
            kernel=kernel,
            n_restarts_optimizer=self._settings.n_restarts_optimizer,
            # When output_dimension > 1 and normalize_y is False,
            # the prediction uncertainty
            # (e.g. standard deviation, samples, ...)
            # is wrong (see https://github.com/scikit-learn/scikit-learn/issues/29697).
            # Set normalize_y to True fix this bug.
            # TODO: Remove this bugfix once the bug has been corrected in sklearn.
            normalize_y=True,
            optimizer=self._settings.optimizer,
            random_state=self._settings.random_state,
        )

    @property
    def kernel(self):  # (...) -> Kernel
        """The kernel used for prediction."""
        if self.is_trained:
            return self.algo.kernel_
        return self.algo.kernel

    def __compute_parameter_length_scale_bounds(
        self,
        bounds: __Bounds | Mapping[str, __Bounds],
    ) -> list[tuple[float, float]]:
        """Return the lower and upper bounds for the parameter length scales.

        Args:
            bounds: The lower and upper bounds of the parameter length scales.
                Either a unique lower-upper pair common to all the inputs
                or lower-upper pairs for some of them.
                When an input has no pair,
                the lower bound is 0.01 and the upper bound is 100.

        Returns:
            The lower and upper bounds of the parameter length scales.
        """
        dimension = self._reduced_input_dimension
        if not bounds:
            return [self.__DEFAULT_BOUNDS] * dimension

        if isinstance(bounds, tuple):
            bounds: tuple[float, float]
            return [bounds] * dimension

        bounds_ = []
        for name in self.input_names:
            name_bounds = bounds.get(name, self.__DEFAULT_BOUNDS)
            bounds_.extend([name_bounds] * self.sizes[name])

        return bounds_

    def _fit(
        self,
        input_data: RealArray,
        output_data: RealArray,
    ) -> None:
        self.algo.fit(input_data, output_data)

    def _predict(
        self,
        input_data: RealArray,
    ) -> RealArray:
        return self.algo.predict(input_data).reshape((len(input_data), -1))

    def predict_std(self, input_data: DataType) -> RealArray:  # noqa: D102
        if isinstance(input_data, Mapping):
            input_data = concatenate_dict_of_arrays_to_array(
                input_data, self.input_names
            )

        input_data = atleast_2d(input_data)
        transformer = self.transformer.get(self.learning_set.INPUT_GROUP)
        if transformer:
            input_data = transformer.transform(input_data)

        output_data = self.algo.predict(input_data, return_std=True)[1]
        if output_data.ndim == 1:
            return repeat(output_data[:, newaxis], self._reduced_output_dimension, 1)

        return output_data

    @RegressionDataFormatters.format_input_output(input_axis=1)
    def compute_samples(  # noqa: D102
        self, input_data: RealArray, n_samples: int, seed: int | None = None
    ) -> RealArray:
        samples = self.algo.sample_y(
            input_data, n_samples=n_samples, random_state=self._seeder.get_seed(seed)
        )
        # sklearn.gaussian_process.GaussianProcessRegressor.sample_y
        # returns an array of shape
        # - (len(input_data), n_samples) when the output dimension is 1
        # - (len(input_data), output_dimension, n_samples) otherwise,
        # while gemseo....GaussianProcessRegressor.compute_samples
        # returns an array of shape (n_samples, len(input_data), output_dimension).
        if samples.ndim == 2:
            samples = samples[:, newaxis, :]

        return swapaxes(swapaxes(samples, 0, 2), 1, 2)
