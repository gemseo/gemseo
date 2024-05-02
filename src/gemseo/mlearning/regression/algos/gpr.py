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

Then, the GP conditioned by the learning set
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

from collections.abc import Iterable
from collections.abc import Mapping
from typing import TYPE_CHECKING
from typing import Callable
from typing import ClassVar
from typing import Final

import sklearn.gaussian_process
from numpy import atleast_2d
from numpy import newaxis
from numpy import repeat

from gemseo.mlearning.regression.algos.base_random_process_regressor import (
    BaseRandomProcessRegressor,
)
from gemseo.utils.data_conversion import concatenate_dict_of_arrays_to_array
from gemseo.utils.seeder import SEED

if TYPE_CHECKING:
    from sklearn.gaussian_process.kernels import Kernel

    from gemseo.datasets.io_dataset import IODataset
    from gemseo.mlearning.core.algos.ml_algo import DataType
    from gemseo.mlearning.core.algos.ml_algo import TransformerType
    from gemseo.typing import RealArray

__Bounds = tuple[float, float]


class GaussianProcessRegressor(BaseRandomProcessRegressor):
    """Gaussian process regression model."""

    SHORT_ALGO_NAME: ClassVar[str] = "GPR"
    LIBRARY: ClassVar[str] = "scikit-learn"
    __DEFAULT_BOUNDS: Final[tuple[float, float]] = (0.01, 100.0)

    def __init__(
        self,
        data: IODataset,
        transformer: TransformerType = BaseRandomProcessRegressor.IDENTITY,
        input_names: Iterable[str] | None = None,
        output_names: Iterable[str] | None = None,
        kernel: Kernel | None = None,
        bounds: __Bounds | Mapping[str, __Bounds] | None = None,
        alpha: float | RealArray = 1e-10,
        optimizer: str | Callable = "fmin_l_bfgs_b",
        n_restarts_optimizer: int = 10,
        random_state: int | None = SEED,
    ) -> None:
        """
        Args:
            kernel: The kernel specifying the covariance model.
                If ``None``, use a Matérn(2.5).
            bounds: The lower and upper bounds of the parameter length scales
                when ``kernel`` is ``None``.
                Either a unique lower-upper pair common to all the inputs
                or lower-upper pairs for some of them.
                When ``bounds`` is ``None`` or when an input has no pair,
                the lower bound is 0.01 and the upper bound is 100.
            alpha: The nugget effect to regularize the model.
            optimizer: The optimization algorithm to find the parameter length scales.
            n_restarts_optimizer: The number of restarts of the optimizer.
            random_state: The random state passed to the random number generator.
                Use an integer for reproducible results.
        """  # noqa: D205 D212
        super().__init__(
            data,
            transformer=transformer,
            input_names=input_names,
            output_names=output_names,
            kernel=kernel,
            alpha=alpha,
            optimizer=optimizer,
            n_restarts_optimizer=n_restarts_optimizer,
            random_state=random_state,
        )

        if kernel is None:
            kernel = sklearn.gaussian_process.kernels.Matern(
                (1.0,) * self._reduced_input_dimension,
                self.__compute_parameter_length_scale_bounds(bounds),
                nu=2.5,
            )

        self.algo = sklearn.gaussian_process.GaussianProcessRegressor(
            kernel=kernel,
            alpha=alpha,
            optimizer=optimizer,
            n_restarts_optimizer=n_restarts_optimizer,
            random_state=random_state,
        )
        self.parameters["kernel"] = kernel.__class__.__name__

    @property
    def kernel(self):  # (...) -> Kernel
        """The kernel used for prediction."""
        if self.is_trained:
            return self.algo.kernel_
        return self.algo.kernel

    def __compute_parameter_length_scale_bounds(
        self,
        bounds: __Bounds | Mapping[str, __Bounds] | None,
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
        if bounds is None:
            return [self.__DEFAULT_BOUNDS] * dimension

        if isinstance(bounds, tuple):
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

    def compute_samples(  # noqa: D102
        self, input_data: RealArray, n_samples: int, seed: int = SEED
    ) -> tuple[RealArray]:
        samples = self.algo.sample_y(input_data, n_samples=n_samples, random_state=seed)
        if samples.ndim == 2:
            return (samples,)

        return tuple(samples[:, i, :] for i in range(self._reduced_output_dimension))
