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

from typing import Callable
from typing import ClassVar
from typing import Iterable
from typing import Mapping
from typing import Tuple

import sklearn.gaussian_process
from numpy import atleast_2d
from numpy import ndarray
from numpy import newaxis
from numpy import repeat
from sklearn.gaussian_process.kernels import Kernel

from gemseo.core.dataset import Dataset
from gemseo.mlearning.core.ml_algo import DataType
from gemseo.mlearning.core.ml_algo import TransformerType
from gemseo.mlearning.regression.regression import MLRegressionAlgo
from gemseo.utils.data_conversion import concatenate_dict_of_arrays_to_array
from gemseo.utils.python_compatibility import Final

__Bounds = Tuple[float, float]


class GaussianProcessRegressor(MLRegressionAlgo):
    """Gaussian process regression model."""

    SHORT_ALGO_NAME: ClassVar[str] = "GPR"
    LIBRARY: Final[str] = "scikit-learn"
    __DEFAULT_BOUNDS: Final[tuple[float, float]] = (0.01, 100.0)

    def __init__(
        self,
        data: Dataset,
        transformer: TransformerType = MLRegressionAlgo.IDENTITY,
        input_names: Iterable[str] | None = None,
        output_names: Iterable[str] | None = None,
        kernel: Kernel | None = None,
        bounds: __Bounds | Mapping[str, __Bounds] | None = None,
        alpha: float | ndarray = 1e-10,
        optimizer: str | Callable = "fmin_l_bfgs_b",
        n_restarts_optimizer: int = 10,
        random_state: int | None = None,
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
            random_state: The seed used to initialize the centers.
                If None, the random number generator is the RandomState instance
                used by `numpy.random`.
        """
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
                (1.0,) * self._reduced_dimensions[0],
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
        else:
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
        dimension = self._reduced_dimensions[0]
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
        input_data: ndarray,
        output_data: ndarray,
    ) -> None:
        self.algo.fit(input_data, output_data)

    def _predict(
        self,
        input_data: ndarray,
    ) -> ndarray:
        output_data = self.algo.predict(input_data)
        if output_data.ndim == 1:
            output_data = output_data[:, newaxis]
        return output_data

    def predict_std(
        self,
        input_data: DataType,
    ) -> ndarray:
        """Predict the standard deviation from input data.

        The user can specify these input data either as a NumPy array,
        e.g. ``array([1., 2., 3.])``
        or as a dictionary of NumPy arrays,
        e.g.  ``{'a': array([1.]), 'b': array([2., 3.])}``.

        If the NumPy arrays are of dimension 2,
        their i-th rows represent the input data of the i-th sample;
        while if the NumPy arrays are of dimension 1,
        there is a single sample.

        Args:
            input_data: The input data.

        Returns:
            The standard deviation at the query points.

        Warning:
            If the output variables are transformed before the training stage,
            then the standard deviation is related to this transformed output space
            unlike :meth:`.predict` which returns values in the original output space.
        """
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
            output_data = repeat(
                output_data[:, newaxis], self._reduced_dimensions[1], 1
            )
        return output_data
