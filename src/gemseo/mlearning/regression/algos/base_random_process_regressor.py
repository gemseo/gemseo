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
"""A base class for regressors based on a random process.

A class implementing a Gaussian process regressor must derive from it.
"""

from __future__ import annotations

from abc import abstractmethod
from typing import TYPE_CHECKING

from gemseo.mlearning.regression.algos.base_regressor import BaseRegressor
from gemseo.utils.seeder import Seeder

if TYPE_CHECKING:
    from gemseo.datasets.dataset import DataType
    from gemseo.typing import RealArray

if TYPE_CHECKING:
    from gemseo.datasets.dataset import DataType
    from gemseo.typing import RealArray


class BaseRandomProcessRegressor(BaseRegressor):
    """A base class for regressors based on a random process."""

    _seeder: Seeder
    """A seed generator."""

    def _post_init(self):
        super()._post_init()
        self._seeder = Seeder()

    @abstractmethod
    def predict_std(
        self,
        input_data: DataType,
    ) -> RealArray:
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

        .. warning::

           This statistic is expressed in relation to the transformed output space.
           You can sample the :meth:`.predict` method
           to estimate it in relation to the original output space
           if it is different from the transformed output space.
        """

    @abstractmethod
    def compute_samples(
        self, input_data: RealArray, n_samples: int, seed: int | None = None
    ) -> RealArray:
        """Sample a random vector from the conditioned Gaussian process.

        Args:
            input_data: The :math:`N` input points of dimension :math:`d`
                at which to observe the conditioned Gaussian process;
                shaped as ``(N, d)``.
            n_samples: The number of samples ``M``.
            seed: The seed for reproducible results.

        Returns:
            The output samples shaped as ``(M, N, p)``
            where ``p`` is the output dimension.
        """
