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
#                           documentation
#        :author: Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""The SciPy-based joint probability distribution.

:class:`.SPJointDistribution` is a :class:`.BaseJointDistribution`
based on the `SciPy <https://docs.scipy.org/doc/scipy/tutorial/stats.html>`_ library.

.. warning::

   For the moment,
   there is no copula that can be used with :class:`.SPJointDistribution`;
   if you want to introduce dependency between random variables,
   please consider :class:`.OTJointDistribution`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Any

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Sequence

    from gemseo.typing import RealArray
    from gemseo.typing import StrKeyMapping
    from gemseo.uncertainty.distributions.scipy.distribution import SPDistribution

from numpy import array

from gemseo.uncertainty.distributions.base_joint import BaseJointDistribution


class SPJointDistribution(BaseJointDistribution):
    """The SciPy-based joint probability distribution."""

    def __init__(  # noqa: D107
        self,
        distributions: Sequence[SPDistribution],
        copula: None = None,
    ) -> None:
        """
        Raises:
            NotImplementedError: When the copula is not ``None``.
        """  # noqa: D205 D212 D415
        if copula is not None:
            msg = "There is not copula distribution yet for SciPy-based distributions."
            raise NotImplementedError(msg)

        super().__init__(distributions, copula=copula)

    def _create_distribution(
        self, distribution_name: str, parameters: StrKeyMapping, **kwargs: Any
    ) -> None:
        self.distribution = self.marginals
        self._set_bounds(self.marginals)

    def compute_cdf(  # noqa: D102
        self,
        value: Iterable[float],
    ) -> RealArray:
        return array([
            marginal.distribution.cdf(value_)
            for value_, marginal in zip(value, self.marginals)
        ])

    def compute_inverse_cdf(  # noqa: D102
        self,
        value: Iterable[float],
    ) -> RealArray:
        return array([
            marginal.distribution.ppf(value_)
            for value_, marginal in zip(value, self.marginals)
        ])
