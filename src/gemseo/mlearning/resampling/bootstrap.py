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
"""A bootstrap tool for resampling and surrogate modeling."""

from __future__ import annotations

from typing import TYPE_CHECKING

from numpy import setdiff1d
from numpy import unique
from numpy.random import default_rng

from gemseo import SEED
from gemseo.mlearning.resampling.resampler import Resampler
from gemseo.mlearning.resampling.split import Split
from gemseo.mlearning.resampling.splits import Splits

if TYPE_CHECKING:
    from numpy.typing import NDArray


class Bootstrap(Resampler):
    """A bootstrap tool for resampling and surrogate modeling."""

    def __init__(
        self,
        sample_indices: NDArray[int],
        n_replicates: int = 100,
        seed: int | None = SEED,
    ) -> None:
        """
        Args:
            n_replicates: The number of bootstrap replicates.
        """  # noqa: D205 D212
        super().__init__(sample_indices, n_splits=n_replicates, seed=seed)

    def _create_splits(self) -> Splits:
        splits = []
        n_samples = len(self._sample_indices)
        generator = default_rng(self._seed)
        for _ in range(self._n_splits):
            learning_sample_indices = self._sample_indices[
                unique(generator.choice(n_samples, n_samples))
            ]
            splits.append(
                Split(
                    learning_sample_indices,
                    setdiff1d(self._sample_indices, learning_sample_indices),
                )
            )

        return Splits(*splits)

    @property
    def n_replicates(self) -> int:
        """The number of bootstrap replicates."""
        return self._n_splits
