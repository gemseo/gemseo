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
"""The DOE algorithm to compute the Sobol' indices."""

from __future__ import annotations

from typing import Any

from numpy import array
from numpy import ndarray
from openturns import SobolIndicesExperiment

from gemseo.algos.doe._openturns.base_ot_doe import BaseOTDOE


class OTSobolDOE(BaseOTDOE):
    """The DOE algorithm to compute the Sobol' indices.

    .. note:: This class is a singleton.
    """

    def generate_samples(  # noqa: D102
        self, n_samples: int, dimension: int, **options: Any
    ) -> ndarray:
        # If eval_second_order is set to False, the input design is of size N(2+n_X).
        # If eval_second_order is set to False,
        #   if n_X = 2, the input design is of size N(2+n_X).
        #   if n_X != 2, the input design is of size N(2+2n_X).
        # Ref: https://openturns.github.io/openturns/latest/user_manual/_generated/
        # openturns.SobolIndicesExperiment.html#openturns.SobolIndicesExperiment
        eval_second_order = options["eval_second_order"]
        if eval_second_order and dimension > 2:
            sub_sample_size = int(n_samples / (2 * dimension + 2))
        else:
            sub_sample_size = int(n_samples / (dimension + 2))

        return array(
            SobolIndicesExperiment(
                self._get_uniform_distribution(dimension),
                sub_sample_size,
                eval_second_order,
            ).generate()
        )
