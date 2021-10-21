# -*- coding: utf-8 -*-
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
"""Build a diagonal DOE for scalable model construction."""
from __future__ import division, unicode_literals

import logging
from typing import Dict, Optional, Sequence, Union

from numpy import array, ndarray

from gemseo.algos.doe.doe_lib import DOELibrary

OptionType = Optional[Union[str, int, float, bool, Sequence[str]]]

LOGGER = logging.getLogger(__name__)


class DiagonalDOE(DOELibrary):

    """Class used to create a diagonal DOE."""

    ALGO_LIST = ["DiagonalDOE"]
    ALGO_DESC = {"DiagonalDOE": "Diagonal design of experiments"}

    def __init__(self):
        super(DiagonalDOE, self).__init__()

        for algo in self.ALGO_LIST:
            description = DiagonalDOE.ALGO_DESC[algo]
            self.lib_dict[algo] = {
                DOELibrary.LIB: self.__class__.__name__,
                DOELibrary.INTERNAL_NAME: algo,
                DOELibrary.DESCRIPTION: description,
            }

    def _get_options(
        self,
        eval_jac=False,  # type: bool
        n_processes=1,  # type: int
        wait_time_between_samples=0.0,  # type: float
        n_samples=1,  # type: int
        reverse=None,  # type: Optional[Sequence[str]]
        max_time=0,  # type: int
        **kwargs  # type: OptionType
    ):  # type: (...) -> Dict[str, OptionType] # pylint: disable=W0221
        """Set the options.

        Args:
            eval_jac: Whether to evaluate jacobian.
            n_processes: The number of processes.
            wait_time_between_samples: The waiting time between two samples.
            n_samples: The number of samples.
            reverse: The dimensions or variables to sample from their
                upper bounds to their lower bounds. If None, reverse = [].
            max_time: The maximum runtime in seconds,
                disabled if 0.
            **kwargs: The additional arguments.

        Returns:
            The processed options.
        """
        wtbs = wait_time_between_samples
        return self._process_options(
            eval_jac=eval_jac,
            n_processes=n_processes,
            wait_time_between_samples=wtbs,
            n_samples=n_samples,
            reverse=reverse,
            max_time=max_time,
            **kwargs
        )

    def _generate_samples(
        self, **options  # type: OptionType
    ):  # type: (...) -> ndarray
        """Generate the DOE samples.

        Args:
            **options: The options for the algorithm,
                see the associated JSON file.

        Return:
            The DOE samples.
        """
        reverse = options.get("reverse", [])
        if reverse is None:
            reverse = []
        names = self.problem.design_space.variables_names
        sizes = self.problem.design_space.variables_sizes
        name_by_index = {}
        start = 0
        for name in names:
            for index in range(start, start + sizes[name]):
                name_by_index[index] = name
            start += sizes[name]
        samples = []
        for index in range(self.problem.dimension):
            if str(index) in reverse or name_by_index[index] in reverse:
                samples.append(
                    [
                        point / (options[self.N_SAMPLES] - 1.0)
                        for point in range(options[self.N_SAMPLES] - 1, -1, -1)
                    ]
                )
            else:
                samples.append(
                    [
                        point / (options[self.N_SAMPLES] - 1.0)
                        for point in range(0, options[self.N_SAMPLES])
                    ]
                )
        samples = array(samples).T
        return samples
