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
"""
Build a diagonal DOE for scalable model construction
****************************************************
"""
from __future__ import absolute_import, division, print_function, unicode_literals

from builtins import range, super

from future import standard_library
from numpy import array

from gemseo.algos.doe.doe_lib import DOELibrary

standard_library.install_aliases()


from gemseo import LOGGER


class DiagonalDOE(DOELibrary):

    """Class used for creation of a diagonal DOE."""

    ALGO_LIST = ["DiagonalDOE"]
    ALGO_DESC = {}
    ALGO_DESC["DiagonalDOE"] = "Diagonal design of experiments"

    def __init__(self):
        """
        Constructor, initializes the DOE samples
        """
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
        eval_jac=False,
        n_processes=1,
        wait_time_between_samples=0.0,
        n_samples=1,
        reverse=None,
        **kwargs
    ):  # pylint: disable=W0221
        """Sets the options

        :param eval_jac: evaluate jacobian
        :type eval_jac: bool
        :param n_processes: number of processes
        :type n_processes: int
        :param wait_time_between_samples: waiting time between two samples
        :type wait_time_between_samples: float
        :param n_samples: number of samples
        :type n_samples: int
        :param reverse: list of dimensions or variables to sample from their
            upper bounds to their lower bounds. Default: None.
        :type reverse: list(str)
        :param kwargs: additional arguments

        """
        wtbs = wait_time_between_samples
        return self._process_options(
            eval_jac=eval_jac,
            n_processes=n_processes,
            wait_time_between_samples=wtbs,
            n_samples=n_samples,
            reverse=reverse,
            **kwargs
        )

    def _generate_samples(self, **options):
        """
        Generates the list of x samples

        :param options: the options dict for the algorithm,
            see associated JSON file
        """
        reverse = options.get("reverse", [])
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
