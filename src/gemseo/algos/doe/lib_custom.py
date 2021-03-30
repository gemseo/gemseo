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
#        :author: Damien Guenot
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
#        :author: Francois Gallard
"""
Run a DOE from a file containing samples values
***********************************************
"""
from __future__ import absolute_import, division, print_function, unicode_literals

from future import standard_library
from numpy import atleast_2d, loadtxt

from gemseo.algos.doe.doe_lib import DOELibrary

standard_library.install_aliases()


from gemseo import LOGGER


class CustomDOE(DOELibrary):

    """Class used for creation of DOE samples provided by user
    This samples are provided as file in text or csv format.
    """

    ALGO_LIST = ["CustomDOE"]
    DELIMITER_KEYWORD = "delimiter"
    SKIPROWS_KEYWORD = "skiprows"
    DOE_FILE = "doe_file"

    def __init__(self):
        """
        Constructor, initializes the DOE samples
        For this class of DOE library, samples are provided as file
        in text or csv format. csv file format is assume to have a header
        whereas text file (extension .txt) has not.
        """
        super(CustomDOE, self).__init__()
        self.file_dv_names_list = None

        desc = {}
        desc["CustomDOE"] = (
            "The **CustomDOE** class is used for creation"
            " of DOE samples provided by user. This samples"
            " are provided as file in text or csv format."
        )
        for algo in self.ALGO_LIST:
            self.lib_dict[algo] = {
                DOELibrary.LIB: self.__class__.__name__,
                DOELibrary.INTERNAL_NAME: algo,
                DOELibrary.DESCRIPTION: desc[algo],
            }

    def _get_options(
        self,
        doe_file,
        delimiter=",",  # pylint: disable=W0221
        comments="#",
        skiprows=0,
        eval_jac=False,
        n_processes=1,
        wait_time_between_samples=0.0,
        **kwargs
    ):
        """Sets the options

        :param doe_file: path and name of file
        :type doe_file: str
        :param delimiter: The string used to separate values.
        :type delimiter: str
        :param comments:  the characters or list of characters used to
            indicate the start of a comment
        :type comments: str
        :param skiprows: skip the first `skiprows` lines
        :type skiprows: int
        :param eval_jac: evaluate jacobian
        :type eval_jac: bool
        :param n_processes: number of processes
        :type n_processes: int
        :param wait_time_between_samples: waiting time between two samples
        :type wait_time_between_samples: float
        :param kwargs: additional arguments

        """
        wtbs = wait_time_between_samples
        return self._process_options(
            doe_file=doe_file,
            delimiter=delimiter,
            comments=comments,
            skiprows=skiprows,
            eval_jac=eval_jac,
            n_processes=n_processes,
            wait_time_between_samples=wtbs,
            **kwargs
        )

    def read_file(self, doe_file, delimiter=",", comments="#", skiprows=0):
        """Read a file containing a DOE

        :param doe_file: path and name of file
        :type doe_file: str
        :param delimiter: The string used to separate values.
        :type delimiter: str
        :param comments:  the characters or list of characters used to
            indicate the start of a comment
        :type comments: str
        :param skiprows: skip the first `skiprows` lines
        :type skiprows: int
        :returns: sample (an array of samples)
        :rtype: numpy array
        """
        try:
            samples = loadtxt(
                doe_file,
                comments=comments,
                delimiter=delimiter,
                skiprows=skiprows,
                unpack=False,
            )
            samples = atleast_2d(samples)
            if (
                samples.shape[1] != self.problem.dimension
                and self.problem.dimension == 1
            ):
                samples = samples.T
        except ValueError:
            LOGGER.error("Failed to load DOE input file : %s", str(doe_file))
            raise
        self.__check_input_dv_lenght(samples)

        # Normalize samples
        normalize_vect = self.problem.design_space.normalize_vect
        for i in range(samples.shape[0]):
            samples[i, :] = normalize_vect(samples[i, :])

        return samples

    def __check_input_dv_lenght(self, samples):
        """
        Check that file contains all variables given as design variable
        at initialization
        """
        dim = self.problem.dimension
        if samples.shape[1] != dim:
            raise ValueError(
                "Mismatch between problem design variables "
                + str(dim)
                + " and doe input file dimension : "
                + str(samples.shape[1])
            )

    def _generate_samples(self, **options):
        """
        Generates the list of x samples

        :param options: the options dict for the algorithm,
            see associated JSON file
        """

        samples = self.read_file(
            options[self.DOE_FILE],
            delimiter=options[self.DELIMITER_KEYWORD],
            skiprows=options[self.SKIPROWS_KEYWORD],
        )
        return samples
