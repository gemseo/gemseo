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
"""Design of experiments from custom data."""
from __future__ import division, unicode_literals

import logging
from typing import Dict, List, Optional, Sequence, TextIO, Union

from numpy import apply_along_axis, atleast_2d, loadtxt, ndarray

from gemseo.algos.doe.doe_lib import DOELibrary
from gemseo.utils.py23_compat import Path

OptionType = Optional[Union[str, int, float, bool, List[str], Path, TextIO, ndarray]]

LOGGER = logging.getLogger(__name__)


class CustomDOE(DOELibrary):
    """A design of experiments from samples provided as a file or an array.

    The samples are provided
    either as a file in text or csv format
    or as a sequence of sequences of numbers,
    e.g. a 2D numpy array.

    A csv file format is assumed to have a header
    whereas a text file (extension .txt) does not.
    """

    ALGO_LIST = ["CustomDOE"]
    COMMENTS_KEYWORD = "comments"
    DELIMITER_KEYWORD = "delimiter"
    SKIPROWS_KEYWORD = "skiprows"
    DOE_FILE = "doe_file"
    SAMPLES = "samples"

    def __init__(self):  # type: (...) -> None
        super(CustomDOE, self).__init__()
        self.file_dv_names_list = None

        desc = {
            "CustomDOE": (
                "This samples are provided "
                "either as a file in text or csv format "
                "or as a sequence of sequences of numbers."
            )
        }
        for algo in self.ALGO_LIST:
            self.lib_dict[algo] = {
                DOELibrary.LIB: self.__class__.__name__,
                DOELibrary.INTERNAL_NAME: algo,
                DOELibrary.DESCRIPTION: desc[algo],
            }

    def _get_options(
        self,
        doe_file=None,  # type: Optional[Union[str, Path, TextIO]]
        samples=None,  # type: Optional[ndarray]
        delimiter=",",  # type: Optional[str]
        comments="#",  # type: Optional[Union[str,Sequence[str]]]
        skiprows=0,  # type: int
        max_time=0,  # type: float
        eval_jac=False,  # type: bool
        n_processes=1,  # type: int
        wait_time_between_samples=0.0,  # type: float
        **kwargs
    ):  # type: (...) -> Dict[str,OptionType]
        """Sets the options.

        :param doe_file: Either the file, the filename, or the generator to read.
        :type doe_file: str
        :param samples: The samples.
        :type samples: array
        :param delimiter: The string used to separate values.
            If None, use whitespace.
        :type delimiter: str
        :param comments:  The characters or list of characters
            used to indicate the start of a comment.
            None implies no comments.
        :type comments: str
        :param skiprows: skip the first `skiprows` lines
        :type skiprows: int
        :param eval_jac: evaluate jacobian
        :type eval_jac: bool
        :param n_processes: number of processes
        :type n_processes: int
        :param wait_time_between_samples: waiting time between two samples
        :type wait_time_between_samples: float
        :param max_time: maximum runtime in seconds,
            disabled if 0 (Default value = 0)
        :type max_time: float
        :param kwargs: additional arguments
        """
        wtbs = wait_time_between_samples
        return self._process_options(
            max_time=max_time,
            doe_file=doe_file,
            samples=samples,
            delimiter=delimiter,
            comments=comments,
            skiprows=skiprows,
            eval_jac=eval_jac,
            n_processes=n_processes,
            wait_time_between_samples=wtbs,
            **kwargs
        )

    def read_file(
        self,
        doe_file,  # type: Union[str, Path, TextIO]
        delimiter=",",  # type: Optional[str]
        comments="#",  # type: Optional[Union[str,Sequence[str]]]
        skiprows=0,  # type: int
    ):  # type: (...) -> ndarray
        """Read a file containing several samples (one per line) and return them.

        Args:
            doe_file: Either the file, the filename, or the generator to read.
            delimiter: The string used to separate values.
                If None, use whitespace.
            comments:  The characters or list of characters
                used to indicate the start of a comment.
                None implies no comments.
            skiprows: Skip the first `skiprows` lines.

        Returns:
            The samples.
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
            LOGGER.error("Failed to load DOE input file: %s", doe_file)
            raise

        return samples

    def _generate_samples(
        self, **options  # type: OptionType
    ):  # type: (...) -> ndarray
        """
        Raises:
            ValueError: If the dimension is different from the problem one.
        """
        error_message = (
            "The algorithm CustomDOE requires "
            "either 'doe_file' or 'samples' as option"
        )
        samples = options.get(self.SAMPLES)
        if samples is None:
            doe_file = options.get(self.DOE_FILE)
            if doe_file is None:
                raise ValueError(error_message)
            samples = self.read_file(
                doe_file,
                comments=options[self.COMMENTS_KEYWORD],
                delimiter=options[self.DELIMITER_KEYWORD],
                skiprows=options[self.SKIPROWS_KEYWORD],
            )
        else:
            if options.get(self.DOE_FILE) is not None:
                raise ValueError(error_message)

        if samples.shape[1] != self.problem.dimension:
            raise ValueError(
                "Dimension mismatch between the problem ({}) and "
                " the samples ({})".format(self.problem.dimension, samples.shape[1])
            )

        samples = apply_along_axis(
            self.problem.design_space.normalize_vect, axis=1, arr=samples
        )

        return samples
