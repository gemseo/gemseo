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
from __future__ import annotations

import logging
from pathlib import Path
from typing import ClassVar
from typing import List
from typing import Optional
from typing import Sequence
from typing import TextIO
from typing import Union

from numpy import apply_along_axis
from numpy import atleast_2d
from numpy import loadtxt
from numpy import ndarray

from gemseo.algos.doe.doe_lib import DOEAlgorithmDescription
from gemseo.algos.doe.doe_lib import DOELibrary

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

    COMMENTS_KEYWORD: ClassVar[str] = "comments"
    """The name given to the string indicating a comment line."""

    DELIMITER_KEYWORD: ClassVar[str] = "delimiter"
    """The name given to the string separating two fields."""

    DOE_FILE: ClassVar[str] = "doe_file"
    """The name given to the DOE file."""

    SAMPLES: ClassVar[str] = "samples"
    """The name given to the samples."""

    SKIPROWS_KEYWORD: ClassVar[str] = "skiprows"
    """The name given to the number of skipped rows in the DOE file."""

    LIBRARY_NAME = "GEMSEO"

    def __init__(self) -> None:  # noqa:D107
        super().__init__()
        name = self.__class__.__name__
        self.algo_name = name

        desc = {
            "CustomDOE": (
                "This samples are provided "
                "either as a file in text or csv format "
                "or as a sequence of sequences of numbers."
            )
        }
        self.descriptions[name] = DOEAlgorithmDescription(
            algorithm_name=name,
            description=desc[name],
            internal_algorithm_name=name,
            library_name=name,
        )

    def _get_options(
        self,
        doe_file: str | Path | None = None,
        samples: ndarray | None = None,
        delimiter: str | None = ",",
        comments: str | Sequence[str] | None = "#",
        skiprows: int = 0,
        max_time: float = 0,
        eval_jac: bool = False,
        n_processes: int = 1,
        wait_time_between_samples: float = 0.0,
        **kwargs: OptionType,
    ) -> dict[str, OptionType]:
        """Set the options.

        Args:
            doe_file: Either a file path or the generator to read.
                If None, the samples are used and must be provided.
            samples: The samples.
                They must be at least a 2D-array.
                If None, the `doe_file` is used and must be provided.
            delimiter: The character used to separate values.
                If None, use whitespace.
            comments:  The characters or list of characters
                used to indicate the start of a comment.
                None implies no comments.
            skiprows: The number of first lines to skip.
            eval_jac: Whether to evaluate the jacobian.
            n_processes: The maximum simultaneous number of processes
                used to parallelize the execution.
            wait_time_between_samples: The waiting time between two samples.
            max_time: The maximum runtime in seconds,
                disabled if 0.
            **kwargs: The additional arguments.

        Returns:
            The processed options.
        """
        return self._process_options(
            max_time=max_time,
            doe_file=str(doe_file) if doe_file is not None else None,
            samples=samples,
            delimiter=delimiter,
            comments=comments,
            skiprows=skiprows,
            eval_jac=eval_jac,
            n_processes=n_processes,
            wait_time_between_samples=wait_time_between_samples,
            **kwargs,
        )

    def read_file(
        self,
        doe_file: str | Path | TextIO,
        delimiter: str | None = ",",
        comments: str | Sequence[str] | None = "#",
        skiprows: int = 0,
    ) -> ndarray:
        """Read a file containing several samples (one per line) and return them.

        Args:
            doe_file: Either the file, the filename, or the generator to read.
            delimiter: The character used to separate values.
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
                doe_file, comments=comments, delimiter=delimiter, skiprows=skiprows
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

    def _generate_samples(self, **options: OptionType) -> ndarray:
        """
        Returns:
            The samples.

        Raises:
            ValueError: If no `doe_file` and no `samples` are given.
                If both `doe_file` and `samples` are given.
                If the dimension of `samples` is different from the
                one of the problem.
        """  # noqa: D205, D212, D415
        error_message = (
            "The algorithm CustomDOE requires "
            "either 'doe_file' or 'samples' as option."
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
                " the samples ({}).".format(self.problem.dimension, samples.shape[1])
            )

        return apply_along_axis(
            self.problem.design_space.transform_vect, axis=1, arr=samples
        )
