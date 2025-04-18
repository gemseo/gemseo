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
from collections.abc import Mapping
from collections.abc import Sequence
from pathlib import Path
from typing import TYPE_CHECKING
from typing import ClassVar
from typing import Final
from typing import Optional
from typing import TextIO
from typing import Union

from numpy import apply_along_axis
from numpy import ndarray
from numpy import vstack
from pandas import read_csv

from gemseo.algos.doe.base_doe_library import BaseDOELibrary
from gemseo.algos.doe.base_doe_library import DOEAlgorithmDescription
from gemseo.algos.doe.custom_doe.settings.custom_doe_settings import CustomDOE_Settings
from gemseo.typing import RealArray

if TYPE_CHECKING:
    from gemseo.algos.design_space import DesignSpace

OptionType = Optional[Union[str, int, float, bool, list[str], Path, TextIO, RealArray]]

LOGGER = logging.getLogger(__name__)


class CustomDOE(BaseDOELibrary):
    """A design of experiments from samples provided as a file or an array.

    The samples are provided either as a file in text or csv format or as a sequence of
    sequences of numbers, e.g. a 2D numpy array.

    A csv file format is assumed to have a header whereas a text file (extension .txt)
    does not.
    """

    _COMMENTS_KEYWORD: Final[str] = "comments"
    _DELIMITER_KEYWORD: Final[str] = "delimiter"
    _DOE_FILE: Final[str] = "doe_file"
    _SAMPLES: Final[str] = "samples"
    _SKIPROWS_KEYWORD: Final[str] = "skiprows"

    _USE_UNIT_HYPERCUBE: ClassVar[bool] = False

    ALGORITHM_INFOS: ClassVar[dict[str, DOEAlgorithmDescription]] = {
        "CustomDOE": DOEAlgorithmDescription(
            algorithm_name="CustomDOE",
            description=(
                "This samples are provided "
                "either as a file in text or csv format "
                "or as a sequence of sequences of numbers."
            ),
            internal_algorithm_name="CustomDOE",
            library_name="CustomDOE",
            Settings=CustomDOE_Settings,
        )
    }

    def __init__(self, algo_name: str = "CustomDOE") -> None:  # noqa:D107
        super().__init__(algo_name)

    @staticmethod
    def read_file(
        doe_file: str | Path | TextIO,
        delimiter: str = ",",
        comments: str | Sequence[str] | None = "#",
        skiprows: int = 0,
    ) -> RealArray:
        """Read a file containing several samples (one per line) and return them.

        Args:
            doe_file: Either the file, the filename, or the generator to read.
            delimiter: The character used to separate values.
            comments:  The characters or list of characters
                used to indicate the start of a comment.
                ``None`` implies no comments.
            skiprows: Skip the first ``skiprows`` lines.

        Returns:
            The samples.
        """
        try:
            samples = read_csv(
                doe_file,
                delimiter=delimiter,
                skiprows=skiprows,
                header=None,
                comment=comments,
            ).to_numpy()
        except Exception:
            LOGGER.exception("Failed to load the DOE file %s", doe_file)
            raise

        return samples

    def _generate_unit_samples(
        self, design_space: DesignSpace, **settings: OptionType
    ) -> RealArray:
        """
        Raises:
            ValueError: If the dimension of ``samples`` is different from the
                one of the problem.
        """  # noqa: D205, D212, D415
        samples = settings.get(self._SAMPLES)
        doe_file = settings.get(self._DOE_FILE)
        dimension = design_space.dimension
        if doe_file:
            samples = self.read_file(
                doe_file,
                comments=settings[self._COMMENTS_KEYWORD],
                delimiter=settings[self._DELIMITER_KEYWORD],
                skiprows=settings[self._SKIPROWS_KEYWORD],
            )

        if isinstance(samples, Mapping):
            samples = design_space.convert_dict_to_array(samples)
        elif not isinstance(samples, ndarray):
            samples = vstack([
                design_space.convert_dict_to_array(sample) for sample in samples
            ])

        if samples.shape[1] != dimension:
            msg = (
                f"Dimension mismatch between the variables space ({dimension}) "
                f"and the samples ({samples.shape[1]})."
            )
            raise ValueError(msg)

        return apply_along_axis(design_space.transform_vect, axis=1, arr=samples)
