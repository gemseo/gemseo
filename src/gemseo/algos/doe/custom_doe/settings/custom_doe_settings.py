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
"""Settings for the custom DOE."""

from __future__ import annotations

from collections.abc import Mapping
from collections.abc import Sequence  # noqa: TC003
from pathlib import Path  # noqa: TC003
from typing import TYPE_CHECKING
from typing import Union

from pydantic import Field
from pydantic import NonNegativeInt
from pydantic import model_validator

from gemseo.algos.doe.base_doe_settings import BaseDOESettings
from gemseo.utils.pydantic_ndarray import NDArrayPydantic  # noqa: TC001

if TYPE_CHECKING:
    from typing_extensions import Self

SamplesType = Union[
    NDArrayPydantic,
    Mapping[str, NDArrayPydantic],
    Sequence[Mapping[str, NDArrayPydantic]],
]


class CustomDOE_Settings(BaseDOESettings):  # noqa: N801
    """The settings for the ``CustomDOE``."""

    _TARGET_CLASS_NAME = "CustomDOE"

    doe_file: str | Path = Field(
        default="",
        description="""The path to the file containing the input samples.

If empty, use ``samples``.""",
    )

    samples: SamplesType = Field(
        default_factory=dict,
        description="""The input samples.

They must be at least a 2D-array, a dictionary of 2D-arrays
or a list of dictionaries of 1D-arrays. If empty, use ``doe_file``.""",
    )

    delimiter: str = Field(
        default=",",
        description="""The character used to separate values.""",
    )

    comments: str | Sequence[str] = Field(
        default="#",
        description="""The (list of) characters used to indicate the start of a comment.

No comments if empty.""",
    )

    skiprows: NonNegativeInt = Field(
        default=0,
        description="""The number of first lines to skip.""",
    )

    @model_validator(mode="after")
    def __check_file_or_samples(self) -> Self:
        """Check the consistency of the ``doe_file`` and ``samples`` settings.

        Raises:
            ValueError: If both ``samples`` and ``doe_file`` are ``None``. If both
                ``samples`` and ``doe_file`` were provided.
        """
        samples = self.samples
        doe_file = self.doe_file
        if isinstance(samples, (Mapping, Sequence)):
            has_samples = bool(samples)
        else:
            has_samples = bool(samples.size)

        if (not has_samples and not doe_file) or (has_samples and doe_file):
            error_message = (
                "The algorithm CustomDOE requires either a 'doe_file' or the input"
                " 'samples' as settings."
            )
            raise ValueError(error_message)
        return self
