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
"""Settings of the SVM classification algorithm."""

from __future__ import annotations

from typing import Annotated
from typing import Callable

from pydantic import Field
from pydantic import NonNegativeInt
from pydantic import PositiveFloat
from pydantic import WithJsonSchema

from gemseo.mlearning.classification.algos.base_classifier_settings import (
    BaseClassifierSettings,
)
from gemseo.utils.seeder import SEED


class SVMClassifier_Settings(BaseClassifierSettings):  # noqa: N801
    """The settings of the SV classification algorithm."""

    C: PositiveFloat = Field(
        default=1.0, description="The inverse L2 regularization parameter."
    )

    kernel: str | Annotated[Callable, WithJsonSchema({})] = Field(
        default="rbf",
        description="""The name of the kernel or a callable for the SVM.

Examples of names: "linear", "poly", "rbf", "sigmoid", "precomputed".""",
    )

    probability: bool = Field(
        default=False, description="Whether to enable the probability estimates."
    )

    random_state: NonNegativeInt | None = Field(
        default=SEED,
        description="""The random state parameter.

If ``None``, use the global random state instance from ``numpy.random``.
Creating the model multiple times will produce different results.
If ``int``, use a new random number generator seeded by this integer.
This will produce the same results.""",
    )
