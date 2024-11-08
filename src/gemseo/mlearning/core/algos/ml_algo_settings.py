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
"""Settings for the machine learning algorithms."""

from __future__ import annotations

from collections.abc import MutableMapping
from typing import Union

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field

from gemseo.mlearning.transformers.base_transformer import BaseTransformer
from gemseo.typing import StrKeyMapping

SubTransformerType = Union[str, tuple[str, StrKeyMapping], BaseTransformer]
TransformerType = MutableMapping[str, SubTransformerType]


class BaseMLAlgoSettings(BaseModel):
    """The settings common to all the machine learning algorithms."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True, frozen=True)

    transformer: StrKeyMapping = Field(
        default_factory=dict,
        description="""The strategies to transform the variables.

The values are instances of :class:`.BaseTransformer`
while the keys are the names of
either the variables
or the groups of variables,
e.g. ``"inputs"`` or ``"outputs"``
in the case of the regression algorithms.
If a group is specified,
the :class:`.BaseTransformer` will be applied
to all the variables of this group.
If :attr:`.IDENTITY`, do not transform the variables.""",
    )

    parameters: StrKeyMapping = Field(
        default_factory=dict, description="Other parameters."
    )
