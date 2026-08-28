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
"""The variable of the releases predating the hierarchy of variables."""

from __future__ import annotations

from typing import Any
from warnings import warn

from numpy import inf
from pydantic import BaseModel

from gemseo.space._variable._base import DataType
from gemseo.space._variable._factory import VARIABLE_FACTORY


class Variable(BaseModel):
    """A variable of the releases predating the hierarchy of variables.

    A single `Variable` class used to define a variable of any data type. It has been
    replaced by one class per data type, namely
    [ContinuousVariable][gemseo.space._variable._continuous.ContinuousVariable] and
    [IntegerVariable][gemseo.space._variable._integer.IntegerVariable], built by
    [VariableFactory][gemseo.space._variable._factory.VariableFactory].

    A pickle refers to a class by name, so loading a design space pickled by such a
    release requires this name to still resolve. This class exists for that sole
    purpose: unpickling one turns it into the variable of the matching data type,
    fully validated. Do not use it for anything else.
    """

    size: Any = 1
    """The size of the variable."""

    type: Any = DataType.FLOAT
    """The type of data."""

    lower_bound: Any = -inf
    """The lower bound of the variable."""

    upper_bound: Any = inf
    """The upper bound of the variable."""

    def __setstate__(self, state: dict[str, Any]) -> None:
        """Restore the variable as the variable class pinning its data type.

        The fields of the old class are read from the state and passed to the factory,
        so that the restored variable is validated as a new one would be, e.g. its
        bounds are converted, checked and frozen. Then this instance takes the identity
        of the variable built that way, so that the caller gets it directly.

        Args:
            state: The state of the variable, as pickled by a release predating the
                hierarchy of variables.
        """
        fields = state.get("__dict__", {})
        variable = VARIABLE_FACTORY.create(
            fields.get("type", DataType.FLOAT),
            size=fields.get("size", 1),
            lower_bound=fields.get("lower_bound", -inf),
            upper_bound=fields.get("upper_bound", inf),
        )
        warn(
            "The class 'gemseo.space._variable.Variable' is deprecated; "
            f"the variable has been restored as a {type(variable).__name__}. "
            "Save the design space again "
            "to store it with the current variable classes.",
            DeprecationWarning,
            stacklevel=2,
        )
        # Take the identity of the variable built by the factory.
        # The private attributes of the old class, held by the state, are dropped:
        # the new classes define none.
        # These attributes are set through object
        # because the new classes are frozen.
        object.__setattr__(self, "__dict__", dict(variable.__dict__))
        object.__setattr__(
            self, "__pydantic_fields_set__", set(variable.__pydantic_fields_set__)
        )
        object.__setattr__(self, "__pydantic_extra__", variable.__pydantic_extra__)
        object.__setattr__(self, "__pydantic_private__", variable.__pydantic_private__)
        object.__setattr__(self, "__class__", type(variable))
