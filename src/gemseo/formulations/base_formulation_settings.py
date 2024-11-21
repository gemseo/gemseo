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
"""Base class for the settings of the formulations."""

from __future__ import annotations

from collections.abc import Sequence
from typing import ClassVar

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field


class BaseFormulationSettings(BaseModel):
    """Base class for the settings of the formulations."""

    _TARGET_CLASS_NAME: ClassVar[str]
    """The name of the formulation class."""

    model_config = ConfigDict(extra="forbid", arbitrary_types_allowed=True)

    differentiated_input_names_substitute: Sequence[str] = Field(
        default=(),
        description=r"""The names of the discipline inputs
with respect to which to differentiate the discipline outputs
used as objective, constraints and observables.
If empty, consider the inputs of these functions.
More precisely,
for each function,
an :class:`.MDOFunction` is built from the ``disciplines``,
which depend on input variables :math:`x_1,\ldots,x_d,x_{d+1}`,
and over an input space
spanned by the input variables :math:`x_1,\ldots,x_d`
and depending on both the MDO formulation and the ``design_space``.
Then,
the methods :meth:`.MDOFunction.evaluate` and :meth:`.MDOFunction.jac`
are called at a given point of the input space
and return the output value and the Jacobian matrix,
i.e. the matrix concatenating the partial derivatives
with respect to the inputs :math:`x_1,\ldots,x_d`
at this point of the input space.
This argument can be used to compute the matrix
concatenating the partial derivatives
at the same point of the input space
but with respect to custom inputs,
e.g. :math:`x_{d-1}` and :math:`x_{d+1}`.
Mathematically speaking,
this matrix returned by :meth:`.MDOFunction.jac`
is no longer a Jacobian.""",
    )
