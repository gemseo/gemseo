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
"""Settings of the RBF network for regression."""

from __future__ import annotations

from typing import Annotated
from typing import Callable

from pydantic import Field
from pydantic import WithJsonSchema
from strenum import StrEnum

from gemseo.mlearning.regression.algos.base_regressor_settings import (
    BaseRegressorSettings,
)
from gemseo.utils.pydantic_ndarray import NDArrayPydantic  # noqa: TC001


class RBF(StrEnum):
    """The radial basis functions."""

    MULTIQUADRIC = "multiquadric"
    INVERSE_MULTIQUADRIC = "inverse_multiquadric"
    GAUSSIAN = "gaussian"
    LINEAR = "linear"
    CUBIC = "cubic"
    QUINTIC = "quintic"
    THIN_PLATE = "thin_plate"


# TODO: API: remove Function.
Function = RBF


class RBFRegressor_Settings(BaseRegressorSettings):  # noqa: N801
    """The settings of the RBF network for regression."""

    function: RBF | Annotated[Callable[[float, float], float], WithJsonSchema({})] = (
        Field(
            default=RBF.MULTIQUADRIC,
            description=r"""The radial basis function.

This function takes a radius :math:`r` as input,
representing a distance between two points.
If it is a string,
then it must be one of the following:

- ``"multiquadric"`` for :math:`\sqrt{(r/\epsilon)^2 + 1}`,
- ``"inverse"`` for :math:`1/\sqrt{(r/\epsilon)^2 + 1}`,
- ``"gaussian"`` for :math:`\exp(-(r/\epsilon)^2)`,
- ``"linear"`` for :math:`r`,
- ``"cubic"`` for :math:`r^3`,
- ``"quintic"`` for :math:`r^5`,
- ``"thin_plate"`` for :math:`r^2\log(r)`.

If it is a callable,
then it must take the two arguments ``self`` and ``r`` as inputs,
e.g. ``lambda self, r: sqrt((r/self.epsilon)**2 + 1)``
for the multiquadric function.
The epsilon parameter will be available as ``self.epsilon``.
Other keyword arguments passed in will be available as well.""",
        )
    )

    der_function: (
        Annotated[Callable[[NDArrayPydantic], NDArrayPydantic], WithJsonSchema({})]
        | None
    ) = Field(
        default=None,
        description=r"""The derivative of the radial basis function.

Only to be provided if ``function`` is a callable
and if the use of the model with its derivative is required.
If ``None`` and if ``function`` is a callable,
an error will be raised.
If ``None`` and if ``function`` is a string,
the class will look for its internal implementation
and will raise an error if it is missing.
The ``der_function`` shall take three arguments
(``input_data``, ``norm_input_data``, ``eps``).
For an RBF of the form function(:math:`r`),
der_function(:math:`x`, :math:`|x|`, :math:`\epsilon`) shall
return :math:`\epsilon^{-1} x/|x| f'(|x|/\epsilon)`.""",
    )

    epsilon: float | None = Field(
        default=None,
        description="""An adjustable constant for Gaussian or multiquadric functions.

If ``None``, use the average distance between input data.""",
    )

    smooth: float = Field(
        default=0.0,
        description="""The degree of smoothness.

``0`` involves an interpolation of the learning points.""",
    )

    norm: (
        str
        | Annotated[
            Callable[[NDArrayPydantic, NDArrayPydantic], float], WithJsonSchema({})
        ]
    ) = Field(
        default="euclidean",
        description="""The distance metric.

Either a distance function name `known by SciPy
<https://docs.scipy.org/doc/scipy/reference/generated/
scipy.spatial.distance.cdist.html>`_
or a function that computes the distance between two points.""",
    )
