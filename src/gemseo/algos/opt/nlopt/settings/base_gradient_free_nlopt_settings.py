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
"""Settings for the gradient-free NLopt algorithms."""

from __future__ import annotations

from pydantic import Field
from pydantic import PositiveFloat

from gemseo.algos.opt.nlopt.settings.base_nlopt_settings import BaseNLoptSettings
from gemseo.utils.pydantic_ndarray import NDArrayPydantic  # noqa: TC001


class BaseGradientFreeNLoptSettings(BaseNLoptSettings):
    """The NLopt optimization library settings for gradient-free algorithms."""

    init_step: PositiveFloat | NDArrayPydantic[PositiveFloat] = Field(
        default=0.25,
        description="""The initial step size for derivative-free algorithms.

It can be an array of the initial steps for each dimension, or a single
number if the same step will be used for all of them.

For derivative-free local-optimization algorithms, the optimizer must
somehow decide on some initial step size to perturb `x` by when it begins
the optimization. This step size should be big enough so that the value of
the objective significantly changes, but not too big if you want to find the
local optimum nearest to x.""",
    )
