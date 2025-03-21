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
r"""Settings of the polynomial chaos expansion model.

.. _CleaningStrategy: https://openturns.github.io/openturns/latest/user_manual/response_surface/_generated/openturns.CleaningStrategy.html
.. _FixedStrategy: http://openturns.github.io/openturns/latest/user_manual/response_surface/_generated/openturns.FixedStrategy.html
.. _LARS: https://openturns.github.io/openturns/latest/theory/meta_modeling/polynomial_sparse_least_squares.html#polynomial-sparse-least-squares
"""  # noqa: E501

from __future__ import annotations

from dataclasses import dataclass

from pydantic import Field
from pydantic import NonNegativeInt
from pydantic import PositiveFloat
from pydantic import PositiveInt

from gemseo.algos.parameter_space import ParameterSpace  # noqa: TC001
from gemseo.core.discipline.discipline import Discipline  # noqa: TC001
from gemseo.mlearning.regression.algos.base_regressor_settings import (
    BaseRegressorSettings,
)


@dataclass
class CleaningOptions:
    """The options of the `CleaningStrategy`_."""

    max_considered_terms: int = 100
    """The maximum number of coefficients of the polynomial basis to be considered."""

    most_significant: int = 20
    """The maximum number of efficient coefficients of the polynomial basis to be
    kept."""

    significance_factor: float = 1e-4
    """The threshold to select the efficient coefficients of the polynomial basis."""


class PCERegressor_Settings(BaseRegressorSettings):  # noqa: N801
    """The settings of the polynomial chaos expansion model."""

    # TODO: API: remove in gemseo v7.
    probability_space: ParameterSpace | None = Field(
        default=None,
        description="""The random input variables using :class:`.OTDistribution`.

If ``None``,
:class:`.PCERegressor` uses ``data.misc["input_space"]``
where ``data`` is the :class:`.IODataset` passed at instantiation.
""",
    )

    degree: PositiveInt = Field(
        default=2, description="The polynomial degree of the PCE."
    )

    discipline: Discipline | None = Field(
        default=None,
        description="""The discipline to be sampled.

Used only when ``use_quadrature`` is ``True`` and ``data`` is ``None``.""",
    )

    use_quadrature: bool = Field(
        default=False,
        description="""Whether to estimate the coefficients of the PCE by quadrature.

If so,
use the quadrature points stored in ``data`` or sample ``discipline``.
Otherwise,
estimate the coefficients by least-squares regression.""",
    )

    use_lars: bool = Field(
        default=False,
        description="""Whether to use the `LARS`_ algorithm.

This argument is ignored when ``use_quadrature`` is ``True``.""",
    )

    use_cleaning: bool = Field(
        default=False,
        description="""Whether to use the `CleaningStrategy`_ algorithm.

Otherwise,
use a fixed truncation strategy (`FixedStrategy`_).""",
    )

    hyperbolic_parameter: PositiveFloat = Field(
        default=1.0,
        description="""The :math:`q`-quasi norm parameter of the `hyperbolic and
anisotropic enumerate function`_, defined over the interval:math:`]0,1]`.""",
    )

    n_quadrature_points: NonNegativeInt = Field(
        default=0,
        description="""The total number of quadrature points.

These points are used
to compute the marginal number of points by input dimension
when ``discipline`` is not ``None``.
If ``0``, use :math:`(1+P)^d` points,
where :math:`d` is the dimension of the input space
and :math:`P` is the polynomial degree of the PCE.""",
    )

    cleaning_options: CleaningOptions | None = Field(
        default=None,
        description="""The options of the `CleaningStrategy`_.

        If ``None``, use :attr:`.DEFAULT_CLEANING_OPTIONS`.
        """,
    )
