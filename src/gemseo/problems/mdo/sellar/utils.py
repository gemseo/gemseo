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
"""Utils for the customizable Sellar MDO problem."""

from __future__ import annotations

from contextlib import contextmanager
from typing import TYPE_CHECKING
from typing import Any

from numpy import array
from numpy import atleast_2d
from numpy import float64
from numpy import ndarray
from numpy import ones
from numpy import zeros

from gemseo.core.data_converters.json import JSONGrammarDataConverter
from gemseo.core.grammars.json_grammar import JSONGrammar
from gemseo.problems.mdo.sellar import WITH_2D_ARRAY
from gemseo.problems.mdo.sellar.variables import ALPHA
from gemseo.problems.mdo.sellar.variables import BETA
from gemseo.problems.mdo.sellar.variables import GAMMA
from gemseo.problems.mdo.sellar.variables import X_1
from gemseo.problems.mdo.sellar.variables import X_2
from gemseo.problems.mdo.sellar.variables import X_SHARED
from gemseo.problems.mdo.sellar.variables import Y_1
from gemseo.problems.mdo.sellar.variables import Y_2

if TYPE_CHECKING:
    from collections.abc import Iterable

    from gemseo.mda.base_mda import BaseMDA
    from gemseo.typing import RealArray


def get_initial_data(names: Iterable[str] = (), n: int = 1) -> dict[str, RealArray]:
    """Generate an initial solution for the MDO problem.

    Args:
        names: The names of the discipline inputs.
        n: The size of the local design variables and coupling variables

    Returns:
        The default values of the discipline inputs.
    """
    inputs = {
        X_1: zeros(n),
        X_2: zeros(n),
        X_SHARED: array([1.0, 0.0], dtype=float64),
        Y_1: ones(n, dtype=float64),
        Y_2: ones(n, dtype=float64),
        ALPHA: array([3.16]),
        BETA: array([24.0]),
        GAMMA: array([0.2]),
    }
    if WITH_2D_ARRAY:  # pragma: no cover
        inputs[X_SHARED] = atleast_2d(inputs[X_SHARED])
    if not names:
        return inputs
    return {name: inputs[name] for name in names if name in inputs}


def get_y_opt(mda: BaseMDA) -> ndarray:
    """Return the optimal ``y`` array.

    Args:
        mda: The mda.

    Returns:
        The optimal ``y`` array.
    """
    return array([
        mda.io.data[Y_1][0].real,
        mda.io.data[Y_2][0].real,
    ])


class DataConverterFor2DArray(JSONGrammarDataConverter):
    """A data converter where ``x_shared`` is not a ndarray."""

    def convert_value_to_array(self, name: str, value: Any) -> ndarray:  # noqa: D102 # pragma: no cover
        if name == X_SHARED:
            return value[0]
        return super().convert_value_to_array(name, value)

    def convert_array_to_value(self, name: str, array_: Any) -> Any:  # noqa: D102 # pragma: no cover
        if name == X_SHARED:
            return array([array_])
        return super().convert_array_to_value(name, array_)


@contextmanager
def set_data_converter() -> None:
    """Set the data converter according to whether 2D array shall be used."""
    if WITH_2D_ARRAY:  # pragma: no cover
        JSONGrammar.DATA_CONVERTER_CLASS = DataConverterFor2DArray
        yield
        JSONGrammar.DATA_CONVERTER_CLASS = JSONGrammarDataConverter
    else:
        yield
