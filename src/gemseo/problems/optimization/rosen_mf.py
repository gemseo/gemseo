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
#    INITIAL AUTHORS - API and implementation and/or documentation
#        :author: Damien Guenot
#        :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""A multi-fidelity Rosenbrock discipline."""

from __future__ import annotations

from typing import TYPE_CHECKING

from numpy import atleast_2d
from numpy import zeros
from scipy.optimize import rosen
from scipy.optimize import rosen_der

from gemseo.core.discipline import Discipline

if TYPE_CHECKING:
    from collections.abc import Iterable

    from gemseo.typing import StrKeyMapping


class RosenMF(Discipline):
    r"""A multi-fidelity Rosenbrock discipline.

    Its expression is :math:`\mathrm{fidelity} * \mathrm{Rosenbrock}(x)`
    where both :math:`\mathrm{fidelity}` and :math:`x` are provided as input data.
    """

    auto_detect_grammar_files = True

    def __init__(self, dimension: int = 2) -> None:
        """
        Args:
            dimension: The dimension of the design space.
        """  # noqa: D205 D212
        super().__init__()
        self.io.input_grammar.defaults = {"x": zeros(dimension), "fidelity": 1.0}

    def _run(self, input_data: StrKeyMapping) -> StrKeyMapping | None:
        fidelity = input_data["fidelity"]
        x_val = input_data["x"]
        return {"rosen": fidelity * rosen(x_val)}

    def _compute_jacobian(
        self,
        input_names: Iterable[str] = (),
        output_names: Iterable[str] = (),
    ) -> None:
        x_val = self.io.data["x"]
        fidelity = self.io.data["fidelity"]
        self.jac = {
            "rosen": {
                "x": atleast_2d(fidelity * rosen_der(x_val)),
                "fidelity": atleast_2d(rosen(x_val)),
            }
        }
