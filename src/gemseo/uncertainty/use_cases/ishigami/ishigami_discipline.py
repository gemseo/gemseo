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
"""The Ishigami function as a discipline."""
from __future__ import annotations

from typing import Iterable

from numpy import array

from gemseo.core.discipline import MDODiscipline
from gemseo.uncertainty.use_cases.ishigami.functions import compute_gradient
from gemseo.uncertainty.use_cases.ishigami.functions import compute_output


class IshigamiDiscipline(MDODiscipline):
    r"""The Ishigami function as a discipline.

    .. math::
       f(x_1,_2,x_3) = \sin(x_1)+ 7\sin(x_2)^2 + 0.1x_3^4\sin(X_1)

    See :cite:`ishigami1990`.
    """

    def __init__(self) -> None:  # noqa: D107
        super().__init__()
        self.input_grammar.update(["x1", "x2", "x3"])
        self.output_grammar.update(["y"])
        self.default_inputs.update(
            {name: array([0.0]) for name in self.input_grammar.names}
        )

    def _run(self) -> None:
        self.store_local_data(y=array([compute_output(self.get_inputs_asarray())]))

    def _compute_jacobian(
        self,
        inputs: Iterable[str] | None = None,
        outputs: Iterable[str] | None = None,
    ) -> None:
        self.jac = {
            "y": {
                input_name: array([[derivative]])
                for input_name, derivative in zip(
                    self.get_input_data_names(),
                    compute_gradient(self.get_inputs_asarray()),
                )
            }
        }
