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

from typing import TYPE_CHECKING

from numpy import array
from numpy import concatenate

from gemseo.core.discipline import Discipline
from gemseo.problems.uncertainty.ishigami.functions import compute_gradient
from gemseo.problems.uncertainty.ishigami.functions import compute_output

if TYPE_CHECKING:
    from collections.abc import Iterable

    from gemseo.typing import StrKeyMapping


class IshigamiDiscipline(Discipline):
    r"""The Ishigami function as a discipline."""

    def __init__(self) -> None:  # noqa: D107
        super().__init__()
        self.io.input_grammar.update_from_names(["x1", "x2", "x3"])
        self.io.output_grammar.update_from_names(["y"])
        self.io.input_grammar.defaults.update({
            name: array([0.0]) for name in self.io.input_grammar
        })

    def _run(self, input_data: StrKeyMapping) -> StrKeyMapping | None:
        return {"y": array([compute_output(concatenate(list(input_data.values())))])}

    def _compute_jacobian(
        self,
        input_names: Iterable[str] = (),
        output_names: Iterable[str] = (),
    ) -> None:
        local_data = self.io.data
        inputs_array = concatenate([local_data[name] for name in self.io.input_grammar])
        self.jac = {
            "y": {
                input_name: array([[derivative]])
                for input_name, derivative in zip(
                    self.io.input_grammar,
                    compute_gradient(inputs_array),
                )
            }
        }
