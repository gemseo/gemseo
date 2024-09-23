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
#
# Copyright 2024 Capgemini Engineering
# Created on 10/09/2024, 14:25
# Contributors:
#    INITIAL AUTHORS - initial API and implementation and/or initial documentation
#        :author:  Vincent Drouet
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import annotations

from pathlib import Path
from typing import ClassVar

from numpy import array
from numpy import sqrt
from numpy.testing import assert_allclose

from gemseo import create_mda
from gemseo.core.discipline import MDODiscipline
from gemseo.problems.mdo.sellar.utils import get_initial_data
from gemseo.problems.mdo.sellar.variables import X_1
from gemseo.problems.mdo.sellar.variables import X_2
from gemseo.problems.mdo.sellar.variables import X_SHARED
from gemseo.problems.mdo.sellar.variables import Y_1
from gemseo.problems.mdo.sellar.variables import Y_2

GEMSEO_PATH = Path(__file__).parent


class DictSellarBase(MDODiscipline):
    """The base class for a Sellar discipline with dictionaries as inputs/outputs."""

    _INPUT_NAMES: ClassVar[tuple[str]]
    """The names of the input variables."""

    _OUTPUT_NAMES: ClassVar[tuple[str]]
    """The names of the output variables."""

    NAME: str
    """The name of the discipline."""

    def __init__(self) -> None:
        super().__init__(self.NAME, grammar_type="DictGrammar")
        default_inputs = {
            name: {name: value}
            for name, value in get_initial_data(self._INPUT_NAMES).items()
        }
        self.input_grammar.update_from_data(default_inputs)
        self.output_grammar.update_from_data({
            name: {name: value}
            for name, value in get_initial_data(self._OUTPUT_NAMES).items()
        })
        self.default_inputs = default_inputs


class DictSellar1(DictSellarBase):
    """The first Sellar discipline with dictionaries as inputs/outputs."""

    _INPUT_NAMES: ClassVar[tuple[str]] = (X_1, X_SHARED, Y_2)

    _OUTPUT_NAMES: ClassVar[tuple[str]] = (Y_1,)

    NAME: str = "DictSellar1"

    def _run(self) -> None:
        x_1 = self.local_data[X_1][X_1]
        x_shared = self.local_data[X_SHARED][X_SHARED]
        y_2 = self.local_data[Y_2][Y_2]

        y_1 = x_shared[0] ** 2 + x_shared[1] + x_1 - 0.2 * y_2
        self.local_data[Y_1] = {Y_1: y_1}

    def _compute_jacobian(self, input_names, output_names) -> None:
        x_shared = self.local_data[X_SHARED][X_SHARED]

        self.jac = {name: {} for name in self._OUTPUT_NAMES}
        jac = self.jac[Y_1]
        jac[X_1] = array([[1]])
        jac[X_2] = array([[0]])
        jac[X_SHARED] = array([[2 * x_shared[0], 1]]).T
        jac[Y_2] = array([[-0.2]])


class DictSellar2(DictSellarBase):
    """The second Sellar discipline with dictionaries as inputs/outputs."""

    _INPUT_NAMES: ClassVar[tuple[str]] = (X_SHARED, Y_1)

    _OUTPUT_NAMES: ClassVar[tuple[str]] = (Y_2,)

    NAME: str = "DictSellar2"

    def _run(self) -> None:
        x_shared = self.local_data[X_SHARED][X_SHARED]
        y_1 = self.local_data[Y_1][Y_1]

        y_2 = sqrt(abs(y_1)) + x_shared[0] + x_shared[1]
        self.local_data[Y_2] = {Y_2: y_2}

    def _compute_jacobian(self, input_names, output_names) -> None:
        y_1 = self.local_data[Y_1][Y_1]
        self.jac = {name: {} for name in self._OUTPUT_NAMES}
        jac = self.jac[Y_2]
        jac[X_SHARED] = array([[1, 1]]).T
        jac[Y_1] = array([0.5 / sqrt(y_1)])


def test_sellar_with_dict(monkeypatch, reset_factory):
    """Test the Sellar MDA with dictionary inputs/outputs."""
    monkeypatch.setenv("GEMSEO_PATH", GEMSEO_PATH)
    d1 = DictSellar1()
    d2 = DictSellar2()
    mda = create_mda(
        "MDANewtonRaphson",
        [d1, d2],
        grammar_type="DictGrammar",
    )
    res = mda.execute({X_1: {X_1: array([5])}})
    assert_allclose(res[Y_1][Y_1], array([5.33792068]))
    assert_allclose(res[Y_2][Y_2], array([3.31039405]))
