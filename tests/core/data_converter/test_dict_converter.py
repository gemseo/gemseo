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
from typing import TYPE_CHECKING
from typing import ClassVar

import pytest
from numpy import array
from numpy import sqrt
from numpy.testing import assert_allclose

from gemseo import create_mda
from gemseo.core.data_converter.factory import DataConverterFactory
from gemseo.core.discipline.discipline import Discipline
from gemseo.core.grammar.factory import grammar_factory
from gemseo.mda.core.base import BaseMDA
from gemseo.problem.mdo.sellar import variable
from gemseo.problem.mdo.sellar.util import get_initial_data

if TYPE_CHECKING:
    from gemseo.util.typing import StrKeyMapping

gemseo_path = str(Path(__file__).parent)


@pytest.fixture
def change_grammar(monkeypatch):
    # Get the new grammar visible.
    monkeypatch.setenv("GEMSEO_PATH", gemseo_path)
    grammar_factory.update()
    DataConverterFactory().update()
    old_disc_grammar_type = Discipline.default_grammar_type
    old_mda_grammar_type = BaseMDA.default_grammar_type
    Discipline.default_grammar_type = "DictGrammar"
    BaseMDA.default_grammar_type = "DictGrammar"
    yield
    Discipline.default_grammar_type = old_disc_grammar_type
    BaseMDA.default_grammar_type = old_mda_grammar_type
    monkeypatch.delenv("GEMSEO_PATH")
    grammar_factory.update()
    DataConverterFactory().update()


class DictSellarBase(Discipline):
    """The base class for a Sellar discipline with dictionaries as inputs/outputs."""

    _input_names: ClassVar[tuple[str]]
    """The names of the input variables."""

    _output_names: ClassVar[tuple[str]]
    """The names of the output variables."""

    NAME: str
    """The name of the discipline."""

    default_grammar_type = "DictGrammar"

    def __init__(self) -> None:
        super().__init__(self.NAME)
        default_input_data = {
            name: {name: value}
            for name, value in get_initial_data(self._input_names).items()
        }
        self.io.input_grammar.update_from_data(default_input_data)
        self.io.output_grammar.update_from_data({
            name: {name: value}
            for name, value in get_initial_data(self._output_names).items()
        })
        self.io.input_grammar.defaults = default_input_data


class DictSellar1(DictSellarBase):
    """The first Sellar discipline with dictionaries as inputs/outputs."""

    _input_names: ClassVar[tuple[str, ...]] = (
        variable.x_1,
        variable.x_shared,
        variable.y_2,
    )

    _output_names: ClassVar[tuple[str]] = (variable.y_1,)

    NAME: str = "DictSellar1"

    def _run(self, input_data: StrKeyMapping) -> StrKeyMapping | None:
        x_1 = input_data[variable.x_1][variable.x_1]
        x_shared = input_data[variable.x_shared][variable.x_shared]
        y_2 = input_data[variable.y_2][variable.y_2]
        return {
            variable.y_1: {
                variable.y_1: x_shared[0] ** 2 + x_shared[1] + x_1 - 0.2 * y_2
            }
        }

    def _compute_jacobian(self, input_names, output_names) -> None:
        x_shared = self.io.input_data[variable.x_shared][variable.x_shared]

        self.jac = {name: {} for name in self._output_names}
        jac = self.jac[variable.y_1]
        jac[variable.x_1] = array([[1]])
        jac[variable.x_2] = array([[0]])
        jac[variable.x_shared] = array([[2 * x_shared[0], 1]]).T
        jac[variable.y_2] = array([[-0.2]])


class DictSellar2(DictSellarBase):
    """The second Sellar discipline with dictionaries as inputs/outputs."""

    _input_names: ClassVar[tuple[str, ...]] = (variable.x_shared, variable.y_1)

    _output_names: ClassVar[tuple[str]] = (variable.y_2,)

    NAME: str = "DictSellar2"

    def _run(self, input_data: StrKeyMapping) -> StrKeyMapping | None:
        x_shared = input_data[variable.x_shared][variable.x_shared]
        y_1 = input_data[variable.y_1][variable.y_1]
        return {
            variable.y_2: {variable.y_2: sqrt(abs(y_1)) + x_shared[0] + x_shared[1]}
        }

    def _compute_jacobian(self, input_names, output_names) -> None:
        y_1 = self.io.input_data[variable.y_1][variable.y_1]
        self.jac = {name: {} for name in self._output_names}
        jac = self.jac[variable.y_2]
        jac[variable.x_shared] = array([[1, 1]]).T
        jac[variable.y_1] = array([0.5 / sqrt(y_1)])


def test_sellar_with_dict(change_grammar):
    """Test the Sellar MDA with dictionary inputs/outputs."""
    d1 = DictSellar1()
    d2 = DictSellar2()
    mda = create_mda(
        "MDANewtonRaphson",
        [d1, d2],
    )
    res = mda.execute({variable.x_1: {variable.x_1: array([5])}})
    assert_allclose(res[variable.y_1][variable.y_1], array([5.33792068]))
    assert_allclose(res[variable.y_2][variable.y_2], array([3.31039405]))
