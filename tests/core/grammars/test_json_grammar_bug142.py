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
#    INITIAL AUTHORS - initial API and implementation and/or initial
#                         documentation
#        :author: Jean-Christophe Giret
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Reproducer for issue #142."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from gemseo.core.chains.chain import MDOChain
from gemseo.core.discipline import Discipline

if TYPE_CHECKING:
    from gemseo.typing import StrKeyMapping

TEST_PATH = Path(__file__).parent / "data"


class _MyDisciplineA(Discipline):
    """A test class."""

    def __init__(self) -> None:
        super().__init__()
        self.io.input_grammar.update_from_names(["A"])
        self.io.output_grammar.update_from_file(TEST_PATH / "grammar_test_bug142.json")

    def _run(self, input_data: StrKeyMapping):
        pass


class _MyDisciplineB(Discipline):
    """A test class."""

    def __init__(self) -> None:
        super().__init__()
        self.io.input_grammar.update_from_file(TEST_PATH / "grammar_test_bug142.json")
        self.io.output_grammar.update_from_names(["B"])

    def _run(self, input_data: StrKeyMapping):
        pass


def test_bug142() -> None:
    """This test is a reproducer for issue #142.

    An AttributeError was raised at the MDOChain construction,
    during the creation of the MDOChain grammar,
    from the disciplines ones.
    The issue was observed when specific conditions were met:
        - two or more disciplines.
        - couplings defined ONLY with JSON Grammar at discipline level.
        - couplings set as non-required in the JSON Grammar.

    Such a condition was not properly handled
    and led to an unexpected AttributeException error.
    """
    disciplines = [_MyDisciplineA(), _MyDisciplineB()]
    MDOChain(disciplines)
