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
"""Benchmark for comparing grammar creation."""

from __future__ import annotations

from grammar_validation import Benchmark as ValidationBenchmark

from gemseo.core.grammars.json_grammar import JSONGrammar
from gemseo.core.grammars.pydantic_grammar import PydanticGrammar
from gemseo.core.grammars.simple_grammar import SimpleGrammar


class Benchmark(ValidationBenchmark):
    """Benchmark for creating a grammar."""

    def _setup(self) -> None:
        super()._setup()

        if self.grammar_class == PydanticGrammar:
            self.init_data = self.grammar._PydanticGrammar__model
        elif self.grammar_class == SimpleGrammar:
            self.init_data = self.grammar
        elif self.grammar_class == JSONGrammar:
            filename = "benchmark-grammar.json"
            self.grammar.write(filename)
            self.init_data = filename

    def _benchmark(self) -> None:
        self.grammar_class("grammar", self.init_data)


if __name__ == "__main__":
    Benchmark().run()
