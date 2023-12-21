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
"""Benchmark for comparing the creation of the grammar data description."""

from __future__ import annotations

from grammar_validation import Benchmark as ValidationBenchmark
from numpy import ndarray  # noqa: F401
from numpy.typing import NDArray  # noqa: F401
from pydantic import BaseModel  # noqa: F401
from src.gemseo.utils.string_tools import MultiLineString

from gemseo.core.grammars.pydantic_grammar import PydanticGrammar
from gemseo.core.grammars.simple_grammar import SimpleGrammar

# The above imports must be kept to the exec'd code in the benchmark works.


class Benchmark(ValidationBenchmark):
    """Benchmark for preparing a grammar."""

    def _setup(self) -> None:
        super()._setup()

        # The code string must define the variable 'model'.
        code = MultiLineString()
        if self.grammar_class == PydanticGrammar:
            code.add("class model(BaseModel):")
            code.indent()
            for (
                name,
                field,
            ) in self.grammar._PydanticGrammar__model.model_fields.items():
                code.add(f"{name}: {field.outer_type_.__name__}")
        elif self.grammar_class == SimpleGrammar:
            self.args = dict(self.grammar)
            code.add("model = {")
            code.indent()
            for name, type_ in self.grammar.items():
                code.add(f"'{name}': {type_.__name__},")
            code.dedent()
            code.add("}")
        else:
            filename = "benchmark-grammar.json"
            self.grammar.write(filename)
            code.add(f"model = '{filename}'")

        self.code = str(code)

    def _show(self) -> None:
        super()._show()
        print(self.code)  # noqa: T201

    def _benchmark(self) -> None:
        exec(self.code)


if __name__ == "__main__":
    Benchmark().run()
