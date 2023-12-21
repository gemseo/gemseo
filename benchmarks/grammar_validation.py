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
"""Benchmark for comparing grammar validation."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Final

from base_benchmark import BaseBenchmark
from data_factory import DataFactory

from gemseo.core.grammars.json_grammar import JSONGrammar
from gemseo.core.grammars.pydantic_grammar import PydanticGrammar
from gemseo.core.grammars.simple_grammar import SimpleGrammar

if TYPE_CHECKING:
    from argparse import ArgumentParser

    from gemseo.core.grammars.base_grammar import BaseGrammar


class Benchmark(BaseBenchmark):
    """Benchmark for data validation."""

    DATA_CLASS = DataFactory

    __GRAMMAR_NICKNAMES_TO_CLASSES: Final[dict[str, BaseGrammar]] = {
        "simple": SimpleGrammar,
        "json": JSONGrammar,
        "pydantic": PydanticGrammar,
    }

    grammar_class: type[BaseGrammar]
    """The grammar class to benchmark."""

    grammar: BaseGrammar
    """The grammar instance to benchmark."""

    def _setup(self) -> None:
        self.DATA_CLASS.with_only_ndarrays = self._args.with_ndarrays_only
        super()._setup()
        self.data = self._data.data
        self.grammar_class = self.__GRAMMAR_NICKNAMES_TO_CLASSES[
            self._args.grammar_type
        ]
        self.grammar = self.grammar_class("grammar")
        self.grammar.update_from_data(self.data)
        if isinstance(self.grammar, JSONGrammar):
            self.grammar._create_validator()

    def _benchmark(self) -> None:
        self.grammar.validate(self.data)

    def _show(self) -> None:
        print(self._data)  # noqa: T201
        print(repr(self.grammar))  # noqa: T201

    def __str__(self) -> str:
        return (
            f"ndarrays_only_is_{self._args.with_ndarrays_only}-"
            f"{self._data.items_nb}-{self._data.keys_nb}-{self._data.depth}"
        )

    def _get_args_parser(self) -> ArgumentParser:
        parser = super()._get_args_parser()
        parser.add_argument(
            "--grammar",
            required=True,
            choices=self.__GRAMMAR_NICKNAMES_TO_CLASSES.keys(),
        )
        parser.add_argument(
            "--with-ndarrays-only",
            action="store_true",
            help="Whether to use data with ndarrays only.",
        )
        return parser


if __name__ == "__main__":
    Benchmark().run()
