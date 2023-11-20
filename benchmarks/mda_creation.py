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
"""Benchmark for compare_dict_of_arrays."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import Final

from base_benchmark import BaseBenchmark
from data_factory import DataFactory

from gemseo.core.grammars.json_grammar import JSONGrammar
from gemseo.core.grammars.simple_grammar import SimpleGrammar
from gemseo.mda.mda_chain import MDAChain
from gemseo.problems.scalable.linear.disciplines_generator import (
    create_disciplines_from_sizes,
)

if TYPE_CHECKING:
    from argparse import ArgumentParser

    from gemseo.core.grammars.base_grammar import BaseGrammar


class Benchmark(BaseBenchmark):
    """Benchmark for data validation."""

    DATA_CLASS = DataFactory

    __GRAMMAR_NICKNAMES_TO_CLASSES: Final[dict[str, BaseGrammar]] = {
        "simple": SimpleGrammar,
        "json": JSONGrammar,
    }

    grammar_name: type[BaseGrammar]
    """The grammar class to benchmark."""

    def _setup(self) -> None:
        super()._setup()
        self.grammar_name = self.__GRAMMAR_NICKNAMES_TO_CLASSES[
            self._args.grammar
        ].__name__
        assert self._args.nb_of_disc_ios <= self._args.nb_of_total_disc_io
        self.disciplines = create_disciplines_from_sizes(
            self._args.nb_of_disc,
            nb_of_total_disc_io=self._args.nb_of_total_disc_io,
            nb_of_disc_inputs=self._args.nb_of_disc_ios,
            nb_of_disc_outputs=self._args.nb_of_disc_ios,
            unique_disc_per_output=True,
            no_self_coupled=True,
            grammar_type=self.grammar_name,
        )

    def _benchmark(self) -> None:
        MDAChain(self.disciplines, grammar_type=self.grammar_name)

    def __str__(self) -> str:
        return (
            f"{self._args.nb_of_disc}-{self._args.nb_of_total_disc_io}-"
            f"{self._args.nb_of_disc_ios}"
        )

    def _get_args_parser(self) -> ArgumentParser:
        parser = super()._get_args_parser()
        parser.add_argument(
            "--nb-of-disc",
            required=True,
            type=int,
        )
        parser.add_argument(
            "--nb-of-total-disc-io",
            required=True,
            type=int,
        )
        parser.add_argument(
            "--nb-of-disc-ios",
            required=True,
            type=int,
        )
        parser.add_argument(
            "--grammar",
            required=True,
            choices=self.__GRAMMAR_NICKNAMES_TO_CLASSES.keys(),
        )
        return parser


if __name__ == "__main__":
    Benchmark().run()
