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
"""Base class for performance benchmarks."""

from __future__ import annotations

import abc
import sys
from argparse import ArgumentParser
from argparse import Namespace
from dataclasses import asdict
from dataclasses import dataclass
from pprint import pprint
from typing import Any
from typing import ClassVar

from pyperf import Runner


class BaseBenchmark(abc.ABC):
    """Abstract base class for benchmarking code."""

    DATA_CLASS: type[dataclass]
    """A dataclass with default values that contains to benchmark data.

    The fields and their default values will be exposed to the CLI arguments.
    """

    _data_defaults: dict[str, Any]
    """The default values of the benchmark data."""

    _args_parser: ArgumentParser
    """The CLI argument parser."""

    _args: Namespace
    """The parsed CLI arguments."""

    _runner: Runner
    """The pyperf runner."""

    __RECURSION_LIMIT: ClassVar[int] = 10000
    """The recursion limit, presumably for Windows."""

    def __init__(self) -> None:  # noqa: D107
        sys.setrecursionlimit(self.__RECURSION_LIMIT)
        self._data_defaults = asdict(self.DATA_CLASS())
        self._args_parser = self._get_args_parser()
        self._args = Namespace()
        self._runner = Runner(_argparser=self._args_parser, program_args=sys.argv)

    def _get_args_parser(self) -> ArgumentParser:
        """Return the CLI argument parser."""
        parser = ArgumentParser(description=self.__class__.__doc__)

        group = parser.add_mutually_exclusive_group()
        group.add_argument(
            "--run",
            action="store_true",
            help="Whether to simply run the code without benchmarking or profiling",
        )
        parser.add_argument(
            "--show",
            action="store_true",
            help="Whether to show the data to compare and exit",
        )

        for name, value in self._data_defaults.items():
            parser.add_argument(
                f"--{name}",
                default=value,
                type=type(value),
                help="default = %(default)s",
            )

        return parser

    def _setup(self) -> None:
        """Prepare the data for the benchmark."""
        self._data = self.DATA_CLASS(**{
            k: getattr(self._args, k) for k in self._data_defaults
        })

    def _parse_args(self) -> None:
        """Parse the CLI arguments."""
        self._args = self._args_parser.parse_args()

    @abc.abstractmethod
    def _benchmark(self) -> None:
        """Run the code to be benchmarked."""

    def _show(self) -> None:
        """Show the benchmark data."""
        pprint(self._data)  # noqa: T203

    def run(self) -> None:
        """Run the benchmark."""
        self._parse_args()
        self._setup()

        args = self._args

        if args.show:
            self._show()
        elif args.run:
            self._benchmark()
        else:
            self._runner._set_args(args)
            self._runner.bench_func(str(self), self._benchmark)

    def __str__(self) -> str:
        return ""
