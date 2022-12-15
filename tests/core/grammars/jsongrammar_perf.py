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
#        :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import annotations

import argparse
import cProfile
import string
import sys
import timeit
from itertools import combinations
from itertools import islice

import numpy as np
from gemseo.core.discipline import MDODiscipline
from gemseo.core.grammars.json_grammar import JSONGrammar
from gemseo.mda.mda_chain import MDAChain
from numpy import ones
from numpy.random import choice
from numpy.random import seed
from pyperf import Runner

seed(1)

DEFAULT_ARGS = dict(
    nb_of_disc=10,
    nb_of_total_disc_io=1000,
    nb_of_disc_io=20,
    data_size=1000,
)

ALPHABET = np.array(list(string.ascii_uppercase))


def generate_bench_many_io() -> JSONGrammar:
    """Create a JSON grammar.

    Returns:
        A JSON grammar.
    """
    grammar = JSONGrammar("manyinpt")
    grammar.update(["t"])
    return grammar


def test_bench_many_io():
    grammar = generate_bench_many_io()

    ref_check_times = {}
    for n_t in (10, 1000, 100000):
        data_dict = {"t": ones(n_t)}

        def run_check():
            return grammar.validate(data_dict)  # noqa: B023

        tref = timeit.timeit(stmt=run_check, number=100)
        ref_check_times[n_t] = tref

    return ref_check_times


def test_large_data_validation(sizes=(10, 1000, 100000), n_repeats=5):
    """Benchmarks the validation time of large input data."""
    ref_check_times = {}
    grammar = generate_bench_many_io()
    for n_t in sizes:
        inputs = {"t": ones(n_t)}

        def create_chain():
            grammar.validate(inputs)  # noqa: B023

        tref = timeit.timeit(stmt=create_chain, number=n_repeats)
        ref_check_times[n_t] = tref / n_repeats

    print(ref_check_times)  # noqa: T201
    return ref_check_times


class BaseBenchmarkee:
    """Abstract base class for benchmarked code."""

    def __init__(self):
        self.setup()

    def setup(self):
        """Prepare data for the benchmarked run method."""

    def run(self):
        """Run the code to be benchmarked."""

    def __str__(self):
        return ""


class ManyDisciplinesBenchmark(BaseBenchmarkee):
    """To benchmark many disciplines classes."""

    ALPHABET = np.array(list(string.ascii_uppercase))

    def __init__(
        self, class_, nb_of_disc, nb_of_total_disc_io, nb_of_disc_io, data_size
    ):
        """
        Args:
            class_: The discipline class to benchmark.
            nb_of_disc: The number of disciplines.
            nb_of_total_disc_io: The total number of disciplines inputs and outputs.
            nb_of_disc_io: The number of inputs and outputs per disciplines.
            data_size: The size of one input data.
        """
        assert nb_of_total_disc_io > nb_of_disc_io
        self.class_ = class_
        self.nb_of_disc = nb_of_disc
        self.nb_of_total_disc_io = nb_of_total_disc_io
        self.nb_of_disc_io = nb_of_disc_io
        self.data_size = data_size
        self.disciplines = {}
        super().__init__()

    @classmethod
    def __get_disc_names(cls, total_len):
        """Return the names of the disciplines."""
        n_letters = 1

        while len(cls.ALPHABET) ** n_letters < total_len:
            n_letters += 1

        return [
            "".join(c) for c in islice(combinations(cls.ALPHABET, n_letters), total_len)
        ]

    @staticmethod
    def __disc_run(disc):
        """Set the local data of a discipline."""
        data = ones(1)
        disc.local_data = {key: data for key in disc.get_output_data_names()}

    def setup(self):
        """Prepare the data for running the benchmark."""
        disc_names = self.__get_disc_names(self.nb_of_disc)
        input_data = ones(self.data_size)
        disciplines = {}
        for disc_name in disc_names:
            # Choose inputs among all io
            in_names = [
                str(i) for i in choice(self.nb_of_total_disc_io, self.nb_of_disc_io)
            ]
            out_names = [
                str(i) for i in choice(self.nb_of_total_disc_io, self.nb_of_disc_io)
            ]
            disc = MDODiscipline(disc_name)
            disc._run = self.__disc_run
            disc.input_grammar.initialize_from_data_names(in_names)
            disc.output_grammar.initialize_from_data_names(out_names)
            disc.default_inputs = {
                key: input_data for key in disc.get_input_data_names()
            }
            disciplines[disc_name] = disc

        self.disciplines = disciplines

    def run(self):
        """Run the benchmark payload."""
        self.class_(list(self.disciplines.values()))
        # inst = self.class_(list(self.disciplines.values()))
        # inst.input_grammar.validate(inst.default_inputs)

    def __str__(self):
        return f"{self.class_.__name__}-{self.nb_of_disc}"


if __name__ == "__main__":

    sys.setrecursionlimit(10000)

    class_ = MDAChain

    # CLI parser to control the benchmark
    parser = argparse.ArgumentParser(description="json_grammar benchmark")

    parser.add_argument("--profile", action="store_true")

    # add arguments for our discipline benchmark parameters
    for name, value in DEFAULT_ARGS.items():
        parser.add_argument(
            f"--{name}",
            default=value,
            type=type(value),
            help="default = %(default)s",
        )

    # create the pyperf runner, add its CLI info to our CLI parser and pass it
    # how our CLI was called
    runner = Runner(_argparser=parser, program_args=sys.argv)

    args = parser.parse_args()

    # filter and convert the parsed args for our constructor
    disc_bench_args = {k: getattr(args, k) for k in DEFAULT_ARGS}
    bench = ManyDisciplinesBenchmark(class_, **disc_bench_args)

    bench_name = str(bench)

    if args.profile:
        args.p = 1
        args.n = 1

    runner._set_args(args)

    if args.profile:
        cProfile.run("bench.run()", filename=args.output)
    else:
        runner.bench_func(bench_name, bench.run)
