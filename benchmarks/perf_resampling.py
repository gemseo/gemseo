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
"""Benchmark for resampling."""
from __future__ import annotations

import argparse
import cProfile  # noqa: F401
import sys

from base_benchmarkee import BaseBenchmarkee
from gemseo.algos.design_space import DesignSpace
from gemseo.core.doe_scenario import DOEScenario
from gemseo.disciplines.analytic import AnalyticDiscipline
from pyperf import Runner


def _execute_doe_scenario(scenario: DOEScenario) -> None:
    """Execute a DOE scenario several times.

    Args:
        scenario: The scenario to be executed.
    """
    scenario.formulation.opt_problem.reset()
    scenario.execute(
        {
            "algo": "OT_MONTE_CARLO",
            "n_samples": 10,
        }
    )


class ResamplingBenchmarkee(BaseBenchmarkee):
    """To benchmark many disciplines classes."""

    def __init__(self, use_configure: bool):
        """Constructor."""
        if use_configure:
            from gemseo.api import configure

            configure()

        self.scenario = None
        super().__init__()

    def setup(self):
        """Set up the benchmark."""
        discipline = AnalyticDiscipline({"y": "u"}, "func")
        space = DesignSpace()
        space.add_variable("u", l_b=0.0, u_b=1.0, value=0.5)
        self.scenario = DOEScenario([discipline], "DisciplinaryOpt", "y", space)

    def run(self):
        """Run the benchmark."""
        _execute_doe_scenario(self.scenario)


if __name__ == "__main__":

    sys.setrecursionlimit(10000)

    # CLI parser to control the benchmark
    parser = argparse.ArgumentParser(description="_sampling_doe_scenario benchmark")
    parser.add_argument("--configure", type=bool)

    # create the pyperf runner, add its CLI info to our CLI parser
    runner = Runner(_argparser=parser, program_args=sys.argv)

    args = parser.parse_args()
    args.copy_env = True

    bench = ResamplingBenchmarkee(args.configure)
    runner._set_args(args)
    runner.bench_func("sampling_doe_scenario", bench.run)
