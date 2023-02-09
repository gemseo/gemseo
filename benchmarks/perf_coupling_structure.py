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
"""Benchmark for coupling structure."""
from __future__ import annotations

import argparse
import cProfile  # noqa: F401
import pickle
import sys
from collections import defaultdict
from pathlib import Path

from base_benchmarkee import BaseBenchmarkee
from pyperf import Runner


def _compute_graph(nodes):
    """Computes the successors_i of each node and the edges between the nodes.

    :param nodes: the nodes of the graph
    """
    graph = {}
    edges = {}
    defaultdict_of_list = defaultdict(list)

    for disc_i, (_, outputs_i) in enumerate(nodes):
        successors_i = list()
        edges_i = edges[disc_i] = defaultdict_of_list
        # find out in which discipline(s) the outputs_i are used
        for disc_j, (inputs_j, _) in enumerate(nodes):
            if disc_i != disc_j:
                inter = outputs_i & inputs_j
                if inter:
                    successors_i += [disc_j]
                    edges_i[disc_j].extend(inter)

        graph[disc_i] = set(successors_i)

    return graph, edges


# def _compute_graph(nodes):
#     """Computes the successors_i of each node and the edges between
#     the nodes.
#
#     :param nodes: the nodes of the graph
#     """
#     graph = {}
#     disc_i = 0
#     edges = {}
#
#     for (_, outputs_i) in nodes:
#         successors_i = set()
#         # find out in which discipline(s) the outputs_i are used
#         for output_i in outputs_i:
#             disc_j = 0
#             for (inputs_j, _) in nodes:
#                 if disc_i != disc_j and output_i in inputs_j:
#                     successors_i.add(disc_j)
#                     # add the edge disc_i -> disc_j with
#                     # label output_i
#                     if disc_i not in edges:
#                         edges[disc_i] = {}
#                     if disc_j not in edges[disc_i]:
#                         edges[disc_i][disc_j] = []
#                     edges[disc_i][disc_j].append(output_i)
#                 disc_j += 1
#
#         graph[disc_i] = successors_i
#         disc_i += 1
#     return graph, edges


class ComputeGraphBenchmarkee(BaseBenchmarkee):
    """To benchmark many disciplines classes."""

    def __init__(self, file_path):
        """Constructor."""
        self.file_path = Path(file_path)
        self.nodes = None
        super().__init__()

    def setup(self):
        """Set up the benchmark."""
        self.nodes = pickle.load(open(self.file_path, "rb"))

    def run(self):
        """Run the benchmark."""
        _compute_graph(self.nodes)

    def __str__(self):
        return f"_compute_graph-{self.file_path.stem}"


if __name__ == "__main__":
    sys.setrecursionlimit(10000)

    # CLI parser to control the benchmark
    parser = argparse.ArgumentParser(description="coupling_structure benchmark")
    parser.add_argument("--nodes", type=str)

    # create the pyperf runner, add its CLI info to our CLI parser
    runner = Runner(_argparser=parser, program_args=sys.argv)

    args = parser.parse_args()

    bench = ComputeGraphBenchmarkee(args.nodes)

    bench_name = str(bench)

    runner._set_args(args)
    runner.bench_func(bench_name, bench.run)
    # cProfile.run("bench.run()", filename="perf.stats")
