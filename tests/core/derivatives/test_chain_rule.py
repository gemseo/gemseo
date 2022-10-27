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
#    INITIAL AUTHORS - API and implementation and/or documentation
#        :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import annotations

import pytest
from gemseo.core.chain import MDOChain
from gemseo.core.dependency_graph import DependencyGraph
from gemseo.core.derivatives.chain_rule import traverse_add_diff_io
from gemseo.problems.scalable.linear.disciplines_generator import (
    create_disciplines_from_desc,
)
from gemseo.problems.scalable.linear.disciplines_generator import (
    create_disciplines_from_sizes,
)
from numpy.random import seed


DISC_DESCR_1 = [
    ("A", ["x", "a"], ["p", "q", "xx"]),
    ("B", ["x", "y"], ["r"]),
    ("C", ["x"], ["m"]),
    ("D", ["q", "r", "w"], ["s"]),
    ("E", ["s"], ["o"]),
    ("H", ["z"], ["w"]),
    ("I", ["m"], ["n"]),
]


def test_traverse_add_diff_io_basic():
    """Test the differentiated inputs and outputs graph calculations."""
    disciplines = create_disciplines_from_desc(DISC_DESCR_1)
    graph = DependencyGraph(disciplines).graph

    traverse_add_diff_io(graph, ["x"], ["o"])

    # A
    assert disciplines[0]._differentiated_inputs == ["x"]
    assert disciplines[0]._differentiated_outputs == ["q"]

    # B
    assert disciplines[1]._differentiated_inputs == ["x"]
    assert disciplines[1]._differentiated_outputs == ["r"]

    # C
    assert disciplines[2]._differentiated_inputs == []
    assert disciplines[2]._differentiated_outputs == []

    # D
    assert sorted(disciplines[3]._differentiated_inputs) == ["q", "r"]
    assert disciplines[3]._differentiated_outputs == ["s"]

    # E
    assert disciplines[4]._differentiated_inputs == ["s"]
    assert disciplines[4]._differentiated_outputs == ["o"]

    # H
    assert disciplines[5]._differentiated_inputs == []
    assert disciplines[5]._differentiated_outputs == []

    # I
    assert disciplines[6]._differentiated_inputs == []
    assert disciplines[6]._differentiated_outputs == []


def test_chain_jac_basic():
    """Test the jacobian from the MDOChain on a basic case."""
    seed(1)
    disciplines = create_disciplines_from_desc(DISC_DESCR_1)
    chain = MDOChain(disciplines)
    assert chain.check_jacobian(inputs=["x"], outputs=["o"])


@pytest.mark.parametrize("nb_of_disc", [1, 5, 10, 20])
@pytest.mark.parametrize("nb_of_total_disc_io", [3, 10, 20, 100])
@pytest.mark.parametrize("nb_of_disc_ios", [1, 2, 10])
def test_chain_jac_random(nb_of_disc, nb_of_total_disc_io, nb_of_disc_ios):
    if nb_of_disc_ios > nb_of_total_disc_io:
        return
    seed(1)
    disciplines = create_disciplines_from_sizes(
        nb_of_disc,
        nb_of_total_disc_io=nb_of_total_disc_io,
        nb_of_disc_inputs=nb_of_disc_ios,
        nb_of_disc_outputs=nb_of_disc_ios,
        inputs_size=1,
        outputs_size=1,
        unique_disc_per_output=True,
        no_strong_couplings=True,
        no_self_coupled=True,
        grammar_type=MDOChain.SIMPLE_GRAMMAR_TYPE,
    )
    assert MDOChain(
        disciplines, grammar_type=MDOChain.SIMPLE_GRAMMAR_TYPE
    ).check_jacobian()


@pytest.mark.parametrize("inputs_size", [1, 2])
@pytest.mark.parametrize("outputs_size", [1, 3])
@pytest.mark.parametrize("unique_disc_per_output", [True, False])
def test_chain_jac_io_sizes(inputs_size, outputs_size, unique_disc_per_output):
    seed(1)
    disciplines = create_disciplines_from_sizes(
        5,
        nb_of_total_disc_io=20,
        nb_of_disc_inputs=3,
        nb_of_disc_outputs=3,
        inputs_size=inputs_size,
        outputs_size=outputs_size,
        unique_disc_per_output=unique_disc_per_output,
        no_strong_couplings=True,
        no_self_coupled=True,
        grammar_type=MDOChain.SIMPLE_GRAMMAR_TYPE,
    )
    assert MDOChain(
        disciplines, grammar_type=MDOChain.SIMPLE_GRAMMAR_TYPE
    ).check_jacobian()


@pytest.mark.parametrize("nb_of_disc", [5, 10])
@pytest.mark.parametrize("nb_of_total_disc_io", [5, 10, 40])
@pytest.mark.parametrize("no_self_coupled", [True, False])
def test_chain_jac_random_with_couplings(
    nb_of_disc,
    nb_of_total_disc_io,
    no_self_coupled,
):
    seed(1)
    disciplines = create_disciplines_from_sizes(
        nb_of_disc,
        nb_of_total_disc_io=nb_of_total_disc_io,
        nb_of_disc_inputs=2,
        nb_of_disc_outputs=2,
        inputs_size=1,
        outputs_size=1,
        unique_disc_per_output=True,
        no_strong_couplings=True,
        no_self_coupled=no_self_coupled,
        grammar_type=MDOChain.SIMPLE_GRAMMAR_TYPE,
    )
    assert MDOChain(
        disciplines, grammar_type=MDOChain.SIMPLE_GRAMMAR_TYPE
    ).check_jacobian()


# def test_chain_fail_multiple_io(
# ):
#     disc_descriptions = [
#         ("B", ["2" ], ["6"]),
#         ("C", ["3" ], ["2"]),
#         ("G", [ "2"], ["0"])]
#     disciplines = create_disciplines_from_desc(
#     disc_descriptions ,grammar_type=MDOChain.SIMPLE_GRAMMAR_TYPE
#     )
#     assert MDOChain(disciplines,
#         grammar_type=MDOChain.SIMPLE_GRAMMAR_TYPE).check_jacobian()


# def test_chain_jac_big( ):
#     seed(2)
#     disciplines = create_disciplines_from_sizes(
#         1000,
#         nb_of_total_disc_io=1000,
#         nb_of_disc_inputs=10,
#         nb_of_disc_outputs=10,
#         inputs_size=1,
#         outputs_size=1,
#         unique_disc_per_output=False,
#         no_strong_couplings=False,
#         no_self_coupled=False,
#         grammar_type=MDOChain.SIMPLE_GRAMMAR_TYPE
#     )
#     coupling_structure = MDOCouplingStructure(disciplines)
#     print("N Disc",len(disciplines))
#     from time import time
#     t0=time()
#     traverse_add_diff_io(coupling_structure.graph.graph,
#     disciplines[0].get_input_data_names(),
#         disciplines[-1].get_output_data_names())
#
#     raise ValueError(str(len(disciplines))+ " time = "+str(time()-t0))
