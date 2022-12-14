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
from gemseo.core.coupling_structure import MDOCouplingStructure
from gemseo.core.derivatives.mda_derivatives import traverse_add_diff_io_mda
from gemseo.core.discipline import MDODiscipline
from gemseo.mda.mda_chain import MDAChain
from gemseo.problems.scalable.linear.disciplines_generator import (
    create_disciplines_from_desc,
)
from gemseo.problems.scalable.linear.disciplines_generator import (
    create_disciplines_from_sizes,
)
from numpy.random import seed

from tests.mda.test_mda import analytic_disciplines_from_desc


DISC_DESCR_1 = [
    ("A", ["x1", "x2"], ["a"]),
    ("B", ["x3"], ["b"]),
    ("C", ["a", "b", "y1"], ["y2", "y3"]),
    ("D", ["y2"], ["y1", "o1"]),
    ("E", ["y1", "y2", "y3", "x1"], ["o2"]),
]

DISC_DESCR_SELF_STRONG_C = [
    ("A", ["x", "ys", "yb"], ["ya", "ys"]),
    ("B", ["x", "ya"], ["yb"]),
    ("E", ["ya", "yb", "ys", "x"], ["o"]),
]

DISC_DESCR_SELF_C = [("A", ["x", "y"], ["y", "o"])]


def test_traverse_add_diff_io_basic():
    """Test the differentiated inputs and outputs graph calculations."""
    disciplines = create_disciplines_from_desc(DISC_DESCR_1)
    coupl = MDOCouplingStructure(disciplines)

    traverse_add_diff_io_mda(coupl, ["x1"], ["o1"])

    # A
    assert disciplines[0]._differentiated_inputs == ["x1"]
    assert disciplines[0]._differentiated_outputs == ["a"]

    # B
    assert not disciplines[1]._differentiated_inputs
    assert not disciplines[1]._differentiated_outputs

    # C
    assert sorted(disciplines[2]._differentiated_inputs) == ["a", "y1"]
    assert disciplines[2]._differentiated_outputs == ["y2"]

    # D
    assert sorted(disciplines[3]._differentiated_inputs) == ["y2"]
    assert sorted(disciplines[3]._differentiated_outputs) == ["o1", "y1"]

    # E
    assert not disciplines[4]._differentiated_inputs
    assert not disciplines[4]._differentiated_outputs


@pytest.mark.parametrize(
    "grammar_type", [MDODiscipline.SIMPLE_GRAMMAR_TYPE, MDODiscipline.JSON_GRAMMAR_TYPE]
)
def test_chain_jac_basic_grammars(grammar_type):
    """Test the jacobian from the MDOChain on a basic case with different grammars."""
    seed(1)
    disciplines = create_disciplines_from_desc(DISC_DESCR_1, grammar_type=grammar_type)
    mda = MDAChain(disciplines, grammar_type=grammar_type)
    assert mda.check_jacobian(inputs=["x1"], outputs=["o1"])


@pytest.mark.parametrize("input", [["x1"], ["x2"], ["x1", "x2"]])
@pytest.mark.parametrize("output", [["o1"], ["o2"], ["o1", "o2"]])
def test_chain_jac_basic(input, output):
    """Test the jacobian from the MDOChain on a basic case."""
    seed(1)
    disciplines = create_disciplines_from_desc(
        DISC_DESCR_1, grammar_type=MDODiscipline.SIMPLE_GRAMMAR_TYPE
    )
    mda = MDAChain(disciplines, grammar_type=MDODiscipline.SIMPLE_GRAMMAR_TYPE)
    assert mda.check_jacobian(inputs=input, outputs=output)


@pytest.mark.parametrize("descriptions", [DISC_DESCR_SELF_C, DISC_DESCR_SELF_STRONG_C])
def test_chain_jac_self_coupled(descriptions):
    """Test the jacobian with self-couplings."""
    seed(1)
    disciplines = create_disciplines_from_desc(descriptions)
    mda = MDAChain(disciplines)
    assert mda.check_jacobian(inputs=["x"], outputs=["o"])


def test_double_mda():
    disciplines = analytic_disciplines_from_desc(
        (
            {"a": "x"},
            {"y1": "x1", "b": "a+1"},
            {"x1": "1.-0.3*y1"},
            {"y2": "x2", "c": "a+2"},
            {"x2": "1.-0.3*y2"},
            {"obj1": "x1+x2"},
            {"obj2": "b+c"},
            {"obj": "obj1+obj2"},
        )
    )
    mdachain = MDAChain(disciplines)
    assert mdachain.check_jacobian(inputs=["x"], outputs=["obj"])


@pytest.mark.parametrize("nb_of_disc", [1, 5, 10, 20])
@pytest.mark.parametrize("nb_of_total_disc_io", [3, 10, 20, 100])
@pytest.mark.parametrize("nb_of_disc_ios", [1, 2, 10])
def test_chain_jac_random(nb_of_disc, nb_of_total_disc_io, nb_of_disc_ios):
    """Test the Jacobian of MDA with various IOs variables set sizes."""
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
        no_strong_couplings=False,
        no_self_coupled=True,
        grammar_type=MDODiscipline.SIMPLE_GRAMMAR_TYPE,
    )
    assert MDAChain(
        disciplines, grammar_type=MDODiscipline.SIMPLE_GRAMMAR_TYPE
    ).check_jacobian()


@pytest.mark.parametrize("inputs_size", [1, 2])
@pytest.mark.parametrize("outputs_size", [1, 3])
def test_chain_jac_io_sizes(inputs_size, outputs_size):
    """Test the Jacobian of MDA with various IOs sizes."""
    seed(1)
    disciplines = create_disciplines_from_sizes(
        5,
        nb_of_total_disc_io=20,
        nb_of_disc_inputs=3,
        nb_of_disc_outputs=3,
        inputs_size=inputs_size,
        outputs_size=outputs_size,
        unique_disc_per_output=True,
        no_strong_couplings=True,
        no_self_coupled=True,
        grammar_type=MDODiscipline.SIMPLE_GRAMMAR_TYPE,
    )
    assert MDAChain(
        disciplines, grammar_type=MDODiscipline.SIMPLE_GRAMMAR_TYPE
    ).check_jacobian()
