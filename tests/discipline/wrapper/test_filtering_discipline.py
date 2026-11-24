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
#                           documentation
#        :author: Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import annotations

import pytest
from numpy import array
from numpy.testing import assert_array_equal

from gemseo import create_discipline
from gemseo.discipline import AnalyticDiscipline
from gemseo.discipline import DisciplineChain
from gemseo.discipline.wrapper.filtering_discipline import FilteringDiscipline


@pytest.fixture
def discipline():
    expressions = {"y1": "x1+x2+x3", "y2": "-x1-x2-x3"}
    disc = create_discipline("AnalyticDiscipline", expressions, name="foo")
    disc.io.input_grammar.descriptions = {"x1": "This is x1.", "x2": "This is x2."}
    disc.io.output_grammar.descriptions = {"y1": "This is y1.", "y2": "This is y2."}
    disc.add_differentiated_inputs(["x1", "x2", "x3"])
    disc.add_differentiated_outputs(["y1", "y2"])
    return disc


def test_standard(discipline) -> None:
    fdisc = FilteringDiscipline(discipline)
    assert set(fdisc.io.input_grammar) == set(discipline.io.input_grammar)
    assert set(fdisc.io.output_grammar) == set(discipline.io.output_grammar)
    fdisc.execute()
    for name in ["x1", "x2", "x3", "y1", "y2"]:
        assert name in fdisc.io.get_merged_data(as_dict=False)
    fdisc.linearize()
    for output_name in ["y1", "y2"]:
        assert output_name in fdisc.jac
        for input_name in ["x1", "x2", "x3"]:
            assert input_name in fdisc.jac[output_name]

    assert (
        fdisc.io.input_grammar.descriptions == discipline.io.input_grammar.descriptions
    )
    assert (
        fdisc.io.output_grammar.descriptions
        == discipline.io.output_grammar.descriptions
    )


def test_keep_in_keep_out(discipline) -> None:
    fdisc = FilteringDiscipline(discipline, input_names=["x1"], output_names=["y1"])
    assert set(fdisc.io.input_grammar) == {"x1"}
    assert set(fdisc.io.output_grammar) == {"y1"}
    fdisc.execute()
    for name in ["x2", "x3", "y2"]:
        assert name not in fdisc.io.get_merged_data(as_dict=False)
    fdisc.linearize()
    assert "y2" not in fdisc.jac
    for input_name in ["x2", "x3"]:
        assert input_name not in fdisc.jac["y1"]

    assert fdisc.io.input_grammar.descriptions == {"x1": "This is x1."}
    assert fdisc.io.output_grammar.descriptions == {"y1": "This is y1."}


def test_remove_in_keep_out(discipline) -> None:
    fdisc = FilteringDiscipline(
        discipline, input_names=["x1"], output_names=["y1"], use_input_names=False
    )
    assert set(fdisc.io.input_grammar) == {"x2", "x3"}
    assert set(fdisc.io.output_grammar) == {"y1"}
    fdisc.execute()
    for name in ["x1", "y2"]:
        assert name not in fdisc.io.get_merged_data(as_dict=False)
    fdisc.linearize()
    assert "y2" not in fdisc.jac
    assert "x1" not in fdisc.jac["y1"]


def test_keep_in_remove_out(discipline) -> None:
    fdisc = FilteringDiscipline(
        discipline, input_names=["x1"], output_names=["y1"], use_output_names=False
    )
    assert set(fdisc.io.input_grammar) == {"x1"}
    assert set(fdisc.io.output_grammar) == {"y2"}
    fdisc.execute()
    for name in ["x2", "x3", "y1"]:
        assert name not in fdisc.io.get_merged_data(as_dict=False)
    fdisc.linearize()
    assert "y1" not in fdisc.jac
    for input_name in ["x2", "x3"]:
        assert input_name not in fdisc.jac["y2"]


def test_linearize_with_namespaces(discipline) -> None:
    """Check the Jacobian when the wrapped discipline has namespaces."""
    discipline.add_namespace_to_input("x1", "n")
    discipline.add_namespace_to_output("y1", "m")
    jac = FilteringDiscipline(discipline, ["n:x1", "x2"], ["m:y1"]).linearize(
        compute_all_jacobians=True
    )
    assert {k: sorted(v) for k, v in jac.items()} == {"m:y1": ["n:x1", "x2"]}
    assert_array_equal(jac["m:y1"]["n:x1"], array([[1.0]]))


@pytest.fixture(params=[False, True], ids=["on_wrapped", "on_wrapper"])
def namespaced_wrapper(request) -> FilteringDiscipline:
    """A filtering discipline whose wrapped discipline has namespaces.

    The namespaces are added either on the wrapped discipline or on the wrapper.
    """
    discipline = AnalyticDiscipline({"y": "2*x"})
    if request.param:
        wrapper = FilteringDiscipline(discipline)
        wrapper.add_namespace_to_input("x", "n")
        wrapper.add_namespace_to_output("y", "m")
    else:
        discipline.add_namespace_to_input("x", "n")
        discipline.add_namespace_to_output("y", "m")
        wrapper = FilteringDiscipline(discipline)

    return wrapper


def test_namespaces_execution(namespaced_wrapper) -> None:
    """Check the execution with namespaces."""
    output_data = namespaced_wrapper.execute({"n:x": array([3.0])})
    assert_array_equal(output_data["m:y"], array([6.0]))


def test_namespaces_linearization(namespaced_wrapper) -> None:
    """Check the Jacobian with namespaces."""
    jacobian = namespaced_wrapper.linearize(
        {"n:x": array([3.0])}, compute_all_jacobians=True
    )
    assert {k: list(v) for k, v in jacobian.items()} == {"m:y": ["n:x"]}
    assert_array_equal(jacobian["m:y"]["n:x"], array([[2.0]]))


def test_linearize_chain_with_shared_name_in_namespaces() -> None:
    """Check the Jacobian of a chain whose input has several namespace prefixes."""
    discipline_1 = AnalyticDiscipline({"y1": "2*x"})
    discipline_1.add_namespace_to_input("x", "ns1")
    discipline_2 = AnalyticDiscipline({"y2": "3*x"})
    discipline_2.add_namespace_to_input("x", "ns2")
    chain = DisciplineChain([discipline_1, discipline_2])
    wrapper = FilteringDiscipline(chain)
    wrapper.execute({"ns1:x": array([1.0]), "ns2:x": array([2.0])})
    jac = wrapper.linearize(compute_all_jacobians=True)
    expected = {
        "y1": {"ns1:x": [[2.0]], "ns2:x": [[0.0]]},
        "y2": {"ns1:x": [[0.0]], "ns2:x": [[3.0]]},
    }
    assert {k: sorted(v) for k, v in jac.items()} == {
        k: sorted(v) for k, v in expected.items()
    }
    for output_name, input_name_to_value in expected.items():
        for input_name, value in input_name_to_value.items():
            assert_array_equal(jac[output_name][input_name], array(value))
