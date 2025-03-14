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
from numpy import array

from gemseo import create_discipline
from gemseo import create_mda
from gemseo.core.chains.chain import MDOChain
from gemseo.core.chains.parallel_chain import MDOParallelChain
from gemseo.core.coupling_structure import CouplingStructure
from gemseo.core.discipline import Discipline
from gemseo.core.namespaces import remove_prefix
from gemseo.core.namespaces import split_namespace
from gemseo.core.namespaces import update_namespaces
from gemseo.disciplines.auto_py import AutoPyDiscipline


def func_1(x=1.0, u=2.0):
    y = 2.0 * x + u
    return y  # noqa: RET504


def dfunc_1(x=1.0, u=2.0):
    return array([[2.0, 1.0]])


def func_2(y=3.0, a=2.0):
    z = 2.0 * y
    return z  # noqa: RET504


def func_3(y=2.0):
    u = 1.0 - 0.01 * y
    f = 2 * u
    return u, f


def dfunc_3(y=2.0):
    return array([[-0.01], [-0.02]])


def test_remove_ns_prefix() -> None:
    """Test the remove_ns_prefix method."""
    data_dict = {"ac": 1, "a:b": 2}
    assert list(remove_prefix(data_dict.keys())) == ["ac", "b"]


def test_split_namespace() -> None:
    assert split_namespace("ns:bar") == ["ns", "bar"]
    assert split_namespace("bar") == ["bar"]
    assert split_namespace("ns:") == ["ns", ""]
    assert split_namespace(":bar") == ["", "bar"]


@pytest.fixture(params=[Discipline.GrammarType.SIMPLE, Discipline.GrammarType.JSON])
def grammar_type(request):
    old_grammar_type = AutoPyDiscipline.default_grammar_type
    AutoPyDiscipline.default_grammar_type = request.param
    yield
    AutoPyDiscipline.default_grammar_type = old_grammar_type


@pytest.mark.parametrize("use_defaults", [True, False])
def test_analytic_disc_ns(grammar_type, use_defaults) -> None:
    """Tests basic namespaces features using analytic disciplines."""
    disc = create_discipline("AutoPyDiscipline", py_func=func_1)
    disc_ns = create_discipline("AutoPyDiscipline", py_func=func_1)

    disc_ns.add_namespace_to_input("x", "ns")
    assert sorted(disc_ns.io.input_grammar) == ["ns:x", "u"]
    assert sorted(disc_ns.io.input_grammar.names_without_namespace) == ["u", "x"]

    outs_ref = disc.execute({"x": array([1.0])})
    if use_defaults:
        assert "ns:x" in disc_ns.io.input_grammar.defaults
        outs_ns = disc_ns.execute()
    else:
        outs_ns = disc_ns.execute({"ns:x": array([1.0])})

    assert outs_ref["x"] == outs_ns["ns:x"]


def test_chain_disc_ns(grammar_type) -> None:
    """Tests MDOChain features with namespaces."""
    disc_1 = create_discipline("AutoPyDiscipline", py_func=func_1)
    disc_2 = create_discipline("AutoPyDiscipline", py_func=func_2)

    disc_1.add_namespace_to_input("x", "ns_in")
    disc_1.add_namespace_to_output("y", "ns_out")
    disc_2.add_namespace_to_input("y", "ns_out")

    chain = MDOChain([disc_1, disc_2])

    assert sorted(chain.io.input_grammar) == ["a", "ns_in:x", "u"]
    assert sorted(remove_prefix(chain.io.input_grammar)) == [
        "a",
        "u",
        "x",
    ]
    assert sorted(chain.io.input_grammar.names_without_namespace) == [
        "a",
        "u",
        "x",
    ]
    assert sorted(chain.io.output_grammar) == ["ns_out:y", "z"]

    out = chain.execute({"ns_in:x": array([3.0]), "u": array([4.0])})
    assert out["ns_out:y"] == array([10.0])
    assert out["z"] == array([20.0])


@pytest.mark.parametrize("chain_type", [MDOChain, MDOParallelChain])
def test_chain_disc_ns_twice(grammar_type, chain_type) -> None:
    """Tests MDOChain and MDOParallelChain with twice the same disciplines and different
    namespaces."""
    disc_1 = create_discipline("AutoPyDiscipline", py_func=func_1, py_jac=dfunc_1)
    disc_2 = create_discipline("AutoPyDiscipline", py_func=func_1, py_jac=dfunc_1)

    disc_1.add_namespace_to_input("x", "ns1")
    disc_1.add_namespace_to_output("y", "ns1")
    disc_2.add_namespace_to_input("x", "ns2")
    disc_2.add_namespace_to_output("y", "ns2")

    chain = chain_type([disc_1, disc_2])

    assert sorted(chain.io.input_grammar) == sorted(["ns2:x", "ns1:x", "u"])
    assert sorted(chain.io.output_grammar) == sorted(["ns2:y", "ns1:y"])

    assert sorted(chain.io.input_grammar.names_without_namespace) == sorted([
        "x",
        "x",
        "u",
    ])
    assert sorted(chain.io.output_grammar.names_without_namespace) == sorted([
        "y",
        "y",
    ])

    out = chain.execute({
        "ns1:x": array([5.0]),
        "ns2:x": array([3.0]),
        "u": array([4.0]),
    })
    assert out["ns1:y"] == array([14.0])
    assert out["ns2:y"] == array([10.0])

    in_out_data = remove_prefix(
        chain.io.output_grammar.keys() | chain.io.input_grammar.keys()
    )
    assert sorted(in_out_data) == [
        "u",
        "x",
        "x",
        "y",
        "y",
    ]

    out_no_ns = chain.io.get_output_data(with_namespaces=False)
    assert "y" in out_no_ns

    assert chain.check_jacobian(
        input_names=["ns2:x", "ns1:x", "u"], output_names=["ns1:y", "ns2:y"]
    )


def test_mda_with_namespaces(grammar_type) -> None:
    """Tests MDAs and namespaces."""
    disc_1 = create_discipline("AutoPyDiscipline", py_func=func_1, py_jac=dfunc_1)
    disc_2 = create_discipline("AutoPyDiscipline", py_func=func_3, py_jac=dfunc_3)

    disciplines = [disc_1, disc_2]
    mda = create_mda(
        "MDAGaussSeidel",
        disciplines=disciplines,
        tolerance=1e-10,
    )
    mda.execute()

    struct = CouplingStructure(disciplines)
    assert len(struct.get_strongly_coupled_disciplines()) == 2
    disc_1.add_namespace_to_output("y", "ns")

    struct = CouplingStructure(disciplines)
    assert not struct.get_strongly_coupled_disciplines()

    disc_2.add_namespace_to_input("y", "ns")
    struct = CouplingStructure(disciplines)
    assert len(struct.get_strongly_coupled_disciplines()) == 2

    disc_1.add_namespace_to_input("u", "ns")
    struct = CouplingStructure(disciplines)
    assert not struct.get_strongly_coupled_disciplines()

    disc_2.add_namespace_to_output("u", "ns")
    struct = CouplingStructure(disciplines)
    assert len(struct.get_strongly_coupled_disciplines()) == 2

    mda_ns = create_mda(
        "MDAGaussSeidel",
        disciplines=disciplines,
        tolerance=1e-10,
    )
    mda_ns.execute()

    assert disc_1.check_jacobian()
    assert disc_2.check_jacobian()

    assert mda_ns.check_jacobian(input_names=["x"], output_names=["f"], threshold=1e-5)


def a_func(x=1.0):
    y = x + 1
    return y  # noqa: RET504


def b_func(y=1.0):
    z = y + 1
    return z  # noqa: RET504


def test_namespaces_chain() -> None:
    """Tests MDOChain namespaces and jacobian."""
    a_disc = create_discipline("AutoPyDiscipline", py_func=a_func)
    b_disc = create_discipline("AutoPyDiscipline", py_func=b_func)
    chain = MDOChain(disciplines=[a_disc, b_disc])
    assert chain.execute()["z"][0] == 3.0

    a_disc_ns = create_discipline("AutoPyDiscipline", py_func=a_func)
    a_disc_ns.add_namespace_to_output("y", "ns")
    chain_ns = MDOChain(disciplines=[a_disc_ns, b_disc])
    assert chain_ns.execute()["z"][0] == 2.0

    b_disc.linearization_mode = "finite_differences"
    a_disc_ns.linearization_mode = "finite_differences"
    assert chain_ns.check_jacobian(input_names=["x", "y"], output_names=["z", "ns:y"])


def test_update_namespaces() -> None:
    namespaces = {"a": "b", "c": ["a", "b"], "d": "e", "f": ["g"], "g": ["h"], "i": "j"}
    update_namespaces(
        namespaces, {"a": "1", "c": ["1"], "x": "1", "g": "1", "i": ["1"]}
    )
    assert namespaces == {
        "a": ["b", "1"],
        "c": ["a", "b", "1"],
        "d": "e",
        "f": ["g"],
        "x": "1",
        "g": ["h", "1"],
        "i": ["j", "1"],
    }
