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
# Copyright 2022 IRT Saint Exupéry, https://www.irt-saintexupery.com
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
from gemseo.api import configure_logger
from gemseo.api import create_discipline
from gemseo.api import create_mda
from gemseo.core.chain import MDOChain
from gemseo.core.chain import MDOParallelChain
from gemseo.core.coupling_structure import MDOCouplingStructure
from gemseo.core.discipline import MDODiscipline
from gemseo.core.grammars.json_grammar import JSONGrammar
from gemseo.core.grammars.simple_grammar import SimpleGrammar
from gemseo.core.namespaces import namespaces_separator
from gemseo.core.namespaces import remove_prefix_from_dict
from gemseo.core.namespaces import remove_prefix_from_list
from gemseo.core.namespaces import remove_prefix_from_name
from gemseo.core.namespaces import split_namespace
from gemseo.core.namespaces import update_namespaces
from numpy import array


def func_1(x=1.0, u=2.0):
    y = 2.0 * x + u
    return y


def dfunc_1(x=1.0, u=2.0):
    return array([[2.0, 1.0]])


def func_2(y=3.0, a=2.0):
    z = 2.0 * y
    return z


def func_3(y=2.0):
    u = 1.0 - 0.01 * y
    f = 2 * u
    return u, f


def dfunc_3(y=2.0):
    return array([[-0.01], [-0.02]])


def test_remove_ns_prefix():
    """Test the remove_ns_prefix method."""
    assert remove_prefix_from_name("ab:c") == "c"
    assert remove_prefix_from_name("ac") == "ac"

    data_dict = {"ac": 1, "a:b": 2}
    assert remove_prefix_from_dict(data_dict) == {"ac": 1, "b": 2}
    assert remove_prefix_from_list(data_dict.keys()) == ["ac", "b"]


def test_split_namespace():
    assert split_namespace("ns:bar") == ["ns", "bar"]
    assert split_namespace("bar") == ["bar"]
    assert split_namespace("ns:") == ["ns", ""]
    assert split_namespace(":bar") == ["", "bar"]


@pytest.mark.parametrize(
    "grammar_type", [MDODiscipline.SIMPLE_GRAMMAR_TYPE, MDODiscipline.JSON_GRAMMAR_TYPE]
)
@pytest.mark.parametrize("use_defaults", [True, False])
def test_analytic_disc_ns(grammar_type, use_defaults):
    """Tests basic namespaces features using analytic disciplines."""
    disc = create_discipline(
        "AutoPyDiscipline", py_func=func_1, grammar_type=grammar_type
    )
    disc_ns = create_discipline(
        "AutoPyDiscipline", py_func=func_1, grammar_type=grammar_type
    )

    disc_ns.add_namespace_to_input("x", "ns")
    assert sorted(disc_ns.get_input_data_names()) == ["ns:x", "u"]
    assert sorted(disc_ns.get_input_data_names(with_namespaces=False)) == ["u", "x"]

    outs_ref = disc.execute({"x": array([1.0])})
    if use_defaults:
        assert disc_ns.default_inputs["x"] is not None
        assert "ns:x" in disc_ns.default_inputs
        outs_ns = disc_ns.execute()
    else:
        outs_ns = disc_ns.execute({"ns:x": array([1.0])})

    assert outs_ref["x"] == outs_ns["ns:x"]
    assert outs_ref["x"] == outs_ns["x"]


@pytest.mark.parametrize(
    "grammar_type", [MDODiscipline.SIMPLE_GRAMMAR_TYPE, MDODiscipline.JSON_GRAMMAR_TYPE]
)
def test_chain_disc_ns(grammar_type):
    """Tests MDOChain features with namespaces."""
    disc_1 = create_discipline(
        "AutoPyDiscipline", py_func=func_1, grammar_type=grammar_type
    )
    disc_2 = create_discipline(
        "AutoPyDiscipline", py_func=func_2, grammar_type=grammar_type
    )

    disc_1.add_namespace_to_input("x", "ns_in")
    disc_1.add_namespace_to_output("y", "ns_out")
    disc_2.add_namespace_to_input("y", "ns_out")

    chain = MDOChain([disc_1, disc_2], grammar_type=grammar_type)

    assert sorted(chain.get_input_data_names()) == ["a", "ns_in:x", "u"]
    assert sorted(remove_prefix_from_list(chain.get_input_data_names())) == [
        "a",
        "u",
        "x",
    ]
    assert sorted(chain.get_input_data_names(with_namespaces=False)) == [
        "a",
        "u",
        "x",
    ]
    assert sorted(chain.get_output_data_names()) == ["ns_out:y", "z"]

    out = chain.execute({"ns_in:x": array([3.0]), "u": array([4.0])})
    assert out["y"] == out["ns_out:y"]
    assert out["y"] == array([10.0])
    assert out["z"] == array([20.0])


@pytest.mark.parametrize(
    "grammar_type", [MDODiscipline.SIMPLE_GRAMMAR_TYPE, MDODiscipline.JSON_GRAMMAR_TYPE]
)
@pytest.mark.parametrize("chain_type", [MDOChain, MDOParallelChain])
def test_chain_disc_ns_twice(grammar_type, chain_type):
    """Tests MDOChain and MDOParallelChain with twice the same disciplines and different
    namespaces."""
    disc_1 = create_discipline(
        "AutoPyDiscipline", py_func=func_1, py_jac=dfunc_1, grammar_type=grammar_type
    )
    disc_2 = create_discipline(
        "AutoPyDiscipline", py_func=func_1, py_jac=dfunc_1, grammar_type=grammar_type
    )

    disc_1.add_namespace_to_input("x", "ns1")
    disc_1.add_namespace_to_output("y", "ns1")
    disc_2.add_namespace_to_input("x", "ns2")
    disc_2.add_namespace_to_output("y", "ns2")

    chain = chain_type([disc_1, disc_2], grammar_type=grammar_type)

    assert sorted(chain.get_input_data_names()) == sorted(["ns2:x", "ns1:x", "u"])
    assert sorted(chain.get_output_data_names()) == sorted(["ns2:y", "ns1:y"])

    assert sorted(chain.get_input_data_names(with_namespaces=False)) == sorted(
        ["x", "x", "u"]
    )
    assert sorted(chain.get_output_data_names(with_namespaces=False)) == sorted(
        ["y", "y"]
    )

    out = chain.execute(
        {"ns1:x": array([5.0]), "ns2:x": array([3.0]), "u": array([4.0])}
    )
    assert out["ns1:y"] == array([14.0])
    assert out["ns2:y"] == array([10.0])

    assert sorted(chain.get_input_output_data_names(with_namespaces=False)) == [
        "u",
        "x",
        "x",
        "y",
        "y",
    ]

    out_no_ns = chain.get_output_data(with_namespaces=False)
    assert "y" in out_no_ns

    assert chain.check_jacobian(
        inputs=["ns2:x", "ns1:x", "u"], outputs=["ns1:y", "ns2:y"]
    )


@pytest.mark.parametrize(
    "grammar_type", [MDODiscipline.SIMPLE_GRAMMAR_TYPE, MDODiscipline.JSON_GRAMMAR_TYPE]
)
def test_mda_with_namespaces(grammar_type):
    """Tests MDAs and namespaces."""
    configure_logger()
    disc_1 = create_discipline(
        "AutoPyDiscipline", py_func=func_1, py_jac=dfunc_1, grammar_type=grammar_type
    )
    disc_2 = create_discipline(
        "AutoPyDiscipline", py_func=func_3, py_jac=dfunc_3, grammar_type=grammar_type
    )

    disciplines = [disc_1, disc_2]
    mda = create_mda(
        "MDAGaussSeidel", disciplines=disciplines, grammar_type=grammar_type
    )
    out_ref = mda.execute()

    struct = MDOCouplingStructure(disciplines)
    assert len(struct.get_strongly_coupled_disciplines()) == 2
    disc_1.add_namespace_to_output("y", "ns")

    struct = MDOCouplingStructure(disciplines)
    assert not struct.get_strongly_coupled_disciplines()

    disc_2.add_namespace_to_input("y", "ns")
    struct = MDOCouplingStructure(disciplines)
    assert len(struct.get_strongly_coupled_disciplines()) == 2

    disc_1.add_namespace_to_input("u", "ns")
    struct = MDOCouplingStructure(disciplines)
    assert not struct.get_strongly_coupled_disciplines()

    disc_2.add_namespace_to_output("u", "ns")
    struct = MDOCouplingStructure(disciplines)
    assert len(struct.get_strongly_coupled_disciplines()) == 2

    mda_ns = create_mda(
        "MDAGaussSeidel", disciplines=disciplines, grammar_type=grammar_type
    )
    out_ns = mda_ns.execute()
    assert abs(out_ns["y"][0] - out_ref["y"][0]) < 1e-14

    assert disc_1.check_jacobian()
    assert disc_2.check_jacobian()

    assert mda_ns.check_jacobian(inputs=["x"], outputs=["f"], threshold=1e-5)


def test_json_grammar_grammar_add_namespace():
    """Tests JSONGrammar namespaces handling."""
    g = JSONGrammar("g")
    g.update(["name1", "name2"])
    g.required_names.update("name1")
    g.add_namespace("name1", "ns")
    assert "ns" + namespaces_separator + "name1" in g
    assert g.to_namespaced == {"name1": "ns:name1"}
    assert "ns:name1" in g.required_names

    with pytest.raises(ValueError, match="Variable ns:name1 has already a namespace."):
        g.add_namespace("ns:name1", "ns2")


def test_simple_grammar_add_namespace():
    """Tests SimpleGrammar namespaces handling."""
    g = SimpleGrammar(
        "g", names_to_types={"name1": int, "name2": str}, required_names=["name1"]
    )
    g.add_namespace("name1", "ns")
    assert "ns" + namespaces_separator + "name1" in g
    assert g.to_namespaced == {"name1": "ns:name1"}


def a_func(x=1.0):
    y = x + 1
    return y


def b_func(y=1.0):
    z = y + 1
    return z


def test_namespaces_chain():
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
    assert chain_ns.check_jacobian(inputs=["x", "y"], outputs=["z", "ns:y"])


def test_update_namespaces():
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
