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
from __future__ import annotations

import collections
import pickle
from typing import TYPE_CHECKING
from typing import Any

import pytest
from numpy import array
from numpy import ndarray

from gemseo.core.grammars.errors import InvalidDataError
from gemseo.core.grammars.simple_grammar import SimpleGrammar
from gemseo.core.grammars.simpler_grammar import SimplerGrammar
from gemseo.utils.repr_html import REPR_HTML_WRAPPER

if TYPE_CHECKING:
    from collections.abc import Mapping

parametrized_names_to_types = pytest.mark.parametrize(
    "names_to_types",
    [
        {},
        {"name": str},
    ],
)


@pytest.fixture(params=(SimpleGrammar, SimplerGrammar))
def cls(request):
    return request.param


@pytest.mark.parametrize("names_to_types", [None, {}])
def test_init_empty(cls, names_to_types):
    """Verify init with empty inputs."""
    g = cls("g", names_to_types=names_to_types)
    assert g.name == "g"
    assert not g
    assert not g.required_names
    assert not g.defaults


@pytest.mark.parametrize("required_names", [None, [], ["name"]])
def test_init(cls, required_names):
    """Verify init with non-empty inputs."""
    g = cls("g", names_to_types={"name": str}, required_names=required_names)
    assert g
    assert list(g.keys()) == ["name"]
    assert list(g.values()) == [str]
    if required_names is None:
        assert g.required_names == set(g)
    else:
        assert g.required_names == set(required_names)
    assert not g.defaults


def test_init_errors(cls):
    """Verify init errors."""
    msg = "The grammar name cannot be empty."
    with pytest.raises(ValueError, match=msg):
        cls("")

    msg = "The element name must be a type or None: it is 0."
    with pytest.raises(TypeError, match=msg):
        cls("g", names_to_types={"name": 0})

    g = cls("g", names_to_types={"name": str}, required_names=["name"])
    assert g.required_names == {"name"}

    g = cls("g", names_to_types={"name": str}, required_names=[])
    assert not g.required_names

    msg = "foo is not in the grammar."
    with pytest.raises(KeyError, match=msg):
        cls("g", names_to_types={"name": str}, required_names=["foo"])


def test_delitem_error(cls):
    """Verify item deletion errors."""
    g = cls("g")
    with pytest.raises(KeyError, match="foo"):
        del g["foo"]


def test_delitem(cls):
    """Verify item deletion."""
    g = cls("g", names_to_types={"name1": str, "name2": str}, required_names=["name1"])
    g.defaults["name1"] = "foo"

    del g["name1"]

    assert "name1" not in g
    assert "name1" not in g.required_names
    assert "name1" not in g.defaults

    del g["name2"]

    assert "name2" not in g
    assert "name2" not in g.required_names
    assert "name2" not in g.defaults


def test_getitem_error(cls):
    """Verify getitem errors."""
    g = cls("g")
    with pytest.raises(KeyError, match="foo"):
        g["foo"]


def test_getitem(cls):
    """Verify getitem."""
    g = cls("g", names_to_types={"name": str})
    assert g["name"] is str


@parametrized_names_to_types
def test_len(cls, names_to_types):
    """Verify len."""
    g = cls("g", names_to_types=names_to_types)
    assert len(g) == len(names_to_types)


@parametrized_names_to_types
def test_iter(cls, names_to_types):
    """Verify iterator."""
    g = cls("g", names_to_types=names_to_types)
    assert list(iter(g)) == list(names_to_types)


@parametrized_names_to_types
def test_names(cls, names_to_types):
    """Verify names getter."""
    g = cls("g", names_to_types=names_to_types)
    assert list(g.names) == list(names_to_types)


double_names_to_types = pytest.mark.parametrize(
    "names_to_types1,names_to_types2",
    [
        ({}, {}),
        ({}, {"name": int}),
        ({"name": None}, {}),
        ({"name": None}, {"name": int}),
        ({"name1": None}, {"name2": int}),
    ],
)
exclude_names = pytest.mark.parametrize(
    "exclude_names",
    [
        (),
        ["dummy"],
        ["name"],
    ],
)


def create_defaults(names_to_types: Mapping[str, type]) -> dict[str, Any]:
    defaults = {}
    for name, type_ in names_to_types.items():
        value = None if type_ is None else type_(0)
        defaults[name] = value
    return defaults


@double_names_to_types
def test_update_from_types(cls, names_to_types1, names_to_types2):
    """Verify update with a dict."""
    g1 = cls("g1", names_to_types=names_to_types1)
    defaults = create_defaults(names_to_types1)
    g1.defaults.update(defaults)

    g1.update_from_types(names_to_types2)

    assert g1.defaults == defaults

    assert set(g1) == set(names_to_types1) | set(names_to_types2)
    assert g1.required_names == set(g1)

    for name in g1:
        if name in names_to_types2:
            assert g1[name] == names_to_types2[name]
        else:
            assert g1[name] == names_to_types1[name]


@double_names_to_types
@pytest.mark.parametrize(
    ("required_names1", "required_names2"),
    [
        (set(), set()),
        ({"name"}, set()),
        (set(), {"name"}),
        ({"name"}, {"name"}),
    ],
)
@exclude_names
def test_update_with_grammar(
    cls,
    names_to_types1,
    names_to_types2,
    required_names1,
    required_names2,
    exclude_names,
):
    """Verify update with a grammar."""
    g1 = cls(
        "g1",
        names_to_types=names_to_types1,
        required_names=required_names1 & set(names_to_types1),
    )

    g1_required_names_before = set(g1.required_names)

    g2 = cls(
        "g2",
        names_to_types=names_to_types2,
        required_names=required_names2 & set(names_to_types2),
    )

    if g2:
        for name in g2.names:
            g2.defaults[name] = 1

    g1.update(g2, exclude_names=exclude_names)

    exclude_names = set(exclude_names)

    assert set(g1) == set(names_to_types1) | (set(names_to_types2) - exclude_names)
    assert g1.required_names == (g2.required_names - exclude_names) | (
        set(g1) - (set(g2) - exclude_names) & g1_required_names_before
    )

    assert g1.defaults.keys() == g2.defaults.keys() - exclude_names

    for name in g1:
        if name in set(g2) - exclude_names:
            assert g1[name] == g2[name]
        else:
            assert g1[name] == names_to_types1[name]


def test_update_dict_with_mapping(cls):
    """Verify update with mapping."""
    g = cls("g")
    g.update_from_types({"name": dict})
    assert g["name"] == collections.abc.Mapping


def test_update_dict_with_merge(cls):
    """Verify that merge is not supported."""
    g = cls("g")
    with pytest.raises(ValueError, match="Merge is not supported yet."):
        g.update_from_types({}, merge=True)


@pytest.mark.parametrize(
    "names",
    [
        [],
        ["name"],
        ["name1"],
    ],
)
@parametrized_names_to_types
def test_update_with_names(cls, names_to_types, names):
    """Verify update with names."""
    g = cls("g", names_to_types=names_to_types)
    defaults = create_defaults(names_to_types)
    g.defaults.update(defaults)

    g.update_from_names(names)

    assert g.defaults == defaults

    assert set(g) == set(names_to_types) | set(names)
    assert g.required_names == set(g)

    for name in g:
        if name in names:
            assert g[name] == ndarray
        else:
            assert g[name] == names_to_types[name]


def test_update_error(cls):
    """Verify update error."""
    g1 = cls("g1")

    msg = "The element name must be a type or None: it is 0."
    with pytest.raises(TypeError, match=msg):
        g1.update_from_types({"name": 0})


@parametrized_names_to_types
def test_clear(cls, names_to_types):
    """Verify clear."""
    g = cls("g", names_to_types=names_to_types)
    g.defaults.update(create_defaults(names_to_types))
    g.clear()
    assert not g
    assert not g.required_names
    assert not g.defaults


@pytest.mark.parametrize(
    ("names_to_types", "data"),
    [
        # Empty grammar: everything validates.
        ({}, {"name": 0}),
        # Common types.
        ({"name": int}, {"name": 0}),
        ({"name": float}, {"name": 0.0}),
        ({"name": str}, {"name": ""}),
        ({"name": bool}, {"name": True}),
        ({"name": ndarray}, {"name": array([])}),
        # None values element means any type.
        ({"name": None}, {"name": {}}),
    ],
)
def test_validate(cls, names_to_types, data):
    """Verify validate."""
    g = cls("g", names_to_types=names_to_types)
    g.validate(data)


@pytest.mark.parametrize(
    ("data", "error_msg"),
    [
        ({}, r"Missing required names: name1."),
        (
            {"name1": 0, "name2": ""},
            r"Bad type for name2: <class 'str'> instead of <class 'int'>.",
        ),
    ],
)
@pytest.mark.parametrize("raise_exception", [True, False])
def test_validate_error(cls, data, error_msg, raise_exception, caplog):
    """Verify that validate raises the expected errors."""
    g = cls("g", names_to_types={"name1": None, "name2": int}, required_names=["name1"])

    if cls is SimplerGrammar and data:
        g.validate(data)
        return

    if raise_exception:
        with pytest.raises(InvalidDataError, match=error_msg):
            g.validate(data)
    else:
        g.validate(data, raise_exception=False)

    assert caplog.records[0].levelname == "ERROR"
    assert caplog.text.strip().endswith(error_msg)


@parametrized_names_to_types
@pytest.mark.parametrize("required_names", [[], None])
@pytest.mark.parametrize(
    "data",
    [
        {},
        {"name": 0},
        {"name": 0.0},
        {"name": ""},
        {"name": True},
        {"name": ndarray([])},
    ],
)
def test_update_from_data(cls, names_to_types, required_names, data):
    """Verify update_from_data."""
    g = cls("g", names_to_types=names_to_types, required_names=required_names)
    required_names_before = set(g.required_names)
    g.update_from_data(data)

    assert set(g.keys()) == set(names_to_types.keys()) | set(data.keys())
    assert g.required_names == set(data.keys()) | required_names_before

    for name, value in data.items():
        assert g[name] == type(value)


def test_is_array(cls):
    """Verify is_array."""
    g = cls(
        "g",
        names_to_types={
            "a_none": None,
            "a_float": float,
            "a_ndarray": ndarray,
            "a_list": list,
        },
    )

    for name in ("a_none", "a_float"):
        assert not g.is_array(name)

    for name in ("a_ndarray", "a_list"):
        assert g.is_array(name)

    assert g.is_array("a_ndarray", numeric_only=True)
    assert not g.is_array("a_list", numeric_only=True)


def test_is_array_error(cls):
    """Verify is_array error."""
    g = cls("g")
    with pytest.raises(KeyError, match="foo"):
        g.is_array("foo")


NAMES = [
    set(),
    {"name1"},
    {"name1", "name2"},
]


@pytest.mark.parametrize("names", NAMES)
@pytest.mark.parametrize("required_names", [None, *NAMES])
def test_restrict_to(cls, names, required_names):
    """Verify restrict_to."""
    names_to_types = {"name1": None, "name2": int}
    g = cls(
        "g",
        names_to_types=names_to_types,
        required_names=required_names,
    )
    defaults = create_defaults(names_to_types)
    g.defaults.update(defaults)

    g_required_names_before = set(g.required_names)

    g.restrict_to(names)

    assert set(g) == set(names)
    assert g.required_names == g_required_names_before & set(names)

    for name in names:
        assert g.defaults[name] == defaults[name]
    assert len(g.defaults) == len(names)


def test_convert_to_simple_grammar(cls):
    """Verify grammar conversion."""
    g = cls("g")
    assert id(g.to_simple_grammar()) == id(g)


@parametrized_names_to_types
def test_required_names(cls, names_to_types):
    """Verify required_names."""
    g = cls("g", names_to_types=names_to_types)
    assert g.required_names == set(names_to_types.keys())


def test_repr(cls):
    g = cls("g", names_to_types={"name1": int, "name2": str}, required_names=["name1"])
    g.defaults["name2"] = "foo"
    assert (
        repr(g)
        == """
Grammar name: g
   Required elements:
      name1:
         Type: <class 'int'>
   Optional elements:
      name2:
         Type: <class 'str'>
         Default: foo
""".strip()
    )
    assert g._repr_html_() == REPR_HTML_WRAPPER.format(
        "Grammar name: g<br/>"
        "<ul>"
        "<li>Required elements:"
        "<ul>"
        "<li>name1:"
        "<ul>"
        "<li>Type: &lt;class &#x27;int&#x27;&gt;</li>"
        "</ul>"
        "</li>"
        "</ul>"
        "</li>"
        "<li>Optional elements:"
        "<ul>"
        "<li>name2:"
        "<ul>"
        "<li>Type: &lt;class &#x27;str&#x27;&gt;</li>"
        "<li>Default: foo</li>"
        "</ul>"
        "</li>"
        "</ul>"
        "</li>"
        "</ul>"
    )


def test_serialization(cls):
    """Check that the SimpleGrammar can be serialized."""
    g = cls("g", names_to_types={"name1": int, "name2": str}, required_names=["name1"])
    serialized_grammar = pickle.dumps(g)
    deserialized_grammar = pickle.loads(serialized_grammar)

    for k, v in g.items():
        assert deserialized_grammar[k] == v


def test_rename(cls):
    """Verify rename."""
    names_to_types = {"name1": int, "name2": str}
    g = cls("g", names_to_types=names_to_types, required_names=["name1"])
    defaults = create_defaults(names_to_types)
    g.defaults.update(defaults)

    g.rename_element("name1", "n:name1")
    g.rename_element("name2", "n:name2")

    assert list(g.required_names) == ["n:name1"]
    assert not names_to_types.keys() & set(g)
    assert ["n:name1", "n:name2"] == sorted(g.names)

    for name, value in defaults.items():
        assert g.defaults[f"n:{name}"] == value


def test_rename_error(cls):
    """Verify rename error."""
    g = cls("g")
    with pytest.raises(KeyError, match="foo"):
        g.rename_element("foo", "bar")


def test_copy(cls):
    """Verify copy."""
    g = cls("g")
    g.update_from_names(["name"])
    g.defaults["name"] = 1.0
    g_copy = g.copy()
    assert g_copy["name"] is g["name"]
    assert g_copy.defaults["name"] is g.defaults["name"]
    assert next(iter(g_copy.required_names)) is next(iter(g.required_names))
