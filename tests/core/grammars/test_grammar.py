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

import pickle
import re
from copy import deepcopy
from itertools import chain
from itertools import combinations
from typing import TYPE_CHECKING
from typing import Any

import pytest
from numpy import zeros
from pydantic.fields import FieldInfo

from gemseo.core.grammars.errors import InvalidDataError
from gemseo.core.grammars.factory import GrammarFactory
from gemseo.core.grammars.grammar_properties import GrammarProperties
from gemseo.core.grammars.json_grammar import JSONGrammar
from gemseo.core.grammars.pydantic_grammar import PydanticGrammar
from gemseo.core.grammars.simple_grammar import SimpleGrammar
from gemseo.core.grammars.simpler_grammar import SimplerGrammar
from gemseo.utils.repr_html import REPR_HTML_WRAPPER
from gemseo.utils.testing.helpers import do_not_raise

if TYPE_CHECKING:
    from collections.abc import Iterator
    from collections.abc import Mapping

    from gemseo.core.grammars.base_grammar import BaseGrammar

FACTORY = GrammarFactory()


@pytest.fixture(params=FACTORY.class_names)
def grammar_class(request) -> type[BaseGrammar]:
    """Iterate over the grammar classes."""
    return FACTORY.get_class(request.param)


@pytest.fixture
def grammar(grammar_class) -> BaseGrammar:
    """Iterate over empty grammar instances of all types."""
    return grammar_class("g")


def test_serialize(grammar) -> None:
    """Check that a grammar can be properly serialized."""
    grammar.update_from_types({"x": int, "y": bool})
    grammar.add_namespace("x", "n")
    grammar.required_names.remove("y")

    pickled_grammar = pickle.loads(pickle.dumps(grammar))

    assert pickled_grammar.name == grammar.name
    assert pickled_grammar.required_names == grammar.required_names
    assert pickled_grammar.to_namespaced == grammar.to_namespaced
    assert pickled_grammar.from_namespaced == grammar.from_namespaced

    pickled_grammar.validate({"n:x": 1, "y": False})

    if grammar.__class__ != SimplerGrammar:
        for data in [{"y": False}, {"n:x": 1.5}, {"n:x": 1, "y": 3}]:
            with pytest.raises(InvalidDataError):
                pickled_grammar.validate(data)


def test_init_error(grammar_class):
    """Verify __init__ error."""
    msg = "The grammar name cannot be empty."
    with pytest.raises(ValueError, match=msg):
        grammar_class("")


def test_init(grammar):
    """Verify __init__."""
    assert grammar.name == "g"
    assert not grammar
    assert not grammar.defaults
    assert not grammar.descriptions
    assert not grammar.required_names


def test_delitem_error(grammar) -> None:
    """Verify that removing a non-existing item raises an error."""
    msg = "foo"
    with pytest.raises(KeyError, match=msg):
        del grammar["foo"]


def test_delitem(grammar) -> None:
    """Verify __delitem__."""
    grammar.update_from_names(["name"])
    grammar.defaults["name"] = 0
    grammar.descriptions["name"] = "A description."
    del grammar["name"]
    assert "name" not in grammar
    assert "name" not in grammar.required_names
    assert "name" not in grammar.defaults
    assert "name" not in grammar.descriptions


def test_getitem_error(grammar):
    """Verify __getitem__ error."""
    msg = "foo"
    with pytest.raises(KeyError, match=msg):
        grammar["foo"]


parametrized_names_to_types = pytest.mark.parametrize(
    "names_to_types",
    [
        {},
        {"name": str},
    ],
)


@parametrized_names_to_types
def test_len(grammar, names_to_types) -> None:
    """Verify __len__."""
    grammar.update_from_types(names_to_types)
    assert len(grammar) == len(names_to_types)


@parametrized_names_to_types
def test_iter(grammar, names_to_types) -> None:
    """Verify __iter__."""
    grammar.update_from_types(names_to_types)
    assert list(iter(grammar)) == list(names_to_types)


@parametrized_names_to_types
def test_names(grammar, names_to_types) -> None:
    """Verify names getter."""
    grammar.update_from_types(names_to_types)
    assert grammar.names == names_to_types.keys()


@parametrized_names_to_types
def test_names_without_namespace(grammar, names_to_types) -> None:
    """Verify names_without_namespace."""
    grammar.update_from_types(names_to_types)
    assert tuple(grammar.names_without_namespace) == tuple(names_to_types.keys())

    if names_to_types:
        grammar.add_namespace("name", "n")
        assert tuple(grammar.names_without_namespace) == tuple(names_to_types.keys())


def create_defaults(names_to_types: Mapping[str, type]) -> dict[str, Any]:
    """Return default data from names to types.

    Args:
        names_to_types: The mapping from names to types.

    Returns:
        The default data.
    """
    defaults = {}
    for name, type_ in names_to_types.items():
        defaults[name] = type_(0)
    return defaults


@parametrized_names_to_types
def test_clear(grammar, names_to_types) -> None:
    """Verify clear."""
    grammar.update_from_types(names_to_types)
    grammar.defaults.update(create_defaults(names_to_types))
    grammar.descriptions.update(dict.fromkeys(names_to_types, "A description."))
    grammar.clear()
    assert not grammar
    assert not grammar.required_names
    assert not grammar.defaults
    assert not grammar.descriptions


NAMES = [
    set(),
    {"name1"},
    {"name1", "name2"},
]


@pytest.mark.parametrize("names", NAMES)
@pytest.mark.parametrize("required_names", [None, *NAMES])
def test_restrict_to(grammar, names, required_names) -> None:
    """Verify restrict_to."""
    names_to_types = {"name1": int, "name2": int}
    grammar.update_from_names(names_to_types)
    defaults = create_defaults(names_to_types)
    grammar.defaults.update(defaults)
    grammar.descriptions.update(dict.fromkeys(names_to_types, "A description"))
    g_required_names_before = set(grammar.required_names)

    grammar.restrict_to(names)

    assert set(grammar) == set(names)
    assert set(grammar.required_names) == g_required_names_before & set(names)

    for name in names:
        assert grammar.defaults[name] == defaults[name]
        assert grammar.descriptions[name] == "A description"

    assert len(grammar.defaults) == len(names)
    assert len(grammar.descriptions) == len(names)


def test_restrict_to_error(grammar) -> None:
    """Verify that raises the expected error."""
    msg = "The name 'foo' is not in the grammar."
    with pytest.raises(KeyError, match=msg):
        grammar.restrict_to(["foo"])


def test_convert_to_simple_grammar(grammar) -> None:
    """Verify conversion to simple grammar."""
    names_to_types = {"name1": int, "name2": int}
    grammar.update_from_types(names_to_types)
    simple_grammar = grammar.to_simple_grammar()
    assert set(grammar) == set(simple_grammar)
    assert grammar.required_names == simple_grammar.required_names
    assert grammar.defaults == simple_grammar.defaults
    assert grammar.descriptions == simple_grammar.descriptions
    assert isinstance(simple_grammar, SimpleGrammar)
    assert simple_grammar.items() == names_to_types.items()


def test_required_names(grammar) -> None:
    """Verify required_names."""
    names_to_types = {"name1": int, "name2": int}
    grammar.update_from_types(names_to_types)
    assert grammar.required_names == set(names_to_types.keys())


def test_rename(grammar) -> None:
    """Verify rename."""
    grammar.update_from_names(["name"])
    grammar.defaults["name"] = 0
    grammar.descriptions["name"] = "A description."

    grammar.rename_element("name", "new_name")

    assert grammar.required_names == {"new_name"}
    assert list(grammar) == ["new_name"]
    assert grammar.defaults.keys() == {"new_name"}
    assert grammar.defaults["new_name"] == 0
    assert grammar.descriptions["new_name"] == "A description."

    # Cover the case when renaming an element that is not required.
    grammar.required_names.remove("new_name")
    grammar.rename_element("new_name", "new_new_name")


def test_rename_error(grammar) -> None:
    """Verify rename error."""
    with pytest.raises(KeyError, match="foo"):
        grammar.rename_element("foo", "bar")


def test_copy(grammar) -> None:
    """Verify copy."""
    grammar.update_from_names(["name1", "name2"])
    grammar.defaults["name1"] = 0
    grammar.descriptions["name1"] = "A description."
    grammar.add_namespace("name2", "ns")
    grammar_copy = grammar.copy()
    assert grammar_copy.keys() == grammar.keys()
    assert grammar_copy.defaults == grammar.defaults
    assert grammar_copy.descriptions == grammar.descriptions
    assert grammar_copy.required_names == grammar.required_names
    assert grammar_copy.from_namespaced == grammar.from_namespaced
    assert grammar_copy.to_namespaced == grammar.to_namespaced

    # Verify that the copy is deep.
    grammar_copy.add_namespace("name1", "ns")
    assert "ns:name1" in grammar_copy
    assert "ns:name1" not in grammar

    grammar_copy.clear()
    assert grammar.keys()
    assert grammar.defaults
    assert grammar.descriptions
    assert grammar.required_names
    assert grammar.from_namespaced
    assert grammar.to_namespaced


def test_validate_empty_grammar(grammar) -> None:
    """Verify that an empty grammar can validate everything."""
    grammar.validate({"name": 0})


def test_repr(grammar) -> None:
    """Verify __repr__."""
    names_to_types = {"name1": int, "name2": str}
    grammar.update_from_types(names_to_types)
    grammar.required_names.remove("name2")
    grammar.defaults["name2"] = "foo"
    grammar.descriptions["name2"] = "A description."

    is_json_grammar = grammar.__class__ is JSONGrammar
    if is_json_grammar:
        type_repr = {
            "name1": "integer",
            "name2": "string",
        }
    else:
        type_repr = {
            "name1": "<class 'int'>",
            "name2": "<class 'str'>",
        }

    assert (
        repr(grammar)
        == f"""
Grammar name: g
   Required elements:
      name1:
         Type: {type_repr["name1"]}
   Optional elements:
      name2:
         Description: A description.
         Type: {type_repr["name2"]}
         Default: foo
""".strip()
    )

    if not is_json_grammar:
        type_repr = {
            "name1": "&lt;class &#x27;int&#x27;&gt;",
            "name2": "&lt;class &#x27;str&#x27;&gt;",
        }

    assert grammar._repr_html_() == REPR_HTML_WRAPPER.format(
        "Grammar name: g<br/>"
        "<ul>"
        "<li>Required elements:"
        "<ul>"
        "<li>name1:"
        "<ul>"
        f"<li>Type: {type_repr['name1']}</li>"
        "</ul>"
        "</li>"
        "</ul>"
        "</li>"
        "<li>Optional elements:"
        "<ul>"
        "<li>name2:"
        "<ul>"
        "<li>Description: A description.</li>"
        f"<li>Type: {type_repr['name2']}</li>"
        "<li>Default: foo</li>"
        "</ul>"
        "</li>"
        "</ul>"
        "</li>"
        "</ul>"
    )


ARRAY = zeros((1,))
MERGE_TO_VALID_DATA = {
    # Without merged update.
    False: {
        "required_name1": 0,
        "required_name2": ARRAY,
        "optional_name2": ARRAY,
        "new_name": ARRAY,
    },
    # With merged update.
    True: {
        "required_name1": 0,
        "required_name2": 0,
        "optional_name2": 0,
        "new_name": ARRAY,
    },
}
INVALID_DATA = {
    "optional_name1": 0.0,
}


def assert_updated(
    grammar: BaseGrammar,
    merge: bool,
    excluded_names: set[str] = frozenset(()),
) -> None:
    """Assert an updated grammar.

    We check that the elements suffixed with 2 have been updated and that
    a new element is added, while taking into account the fact that they are not
    updated when they belong to the excluded names.

    Args:
        grammar: The grammar to verify.
        merge: Whether the update was done with merge.
        excluded_names: The names excludes from update.
    """
    assert set(grammar) == {
        "required_name1",
        "required_name2",
        "optional_name1",
        "optional_name2",
        "new_name",
    } - (excluded_names & {"new_name"})

    # Currently, changing a required element does not change the bound default.
    assert grammar.defaults == {"optional_name1": 0, "optional_name2": 0}

    assert set(grammar.required_names) == {
        "required_name1",
        "required_name2",
        "optional_name2",
        "new_name",
    } - (excluded_names & {"new_name", "optional_name2"})

    data = {}
    for name, value in MERGE_TO_VALID_DATA[True].items():
        if name in excluded_names:
            data[name] = value
        else:
            data[name] = MERGE_TO_VALID_DATA[False][name]

    grammar.validate(data)

    if merge:
        # The elements that have been merged shall also validate the types existing
        # before the merge (if they were not excluded).
        data = {}
        for name, value in MERGE_TO_VALID_DATA[False].items():
            if name in excluded_names:
                data[name] = MERGE_TO_VALID_DATA[True][name]
            else:
                data[name] = value

        grammar.validate(data)

    with pytest.raises(InvalidDataError):
        grammar.validate({})

    if not isinstance(grammar, SimplerGrammar):
        for name, value in INVALID_DATA.items():
            with pytest.raises(InvalidDataError):
                grammar.validate({name: value})


def prepare_grammar(grammar: BaseGrammar) -> None:
    """Prepare a grammar for testing the update methods.

    The grammar will contain required and optional names, 2 of each.

    Args:
        grammar: The grammar to prepare.
    """
    grammar_type = grammar.__class__

    if grammar_type in {SimpleGrammar, SimplerGrammar}:
        grammar._SimpleGrammar__names_to_types["required_name1"] = int
        grammar._SimpleGrammar__names_to_types["optional_name1"] = int
        grammar._SimpleGrammar__names_to_types["required_name2"] = int
        grammar._SimpleGrammar__names_to_types["optional_name2"] = int
    elif grammar_type is PydanticGrammar:
        grammar._PydanticGrammar__model.model_fields["required_name1"] = FieldInfo(
            annotation=int
        )
        grammar._PydanticGrammar__model.model_fields["optional_name1"] = FieldInfo(
            annotation=int, default=0
        )
        grammar._PydanticGrammar__model.model_fields["required_name2"] = FieldInfo(
            annotation=int
        )
        grammar._PydanticGrammar__model.model_fields["optional_name2"] = FieldInfo(
            annotation=int, default=0
        )
        grammar._PydanticGrammar__model.model_needs_rebuild = True
    elif grammar_type is JSONGrammar:
        grammar._JSONGrammar__schema_builder.add_object({"required_name1": 0}, False)
        grammar._JSONGrammar__schema_builder.add_object({"optional_name1": 0}, False)
        grammar._JSONGrammar__schema_builder.add_object({"required_name2": 0}, False)
        grammar._JSONGrammar__schema_builder.add_object({"optional_name2": 0}, False)

    grammar.required_names.add("required_name1")
    grammar.defaults["optional_name1"] = 0
    grammar.required_names.add("required_name2")
    grammar.defaults["optional_name2"] = 0


parametrized_merge = pytest.mark.parametrize("merge", [True, False])


def check_update_raise(grammar: BaseGrammar, merge: bool):
    """Return a context manager to do nothing or raises.

    Simple{r}Grammar shall do not support merge and shall raise.

    Args:
        grammar: The grammar.
        merge: Whether the update is done with merge.
    """
    if merge and isinstance(grammar, SimpleGrammar):
        return pytest.raises(ValueError)
    return do_not_raise()


UPDATE_DATA = {
    "required_name2": ARRAY,
    "optional_name2": ARRAY,
    "new_name": ARRAY,
}
UPDATE_TYPES = {name: type(value) for name, value in UPDATE_DATA.items()}


def powerset(iterable) -> Iterator[tuple[Any, ...]]:
    """Compute powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3).

    See https://docs.python.org/3/library/itertools.html#itertools-recipes.

    Args:
        iterable: An iterable.

    Returns:
        The powerset.
    """
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(len(s) + 1))


@parametrized_merge
@pytest.mark.parametrize("excluded_names", powerset(UPDATE_DATA))
def test_update(grammar, merge, excluded_names) -> None:
    """Verify update."""
    prepare_grammar(grammar)

    # Update from empty.
    names_before = grammar.names
    grammar.update({}, merge=merge, excluded_names=excluded_names)
    assert grammar.names == names_before

    # Update from non-empty.
    other_grammar = grammar.__class__("g")
    other_grammar.update_from_names(UPDATE_DATA.keys())
    with check_update_raise(grammar, merge):
        grammar.update(other_grammar, merge=merge, excluded_names=excluded_names)
        assert_updated(grammar, merge=merge, excluded_names=set(excluded_names))


def test_update_with_namespace(grammar) -> None:
    """Check update with namespace."""
    grammar.update_from_names(["x"])
    other_grammar = deepcopy(grammar)
    grammar.add_namespace("x", "n")
    other_grammar.add_namespace("x", "other_n")
    grammar.update(other_grammar)
    assert grammar.to_namespaced == {"x": ["n:x", "other_n:x"]}
    assert grammar.from_namespaced == {"n:x": "x", "other_n:x": "x"}


@parametrized_merge
def test_update_from_names(grammar, merge) -> None:
    """Verify update_from_names."""
    prepare_grammar(grammar)

    # Update from empty.
    names_before = grammar.names
    grammar.update_from_names((), merge=merge)
    assert grammar.names == names_before

    # Update from non-empty.
    with check_update_raise(grammar, merge):
        grammar.update_from_names(UPDATE_DATA.keys(), merge=merge)
        assert_updated(grammar, merge=merge)


@parametrized_merge
def test_update_from_types(grammar, merge) -> None:
    """Verify update_from_types."""
    prepare_grammar(grammar)

    # Update from empty.
    names_before = grammar.names
    grammar.update_from_types({}, merge=merge)
    assert grammar.names == names_before

    # Update from non-empty.
    with check_update_raise(grammar, merge):
        grammar.update_from_types(UPDATE_TYPES, merge=merge)
        assert_updated(grammar, merge=merge)


@parametrized_merge
def test_update_from_data(grammar, merge) -> None:
    """Verify update_from_data."""
    prepare_grammar(grammar)

    # Update from empty.
    names_before = grammar.names
    grammar.update_from_data({}, merge=merge)
    assert grammar.names == names_before

    # Update from non-empty.
    with check_update_raise(grammar, merge):
        grammar.update_from_data(UPDATE_DATA, merge=merge)
        assert_updated(grammar, merge=merge)


@pytest.mark.parametrize(
    "data",
    [
        0,
        0.0,
        # TODO: waiting fix support in pydantic
        # 0.0j,
        "0",
        False,
        zeros((1,), dtype=int),
        zeros((1,), dtype=float),
        zeros((1,), dtype=complex),
    ],
)
def test_validate(grammar, data) -> None:
    """Verify validate."""
    data = {"name": data}
    grammar.update_from_data(data)
    grammar.validate(data)


@pytest.mark.parametrize("raises", [True, False])
def test_validate_error_missing_required(grammar, raises, caplog):
    grammar.update_from_names(["name"])

    match = "Grammar g: validation failed.\nMissing required names: name."

    if raises:
        with pytest.raises(InvalidDataError, match=match):
            grammar.validate({})
    else:
        grammar.validate({}, raise_exception=False)

    assert caplog.records[0].levelname == "ERROR"
    assert caplog.text.strip().endswith(match)


def test_validate_empty(grammar) -> None:
    """Verify validate of an empty grammar."""
    # It validates everything.
    data = {"name": 0}
    grammar.validate(data)


def test_add_namespace(grammar) -> None:
    """Check add_namespace."""
    grammar.update_from_types({"x": int, "y": bool})
    grammar.add_namespace("x", "n")

    assert grammar.to_namespaced == {"x": "n:x"}
    assert grammar.from_namespaced == {"n:x": "x"}
    assert "x" not in grammar
    assert "x" not in grammar.required_names
    assert "n:x" in grammar
    assert "n:x" in grammar.required_names

    match = "The name 'dummy' is not in the grammar."
    with pytest.raises(KeyError, match=re.escape(match)):
        grammar.add_namespace("dummy", "n")

    match = "The name 'x' is not in the grammar."
    with pytest.raises(KeyError, match=re.escape(match)):
        grammar.add_namespace("x", "n")

    match = "The variable 'x' already has a namespace ('n')."
    with pytest.raises(ValueError, match=re.escape(match)):
        grammar.add_namespace("n:x", "")


def test_has_names(grammar):
    """Check has_names."""
    assert not grammar.has_names(("name",))
    grammar.update_from_types({"name": str})
    assert not grammar.has_names(("dummy",))
    assert not grammar.has_names(("dummy", "name"))
    assert grammar.has_names(("name",))
    assert grammar.has_names(())


def test_defaults_setter(grammar):
    """Check the defaults' setter."""
    grammar.update_from_types({"x": float})
    grammar.defaults = {"x": 1.0}

    assert isinstance(grammar.defaults, GrammarProperties)
    assert grammar.defaults == {"x": 1.0}


def test_descriptions_setter(grammar):
    """Check the descriptions' setter."""
    grammar.update_from_types({"x": float})
    grammar.descriptions = {"x": "foo"}

    assert isinstance(grammar.descriptions, GrammarProperties)
    assert grammar.descriptions == {"x": "foo"}


def test_name_including_colon(grammar):
    """Check that a grammar does not confuse names inc. colons with namespaced names."""
    g = SimpleGrammar("g")
    # ":" is the namespaces separator
    # and shall never be used by the end-user to define the name of an element.
    # This special character is reserved for namespace management.
    # The end-user must always use the method add_namespace to add namespace.
    # This test is just to check that
    # if the end-user does not follow these instructions,
    # its name will not be interpreted as a namespaced name.
    g.update_from_names(["x:y", "z"])
    g.add_namespace("z", "x")
    assert tuple(g.names) == ("x:y", "x:z")
    assert tuple(g.names_without_namespace) == ("x:y", "z")
