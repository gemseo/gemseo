from __future__ import annotations

import re
from typing import Any
from typing import cast
from typing import Iterable

from docutils import nodes
from docutils.nodes import Element
from sphinx import addnodes
from sphinx.application import Sphinx
from sphinx.util import inspect


def record_defaults(
    app: Sphinx,
    objtype: str,
    name: str,
    obj: Any,
    options: dict,
    args: str,
    retann: str,
) -> None:
    """Record kwargs defaults to env object."""
    try:
        if callable(obj):
            defaults = app.env.temp_data.setdefault("defaults", {})
            obj_defaults = defaults.setdefault(name, {})
            sig = inspect.signature(obj, type_aliases=app.config.autodoc_type_aliases)
            for param in sig.parameters.values():
                if param.default is not param.empty:
                    obj_defaults[param.name] = param.default
    except (TypeError, ValueError):
        pass


def merge_defaults(
    app: Sphinx, domain: str, objtype: str, contentnode: Element
) -> None:
    if domain != "py":
        return
    if not app.config.autodoc_kwargs_defaults:
        return

    try:
        signature = cast(addnodes.desc_signature, contentnode.parent[0])
        if signature["module"]:
            fullname = ".".join([signature["module"], signature["fullname"]])
        else:
            fullname = signature["fullname"]
    except KeyError:
        # signature node does not have valid context info for the target object
        return

    defaults = app.env.temp_data.get("defaults", {})

    if fullname not in defaults:
        return

    field_lists = [n for n in contentnode if isinstance(n, nodes.field_list)]

    if not field_lists:
        field_lists = insert_field_list(contentnode)

    for field_list in field_lists:
        modify_field_list(
            field_list, defaults[fullname], app.config.autodoc_kwargs_defaults_pattern
        )


def insert_field_list(node: Element) -> nodes.field_list:
    field_list = nodes.field_list()
    desc = [n for n in node if isinstance(n, addnodes.desc)]
    if desc:
        # insert just before sub object descriptions (ex. methods, nested classes, etc.)
        index = node.index(desc[0])
        node.insert(index - 1, [field_list])
    else:
        node += field_list

    return field_list


def modify_field_list(
    node: nodes.field_list, defaults: dict[str, Any], pattern: str
) -> None:
    fields = cast(Iterable[nodes.field], node)

    for field in fields:
        field_name = field[0].astext()
        parts = re.split(" +", field_name)
        if parts[0] != "param":
            continue

        if len(parts) == 2:
            # :param xxx:
            name = parts[1]
        elif len(parts) > 2:
            # :param xxx yyy:
            name = " ".join(parts[2:])
        else:
            continue

        default = defaults.get(name)
        if default is None:
            continue

        if isinstance(default, str):
            default = f'"{default}"'

        field[1] += nodes.paragraph("", pattern.format(default))


_KWARGS_DEFAULTS_PATTERN = "By default it is set to {}."


def setup(app: Sphinx) -> dict[str, Any]:
    app.add_config_value("autodoc_kwargs_defaults", False, True)
    app.add_config_value(
        "autodoc_kwargs_defaults_pattern", _KWARGS_DEFAULTS_PATTERN, True
    )
    app.setup_extension("sphinx.ext.autodoc")
    app.connect("autodoc-process-signature", record_defaults)
    app.connect("object-description-transform", merge_defaults)

    return {
        "version": "builtin",
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
