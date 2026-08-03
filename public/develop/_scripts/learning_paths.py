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
"""Assemble the documentation home learning paths into a single JSON asset.

Each goal is described by a ``docs/learning_paths/<goal>.yml`` file and the
display order lives in ``_config.yml``. This script reads them at build time,
validates the content and writes ``assets/learning_paths.json``, which the home
page fetches and renders.
"""

from __future__ import annotations

import json
from pathlib import Path

import mkdocs_gen_files
import yaml

_TYPES = frozenset({"explanation", "tutorial", "howto", "reference"})

_SOURCE_DIR = Path("docs/learning_paths")
_OUTPUT_PATH = "assets/learning_paths.json"


def _load_yaml(path: Path) -> dict:
    """Load a YAML file into a dictionary.

    Args:
        path: The path to the YAML file.

    Returns:
        The parsed content.

    Raises:
        TypeError: When the file is empty or does not hold a mapping.
    """
    with path.open(encoding="utf-8") as file:
        content = yaml.safe_load(file)
    if not isinstance(content, dict):
        msg = f"{path}: expected a YAML mapping, got {type(content).__name__}."
        raise TypeError(msg)
    return content


def _validate_goal(goal: dict, goal_id: str, source: Path) -> None:
    """Validate a single goal definition.

    The internal `path` values (of resources and prerequisites)
    are NOT checked against the site files:
    these links are injected client-side from the JSON,
    so mkdocs `strict` cannot see them and a typo becomes
    a silent 404 on the home page.
    Keep them in sync with the docs by hand when editing the YAML files.

    Args:
        goal: The parsed goal definition.
        goal_id: The identifier from `_config.yml` (also the file stem);
            the goal's own `id` field must match it.
        source: The file the goal was read from, used in error messages.

    Raises:
        ValueError: When a required key is missing,
            an enum value is invalid,
            or
            the goal's `id` does not match `goal_id`.
    """
    for key in ("id", "code", "title", "blurb", "resources"):
        if key not in goal:
            msg = f"{source}: missing required key '{key}'."
            raise ValueError(msg)

    if goal["id"] != goal_id:
        msg = (
            f"{source}: goal id '{goal['id']}' does not match '{goal_id}' "
            f"(from _config.yml order / file name)."
        )
        raise ValueError(msg)

    for prerequisite in goal.get("prerequisites") or ():
        for key in ("title", "path"):
            if key not in prerequisite:
                msg = f"{source}: prerequisite missing required key '{key}'."
                raise ValueError(msg)

    for resource in goal["resources"]:
        for key in ("type", "title", "desc", "path"):
            if key not in resource:
                msg = f"{source}: resource missing required key '{key}'."
                raise ValueError(msg)
        if resource["type"] not in _TYPES:
            msg = f"{source}: invalid type '{resource['type']}' (expected {sorted(_TYPES)})."
            raise ValueError(msg)


def _build_payload() -> dict:
    """Build the JSON payload from the learning path files.

    Returns:
        A dictionary with the ordered goals.

    Raises:
        FileNotFoundError: When a goal listed in the order has no YAML file.
        TypeError: When a YAML file is empty or does not hold a mapping.
        ValueError: When a goal definition is invalid.
    """
    config = _load_yaml(_SOURCE_DIR / "_config.yml")

    goals = []
    for goal_id in config["order"]:
        source = _SOURCE_DIR / f"{goal_id}.yml"
        if not source.is_file():
            msg = f"{_SOURCE_DIR}/_config.yml lists '{goal_id}' but {source} does not exist."
            raise FileNotFoundError(msg)
        goal = _load_yaml(source)
        _validate_goal(goal, goal_id, source)
        goals.append(goal)

    return {"goals": goals}


with mkdocs_gen_files.open(_OUTPUT_PATH, "w") as output_file:
    json.dump(_build_payload(), output_file, ensure_ascii=False)
