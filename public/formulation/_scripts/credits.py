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

# ISC License
#
# Copyright (c) 2021, Timothée Mazzucotelli
#
# Permission to use, copy, modify, and/or distribute this software for any
# purpose with or without fee is hereby granted, provided that the above
# copyright notice and this permission notice appear in all copies.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES
# WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF
# MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR
# ANY SPECIAL, DIRECT, INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES
# WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN
# ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF
# OR IN CONNECTION WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.

# Adapted from https://github.com/mkdocstrings/griffe/blob/main/scripts/gen_credits.py
"""Script to generate the project's credits."""

from __future__ import annotations

import os
import sys
from collections import defaultdict
from collections.abc import Iterable
from importlib.metadata import distributions
from itertools import chain
from pathlib import Path
from textwrap import dedent

from jinja2 import StrictUndefined
from jinja2.sandbox import SandboxedEnvironment
from packaging.requirements import Requirement

# YORE: EOL 3.10: Replace block with line 2.
if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

project_dir = Path(os.getenv("MKDOCS_CONFIG_DIR", "."))
with project_dir.joinpath("pyproject.toml").open("rb") as pyproject_file:
    pyproject = tomllib.load(pyproject_file)
project = pyproject["project"]
project_name = project["name"]
devdeps = [
    dep
    for group in pyproject["dependency-groups"].values()
    for dep in group
    if not dep.startswith("-e")
]

PackageMetadata = dict[str, str | Iterable[str]]
Metadata = dict[str, PackageMetadata]


def _merge_fields(metadata: dict) -> PackageMetadata:
    fields = defaultdict(list)
    for header, value in metadata.items():
        fields[header.lower()].append(value.strip())
    return {
        field: value
        if len(value) > 1 or field in ("classifier", "requires-dist")
        else value[0]
        for field, value in fields.items()
    }


def _norm_name(name: str) -> str:
    return name.replace("_", "-").replace(".", "-").lower()


def _requirements(deps: list[str]) -> dict[str, Requirement]:
    return {_norm_name((req := Requirement(dep)).name): req for dep in deps}


def _get_metadata() -> Metadata:
    metadata = {}
    for pkg in distributions():
        name = _norm_name(pkg.name)  # type: ignore[attr-defined,unused-ignore]
        metadata[name] = _merge_fields(pkg.metadata)  # type: ignore[arg-type]
        metadata[name]["spec"] = set()
        metadata[name].setdefault("summary", "")
        _set_license(metadata[name])
    return metadata


def _set_license(metadata: PackageMetadata) -> None:
    license_field = metadata.get("license-expression", metadata.get("license", ""))
    license_name = (
        license_field if isinstance(license_field, str) else " + ".join(license_field)
    )
    check_classifiers = license_name in (
        "UNKNOWN",
        "Dual License",
        "",
    ) or license_name.count("\n")
    if check_classifiers:
        license_names = [
            classifier.rsplit("::", 1)[1].strip()
            for classifier in metadata["classifier"]
            if classifier.startswith("License ::")
        ]
        license_name = " + ".join(license_names)
    metadata["license"] = license_name or "?"


def _get_deps(base_deps: dict[str, Requirement], metadata: Metadata) -> Metadata:
    deps = {}
    for dep_name, dep_req in base_deps.items():
        if dep_name not in metadata:
            metadata[dep_name] = {
                "spec": set(),
                "name": dep_name,
                "summary": "",
                "version": "",
                "license": "",
            }
        metadata[dep_name]["spec"] |= {str(spec) for spec in dep_req.specifier}  # type: ignore[operator]
        deps[dep_name] = metadata[dep_name]
    return deps


def _render_credits() -> str:
    metadata = _get_metadata()
    dev_dependencies = _get_deps(_requirements(devdeps), metadata)
    prod_dependencies = _get_deps(
        _requirements(
            chain(  # type: ignore[arg-type]
                project.get("dependencies", []),
                chain(*project.get("optional-dependencies", {}).values()),
            ),
        ),
        metadata,
    )

    template_data = {
        "project_name": project_name,
        "prod_dependencies": sorted(
            prod_dependencies.values(), key=lambda dep: str(dep["name"]).lower()
        ),
        "dev_dependencies": sorted(
            dev_dependencies.values(), key=lambda dep: str(dep["name"]).lower()
        ),
    }
    template_text = dedent(
        """
        # Credits

        These projects were used to build *{{ project_name }}*. **Thank you!**

        [Python](https://www.python.org/) |
        [uv](https://docs.astral.sh/uv) |
        [ruff](https://docs.astral.sh/ruff) |
        [pre-commit](https://pre-commit.com)

        {% macro dep_line(dep) -%}
        [{{ dep.name }}](https://pypi.org/project/{{ dep.name }}/) | {{ dep.summary }} | {{ ("`" ~ dep.spec|sort(reverse=True)|join(", ") ~ "`") if dep.spec else "" }} | `{{ dep.version }}` | {{ dep.license }}
        {%- endmacro %}

        {% if prod_dependencies -%}
        ### Runtime dependencies

        Project | Summary | Version (accepted) | Version (last resolved) | License
        ------- | ------- | ------------------ | ----------------------- | -------
        {% for dep in prod_dependencies -%}
        {{ dep_line(dep) }}
        {% endfor %}

        {% endif -%}
        {% if dev_dependencies -%}
        ### Development dependencies

        Project | Summary | Version (accepted) | Version (last resolved) | License
        ------- | ------- | ------------------ | ----------------------- | -------
        {% for dep in dev_dependencies -%}
        {{ dep_line(dep) }}
        {% endfor %}

        {% endif -%}
        """,
    )
    jinja_env = SandboxedEnvironment(undefined=StrictUndefined)
    return jinja_env.from_string(template_text).render(**template_data)


root = Path("_docs")
root.mkdir(exist_ok=True)
with (root / "credits.md").open("w", encoding="utf-8") as f:
    f.write(_render_credits())
