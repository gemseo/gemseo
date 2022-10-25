# Copyright 2021 IRT Saint-Exupéry, https://www.irt-saintexupery.com
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

import csv
from pathlib import Path

import jinja2

ENV = jinja2.Environment(loader=jinja2.FileSystemLoader(Path(__file__).parent))


def create_authors_page(path):
    template = ENV.get_template("authors.tmpl")
    with (path / "_static" / "authors" / "authors.csv").open(
        "r", encoding="UTF-8", newline=""
    ) as f:
        authors = {
            surname: [name, surname, file_name]
            for (name, surname, file_name) in csv.reader(f)
        }
        authors = [authors[surname] for surname in sorted(authors)]

    doc = template.render(authors=authors)

    with open(path / "authors.rst", "w", encoding="utf-8") as f:
        f.write(doc)
