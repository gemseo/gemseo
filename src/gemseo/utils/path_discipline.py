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
# François Gallard
"""A dummy discipline for tests."""

from __future__ import annotations

from pathlib import Path

from gemseo import MDODiscipline


class PathDiscipline(MDODiscipline):
    """A toy discipline that takes Path as input and stores a Path attribute."""

    def __init__(self, tmp_path: Path):
        """Constructor.

        Args:
            tmp_path: any path.
        """
        super().__init__(grammar_type=MDODiscipline.GrammarType.SIMPLE)
        self.input_grammar.update_from_types({"x": Path})
        self.output_grammar.update_from_types({"y": int})
        self.default_inputs["x"] = tmp_path
        self.local_path = tmp_path

    def _run(self):
        self.local_data["y"] = 1
