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
#    INITIAL AUTHORS - initial API and implementation and/or
#                      initial documentation
#        :author:  Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""A coupling study analysis generating an N2 from an Excel specification."""

from __future__ import annotations

from typing import TYPE_CHECKING
from typing import ClassVar

from gemseo import generate_coupling_graph
from gemseo import generate_n2_plot
from gemseo.utils.study_analyses.xls_study_parser import XLSStudyParser

if TYPE_CHECKING:
    from pathlib import Path

    from gemseo.core.discipline import MDODiscipline
    from gemseo.utils.matplotlib_figure import FigSizeType


class CouplingStudyAnalysis:
    """A coupling study analysis from an Excel specification.

    Based on an Excel file defining disciplines in terms of input and output names,
    this analysis generates an N2 (equivalent to the Design Structure Matrix) diagram,
    showing the couplings between the disciplines.

    The Excel file shall contain one sheet per discipline:

    - the name of the sheet shall have the discipline name,
    - the sheet shall define the input names of the discipline
      as a vertical succession of cells starting with ``Inputs``:

        .. table:: Inputs

            +--------------+
            | Inputs       |
            +--------------+
            | input_name_1 |
            +--------------+
            | ...          |
            +--------------+
            | input_name_N |
            +--------------+

    - the sheet shall define the output names of the discipline
      as a vertical succession of cells starting with ``Outputs``:

    .. table:: Outputs

            +---------------+
            | Outputs       |
            +---------------+
            | output_name_1 |
            +---------------+
            | ...           |
            +---------------+
            | output_name_N |
            +---------------+

    - the empty lines of the series ``Inputs`` and ``Outputs`` are ignored,
    - the sheet may contain other data, but these will not be taken into account.
    """  # noqa: E501

    _HAS_SCENARIO: ClassVar[bool] = False
    """Whether the Excel file is supposed to have a scenario sheet."""

    study: XLSStudyParser
    """The XLSStudyParser instance built from the Excel file."""

    disciplines: dict[str, MDODiscipline]
    """The disciplines."""

    def __init__(self, xls_study_path: str | Path) -> None:
        """
        Args:
            xls_study_path: The path to the Excel file describing the study.
        """  # noqa: D205 D212 D415
        self.study = XLSStudyParser(xls_study_path, self._HAS_SCENARIO)
        self.disciplines = self.study.disciplines

    def generate_n2(
        self,
        file_path: str | Path = "n2.pdf",
        show_data_names: bool = True,
        save: bool = True,
        show: bool = False,
        fig_size: FigSizeType = (15, 10),
        show_html: bool = False,
    ) -> None:
        """Generate the N2 based on the disciplines.

        Args:
            file_path: The file path to save the static N2 chart.
            show_data_names: Whether to show the names of the coupling variables
                between two disciplines;
                otherwise,
                circles are drawn,
                whose size depends on the number of coupling names.
            save: Whether to save the static N2 chart.
            show: Whether to show the static N2 chart.
            fig_size: The width and height of the static N2 chart.
            show_html: Whether to display the interactive N2 chart in a web browser.
        """
        generate_n2_plot(
            list(self.disciplines.values()),
            file_path,
            show_data_names,
            save,
            show,
            fig_size,
            show_html,
        )

    def generate_coupling_graph(
        self, file_path: str | Path = "coupling_graph.pdf", full: bool = True
    ) -> None:
        """Generate the coupling graph based on the disciplines.

        Args:
            file_path: The file path to save the coupling graph.
            full: Whether to generate the full coupling graph.
                Otherwise, generate the condensed one.
        """
        generate_coupling_graph(list(self.disciplines.values()), file_path, full)
