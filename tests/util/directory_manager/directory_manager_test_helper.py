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

"""Provide useful functions for DirectoryManager testing."""

from __future__ import annotations

from typing import TYPE_CHECKING

from gemseo import MDOScenario
from gemseo.core.discipline.discipline import Discipline
from gemseo.core.function.array_function import ArrayFunction
from gemseo.discipline.wrapper.disc_from_exe import DiscFromExe
from gemseo.formulation.idf_settings import IDF_Settings
from gemseo.problem.mdo.sobieski.discipline import SobieskiAerodynamics
from gemseo.problem.mdo.sobieski.discipline import SobieskiMission
from gemseo.problem.mdo.sobieski.discipline import SobieskiPropulsion
from gemseo.problem.mdo.sobieski.discipline import SobieskiStructure
from gemseo.problem.mdo.sobieski.standalone.design_space import SobieskiDesignSpace
from gemseo.space.design import DesignSpace

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from gemseo.formulation.core.base_settings import BaseFormulationSettings
    from gemseo.scenario.evaluation import EvaluationScenario
    from gemseo.util.typing import StrKeyMapping


def build_monolevel_scenario(
    formulation_settings_model: BaseFormulationSettings,
    **args,
) -> EvaluationScenario:
    """Build the scenario for SSBJ.

    Args:
        formulation_settings_model: The formulation settings model.

    Returns:
        The MDOScenario.
    """
    disciplines = [
        SobieskiPropulsion(),
        SobieskiAerodynamics(),
        SobieskiMission(),
        SobieskiStructure(),
    ]

    design_space = SobieskiDesignSpace()
    scenario = MDOScenario(
        disciplines=disciplines,
        design_space=design_space,
        formulation_settings=formulation_settings_model,
        **args,
    )
    scenario.add_objective("y_4", minimize=False)
    for c_name in ["g_1", "g_2", "g_3"]:
        scenario.add_constraint(
            c_name, constraint_type=ArrayFunction.ConstraintType.INEQ
        )
    return scenario


class DummyDiscipline1(Discipline):
    """A discipline that does nothing."""

    def __init__(
        self,
        name: str = "",
        input_names: Iterable[str] = (),
        output_names: Iterable[str] = (),
    ) -> None:
        """
        Args:
            input_names: The names of the input variables, if any.
            output_names: The names of the output variables, if any.
        """  # noqa: D205 D212 D415
        super().__init__(name=name)
        self.io.input_grammar.update_from_names(input_names)
        self.io.output_grammar.update_from_names(output_names)

    def _run(self, input_data: StrKeyMapping) -> StrKeyMapping | None:
        y = input_data["a"] * 2

        return {"y": y}


class DummyDiscipline2(DummyDiscipline1):
    """A discipline where the `_run` method calls for the parent's `_run` method."""

    def __init__(
        self,
        name: str = "",
        input_names: Iterable[str] = (),
        output_names: Iterable[str] = (),
    ) -> None:
        """
        Args:
            input_names: The names of the input variables, if any.
            output_names: The names of the output variables, if any.
        """  # noqa: D205 D212 D415
        super().__init__(name=name)
        self.io.input_grammar.update_from_names(input_names)
        self.io.output_grammar.update_from_names(output_names)

    def _run(self, input_data: StrKeyMapping) -> StrKeyMapping | None:
        z = input_data["b"] * super()._run(input_data)

        return {"z": z}


def create_scenario_with_inheriting_disciplines():
    """Create a scenario with one discipline that inherits from another."""
    discipline_a = DummyDiscipline1(
        name="DisciplineA", input_names="a", output_names="y"
    )

    discipline_b = DummyDiscipline2(
        name="DisciplineB", input_names="b", output_names="z"
    )
    disciplines = [discipline_a, discipline_b]

    ds = DesignSpace()
    ds.add_variable("a", lower_bound=-1, upper_bound=1, value=0.0)
    ds.add_variable("b", lower_bound=-1, upper_bound=1, value=0.0)

    scenario = MDOScenario(
        disciplines=disciplines, design_space=ds, formulation_settings=IDF_Settings()
    )
    scenario.add_objective("z", minimize=False)
    return scenario


def read_paths_from_txt(file_path: Path, root_path: Path) -> set[Path]:
    """A function that reads a list of paths from a file.

    This function ignores lines that start with "#". Which allows to insert comments
    in the file to explain the structure of the paths.

    Args:
        file_path: The path to the txt file to read.
        root_path: A root path to add to each of the paths that are read from the file.

    Returns:
        The paths read from the file, starting from the given root path.
    """
    paths = set()
    with file_path.open() as f:
        for line in f:
            if line.startswith("#"):
                continue
            paths.add(root_path / line.replace("\r", "").replace("\n", ""))
    return paths


def create_disc_from_exe(file_path: Path) -> DiscFromExe:
    """Create an executable discipline for testing."""
    sum_path = str(file_path / "sum_data.py")
    exec_cmd = f"python {sum_path} -i input.json -o output.json"

    disc: DiscFromExe = DiscFromExe(
        input_template=str(file_path / "input.json.template"),
        output_template=str(file_path / "output.json.template"),
        root_directory="",
        command_line=exec_cmd,
        input_filename="input.json",
        output_filename="output.json",
    )

    return disc
