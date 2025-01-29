from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from gemseo.disciplines.wrappers._base_disc_from_exe import _BaseDiscFromExe
from gemseo.disciplines.wrappers._base_executable_runner import _BaseExecutableRunner

if TYPE_CHECKING:
    from collections.abc import Mapping

    from gemseo.typing import StrKeyMapping


class ShellExecutableDiscipline(_BaseDiscFromExe):
    """A discipline to execute a shell script as an executable."""

    def __init__(self) -> None:
        # Create the Executable runner
        # The unique directories are created where the script is run (".")
        # We copy the script ``run_discipline.bash`` into the directory to use it
        exec_runner = _BaseExecutableRunner(
            "bash ./run_discipline.bash", ".", files=["./run_discipline.bash"]
        )

        super().__init__(exec_runner, name="ShellDisc")

        # Initialize the grammars
        self.input_grammar.update_from_names(["a", "b"])
        self.output_grammar.update_from_names(["c"])

        # Initialize the default inputs
        self.default_input_data = {"a": np.array([1.0]), "b": np.array([2.0])}

    def _create_inputs(self, input_data: StrKeyMapping) -> None:
        with (self.last_directory / "inputs.txt").open("w") as f:
            for name, value in input_data.items():
                f.write(f"{name}={value[0]}\n")

    def _parse_outputs(self) -> Mapping[str, np.ndarray]:
        data = {}
        with open(self.last_directory / "outputs.txt") as f:
            for line in f:
                if len(line) == 0:
                    continue
                name, value = line.replace("\n", "").split("=")
                data[name] = np.array([float(value)])

        return data
