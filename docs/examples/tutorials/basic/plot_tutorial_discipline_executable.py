# Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com
#
# This work is licensed under a BSD 0-Clause License.
#
# Permission to use, copy, modify, and/or distribute this software
# for any purpose with or without fee is hereby granted.
#
# THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL
# WARRANTIES WITH REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS. IN NO EVENT SHALL
# THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT, INDIRECT,
# OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING
# FROM LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT,
# NEGLIGENCE OR OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION
# WITH THE USE OR PERFORMANCE OF THIS SOFTWARE.
"""# Tutorial - Wrap an executable

## Goal

In this tutorial, you will learn to wrap an executable
into a [Discipline][gemseo.core.discipline.discipline.Discipline].

!!! warning
    This is an experimental feature which uses on-development code.
    Both the API and the behavior may change over time.
    To use with caution.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING

from numpy import array

from gemseo.disciplines.wrappers._base_disc_from_exe import _BaseDiscFromExe
from gemseo.disciplines.wrappers._base_executable_runner import _BaseExecutableRunner

if TYPE_CHECKING:
    from gemseo.typing import StrKeyMapping

# %%
# ## Step 1 — Create an executable
#
# First, let's create an executable representing a model to be run with a CLI.
# Here, let's create a python file `my_script.py`
# which will be run with the command line:
# `python my_script.py -i input_file -o output_file`.
# The script will need to get a JSON input file:
# ```json
# {"a": value_a, "b": value_b}
# ```
# and will return a TXT file:
# ```txt
# output=value_output
# ```
#
# !!! note
#     Any data format can be considered, as you will see latter.
#     You can also consider multiple input/output files.

with open("my_script.py", "w+") as f:
    f.write(
        """from __future__ import annotations

import argparse
import json
import pathlib


def main():
    parser = argparse.ArgumentParser(
        description="Read a JSON file, add two variables, and write the result."
    )
    parser.add_argument("-i", "--input", required=True, help="Input JSON file path")
    parser.add_argument("-o", "--output", required=True, help="Output file path")
    args = parser.parse_args()

    # --- Parse input ---
    with open(args.input) as f:
        data = json.load(f)

    result = data["a"] + data["b"]

    # --- Write output ---
    pathlib.Path(args.output).write_text(f"output={result}")

    print(f"Done. Result written to '{args.output}': output={result}")


if __name__ == "__main__":
    main()
"""
    )


# %%
# ## Step 2 — Create your discipline
#
# The construction of this discipline consists in different steps:
#
# 1. Instantiate the discipline
#     1. Instantiate the `_BaseDiscFromExe` using the super constructor,
#     2. Create the associated `executable_runner`. This indicates how to run the CLI.
#     3. Manage default inputs/outputs within the constructor,
# 2. Implement `define_inputs` to write inputs into the input file, using the specific format.
# 3. Implement `parse_outputs` to read outputs from the formatted file.
#
# !!! warning
#     The `_run()` method should not be modified.
class MyExecutableDiscipline(_BaseDiscFromExe):
    """A discipline to execute a python script as an executable.

    The script is run with the following CLI:
    ``python ./my_script.py -i inputs.json -o outputs.txt``
    """

    def __init__(self) -> None:
        # Create the Executable runner
        # It handles the directory creation,
        # the management of files before the execution
        # and the CLI call.
        #
        # Here,
        # we want our script to be copied within the unique directory
        # that is created with the UUID method (default).
        # Then, our script can be called within the directory.
        exec_runner = _BaseExecutableRunner(
            "python ./my_script.py -i inputs.json -o outputs.txt",
            ".",
            data_paths=["./my_script.py"],
        )

        super().__init__(exec_runner, name="ExecDisc", clean_after_execution=True)

        # Initialize the grammars
        self.input_grammar.update_from_names(["a", "b"])
        self.output_grammar.update_from_names(["output"])

        # Initialize the default inputs
        self.default_input_data = {"a": array([10.0]), "b": array([32.0])}

    def _create_inputs(self, input_data: StrKeyMapping) -> None:
        # Here, we define how the discipline must create the input JSON file.
        # We create the ``inputs.json`` file within the last created directory.
        # Warning: ndarray is not serializable. We must first convert data.
        input = {key: float(value[0]) for key, value in input_data.items()}
        with (
            self._executable_runner.directory_creator.last_directory / "inputs.json"
        ).open("w") as f:
            json.dump(input, f)

    def _parse_outputs(self) -> StrKeyMapping:
        # Here, we define how to read the output file to fill the discipline outputs.
        data = {}
        with Path(
            self._executable_runner.directory_creator.last_directory / "outputs.txt"
        ).open() as f:
            for line in f:
                if len(line) == 0:
                    continue
                name, value = line.replace("\n", "").split("=")
                data[name] = array([float(value)])

        return data


# %%
# ## Step 3 - Use the executable discipline
#
# A discipline based on an executable is mainly driven by 4 different steps:
#
# 1. Create a unique directory to store the files required by the executable,
# 2. Write the input files in this directory,
# 3. Run the executable with the command line,
# 4. Parse the output files.
#
# !!! note
#     The communication between the executable
#     and its discipline is done via input and output files.
#     These files may become a source of errors,
#     since they can be modified by another program while being in use.
#     To avoid such inconveniences,
#     a unique directory can be created for each run.
#     This is also compliant with multi-processing features.
#
# All these steps are completely hidden.
# Therefore, nothing new for you:
# it behaves just as a normal [Discipline][gemseo.core.discipline.discipline.Discipline].
discipline = MyExecutableDiscipline()
discipline.execute()

# %%
discipline.execute({"a": array([5])})

# %%
# For Documentation purposes,
# the initial script file is removed.
Path("my_script.py").unlink()

# %%
# ## Key takeaways
#
# Now you are able to wrap any model that has to be launched with a CLI.
# You first create your specific ``ExecutableRunner``,
# that is given to your ``BaseDiscFromExe``.
