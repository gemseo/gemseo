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
# Contributors:
#    INITIAL AUTHORS - initial API and implementation and/or initial
#                           documentation
#        :author: Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""
Create a discipline from an external executable
===============================================
"""

from __future__ import annotations

import os
import subprocess
from typing import TYPE_CHECKING

from numpy import array

from gemseo import configure_logger
from gemseo.core.discipline import Discipline

if TYPE_CHECKING:
    from gemseo.typing import StrKeyMapping

configure_logger()


# %%
# Introduction
# ------------
#
# Let's consider a binary software computing the float output
# :math:`c = a^2 + b^2` from two float inputs : ``'a'`` and ``'b'``.
#
# The inputs are read in the ``'inputs.txt'`` file
# which looks like: `a=1 b=2` and
# the output is written to: ``'outputs.txt'`` which looks like `c=5`.
#
# Then, the executable can be run using the shell command ``'python run.py'``.
# Let's make a discipline out of this from an initial ``'inputs.txt'``.


# %%
# Implementation of the discipline
# --------------------------------
#
# The construction of :class:`.Discipline` consists in three steps:
#
# 1. Instantiate the :class:`.Discipline` using the super constructor,
# 2. Initialize the grammars using the
#    :meth:`.JSONGrammar.update` method,
# 3. Set the default inputs from the initial ``'inputs.txt'``
#
# The :class:`!Discipline._run` method consists in three steps:
#
# 1. Get the input data from :attr:`!Discipline.local_data` and write the
#    ``'inputs.txt'`` file,
# 2. Run the executable using the ``subprocess.run()`` command (`see more
#    <https://docs.python.org/3/library/subprocess.html#subprocess.run>`_),
# 3. Get the output values and store them to :attr:`!Discipline.local_data`.
#
# Now you can implement the discipline in the following way:


def parse_file(file_path):
    data = {}
    with open(file_path) as inf:
        for line in inf:
            if len(line) == 0:
                continue
            name, value = line.replace("\n", "").split("=")
            data[name] = array([float(value)])

    return data


def write_file(data, file_path) -> None:
    with open(file_path, "w") as outf:
        outf.writelines(
            name + "=" + str(value[0]) + "\n" for name, value in list(data.items())
        )


class ShellExecutableDiscipline(Discipline):
    def __init__(self) -> None:
        super().__init__("ShellDisc")
        self.input_grammar.update_from_names(["a", "b"])
        self.output_grammar.update_from_names(["c"])
        self.default_input_data = {"a": array([1.0]), "b": array([2.0])}

    def _run(self, input_data: StrKeyMapping) -> StrKeyMapping | None:
        cwd = os.getcwd()
        inputs_file = os.path.join(cwd, "inputs.txt")
        outputs_file = os.path.join(cwd, "outputs.txt")
        write_file(input_data, inputs_file)
        subprocess.run(["python", "run.py"], cwd=cwd)
        return parse_file(outputs_file)


# %%
# Execution of the discipline
# ---------------------------
# Now we can run it with default input values:
shell_disc = ShellExecutableDiscipline()
shell_disc.execute()

# %%
# or run it with new input values:
shell_disc.execute({"a": array([2.0]), "b": array([3.0])})
