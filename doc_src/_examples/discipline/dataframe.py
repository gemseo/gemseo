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
# Antoine DECHAUME
"""
Create a discipline that uses pandas DataFrames
===============================================
"""
from __future__ import annotations

from gemseo.api import configure_logger
from gemseo.core.discipline import MDODiscipline
from numpy import ndarray
from pandas import DataFrame

# %%
# Import
# ------

configure_logger()


# %%
# Create a discipline that uses a DataFrame
# -----------------------------------------
#
# We will create a class for a simple discipline that computes an output
# variable ``y = 1 - 0.2 * x`` where ``x`` is an input variable.
# For whatever reason, the business logic of this discipline uses a pandas DataFrame
# to store the input and output values outside |g|.
# Although |g| disciplines only handle input and output variables that are NumPy arrays,
# their local data and default input values can use DataFrame objects.
#
# The input and output grammars of the discipline shall use a naming convention
# to access the names of the columns of a DataFrame.
# The naming convention is built with the name of the input or output,
# the character ``~`` (this can be changed) and
# the name of the DataFrame column.
#
# The code executed by the discipline is in the ``_run`` method,
# where ``self.local_data``, i.e. the local data, has automatically been initialized
# with the default inputs and updated with the inputs passed to the discipline.
# A DataFrame can be retrieved by querying the corresponding key, e.g. ``df``,
# in the local data and then changes can be made to this DataFrame, e.g.
# ``discipline.local_data["df"]["x"] = value``.
#
# The default inputs and local data are instances of :class:`.DisciplineData`.
#
# .. seealso::
#
#  :class:`.DisciplineData` has more information about how DataFrames are handled.


class DataFrameDiscipline(MDODiscipline):
    def __init__(self):
        super().__init__(grammar_type=MDODiscipline.SIMPLE_GRAMMAR_TYPE)
        self.default_inputs = {"df": DataFrame(data={"x": [0.0]})}
        self.input_grammar.update({"df~x": ndarray})
        self.output_grammar.update({"df~y": ndarray})

    def _run(self):
        df = self.local_data["df"]
        df["y"] = 1.0 - 0.2 * df["x"]

        # The code above could also have been written as
        # self.local_data["df~y"] = 1.0 - 0.2 * self.local_data["df~x"]
        # self.local_data["df"]["y"] = 1.0 - 0.2 * self.local_data["df"]["x"]


# %%
# Instantiate the discipline
# --------------------------
discipline = DataFrameDiscipline()

# %%
# Execute the discipline
# ----------------------
# Then, we can execute it easily, either considering default inputs:
print(discipline.execute())

# %%
# or using new inputs:
print(discipline.execute({"df~x": [1.0]}))
