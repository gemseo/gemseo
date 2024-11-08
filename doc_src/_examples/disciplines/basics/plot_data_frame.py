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

from typing import TYPE_CHECKING

import pandera as pa
from pandas import DataFrame
from pandera.typing import DataFrame as DataFrameType
from pandera.typing import Series
from pydantic import BaseModel

from gemseo import configure_logger
from gemseo.core.data_converters.pydantic import PydanticGrammarDataConverter
from gemseo.core.discipline import Discipline
from gemseo.core.grammars.pydantic_grammar import PydanticGrammar

if TYPE_CHECKING:
    from gemseo.typing import StrKeyMapping

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
# where ``self.data``, i.e. the local data, has automatically been initialized
# with the default inputs and updated with the inputs passed to the discipline.
# A DataFrame can be retrieved by querying the corresponding key, e.g. ``df``,
# in the local data and then changes can be made to this DataFrame, e.g.
# ``discipline.data["df"]["x"] = value``.
#
# The default inputs and local data are instances of :class:`.DisciplineData`.
#
# .. seealso::
#
#  :class:`.DisciplineData` has more information about how DataFrames are handled.
class InputDataFrameModel(pa.DataFrameModel):
    x: Series[float] = pa.Field(unique=True)


class OutputDataFrameModel(pa.DataFrameModel):
    y: Series[float] = pa.Field(unique=True)


class InputGrammarModel(BaseModel):
    df: DataFrameType[InputDataFrameModel]


class OutputGrammarModel(BaseModel):
    df: DataFrameType[OutputDataFrameModel]


class DataConverter(PydanticGrammarDataConverter):
    """A data converter where some coupling variables are 2D NumPy arrays."""

    def convert_value_to_array(self, name, value):
        if name == "df":
            return value.to_numpy().flatten()
        return super().convert_value_to_array(name, value)

    def convert_array_to_value(self, name, array_):
        if name == "df":
            return DataFrame({"x": [array_[0]], "y": [array_[1]]})
        return super().convert_array_to_value(name, array_)


PydanticGrammar.DATA_CONVERTER_CLASS = DataConverter


class DataFrameDiscipline(Discipline):
    default_grammar_type = Discipline.GrammarType.PYDANTIC

    def __init__(self) -> None:
        super().__init__()
        self.input_grammar = PydanticGrammar("inputs", model=InputGrammarModel)
        self.output_grammar = PydanticGrammar("outputs", model=OutputGrammarModel)
        self.default_input_data = {"df": DataFrame(data={"x": [0.0]})}

    def _run(self, input_data: StrKeyMapping) -> StrKeyMapping | None:
        df = self.local_data["df"]
        df["y"] = 1.0 - 0.2 * df["x"]


# %%
# Instantiate the discipline
# --------------------------
discipline = DataFrameDiscipline()

# %%
# Execute the discipline
# ----------------------
# Then, we can execute it easily, either considering default inputs:
discipline.execute()

# %%
# or using new inputs:
discipline.execute({"df": DataFrame(data={"x": [1.0]})})
