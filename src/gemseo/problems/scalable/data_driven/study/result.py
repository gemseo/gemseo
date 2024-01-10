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
#    INITIAL AUTHORS - initial API and implementation and/or initial
#         documentation
#        :author:  Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""Scalability study - Result."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any

if TYPE_CHECKING:
    from collections.abc import Iterable
    from collections.abc import Mapping
    from collections.abc import Sequence

RESULTS_DIRECTORY = Path("results")


class ScalabilityResult:
    """Scalability Result."""

    def __init__(self, name: str, id_scaling: int, id_sample: int) -> None:
        """
        Args:
            name: The name of the scalability result.
            id_scaling: The scaling identifier.
            id_sample: The sample identifier.
        """  # noqa: D205, D212, D415
        self.name = name
        self.id_scaling = id_scaling
        self.id_sample = id_sample
        self.algo = None
        self.algo_options = None
        self.formulation_options = None
        self.formulation = None
        self.scaling = None
        self.n_calls = None
        self.n_calls_linearize = None
        self.n_calls_top_level = None
        self.n_calls_linearize_top_level = None
        self.exec_time = None
        self.original_exec_time = None
        self.status = None
        self.is_feasible = None
        self.disc_names = None
        self.old_varsizes = None
        self.new_varsizes = None
        self.output_names = None

    def get(
        self,
        algo: str,
        algo_options: Mapping[str, Any],
        formulation: str,
        formulation_options: Mapping[str, Any],
        scaling: Mapping[str, Any],
        n_calls: Iterable[int],
        n_calls_linearize: Iterable[int],
        n_calls_top_level: Iterable[int],
        n_calls_linearize_top_level: Iterable[int],
        exec_time: float,
        status: int,
        is_feasible: bool,
        disc_names: Sequence[str],
        output_names: Sequence[str],
        old_varsizes: Mapping[str, int],
        new_varsizes: Mapping[str, int],
    ) -> None:
        """Get a scalability result for a given optimization strategy and a given
        scaling strategy.

        Args:
            algo: The name of the optimization algorithm.
            algo_options: The options of the optimization algorithm.
            formulation: The name of the MDO formulation.
            formulation_options: The options of the MDO formulation.
            scaling: The scaling strategy.
            n_calls: The number of calls per discipline.
            n_calls_linearize: The number of linearization per discipline
            n_calls_top_level: The number of calls per discipline
            n_calls_linearize_top_level: The number of linearizations per discipline.
            exec_time: The execution time.
            status: The status of the optimization scenario.
            is_feasible: Whether the solution is feasible.
            disc_names: The names of the disciplines.
            output_names: The names of the outputs.
            old_varsizes: The sizes of the original variables.
            new_varsizes: The sizes of the new variables.
        """
        self.algo = algo
        self.algo_options = algo_options
        self.formulation = formulation
        self.formulation_options = formulation_options
        self.scaling = scaling
        self.n_calls = n_calls
        self.n_calls_linearize = n_calls_linearize
        self.n_calls_top_level = n_calls_top_level
        self.n_calls_linearize_top_level = n_calls_linearize_top_level
        self.exec_time = exec_time
        self.status = status
        self.is_feasible = is_feasible
        self.disc_names = disc_names
        self.output_names = output_names
        self.old_varsizes = old_varsizes
        self.new_varsizes = new_varsizes

    def get_file_path(self, study_directory: str) -> Path:
        """Return file path.

        Args:
            study_directory: The study directory name.
        """
        return (
            Path(study_directory)
            / RESULTS_DIRECTORY
            / Path(self.name).with_suffix(".pkl")
        )

    def to_pickle(self, study_directory: str) -> Path:
        """Save a scalability result into a pickle file whose name is the name of the
        ScalabilityResult instance.

        Args:
            study_directory: The study directory name.

        Returns:
            The path to the result.
        """
        fpath = self.get_file_path(study_directory)
        result = {
            "algo": self.algo,
            "algo_options": self.algo_options,
            "formulation": self.formulation,
            "formulation_options": self.formulation_options,
            "scaling": self.scaling,
            "n_calls": self.n_calls,
            "n_calls_linearize": self.n_calls_linearize,
            "n_calls_top_level": self.n_calls_top_level,
            "n_calls_linearize_top_level": self.n_calls_linearize_top_level,
            "exec_time": self.exec_time,
            "status": self.status,
            "is_feasible": self.is_feasible,
            "disc_names": self.disc_names,
            "output_names": self.output_names,
            "old_varsizes": self.old_varsizes,
            "new_varsizes": self.new_varsizes,
        }
        with fpath.open("wb") as fout:
            pickle.dump(result, fout)
        return fpath

    def load(self, study_directory) -> None:
        """Load a scalability result from a pickle file whose name is the name of the
        ScalabilityResult instance."""
        fname = self.name + ".pkl"
        fpath = Path(study_directory) / RESULTS_DIRECTORY / fname
        with fpath.open("rb") as fin:
            result = pickle.load(fin)
        self.algo = result["algo"]
        self.formulation = result["formulation"]
        self.scaling = result["scaling"]
        self.n_calls = result["n_calls"]
        self.n_calls_linearize = result["n_calls_linearize"]
        self.n_calls_top_level = result["n_calls_top_level"]
        self.n_calls_linearize_top_level = result["n_calls_linearize_top_level"]
        self.exec_time = result["exec_time"]
        self.status = result["status"]
        self.is_feasible = result["is_feasible"]
        self.disc_names = result["disc_names"]
        self.output_names = result["output_names"]
        self.old_varsizes = result["old_varsizes"]
        self.new_varsizes = result["new_varsizes"]
