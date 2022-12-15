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
"""
Scalability study - Result
==========================
"""
from __future__ import annotations

import pickle
from pathlib import Path

RESULTS_DIRECTORY = Path("results")


class ScalabilityResult:
    """Scalability Result."""

    def __init__(self, name, id_scaling, id_sample):
        """Constructor.

        :param str name: name of the scalability result.
        :param int id_scaling: scaling identifier
        :param int id_sample: sample identifier
        """
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
        algo,
        algo_options,
        formulation,
        formulation_options,
        scaling,
        n_calls,
        n_calls_linearize,
        n_calls_top_level,
        n_calls_linearize_top_level,
        exec_time,
        status,
        is_feasible,
        disc_names,
        output_names,
        old_varsizes,
        new_varsizes,
    ):
        """Get a scalability result for a given optimization strategy and a given scaling
        strategy.

        :param str algo: name of the optimization algorithm
        :param dict algo_options: options of the optimization algorithm
        :param str formulation: name of the MDO formulation
        :param dict formulation_options: options of the MDO formulation
        :param scaling: scaling strategy
        :param list(int) n_calls: number of calls for each discipline
        :param list(int) n_calls_linearize: number of linearization
            for each discipline
        :param list(int) n_calls_top_level: number of calls for each discipline
        :param list(int) n_calls_linearize_top_level: number of linearization
            for each discipline
        :param float exec_time: execution time
        :param int status: status of the optimization scenario
        :param bool is_feasible: feasibility of the optimization solution
        :param list(str) disc_names: list of discipline names
        :param dict output_names: list of output names
        :param dict old_varsizes: old variable sizes
        :param dict new_varsizes: new variable sizes
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

    def get_file_path(self, study_directory):
        """Get file path.

        :param str study_directory: study directory name.
        """
        fname = Path(self.name).with_suffix(".pkl")
        fpath = Path(study_directory) / RESULTS_DIRECTORY / fname
        return fpath

    def save(self, study_directory):
        """Save a scalability result into a pickle file whose name is the name of the
        ScalabilityResult instance.

        :param str study_directory: study directory name.
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

    def load(self, study_directory):
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
