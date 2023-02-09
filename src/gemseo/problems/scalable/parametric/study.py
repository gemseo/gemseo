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
#                         documentation
#        :author: Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""
Scalable study
==============
"""
from __future__ import annotations

import logging
import os
import pickle

from matplotlib import pyplot as plt
from numpy import arange
from numpy import where
from numpy.random import rand

from gemseo.core.coupling_structure import MDOCouplingStructure
from gemseo.core.mdo_scenario import MDOScenario
from gemseo.mda.mda_factory import MDAFactory
from gemseo.problems.scalable.parametric.core.variables import get_constraint_name
from gemseo.problems.scalable.parametric.problem import TMScalableProblem
from gemseo.utils.matplotlib_figure import save_show_figure

LOGGER = logging.getLogger(__name__)

MDA_TOLERANCE = {"tolerance": 1e-14, "linear_solver_tolerance": 1e-14}
ALGO_OPTIONS = {
    "xtol_rel": 1e-4,
    "ftol_rel": 1e-4,
    "xtol_abs": 1e-4,
    "ftol_abs": 1e-4,
    "ineq_tolerance": 1e-3,
    "eq_tolerance": 1e-3,
}


def save_matrix_plot(matrix, disc, name, directory="."):
    """Save the graphical representation of a matrix.

    :param ndarray matrix: matrix.
    :param str disc: discipline name.
    :param str name: name of the matrix.
    :param str directory: name of the directory to write the file.
        Default: '.'.
    """
    plt.matshow(matrix, cmap=plt.get_cmap("binary"), vmin=0.0, vmax=1.0)
    plt.colorbar()
    filename = disc + "_" + name
    plt.savefig(os.path.join(directory, filename + ".pdf"))
    plt.close()


OBJECTIVE_NAME = "obj"

COUPLING_DIR = "coupling"
COEFF_DIR = "coefficients"
OPTIM_DIR = "opthistoryview"


class TMParamSS:
    """This scalable parametric study realizes scalable studies with different scaling
    strategies.

    E.g. comparison of MDF and IDF formulations in terms of execution time for different
    number of coupling variables.
    """

    def __init__(
        self,
        n_disciplines,
        n_shared,
        n_local,
        n_coupling,
        full_coupling=True,
        active_probability=0.1,
        feasibility_level=0.8,
        seed=1,
        directory="results",
    ):
        """The TMParamSS constructor depends on:

        - the number of disciplines,
        - the number of shared design parameters,
        - the number of local design parameters for each discipline,
        - the number of coupling variables for each discipline.

        One of the three latter should be a list of integers
        whose components define scaling strategies.

        The constructor instantiates as many TMScalableStudy
        as scaling strategies.

        :param int n_disciplines: number of disciplines.
        :param n_shared: (list of) number(s) of shared design parameters.
        :param n_local: (list of) number(s) of local design parameters
            for each discipline.
        :param n_coupling: (list of) number(s) of coupling parameters
            for each discipline.
        :param bool full_coupling: fully couple the disciplines,
            ie each TMDiscipline depends on each other. Default: True.
        :param float active_probability: active probability
        :param float feasibility_level: level of feasibility
        :param int seed: seed for reproducibility.
        :param str directory: directory to store results

        See also
        --------
        .TMScalableStudy : standard scalable study launched
            with different configurations
            in the case of parametric scalable study.
        """
        self.n_shared = n_shared
        self.n_local = n_local
        self.n_coupling = n_coupling
        self.n_disciplines = n_disciplines
        assert isinstance(n_disciplines, int)
        assert isinstance(n_shared, (int, list))
        assert isinstance(n_local, (int, list))
        assert isinstance(n_coupling, (int, list))

        n_lists = sum(
            1 if isinstance(val, list) else 0 for val in (n_shared, n_local, n_coupling)
        )
        if n_lists > 1:
            msg = "At most 1 value among (n_shared,n_local,n_coupling) can be a list !"
            raise ValueError(msg)

        if isinstance(n_shared, list):
            self.param_label = "n_shared"
            self.param = n_shared
            self.studies = [
                TMScalableStudy(
                    n_disciplines,
                    n_shared_i,
                    n_local,
                    n_coupling,
                    full_coupling,
                    active_probability,
                    feasibility_level,
                    seed=seed,
                    directory=directory,
                )
                for n_shared_i in n_shared
            ]
        elif isinstance(n_local, list):
            self.param_label = "n_local"
            self.param = n_local
            self.studies = [
                TMScalableStudy(
                    n_disciplines,
                    n_shared,
                    n_local_i,
                    n_coupling,
                    full_coupling,
                    active_probability,
                    feasibility_level,
                    seed=seed,
                    directory=directory,
                )
                for n_local_i in n_local
            ]
        elif isinstance(n_coupling, list):
            self.param_label = "n_coupling"
            self.param = n_coupling
            self.studies = [
                TMScalableStudy(
                    n_disciplines,
                    n_shared,
                    n_local,
                    n_coupling_i,
                    full_coupling,
                    active_probability,
                    feasibility_level,
                    seed,
                    directory=directory,
                )
                for n_coupling_i in n_coupling
            ]
        else:
            self.param_label = "None"
            self.param = [1]
            self.studies = [
                TMScalableStudy(
                    n_disciplines,
                    n_shared,
                    n_local,
                    n_coupling,
                    full_coupling,
                    active_probability,
                    feasibility_level,
                    seed,
                    directory=directory,
                )
            ]

    def __str__(self):
        msg = "Parametric scalable study\n"
        msg += "> " + str(self.n_disciplines) + " disciplines\n"
        if hasattr(self.n_shared, "__len__"):
            n_shared = [str(val) for val in self.n_shared]
            n_shared = ", ".join(n_shared)
            tmp = n_shared.split(", ")
            n_shared = ", ".join(tmp[0:-1]) + " or " + tmp[-1]
        else:
            n_shared = str(self.n_shared)
        if hasattr(self.n_local, "__len__"):
            n_local = [str(val) for val in self.n_local]
            n_local = ", ".join(n_local)
            tmp = n_local.split(", ")
            n_local = ", ".join(tmp[0:-1]) + " or " + tmp[-1]
        else:
            n_local = str(self.n_local)
        if hasattr(self.n_coupling, "__len__"):
            n_coupling = [str(val) for val in self.n_coupling]
            n_coupling = ", ".join(n_coupling)
            tmp = n_coupling.split(", ")
            n_coupling = ", ".join(tmp[0:-1]) + " or " + tmp[-1]
        else:
            n_coupling = str(self.n_coupling)
        msg += "> " + str(n_shared) + " shared design parameters\n"
        msg += "> " + str(n_local)
        msg += " local design parameters per discipline\n"
        msg += "> " + str(n_coupling)
        msg += " coupling variables per discipline\n"
        return msg

    def run_formulation(
        self,
        formulation,
        max_iter=100,
        post_coupling=True,
        post_optim=True,
        post_coeff=True,
        algo="NLOPT_SLSQP",
        algo_options=None,
    ):
        """This method solves the scalable problems with a particular MDO formulation.

        :param str formulation: MDO formulation name
        :param int max_iter: maximum number of iterations
        :param bool post_coupling: store coupling plots
        :param bool post_optim: store optimization plots
        :param bool post_coeff: store coefficients plots
        :param algo: algorithm name to solve the problem
        :param algo_options: inequality and equality tolerance,
            xtol etc..
        """
        if algo_options is None:
            algo_options = ALGO_OPTIONS
        for study in self.studies:
            study.run_formulation(
                formulation,
                max_iter,
                post_coupling,
                post_optim,
                post_coeff,
                algo,
                algo_options,
            )

    def save(self, file_path):
        """This method saves the results into a pickle file.

        :param str file_path: pickle file path to store the results.
        """
        exec_time = {}
        sizes = []
        for idx, _ in enumerate(self.studies):
            if self.param_label in ["n_local", "n_coupling"]:
                pos = self.param[idx] * self.studies[idx].n_disciplines
            else:
                pos = self.param[idx]
            exec_time[pos] = {}
            for formulation in self.studies[idx].formulations:
                e_t = self.studies[idx].exec_time[formulation]
                exec_time[pos][formulation] = e_t["scenario"]
            sizes.append(pos)
        with open(file_path, "wb") as f_out:
            pickle.dump(
                {
                    "exec_time": exec_time,
                    "sizes": sizes,
                    "scaling": self.param_label,
                    "formulations": self.studies[0].formulations,
                },
                f_out,
            )


class TMParamSSPost:
    """This class is dedicated to the post-treatment of TMParamSS results."""

    def __init__(self, file_path):
        """The constructor reads data stored in a pickle file.

        :param str file_path: file path where data are stored.
        """
        with open(file_path, "rb") as f_in:
            results = pickle.load(f_in)
            self.exec_time = results["exec_time"]
            self.sizes = results["sizes"]
            self.scaling = results["scaling"]
            self.formulations = results["formulations"]

    def plot(
        self,
        title="A scalable comparison of MDO formulations",
        save=False,
        show=True,
        file_path="comparison.pdf",
    ):
        """Plot one line per MDO formulation where the y-axis represents the execution
        time and the x-axis the scaling strategies.

        :param str title: title of the figure.
            Default: 'A scalable comparison of MDO formulations'.
        :param bool save: save the plot. Default: False.
        :param bool show: show the plot. Default: True.
        :param str file_path: file path to store the figure.
            Default: 'comparison.pdf'.
        """
        _, axis = plt.subplots()
        colors = ["b", "r", "g", "c", "m", "y", "k", "w"]
        idx = 0
        for formulation in self.formulations:
            exec_time = [self.exec_time[size][formulation] for size in self.sizes]
            axis.plot(self.sizes, exec_time, color=colors[idx], lw=2, label=formulation)
            idx += 1
        axis.set_xscale("log")
        axis.set_yscale("log")
        if self.scaling == "n_local":
            axis.set_xlabel("Number of Local Design Variables")
        elif self.scaling == "n_coupling":
            axis.set_xlabel("Number of Coupling Variables")
        elif self.scaling == "n_shared":
            axis.set_xlabel("Number of Shared Design Variables")
        else:
            axis.set_xlabel("Scalable strategy")
        axis.set_ylabel("Optimization Time (s)")
        axis.set_title(title)
        axis.set_axisbelow(True)
        axis.grid(which="both", lw=1, color="gray")
        axis.legend()
        if save:
            plt.savefig(file_path)
        if show:
            plt.show()


class TMScalableStudy:
    """This scalable study creates a scalable MDO problem from Tedford and Martins, 2010
    and compares its resolution according to different MDO formulations."""

    def __init__(
        self,
        n_disciplines,
        n_shared,
        n_local,
        n_coupling,
        full_coupling=True,
        active_probability=0.1,
        feasibility_level=0.8,
        seed=1,
        directory="results",
    ):
        """The TMScalableStudy constructor depends on:

        - the number of disciplines,
        - the number of shared design parameters,
        - the number of local design parameters for each discipline,
        - the number of coupling variables for each discipline.

        :param int n_disciplines: number of disciplines.
        :param int n_shared: number of shared design parameters.
        :param int n_local: number of local design parameters
            for each discipline.
        :param int n_coupling: number of coupling parameters
            for each discipline.
        :param bool full_coupling: fully couple the disciplines,
            ie each TMDiscipline depends on each other. Default: True.
        :param float active_probability: active probability
        :param float feasibility_level: level of feasibility
        :param str directory: directory to store results
        :param int seed: seed for reproducibility.

        See also
        --------
        .TMScalableProblem : Scalable problem managed by the scalable study
            and providing both disciplines and design space.
        """
        self.n_disciplines = n_disciplines
        self.n_local = n_local
        self.n_shared = n_shared
        self.n_coupling = n_coupling
        n_local = [n_local] * n_disciplines
        n_coupling = [n_coupling] * n_disciplines
        self.problem = TMScalableProblem(
            n_shared, n_local, n_coupling, full_coupling, seed=seed
        )
        self.n_calls = {}
        self.n_calls_linearize = {}
        self.exec_time = {}
        self.formulation_options = {"MDF": {"inner_mda_name": "MDAGaussSeidel"}}
        self.formulation_options["MDF"].update(MDA_TOLERANCE)
        self.disc_names = ["scenario", "mda", "mdo_chain", "sub_mda"]
        tmp = sorted(disc.name for disc in self.problem.disciplines)
        self.disc_names += tmp
        self.active_probability = active_probability
        self.feasibility_level = feasibility_level
        self.directory = directory

    def __store_statistics(self, formulation, scenario):
        """Store statistics in dictionaries.

        :param str formulation: MDO formulation.
        :param Scenario scenario: scenario.
        """
        self.n_calls[formulation] = {}
        self.n_calls_linearize[formulation] = {}
        self.exec_time[formulation] = {}
        for disc in self.problem.disciplines:
            self.n_calls[formulation][disc.name] = disc.n_calls
            ncl = disc.n_calls_linearize
            self.n_calls_linearize[formulation][disc.name] = ncl
            self.exec_time[formulation][disc.name] = disc.exec_time
        self.exec_time[formulation]["scenario"] = scenario.exec_time
        ncl = scenario.n_calls_linearize
        self.n_calls_linearize[formulation]["scenario"] = ncl
        self.n_calls[formulation]["scenario"] = scenario.n_calls
        if hasattr(scenario.formulation, "mda"):
            mda = scenario.formulation.mda
            mda_exec_time = mda.exec_time
            mda_nc = mda.n_calls
            mda_ncl = mda.n_calls_linearize
            mdo_chain_exec_time = mda.mdo_chain.exec_time
            mdo_chain_nc = mda.mdo_chain.n_calls
            mdo_chain_ncl = mda.mdo_chain.n_calls_linearize
            sub_mda_exec_time = mda.inner_mdas[0].exec_time
            sub_mda_nc = mda.inner_mdas[0].n_calls
            sub_mda_ncl = mda.inner_mdas[0].n_calls_linearize
        else:
            mda_exec_time = 0.0
            mda_nc = 0
            mda_ncl = 0
            mdo_chain_exec_time = 0.0
            mdo_chain_nc = 0
            mdo_chain_ncl = 0
            sub_mda_exec_time = 0.0
            sub_mda_nc = 0
            sub_mda_ncl = 0
        self.exec_time[formulation]["mda"] = mda_exec_time
        self.exec_time[formulation]["sub_mda"] = sub_mda_exec_time
        self.exec_time[formulation]["mdo_chain"] = mdo_chain_exec_time
        self.n_calls[formulation]["mda"] = mda_nc
        self.n_calls[formulation]["sub_mda"] = sub_mda_nc
        self.n_calls[formulation]["mdo_chain"] = mdo_chain_nc
        self.n_calls_linearize[formulation]["mda"] = mda_ncl
        self.n_calls_linearize[formulation]["sub_mda"] = sub_mda_ncl
        self.n_calls_linearize[formulation]["mdo_chain"] = mdo_chain_ncl

    def run_formulation(
        self,
        formulation,
        max_iter=100,
        post_coupling=True,
        post_optim=True,
        post_coeff=True,
        algo="NLOPT_SLSQP",
        algo_options=None,
        xdsm_pdf=False,
    ):
        """Solve the scalable problem with a particular MDO formulation.

        Args:
            formulation: The name of the MDO formulation.
            max_iter: THe maximum number of iterations.
            post_coupling: Whether to store the coupling plots.
            post_optim: Whether to store the optimization plots.
            post_coeff: Whether to store the coefficients plots.
            algo: The name of the algorithm used to solve the problem.
            algo_options: The options for the algorithm.
            xdsm_pdf: Whether to export the xdsm in pdf.
        """
        if algo_options is None:
            algo_options = ALGO_OPTIONS
        self.problem.reset_design_space()
        self.problem.reset_disciplines()
        factory = MDAFactory()
        mda = factory.create(
            "MDAGaussSeidel", self.problem.disciplines, **MDA_TOLERANCE
        )
        equilibrium = mda.execute()
        scenario = MDOScenario(
            self.problem.disciplines,
            formulation,
            OBJECTIVE_NAME,
            self.problem.design_space,
            **self.formulation_options.get(formulation, {}),
        )
        LOGGER.info("Make the starting point feasible.")
        for disc in range(self.n_disciplines):
            if get_constraint_name(disc) in list(equilibrium.keys()):
                urand = rand(len(equilibrium[get_constraint_name(disc)]))
                val = equilibrium[get_constraint_name(disc)]
                alt = self.feasibility_level
                alt += (1 - self.feasibility_level) * val
                tau = where(urand < self.active_probability, val, alt)
            else:
                tau = 0.0
            scenario.add_constraint(get_constraint_name(disc), "ineq", value=tau)
        input_data = {"algo": algo, "max_iter": max_iter, "algo_options": algo_options}
        scenario.execute(input_data)
        if post_coeff:
            mkdir(self.directory)
            path = mkdir(self.directory, COEFF_DIR)
            for disc in self.problem.disciplines[1:]:
                save_matrix_plot(disc.model.c_shared, disc.name, "c_shared", path)
                save_matrix_plot(disc.model.c_local, disc.name, "c_local", path)
                for name, c_coupling_val in disc.model.c_coupling.items():
                    save_matrix_plot(
                        c_coupling_val, disc.name, "c_coupling_" + name, path
                    )
        if post_coupling:
            path = mkdir(self.directory, COUPLING_DIR)
            scenario.xdsmize(
                latex_output=xdsm_pdf,
                outdir=path,
                outfilename=formulation + "_xdsm",
            )
            coupling_structure = MDOCouplingStructure(scenario.disciplines)
            coupling_structure.plot_n2_chart(file_path=os.path.join(path, "n2.pdf"))
        if post_optim:
            path = mkdir(self.directory, OPTIM_DIR) + "/"
            scenario.post_process(
                "OptHistoryView", save=True, show=False, file_path=path
            )
        self.__store_statistics(formulation, scenario)
        n_iter = 0
        for discipline in scenario.disciplines:
            n_iter += discipline.n_calls
            n_iter += discipline.n_calls_linearize
        return {
            "x_opt": scenario.optimization_result.x_opt,
            "f_opt": scenario.optimization_result.f_opt,
            "status": scenario.optimization_result.status,
            "n_iter": n_iter,
            "is_feas": scenario.optimization_result.is_feasible,
            "exec_time": scenario.exec_time,
        }

    @property
    def formulations(self):
        """Names of the MDO formulations.

        :return: list of MDO formulations names
        :rtype: list(str)
        """
        return list(self.n_calls.keys())

    def __str__(self):
        """String representation of the results (number of calls, number of
        linearizations and execution time) for each discipline.

        :return: string representation
        :rtype: str
        """
        msg = [
            "Scalable study",
            f".... {self.n_disciplines} disciplines",
            f".... {self.n_shared} shared design parameters",
            f".... {self.n_local} local design parameters per discipline",
            f".... {self.n_coupling} coupling variables per discipline",
        ]
        if self.formulations:
            msg.append("MDO formulations")
        for formulation in self.formulations:
            msg.append(f".... {formulation}")
            for discipline in self.problem.disciplines:
                msg.append(self.__elementary_str(formulation, discipline.name))
            if "mda" in self.exec_time[formulation]:
                msg.append(self.__elementary_str(formulation, "mda"))
            if "mdo_chain" in self.exec_time[formulation]:
                msg.append(self.__elementary_str(formulation, "mdo_chain"))
            if "sub_mda" in self.exec_time[formulation]:
                msg.append(self.__elementary_str(formulation, "sub_mda"))
            msg.append(self.__elementary_str(formulation, "scenario"))
        return "\n".join(msg)

    def __elementary_str(self, formulation, discipline):
        """String representation of the result for a given formulation and a given
        discipline.

        :param str formulation: MDO formulation name
        :param str discipline: discipline name
        :return: elementary string representation
        :rtype: str
        """
        n_calls = self.n_calls[formulation].get(discipline, "NA")
        n_lin = self.n_calls_linearize[formulation].get(discipline, "NA")
        exec_time = self.exec_time[formulation].get(discipline, "NA")
        msg = "........ {} = {} calls / {} linearizations / {} seconds"
        return msg.format(discipline, n_calls, n_lin, exec_time)

    def plot_exec_time(self, show=True, save=False, file_path="exec_time.pdf"):
        """Barplot of the execution time of the different disciplines for the different
        formulations. When the formulation is based on a MDA, the MDO scenario is
        detailed in terms of MDA, MDO chain and sub-MDA.

        :param bool show: if True, show plot. Default: False.
        :param bool save: if True, save plot. Default: False.
        :parma str file_path: file path. Default: "exec_time.pdf"
        """
        series = [
            [self.exec_time[formulation][disc] for disc in self.disc_names]
            for formulation in self.formulations
        ]
        bar_width = 0.9 / len(self.formulations)
        indices = arange(len(self.exec_time[self.formulations[0]]))
        factor = 0
        colors = ["b", "g", "r", "c", "m", "y", "k", "w"]
        fig, ax = plt.subplots()
        for formulation in self.formulations:
            ax.bar(
                indices + bar_width * factor,
                series[factor],
                bar_width,
                alpha=0.8,
                color=colors[factor],
                label=formulation,
            )
            for name in ["mda", "mdo_chain", "sub_mda"]:
                ax.bar(
                    [0.0 + bar_width * factor],
                    [self.exec_time[formulation][name]],
                    bar_width,
                    fill=False,
                )
            factor += 1
        ax.set_xlabel("Disciplines")
        ax.set_ylabel("Execution time")
        ax.set_xticks(indices + bar_width)
        ax.set_xticklabels(self.disc_names)
        ax.legend()
        save_show_figure(fig, show, file_path if save else None)


def mkdir(dirname, subdirname=None):
    """Create a directory if not exists.

    :param str dirname: name of the directory.
    :param str subdirname: name of the subdirectory. If None, only considers
        the directory. Default: None.
    """
    if subdirname is not None:
        dirpath = os.path.join(dirname, subdirname)
    else:
        dirpath = dirname
    if not os.path.exists(dirpath):
        os.mkdir(dirpath)
    return dirpath
