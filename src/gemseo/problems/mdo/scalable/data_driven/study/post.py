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
"""Post-processing for scalability study.

The :class:`.PostScalabilityStudy` class implements the way as the set of
:class:`.ScalabilityResult`-based result files
contained in the study directory are graphically post-processed. This class
provides several methods to easily change graphical properties, notably
the plot labels. It also makes it possible to define a cost function per
MDO formulation, converting the numbers of executions and linearizations
of the different disciplines required by an MDO process in an estimation
of the computational cost associated with what would be a scaled version
of the true problem.

.. warning::

   Comparing MDO formulations in terms of estimated true computational time
   rather than CPU time of the :class:`.ScalabilityStudy` is highly
   recommended.
   Indeed, time is often an obviousness criterion to distinguish between
   MDO formulations having the same performance in terms of distance to the
   optimum: look at our calculation budget and choose the best formulation
   that satisfies this budget, or even saves us time. Thus, it is important
   to carefully define these cost functions.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import ClassVar

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from numpy import array
from numpy import atleast_3d
from numpy import median
from numpy import poly1d
from numpy import polyfit

from gemseo.problems.mdo.scalable.data_driven.study.result import ScalabilityResult
from gemseo.utils.string_tools import MultiLineString

if TYPE_CHECKING:
    from collections.abc import Mapping
    from collections.abc import Sequence

    from numpy._typing import NDArray

LOGGER = logging.getLogger(__name__)

CURRENT_DIRECTORY = Path.cwd()
PLOT_EXTENSION = "png"
RESULTS_DIRECTORY = Path("results")
POST_DIRECTORY = Path("visualization")
POSTSTUDY_DIRECTORY = POST_DIRECTORY / "scalability_study"


class PostScalabilityStudy:
    """Post-processing of scalability results."""

    NOMENCLATURE: ClassVar[dict[str, str]] = {
        "exec_time": "Execution time (s)",
        "original_exec_time": "Pseudo-original execution time",
        "n_calls": "Number of discipline evaluations",
        "n_calls_linearize": "Number of gradient evaluations",
        "status": "Optimizatin status",
        "is_feasible": "Feasibility of the solution (0 or 1)",
        "scaling_strategy": "Scaling strategy index",
        "total_calls": "Total number of evaluations",
    }

    def __init__(self, study_directory: str) -> None:
        """
        Args:
            study_directory: The directory of the scalability study.
        """  # noqa: D205, D212, D415
        msg = MultiLineString()
        msg.add("Post-process for scalability study")
        msg.indent()
        msg.add("Working directory: {}", study_directory)
        LOGGER.info("%s", msg)
        self.study_directory = Path(study_directory)
        self.scalability_results = self.__load_results()
        self.n_results = len(self.scalability_results)
        self.descriptions = self.NOMENCLATURE
        self.cost_function = {}
        self.unit_cost = None
        for result in self.scalability_results:
            result.total_calls = sum(result.n_calls.values())
            result.total_calls += sum(result.n_calls_linearize.values())

    def set_cost_function(
        self,
        formulation: str,
        cost: Callable[[Mapping[str, int], int, int, int, int], float],
    ) -> None:
        """Set cost function for each formulation.

        Args:
            formulation: The name of the formulation.
            cost: The cost function.
        """
        self.cost_function[formulation] = cost

    def set_cost_unit(self, cost_unit: str) -> None:
        """Set the measurement unit for cost evaluation.

        Args:
            cost_unit: The cost unit, e.g. 'h' or 'min'.
        """
        self.unit_cost = cost_unit
        description = self.descriptions["original_exec_time"].split(" (")[0]
        description = f"{description} ({cost_unit})"
        self.descriptions["original_exec_time"] = description

    def labelize_exec_time(self, description: str) -> None:
        """Change the description of execution time.

        Args:
            description: The description.
        """
        self._update_descriptions("exec_time", description)

    def labelize_original_exec_time(self, description: str) -> None:
        """Change the description of original execution time.

        Args:
            description: The description.
        """
        self._update_descriptions("original_exec_time", description)
        if self.unit_cost is not None:
            self.set_cost_unit(self.unit_cost)

    def labelize_n_calls(self, description: str) -> None:
        """Change the description of number of calls.

        Args:
            description: The description.
        """
        self._update_descriptions("n_calls", description)

    def labelize_n_calls_linearize(self, description: str) -> None:
        """Change the description of number of calls for linearization.

        Args:
            description: The description.
        """
        self._update_descriptions("n_calls_linearize", description)

    def labelize_status(self, description: str) -> None:
        """Change the description of status.

        Args:
            description: The description.
        """
        self._update_descriptions("status", description)

    def labelize_is_feasible(self, description: str) -> None:
        """Change the description of feasibility.

        Args:
            description: The description.
        """
        self._update_descriptions("is_feasible", description)

    def labelize_scaling_strategy(self, description: str) -> None:
        """Change the description of scaling strategy.

        Args:
            description: The description.
        """
        self._update_descriptions("scaling_strategy", description)

    def _update_descriptions(self, keyword: str, description: str) -> None:
        """Update the description initialized with the NOMENCLATURE class attribute.

        Args:
            keyword: The keyword of the considered object.
            description: The description.

        Raises:
            ValueError: When the keyword is not available.
            TypeError: When the description is not a string.
        """
        if not self.descriptions.get(keyword):
            keywords = ", ".join(list(self.descriptions.keys()))
            msg = f"The keyword {keyword} is not in the list: {keywords}"
            raise ValueError(msg)
        if not isinstance(description, str):
            msg = (
                'The argument "description" must be '
                "of type string, "
                f"not of type {description.__class__.__name__}"
            )
            raise TypeError(msg)
        self.descriptions[keyword] = description

    def __load_results(self) -> list[ScalabilityResult]:
        """Load results from the results directory of the study path.

        Returns:
            The scalability results.

        Raises:
            ValueError: When the results directory does not exist or is empty.
        """
        if not self.study_directory.is_dir():
            msg = f'Directory "{self.study_directory}" does not exist.'
            raise ValueError(msg)
        directory = self.study_directory / RESULTS_DIRECTORY
        if not directory.is_dir():
            msg = f'Directory "{directory}" does not exist.'
            raise ValueError(msg)
        filenames = [
            filename.name for filename in directory.iterdir() if filename.is_file()
        ]
        results = []
        for filename in filenames:
            name = filename.split(".")[0]
            id_scaling = int(name.split("_")[-2])
            id_sample = int(name.split("_")[-1])
            result = ScalabilityResult(name, id_scaling, id_sample)
            result.load(self.study_directory)
            results.append(result)
        if not results:
            msg = f"Directory {directory} is empty."
            raise ValueError(msg)
        return results

    def plot(
        self,
        legend_loc: str = "upper left",
        xticks: Sequence[float] | None = None,
        xticks_labels: Sequence[str] | None = None,
        xmargin: float = 0.0,
        **options: Any,
    ) -> None:
        """Plot the results.

        Args:
            legend_loc: The location of the legend.
            xticks: The x-ticks.
            xticks_labels: The labels for the x-ticks.
            xmargin: The margin on the left and right sides of the x-axis.
            **options: The options for the specialized plot methods
        """
        msg = MultiLineString()
        msg.add("Execute post-processing")
        msg.indent()
        self.__has_scaling_dimension(xticks)
        self.__has_scaling_dimension(xticks_labels)
        there_are_replicates = len(self.get_replicates(True)) > 1
        if self.cost_function:
            self._estimate_original_time()
        if there_are_replicates:
            msg.add("Type: replicate")
            self._plot_boxes(legend_loc, xticks, xticks_labels, xmargin, **options)
        else:
            msg.add("Type: standard")
            self._plot_lines(legend_loc, xticks, xticks_labels, xmargin, **options)
        LOGGER.info("%s", msg)

    def __has_scaling_dimension(self, value: Any) -> None:
        """Assert if a value has the scaling dimension.

        Args:
            value: A value to be tested.
        """
        if value is not None and hasattr(value, "__len__"):
            assert len(value) == len(self.get_scaling_strategies(True))

    def _plot_lines(
        self,
        legend_loc: str = "upper left",
        xticks: Sequence[float] | None = None,
        xticks_labels: Sequence[str] | None = None,
        xmargin: float = 0.0,
    ) -> None:
        """Plot lines.

        Args:
            legend_loc: The location of the legend.
            xticks: The x-ticks.
            xticks_labels: The labels for the x-ticks.
            xmargin: The margin on the left and right sides of the x-axis.
        """
        colors = ["blue", "red", "green"]
        handles = [
            Line2D([0], [0], color=colors[idx], lw=4, label=optim_strategy)
            for idx, optim_strategy in enumerate(self.optimization_strategies)
        ]
        for index, strategy in enumerate(self.optimization_strategies):
            color = colors[index]
            scales, criteria = self.__get_scales_and_criteria(strategy)
            for name, value in criteria.items():
                self.__draw(name, value, xticks, xticks_labels, scales, color, xmargin)
        for criterion in criteria:
            plt.figure(criterion)
            plt.legend(handles=handles, loc=legend_loc, frameon=False, framealpha=0.5)
            fname = Path(criterion).with_suffix("." + PLOT_EXTENSION)
            fpath = self.study_directory / POSTSTUDY_DIRECTORY / fname
            plt.savefig(str(fpath))

    def __draw(
        self,
        name: str,
        value: NDArray[float],
        xticks: Sequence[float],
        labels: Sequence[str],
        scales: Sequence[float],
        color: str,
        xmargin: float,
    ) -> None:
        """Create plot for specific criterion when r=1 replicate.

        name: The name of the criterion.
        value: The criterion value.
        xticks: The x-ticks.
        labels: The labels for the x-ticks.
        scales: The scales.
        color: line color.
        xmargin: The margin on the left and right sides of the x-axis.
        """
        plt.figure(name)
        value = atleast_3d(value.T).T
        n_lines = value.shape[0]
        for id_line in range(n_lines):
            yvalues = value[id_line, :, 0].flatten()
            xvalues = scales
            linestyle = "-" if n_lines == 1 else (0, (id_line + 1, n_lines))
            if xticks is not None:
                xvalues = xticks
            else:
                xticks = xvalues
            plt.plot(xvalues, yvalues, "o", color=color)
            coef = polyfit(xvalues, yvalues, 1)
            poly1d_fn = poly1d(coef)
            plt.plot(xvalues, poly1d_fn(xvalues), linestyle=linestyle, color=color)
            labels = xticks if labels is None else xvalues
            plt.xticks(xticks, labels)
            plt.xlabel(self.descriptions["scaling_strategy"])
            plt.ylabel(self.descriptions[name])
            plt.xlim(xvalues[0] - xmargin, xvalues[-1] + xmargin)
            plt.grid(True, "both")

    @property
    def n_samples(self) -> int:
        """The number of samples."""
        return len(self.get_replicates(True))

    def _plot_boxes(
        self,
        legend_loc: str = "upper left",
        xticks: Sequence[float] | None = None,
        xticks_labels: Sequence[str] | None = None,
        xmargin: float = 0.0,
        minbox: int = 2,
        notch: bool = False,
        widths: float = 0.25,
        whis: float = 1.5,
    ) -> None:
        """Plot results with boxplots.

        Args:
            legend_loc: The location of the legend.
            xticks: The x-ticks.
            xticks_labels: The labels for the x-ticks.
            xmargin: The margin on the left and right sides of the x-axis.
            minbox: The minimal number of values for boxplot.
            notch: Whether to use notched boxplots.
            whis: The reach of the whiskers to the beyond the first and third quartiles.
        """
        if not hasattr(widths, "__len__"):
            widths = [widths] * len(self.get_scaling_strategies(True))
        else:
            self.__has_scaling_dimension(widths)

        colors = ["blue", "red", "green"]
        for index, strategy in enumerate(self.optimization_strategies):
            color = colors[index]
            scales, criteria = self.__get_scales_and_criteria(strategy)
            for name, value in criteria.items():
                self.__drawb(
                    name,
                    index,
                    value,
                    xticks,
                    xticks_labels,
                    scales,
                    widths,
                    color,
                    notch,
                    whis,
                    xmargin,
                )
        handles = [
            Line2D([0], [0], color=colors[idx], lw=4, label=strategy)
            for idx, strategy in enumerate(self.optimization_strategies)
        ]
        for criterion in criteria:
            plt.figure(criterion)
            plt.legend(
                handles=handles,
                loc=legend_loc,
                frameon=False,
                framealpha=0.5,
            )
            fname = Path(criterion).with_suffix("." + PLOT_EXTENSION)
            fpath = self.study_directory / POSTSTUDY_DIRECTORY / fname
            indentation = MultiLineString.INDENTATION
            LOGGER.info("%sSave %s plot in %s", indentation, criterion, fpath)
            plt.savefig(str(fpath))

    def __drawb(
        self,
        name: str,
        index: int,
        value: NDArray[float],
        xticks: Sequence[float],
        labels: Sequence[str],
        scales: Sequence[float],
        widths: Sequence[int],
        color: str,
        notch: bool,
        whis: float,
        xmargin: float,
    ) -> None:
        """Create plot for specific criterion when r=1 replicate.

        Args:
            name: The name of the criterion
            index: The strategy index.
            value: The value of the criterion
            xticks: The x-ticks.
            labels: The labels for the x-ticks.
            scales: The scales
            color: The line color.
            notch: Whether to use notched boxplots.
            whis: The reach of the whiskers to the beyond the first and third quartiles.
            xmargin: The margin on the left and right sides of the x-axis.
        """
        plt.figure(name)
        xvalues = scales
        if xticks is not None:
            xvalues = xticks
        n_strategies = len(self.optimization_strategies)
        tmp = []
        for idx, xtick in enumerate(xvalues):
            tmp.append(float(index) / n_strategies)
            tmp[-1] *= widths[idx] * 3
            tmp[-1] += xtick
        self.__draw_boxplot(value, tmp, color, color, notch, widths, whis)
        zvalues = atleast_3d(value.T).T[0]
        xval_offset = [
            xtick + float(index) / n_strategies * widths[idx] * 3
            for idx, xtick in enumerate(xvalues)
        ]
        plt.plot(xval_offset, median(zvalues, 1), "--", color=color)
        if labels is None:
            labels = xvalues
        plt.xticks(xvalues, labels)
        plt.xlim(xvalues[0] - xmargin, xvalues[-1] + xmargin)
        plt.xlabel(self.descriptions["scaling_strategy"])
        plt.ylabel(self.descriptions[name])
        plt.grid(True, "both")

    @staticmethod
    def __draw_boxplot(
        data: NDArray[float],
        xticks: Sequence[float],
        edge_color: str,
        fill_color: str,
        notch: bool,
        widths: Sequence[int],
        whis: bool,
    ) -> None:
        """Draw boxplot from a dataset.

        Args:
            data: The data.
            xticks: The values of the x-ticks.
            edge_color: The edge color.
            fill_color: The fill color.
            notch: Whether to use notched boxplots.
            widths: The widths of the boxplots.
            whis: The reach of the whiskers to the beyond the first and third quartiles.
        """
        if len(data.shape) == 3:
            data = data[0, :, :]

        if data.dtype == bool:  # noqa: E721
            # To prevent error when arrays are substracted with recent numpy
            data = data.astype(int)

        boxplot = plt.boxplot(
            data.T,
            patch_artist=True,
            positions=xticks,
            showfliers=False,
            whiskerprops={"linestyle": "-"},
            notch=notch,
            widths=widths,
            whis=whis,
        )

        for element in ["boxes", "whiskers", "fliers", "means", "caps"]:
            plt.setp(boxplot[element], color=edge_color)
        plt.setp(boxplot["medians"], color="white")

        for patch in boxplot["boxes"]:
            patch.set(facecolor=fill_color)

        return boxplot

    def __get_scales_and_criteria(
        self, optim_strategy: str
    ) -> tuple[list[int], dict[str, NDArray[float]]]:
        """Get values of criteria and corresponding scaling levels .

        Args:
            optim_strategy: The name of the optimization strategy.

        Returns:
            The scaling levels, the criteria values.
        """
        exec_time = []
        original_exec_time = []
        n_calls = []
        n_calls_linearize = []
        status = []
        is_feasible = []
        scaling = []
        total_calls = []
        for replicate in self.get_replicates(True):
            xcoord, ycoord = self.__get_replicate_values(optim_strategy, replicate)
            exec_time.append(ycoord["exec_time"])
            total_calls.append(ycoord["total_calls"])
            if self.cost_function:
                original_exec_time.append(ycoord["original_exec_time"])
            tmp = [
                list(ycoord["n_calls"][idx].values())
                for idx in range(len(ycoord["n_calls"]))
            ]
            n_calls.append(tmp)
            tmp = [
                list(ycoord["n_calls_linearize"][idx].values())
                for idx in range(len(ycoord["n_calls_linearize"]))
            ]
            n_calls_linearize.append(tmp)
            status.append(ycoord["status"])
            is_feasible.append(ycoord["is_feasible"])
            scaling.append(ycoord["scaling"])
        scaling_levels = xcoord
        values = {"exec_time": array(exec_time).T}  # (n_scal, n_rep)
        if self.cost_function:
            values["original_exec_time"] = array(original_exec_time).T  # (n_s, n_r)
        values["n_calls"] = array(n_calls).T  # (n_d, n_s, n_rep)
        values["n_calls_linearize"] = array(n_calls_linearize).T  # (n_d, n_s, n_r)
        values["total_calls"] = array(total_calls).T  # (n_scal, n_rep)
        values["is_feasible"] = array(is_feasible).T  # (n_scal, n_rep)
        return scaling_levels, values

    def __get_replicate_values(
        self, optim_strategy: str, replicate: int
    ) -> tuple[list[int], dict[str, NDArray[float]]]:
        """Get values of criteria and corresponding scaling levels.

        Args:
            optim_strategy: optimization strategy.
            replicate: replicate index.

        Returns:The scaling levels, the criteria values.
        """
        are_replicate = [value == replicate for value in self.get_replicates()]
        are_optim_strategy = [
            value == optim_strategy for value in self.get_optimization_strategies()
        ]
        are_ok = [
            is_rep and is_oo for is_rep, is_oo in zip(are_replicate, are_optim_strategy)
        ]
        indices = [index for index, is_ok in enumerate(are_ok) if is_ok]
        scaling_levels = self.get_scaling_strategies()
        scaling_levels = [scaling_levels[index] for index in indices]
        tmp = sorted(range(len(scaling_levels)), key=lambda k: scaling_levels[k])
        indices = [indices[index] for index in tmp]
        scaling_levels = [scaling_levels[index] for index in tmp]
        results = [self.scalability_results[index] for index in indices]
        exec_time = [result.exec_time for result in results]
        total_calls = [result.total_calls for result in results]
        original_exec_time = [result.original_exec_time for result in results]
        n_calls = [result.n_calls for result in results]
        n_calls_linearize = [result.n_calls_linearize for result in results]
        status = [result.status for result in results]
        is_feasible = [result.is_feasible for result in results]
        scaling = [result.scaling for result in results]
        values = {
            "exec_time": exec_time,
            "total_calls": total_calls,
            "n_calls": n_calls,
            "n_calls_linearize": n_calls_linearize,
            "status": status,
            "is_feasible": is_feasible,
            "scaling": scaling,
        }
        if self.cost_function:
            values["original_exec_time"] = original_exec_time
        return scaling_levels, values

    @property
    def names(self) -> list[str]:
        """The names of the scalability results."""
        return [value.name for value in self.scalability_results]

    def get_optimization_strategies(self, unique: bool = False) -> list[str]:
        """Return the names of the optimization strategies.

        Args:
            unique: Whether to return unique values.
                Otherwise, one value per scalability result.

        Returns:
            The names of the optimization strategies.
        """
        strategy_names = ["_".join(name.split("_")[0:-2]) for name in self.names]
        if unique:
            return sorted(set(strategy_names))

        return strategy_names

    @property
    def optimization_strategies(self) -> list[str]:
        """The names of the optimization strategies."""
        return self.get_optimization_strategies(True)

    def get_scaling_strategies(self, unique: bool = False) -> list[int]:
        """Return the identifiers of the scaling strategies.

        Args:
            unique: Whether to return unique values.
                Otherwise, one value per scalability result.

        Returns:
            The names of the scaling strategies.
        """
        strategy_identifiers = [int(name.split("_")[-2]) for name in self.names]
        if unique:
            return sorted(set(strategy_identifiers))

        return strategy_identifiers

    def get_replicates(self, unique: bool = False) -> list[int]:
        """Return the replicate identifiers.

        Args:
            unique: Whether to return unique values.
                Otherwise, one value per scalability result.

        Returns:
            The names of the replicate identifiers.
        """
        rep = [int(name.split("_")[-1]) for name in self.names]
        if unique:
            rep = sorted(set(rep))
        return rep

    def _estimate_original_time(self) -> float:
        """Estimate the original execution time.

        From the number of calls and
        linearizations of the different disciplines and top-level disciplines and from
        the cost functions provided by the user.

        Returns:
            The original time.

        Raises:
            ValueError: When there is no cost function for the formulation.
        """
        for scalability_result in self.scalability_results:
            n_c = scalability_result.n_calls
            n_cl = scalability_result.n_calls_linearize
            n_tl_c = scalability_result.n_calls_top_level
            n_tl_cl = scalability_result.n_calls_linearize_top_level
            varsizes = scalability_result.new_varsizes
            formulation = scalability_result.formulation
            if formulation not in self.cost_function:
                msg = (
                    f"The cost function of {formulation} must be defined "
                    "in order to compute "
                    "the estimated original time."
                )
                raise ValueError(msg)
            result = self.cost_function[formulation](
                varsizes, n_c, n_cl, n_tl_c, n_tl_cl
            )
            scalability_result.original_exec_time = result
