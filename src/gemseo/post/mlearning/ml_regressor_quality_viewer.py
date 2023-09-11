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
"""Visualization of the quality of a regression model."""
from __future__ import annotations

from types import MappingProxyType
from typing import Any
from typing import Sequence

from gemseo.datasets.dataset import Dataset
from gemseo.mlearning.regression.regression import MLRegressionAlgo
from gemseo.post.dataset.scatter import Scatter
from gemseo.post.dataset.scatter_plot_matrix import ScatterMatrix
from gemseo.post.dataset.scatter_plot_matrix import ScatterMatrixOption

DatasetPlotOption = ScatterMatrixOption


class MLRegressorQualityViewer:
    """Visualization of the quality of a regression model."""

    __algo: MLRegressionAlgo
    """The regression algorithm."""

    def __init__(self, algo: MLRegressionAlgo) -> None:
        """
        Args:
            algo: The regression algorithm.
        """  # noqa: D205 D212 D415
        self.__algo = algo

    def __plot_model_data(
        self,
        output: str | tuple[str, int],
        plot_residuals: bool,
        default_file_name: str,
        test_dataset: Dataset | None = None,
        input_names: Sequence[str] | str = (),
        use_scatter_matrix: bool = True,
        filter_scatter_plots: bool = True,
        dataset_plot_options: dict[str, DatasetPlotOption] = MappingProxyType({}),
        **execution_options: Any,
    ) -> list[Scatter] | ScatterMatrix:
        """Plot model data versus observations.

        Args:
            output: The name of the output of interest,
                and possibly the component of interest;
                if the latter is missing,
                use all the components of the output.
            plot_residuals: Whether the model data are residuals.
                Otherwise, the model data are predictions.
            default_file_name: The default file name.
            input_names: The names of the inputs to plot in addition to the model data;
                if empty, plot the observations instead.
            test_dataset: The validation dataset.
                If ``None``, use the learning dataset.
            use_scatter_matrix: Whether the method outputs a :class:`.ScatterMatrix`.
                Otherwise, it outputs a list of :class:`.Scatter`.
            filter_scatter_plots: Whether the scatters
                without the model data are removed.
            dataset_plot_options: The options
                of the underlying :class:`.DatasetPlot`.
            **execution_options: The execution options
                of the underlying :class:`.DatasetPlot`.

        Returns:
            The plot of the model data versus the observations.
        """
        dataset_plot_options = dataset_plot_options or {}
        output_name, output_component = (
            (output, ()) if isinstance(output, str) else output
        )

        if test_dataset is None:
            dataset = self.__algo.learning_set
        else:
            dataset = test_dataset

        observed_output_data = dataset.get_view(
            group_names=dataset.OUTPUT_GROUP,
            variable_names=output_name,
            components=output_component,
        ).to_numpy()
        observed_input_data = dataset.get_view(
            group_names=dataset.INPUT_GROUP,
            variable_names=self.__algo.input_names,
        ).to_numpy()
        predicted_output_data = self.__algo.predict(observed_input_data)

        if plot_residuals:
            model_data = predicted_output_data - observed_output_data
            prefix = "R"
        else:
            model_data = predicted_output_data
            prefix = "P"

        if isinstance(output_component, int):
            model_data = model_data[:, [output_component]]
            output_name = f"{output_name}[{output_component}]"

        model_data_name = f"{prefix}[{output_name}]"
        dataset_to_plot = Dataset()
        dataset_to_plot.add_variable(model_data_name, model_data)
        variable_names = [model_data_name]
        if input_names:
            if isinstance(input_names, str):
                input_names = [input_names]

            variable_names.extend(input_names)
            for input_name in input_names:
                dataset_to_plot.add_variable(
                    input_name,
                    dataset.get_view(
                        group_names=dataset.INPUT_GROUP,
                        variable_names=input_name,
                    ).to_numpy(),
                )
        else:
            variable_names.append(output_name)
            dataset_to_plot.add_variable(output_name, observed_output_data)

        trend = dataset_plot_options.pop("trend", ScatterMatrix.Trend.LINEAR)
        file_name = execution_options.pop("file_name", default_file_name)
        if use_scatter_matrix:
            kde = dataset_plot_options.pop("kde", True)
            range_padding = dataset_plot_options.pop("range_padding", 0.2)
            alpha = dataset_plot_options.pop("alpha", 1.0)
            scatter_matrix = ScatterMatrix(
                dataset_to_plot,
                variable_names,
                kde=kde,
                trend=trend,
                range_padding=range_padding,
                alpha=alpha,
                **dataset_plot_options,
            )
            scatter_matrix.execute(file_name=file_name, **execution_options)
            return scatter_matrix
        else:
            scatters = []
            variable_names = [
                (column[1], column[2])
                for column in dataset_to_plot.get_columns(variable_names, True)
            ]
            file_index = 0
            for variable_name in variable_names:
                for other_variable_name in variable_names:
                    if other_variable_name == variable_name:
                        continue

                    if filter_scatter_plots and model_data_name not in [
                        variable_name[0],
                        other_variable_name[0],
                    ]:
                        continue

                    scatter = Scatter(
                        dataset_to_plot, variable_name, other_variable_name, trend
                    )
                    scatter.execute(
                        file_name=file_name,
                        file_name_suffix=f"_{file_index}",
                        **execution_options,
                    )
                    scatters.append(scatter)
                    file_index += 1

            return scatters

    def plot_residuals_vs_observations(
        self,
        output: str | tuple[str, int],
        test_dataset: Dataset | None = None,
        use_scatter_matrix: bool = True,
        filter_scatter_plots: bool = True,
        dataset_plot_options: dict[str, DatasetPlotOption] = MappingProxyType({}),
        **execution_options: Any,
    ) -> list[Scatter] | ScatterMatrix:
        """Plot the residuals of the model versus the observations.

        Args:
            output: The name of the output of interest,
                and possibly the component of interest;
                if the latter is missing,
                use all the components of the output.
            test_dataset: The validation dataset.
                If ``None``, use the learning dataset.
            use_scatter_matrix: Whether the method outputs a :class:`.ScatterMatrix`.
                Otherwise, it outputs a list of :class:`.Scatter`.
            filter_scatter_plots: Whether the scatters
                without the model data are removed.
            dataset_plot_options: The options
                of the underlying :class:`.DatasetPlot`.
            **execution_options: The execution options
                of the underlying :class:`.DatasetPlot`.

        Returns:
            The plots of the residuals of the model versus the observations.
        """
        return self.__plot_model_data(
            output,
            True,
            "residuals_vs_observations",
            test_dataset=test_dataset,
            input_names=(),
            use_scatter_matrix=use_scatter_matrix,
            filter_scatter_plots=filter_scatter_plots,
            dataset_plot_options=dataset_plot_options,
            **execution_options,
        )

    def plot_residuals_vs_inputs(
        self,
        output: str | tuple[str, int],
        input_names: Sequence[str],
        test_dataset: Dataset | None = None,
        use_scatter_matrix: bool = True,
        filter_scatter_plots: bool = True,
        dataset_plot_options: dict[str, DatasetPlotOption] = MappingProxyType({}),
        **execution_options: Any,
    ) -> list[Scatter] | ScatterMatrix:
        """Plot the residuals of the model versus the inputs.

        Args:
            output: The name of the output of interest,
                and possibly the component of interest;
                if the latter is missing,
                use all the components of the output.
            input_names: The names of the inputs to plot in addition to the model data.
            test_dataset: The validation dataset.
                If ``None``, use the learning dataset.
            use_scatter_matrix: Whether the method outputs a :class:`.ScatterMatrix`.
                Otherwise, it outputs a list of :class:`.Scatter`.
            filter_scatter_plots: Whether the scatters
                without the model data are removed.
            dataset_plot_options: The options
                of the underlying :class:`.DatasetPlot`.
            **execution_options: The execution options
                of the underlying :class:`.DatasetPlot`.

        Returns:
            The plots of the residuals of the model versus the inputs.
        """
        return self.__plot_model_data(
            output,
            True,
            "residuals_vs_inputs",
            test_dataset=test_dataset,
            input_names=input_names,
            use_scatter_matrix=use_scatter_matrix,
            filter_scatter_plots=filter_scatter_plots,
            dataset_plot_options=dataset_plot_options,
            **execution_options,
        )

    def plot_predictions_vs_observations(
        self,
        output: str | tuple[str, int],
        test_dataset: Dataset | None = None,
        use_scatter_matrix: bool = True,
        filter_scatter_plots: bool = True,
        dataset_plot_options: dict[str, DatasetPlotOption] = MappingProxyType({}),
        **execution_options: Any,
    ) -> list[Scatter] | ScatterMatrix:
        """Plot the predictions versus the observations.

        Args:
            output: The name of the output of interest,
                and possibly the component of interest;
                if the latter is missing,
                use all the components of the output.
            test_dataset: The validation dataset.
                If ``None``, use the learning dataset.
            use_scatter_matrix: Whether the method outputs a :class:`.ScatterMatrix`.
                Otherwise, it outputs a list of :class:`.Scatter`.
            filter_scatter_plots: Whether the scatters
                without the model data are removed.
            dataset_plot_options: The options
                of the underlying :class:`.DatasetPlot`.
            **execution_options: The execution options
                of the underlying :class:`.DatasetPlot`.

        Returns:
            The plots of the predictions versus the observations.
        """
        return self.__plot_model_data(
            output,
            False,
            "predictions_vs_observations",
            test_dataset=test_dataset,
            input_names=(),
            use_scatter_matrix=use_scatter_matrix,
            filter_scatter_plots=filter_scatter_plots,
            dataset_plot_options=dataset_plot_options,
            **execution_options,
        )
