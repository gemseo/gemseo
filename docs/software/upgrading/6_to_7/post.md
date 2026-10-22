---
description: "Migrating post-processing from GEMSEO 6.3 to GEMSEO 7: what bump-gemseo does not migrate."
search:
  boost: 1
---

<!--
Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
Commons, PO Box 1866, Mountain View, CA 94042, USA.
-->

# Post-processing

The changes of GEMSEO 7 about post-processors and dataset plots that bump-gemseo does not migrate on its own. Run bump-gemseo first (see [Running bump-gemseo](bump_gemseo.md)), then use this page. Each example was executed: the code for GEMSEO 6.3 with GEMSEO 6.3.3, the code for GEMSEO 7 with GEMSEO 7.0.0rc0; both print the same values unless the entry says otherwise. The code for GEMSEO 7 also contains the renames bump-gemseo does on its own.

## Changes that fail without a comment

bump-gemseo writes no comment; the code fails with GEMSEO 7, with the error quoted.

### `Boxplot` options reject `orientation`, `vert` and `labels`

In GEMSEO 6.3, `Boxplot` passed its `**boxplot_options` keyword arguments to matplotlib, including `orientation`, `vert` and `labels`.
In GEMSEO 7, they are the `options` setting of `Boxplot_Settings`, which no longer accepts `orientation`, `vert` and `labels`.

bump-gemseo leaves the call unchanged and writes no comment, so the code fails with a `ValidationError` for `Boxplot_Settings` saying that the boxplot options `['labels']` are not supported.
Set the orientation with the setting `use_vertical_bars`, and replace `labels` by the matplotlib option `tick_labels` in `options`, as in the example below.

!!! info "After bump-gemseo"

    - **Comment**: none.
    - **Error with GEMSEO 7**: `ValidationError: 1 validation error for Boxplot_Settings options Value error, The boxplot options ['labels'] are not supported; use the matplotlib option tick_labels instead. [type=value_error, input_value={'labels': ['f`
    - **See** merge request [!2768](https://gitlab.com/gemseo/dev/gemseo/-/merge_requests/2768).

=== "GEMSEO 6.3"

    ```python
    import numpy as np
    from gemseo.datasets.dataset import Dataset
    from gemseo.post.dataset.boxplot import Boxplot
    dataset = Dataset.from_array(np.array([[1.0, 2.0], [2.0, 1.0], [0.5, 3.0]]), ["x", "y"])
    plot = Boxplot(dataset, use_vertical_bars=True, labels=["first", "second"])
    print(len(plot.execute(save=False)))
    ```

=== "GEMSEO 7"

    ```python
    import numpy as np
    from gemseo.dataset import Dataset
    from gemseo.post.dataset import Boxplot
    from gemseo.post.dataset.boxplot_settings import Boxplot_Settings
    dataset = Dataset.from_array(np.array([[1.0, 2.0], [2.0, 1.0], [0.5, 3.0]]), ["x", "y"])
    plot = Boxplot(dataset, settings=Boxplot_Settings(use_vertical_bars=True, options={"tick_labels": ["first", "second"]}))
    print(len(plot.execute(save=False)))
    ```

### Dataset plots renamed `PairPlot`, `Heatmap` and `RadViz`

GEMSEO 7 renamed three dataset plots: `ScatterMatrix` is now `PairPlot`, `Radar` is now `RadViz`, and `ColorEvolution` was merged into `Heatmap`, where `Heatmap_Settings.symmetric` set to `False` reproduces the old `ColorEvolution` behavior.
The names given to `DatasetPlotFactory` change accordingly.

bump-gemseo does not change these strings and writes no comment, so the factory fails with an `ImportError` saying `The class ScatterMatrix is not available` and listing the available ones.
Replace the names, and pass the specific arguments of a plot through its settings model, e.g. `RadViz_Settings(classifier="c")` for `RadViz`, as in the example below.

!!! info "After bump-gemseo"

    - **Comment**: none.
    - **Error with GEMSEO 7**: `ImportError: The class ScatterMatrix is not available; the available ones are: AndrewsCurves, BarPlot, BaseDatasetPlot, Boxplot, Curves, Heatmap, Lines, PairPlot, ParallelCoordinates, RadViz, RadarChart, Scatter, Surface`
    - **See** issue [#1780](https://gitlab.com/gemseo/dev/gemseo/-/issues/1780), merge request [!2436](https://gitlab.com/gemseo/dev/gemseo/-/merge_requests/2436); issue [#1854](https://gitlab.com/gemseo/dev/gemseo/-/issues/1854), merge request [!2621](https://gitlab.com/gemseo/dev/gemseo/-/merge_requests/2621); issue [#1720](https://gitlab.com/gemseo/dev/gemseo/-/issues/1720), merge request [!2325](https://gitlab.com/gemseo/dev/gemseo/-/merge_requests/2325).

=== "GEMSEO 6.3"

    ```python
    import numpy as np
    from gemseo.datasets.dataset import Dataset
    from gemseo.post.dataset.factory import DatasetPlotFactory
    data = np.array([[0.0, 1.0, 2.0, 0.0], [1.0, 2.0, 0.5, 1.0], [2.0, 0.5, 1.0, 0.0], [3.0, 1.5, 3.0, 1.0]])
    dataset = Dataset.from_array(data, ["x", "y", "z", "c"])
    factory = DatasetPlotFactory()
    counts = [len(factory.create("ScatterMatrix", dataset).execute(save=False))]
    counts.append(len(factory.create("ColorEvolution", dataset).execute(save=False)))
    counts.append(len(factory.create("Radar", dataset, "c").execute(save=False)))
    print(counts)
    ```

=== "GEMSEO 7"

    ```python
    import numpy as np
    from gemseo.dataset import Dataset
    from gemseo.post.dataset.factory import DatasetPlotFactory
    from gemseo.post.dataset.radviz_settings import RadViz_Settings
    data = np.array([[0.0, 1.0, 2.0, 0.0], [1.0, 2.0, 0.5, 1.0], [2.0, 0.5, 1.0, 0.0], [3.0, 1.5, 3.0, 1.0]])
    dataset = Dataset.from_array(data, ["x", "y", "z", "c"])
    factory = DatasetPlotFactory()
    counts = [len(factory.create("PairPlot", dataset).execute(save=False))]
    counts.append(len(factory.create("Heatmap", dataset).execute(save=False)))
    counts.append(len(factory.create("RadViz", dataset, settings=RadViz_Settings(classifier="c")).execute(save=False)))
    print(counts)
    ```

### `Trend` is no longer nested in `Scatter`

In GEMSEO 6.3, the trends of `Scatter` and `ScatterMatrix` were the nested enumerations `Scatter.Trend` and `ScatterMatrix.Trend`.
GEMSEO 7 removed them; use the enumeration `Trend` of `gemseo.post.dataset.trend`, for `Scatter` as for `PairPlot`, the new name of `ScatterMatrix`.

bump-gemseo renames `DatasetPlot` to `BaseDatasetPlot` but leaves the nested enumerations unchanged and writes no comment, so the code fails with `AttributeError: type object 'Scatter' has no attribute 'Trend'`.
Import `Trend` and give it to the `trend` field of the settings model of the plot, e.g. `Scatter_Settings`, since the keyword arguments of the dataset plots were replaced by `settings`, as in the example below.

!!! info "After bump-gemseo"

    - **Comment**: none.
    - **Error with GEMSEO 7**: `AttributeError: type object 'Scatter' has no attribute 'Trend'`
    - **See** issue [#1720](https://gitlab.com/gemseo/dev/gemseo/-/issues/1720), merge request [!2325](https://gitlab.com/gemseo/dev/gemseo/-/merge_requests/2325).

=== "GEMSEO 6.3"

    ```python
    import numpy as np
    from gemseo.datasets.dataset import Dataset
    from gemseo.post.dataset.dataset_plot import DatasetPlot
    from gemseo.post.dataset.scatter import Scatter
    from gemseo.post.dataset.scatter_plot_matrix import ScatterMatrix
    rng = np.random.default_rng(2)
    dataset = Dataset.from_array(rng.random((10, 2)), ["x", "y"])
    plot = Scatter(dataset, "x", "y", trend=Scatter.Trend.LINEAR)
    print([DatasetPlot.PlotEngine.PLOTLY, ScatterMatrix.Trend.QUADRATIC, len(plot.execute(save=False))])
    ```

=== "GEMSEO 7"

    ```python
    import numpy as np
    from gemseo.dataset import Dataset
    from gemseo.post.dataset import Scatter
    from gemseo.post.dataset.base import BaseDatasetPlot
    from gemseo.post.dataset.scatter_settings import Scatter_Settings
    from gemseo.post.dataset.trend import Trend
    rng = np.random.default_rng(2)
    dataset = Dataset.from_array(rng.random((10, 2)), ["x", "y"])
    plot = Scatter(dataset, settings=Scatter_Settings(x="x", y="y", trend=Trend.LINEAR))
    print([BaseDatasetPlot.PlotEngine.PLOTLY, Trend.QUADRATIC, len(plot.execute(save=False))])
    ```

### `RadarChart` post-processor renamed `ConstraintRadar`

GEMSEO 7 renamed the post-processor `RadarChart` to `ConstraintRadar`, its settings class `RadarChart_Settings` to `ConstraintRadar_Settings`, and the module `gemseo.post.radar_chart` to `gemseo.post.constraint_radar`.

bump-gemseo does not change the string `"RadarChart"` given to `execute_post` and writes no comment, so the code fails with an `ImportError` saying `The class RadarChart is not available`.
Build a `ConstraintRadar_Settings` and pass it to `execute_post` as `settings_model`, as in the example below.

!!! info "After bump-gemseo"

    - **Comment**: none.
    - **Error with GEMSEO 7**: `ImportError: The class RadarChart is not available; the available ones are: Animation, BasicHistory, ConstraintRadar, ConstraintsHistory, Correlations, GradientSensitivity, HessianHistory, ObjConstrHist, OptHistoryView,`
    - **See** issue [#1720](https://gitlab.com/gemseo/dev/gemseo/-/issues/1720), merge request [!2325](https://gitlab.com/gemseo/dev/gemseo/-/merge_requests/2325).

=== "GEMSEO 6.3"

    ```python
    from gemseo import execute_post
    from gemseo.algos.opt.factory import OptimizationLibraryFactory
    from gemseo.problems.optimization.power_2 import Power2
    problem = Power2()
    OptimizationLibraryFactory().execute(problem, algo_name="SLSQP", max_iter=10)
    post = execute_post(problem, post_name="RadarChart", save=False, show=False)
    print([post.__class__.__name__.replace("RadarChart", "ConstraintRadar"), len(post.figures)])
    ```

=== "GEMSEO 7"

    ```python
    from gemseo import execute_post
    from gemseo.optimization import SLSQP_Settings
    from gemseo.optimization.factory import OptimizationLibraryFactory
    from gemseo.post import ConstraintRadar_Settings
    from gemseo.problem.optimization.power_2 import Power2
    problem = Power2()
    OptimizationLibraryFactory().execute(problem, settings=SLSQP_Settings(max_iter=10))
    post = execute_post(problem, settings_model=ConstraintRadar_Settings(save=False, show=False))
    print([post.__class__.__name__, len(post.figures)])
    ```

## Changes flagged by bump-gemseo

bump-gemseo leaves a `# TODO(bump-gemseo)` comment: search for its text below.

### `fig_size_x` and `fig_size_y` replaced by `fig_size`

GEMSEO 7 removed the attributes `fig_size_x` and `fig_size_y` of `BaseDatasetPlot`, formerly `DatasetPlot`; use the setting `fig_size` instead, the pair of the width and the height, available as `settings.fig_size`.

bump-gemseo marks these lines with a comment.
Left as it is, the code raises no error, so do not skip the marked lines: replace the two assignments by a single one, e.g. `plot.settings.fig_size = (7.0, 3.0)`, and read the width and the height from this pair, as in the example below.

!!! info "After bump-gemseo"

    - **Comment**: `use settings.fig_size, the pair of the width and the height, instead`
    - **See** issue [#1720](https://gitlab.com/gemseo/dev/gemseo/-/issues/1720), merge request [!2325](https://gitlab.com/gemseo/dev/gemseo/-/merge_requests/2325).

=== "GEMSEO 6.3"

    ```python
    import numpy as np
    from gemseo.datasets.dataset import Dataset
    from gemseo.post.dataset.yvsx import YvsX
    dataset = Dataset.from_array(np.array([[0.0, 1.0], [1.0, 2.0]]), ["x", "y"])
    plot = YvsX(dataset, "x", "y")
    plot.fig_size_x = 7.0
    plot.fig_size_y = 3.0
    plot.execute(save=False)
    print([plot.fig_size_x, plot.fig_size_y])
    ```

=== "GEMSEO 7"

    ```python
    import numpy as np
    from gemseo.dataset import Dataset
    from gemseo.post.dataset import YvsX
    from gemseo.post.dataset.yvsx_settings import YvsX_Settings
    dataset = Dataset.from_array(np.array([[0.0, 1.0], [1.0, 2.0]]), ["x", "y"])
    plot = YvsX(dataset, settings=YvsX_Settings(x="x", y="y"))
    plot.settings.fig_size = (7.0, 3.0)
    plot.execute(save=False)
    print(list(plot.settings.fig_size))
    ```

### Dataset plot settings are a model read from `settings`

In GEMSEO 6.3, a dataset plot kept its settings in the private attributes `_common_settings` and `_specific_settings`, and its number of items in `_n_items`.
In GEMSEO 7, the plot classes take a single `settings` argument, a model whose class is the `settings_class` attribute of the plot, e.g. `YvsX_Settings` for `YvsX`, gathering the former keyword arguments and attributes such as `color`.

bump-gemseo replaces `_common_settings` by `settings` and `_n_items` by `settings.n_items`, but marks `_specific_settings` with a comment; left as it is, the code raises `AttributeError: 'YvsX' object has no attribute '_specific_settings'`.
Read these fields from `self.settings`, and build the plot from its settings model, as in the example below.
For a custom plot, declare its fields in a `<Plot>_Settings` class set as the `settings_class` attribute of the plot, and remove its `__init__`.

!!! info "After bump-gemseo"

    - **Comment**: `the settings of a plot are now a model, so declare its fields in a <Plot>_Settings class set as the settings_class attribute of the plot, remove its __init__, and read them from self.settings`
    - **Error with GEMSEO 7**: `AttributeError: 'YvsX' object has no attribute '_specific_settings'`

=== "GEMSEO 6.3"

    ```python
    import numpy as np
    from gemseo.datasets.dataset import Dataset
    from gemseo.post.dataset.yvsx import YvsX
    dataset = Dataset.from_array(np.array([[1.0, 2.0], [3.0, 4.0]]), ["x", "y"])
    plot = YvsX(dataset, "x", "y")
    plot.color = "red"
    print([plot._common_settings.color, plot._n_items, list(plot._specific_settings.x)])
    ```

=== "GEMSEO 7"

    ```python
    import numpy as np
    from gemseo.dataset import Dataset
    from gemseo.post.dataset.yvsx import YvsX
    from gemseo.post.dataset.yvsx_settings import YvsX_Settings
    dataset = Dataset.from_array(np.array([[1.0, 2.0], [3.0, 4.0]]), ["x", "y"])
    plot = YvsX(dataset, YvsX_Settings(x="x", y="y", color="red"))
    print([plot.settings.color, plot.settings.n_items, list(plot.settings.x)])
    ```

## Deprecated names

The code still works with GEMSEO 7, with a deprecation warning.

### `execute_post` takes a settings model

In GEMSEO 6.3, `execute_post` took the name of the post-processor as `post_name` and its settings as keyword arguments.
In GEMSEO 7, a post-processor is configured with a settings model, e.g. `BasicHistory_Settings`, passed here as `settings_model`; the `**settings` arguments of `PostFactory.execute`, including `post_name`, were removed.

bump-gemseo does not convert these loose settings: it leaves `post_name` and the keyword arguments as they are and writes no comment.
GEMSEO 7 still runs such a call with the same values and without warning, so nothing tells you that this code was not migrated.
Build the settings model of the post-processor from the keyword arguments and pass it, as in the example below.

!!! info "After bump-gemseo"

    - **Comment**: none.
    - **With GEMSEO 7**: it works, with a deprecation warning.

=== "GEMSEO 6.3"

    ```python
    from gemseo import execute_post
    from gemseo.algos.opt.factory import OptimizationLibraryFactory
    from gemseo.problems.optimization.rosenbrock import Rosenbrock
    problem = Rosenbrock()
    OptimizationLibraryFactory().execute(problem, algo_name="SLSQP", max_iter=5)
    post = execute_post(problem, post_name="BasicHistory", variable_names=["rosen"], save=False, show=False)
    print(len(post.figures))
    ```

=== "GEMSEO 7"

    ```python
    from gemseo import execute_post
    from gemseo.optimization import SLSQP_Settings
    from gemseo.optimization.factory import OptimizationLibraryFactory
    from gemseo.post import BasicHistory_Settings
    from gemseo.problem.optimization.rosenbrock import Rosenbrock
    problem = Rosenbrock()
    OptimizationLibraryFactory().execute(problem, settings=SLSQP_Settings(max_iter=5))
    post = execute_post(problem, settings_model=BasicHistory_Settings(variable_names=["rosen"], save=False, show=False))
    print(len(post.figures))
    ```

## Removed names

bump-gemseo marks each use with a `# TODO(bump-gemseo)` comment. Unless an entry above gives an equivalent, they have no replacement: remove the code that uses them.

- `gemseo.post.dataset.scatter_plot_matrix.ScatterMatrixOption`
