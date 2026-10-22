---
description: "Migrating machine learning from GEMSEO 6.3 to GEMSEO 7: what bump-gemseo does not migrate."
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

# Machine learning

The changes of GEMSEO 7 about regressors, classifiers, clustering and their quality that bump-gemseo does not migrate on its own. Run bump-gemseo first (see [Running bump-gemseo](bump_gemseo.md)), then use this page. Each example was executed: the code for GEMSEO 6.3 with GEMSEO 6.3.3, the code for GEMSEO 7 with GEMSEO 7.0.0rc0; both print the same values unless the entry says otherwise. The code for GEMSEO 7 also contains the renames bump-gemseo does on its own.

## Silent changes

The code runs after bump-gemseo, without error nor comment, and gives other results.

### RBF models invert `epsilon` and add a polynomial term

`RBFRegressor` is now based on SciPy's `RBFInterpolator`. The meaning of `epsilon` is inverted: the kernels are functions of `epsilon*r` instead of `r/epsilon`, so use `epsilon_new = 1/epsilon_old`. Moreover, by default the model now includes a polynomial term whose degree is the per-kernel minimum; the new setting `degree` sets it, `-1` meaning no polynomial.

After bump-gemseo, the code runs without error and without comment, but the model and its predictions differ from GEMSEO 6.3. Check every RBF model whose `epsilon` you set: replace its value by its inverse and, to get a model without polynomial term, set `degree=-1`, as in the example below. The changelog notes that predictions may still differ slightly from previous versions.

!!! info "After bump-gemseo"

    - **Comment**: none.
    - **With GEMSEO 7**: no error, but other results.
    - **See** issue [#1861](https://gitlab.com/gemseo/dev/gemseo/-/issues/1861), merge request [!2685](https://gitlab.com/gemseo/dev/gemseo/-/merge_requests/2685).

=== "GEMSEO 6.3"

    ```python
    import numpy as np
    from gemseo.datasets.io_dataset import IODataset
    from gemseo.mlearning.regression.algos.rbf import RBFRegressor
    x = np.linspace(0.0, 1.0, 7)[:, None]
    dataset = IODataset()
    dataset.add_input_variable("x", x)
    dataset.add_output_variable("y", np.sin(4 * x) + x)
    model = RBFRegressor(dataset, function="multiquadric", epsilon=2.0)
    model.learn()
    print([model.predict(np.array([0.25])), model.predict(np.array([0.9]))])
    ```

=== "GEMSEO 7"

    ```python
    import numpy as np
    from gemseo.dataset import IODataset
    from gemseo.machine_learning.regression.model.rbf import RBFRegressor
    from gemseo.machine_learning.regression.model.rbf_settings import RBFRegressor_Settings
    x = np.linspace(0.0, 1.0, 7)[:, None]
    dataset = IODataset()
    dataset.add_input_variable("x", x)
    dataset.add_output_variable("y", np.sin(4 * x) + x)
    model = RBFRegressor(dataset, settings=RBFRegressor_Settings(kernel="multiquadric", epsilon=1 / 2.0, degree=-1))
    model.learn()
    print([model.predict(np.array([0.25])), model.predict(np.array([0.9]))])
    ```

## Changes that fail without a comment

bump-gemseo writes no comment; the code fails with GEMSEO 7, with the error quoted.

### `KMeans.EPS` was removed

The class attribute `KMeans.EPS` was removed; use the constant `gemseo.util.constant.epsilon` instead, which is now a builtin `float`.

bump-gemseo leaves no comment on this line, and GEMSEO 7 raises `AttributeError: type object 'KMeans' has no attribute 'EPS'`. Import `epsilon` from `gemseo.util.constant` and use it in place of `KMeans.EPS`, as in the example below.

!!! info "After bump-gemseo"

    - **Comment**: none.
    - **Error with GEMSEO 7**: `AttributeError: type object 'KMeans' has no attribute 'EPS'`
    - **See** issue [#1873](https://gitlab.com/gemseo/dev/gemseo/-/issues/1873), merge request [!2698](https://gitlab.com/gemseo/dev/gemseo/-/merge_requests/2698).

=== "GEMSEO 6.3"

    ```python
    from gemseo.mlearning.clustering.algos.kmeans import KMeans
    print(KMeans.EPS)
    ```

=== "GEMSEO 7"

    ```python
    from gemseo.util.constant import epsilon
    print(epsilon)
    ```

### Most attributes of `MLModelAssessor` were removed

`MLAlgoAssessor` was renamed to `MLModelAssessor`, and its attributes `algo`, `parameters`, `transformer`, `measure`, `measure_options` and `dataset` were removed. Its arguments changed too: the model is now described by a `settings` argument of type `BaseMLModelSettings`, whose `transformer` field replaces the `transformer` argument.

bump-gemseo renames the class but leaves no comment on these attributes, so reading one of them raises `AttributeError: 'MLModelAssessor' object has no attribute 'algo'`. GEMSEO 7 has no replacement for them: keep the values you pass to the assessor in your own variables and remove the code that reads them from the assessor.

!!! info "After bump-gemseo"

    - **Comment**: none.
    - **Error with GEMSEO 7**: `AttributeError: 'MLModelAssessor' object has no attribute 'algo'`
    - **See** issue [#1696](https://gitlab.com/gemseo/dev/gemseo/-/issues/1696), merge request [!2285](https://gitlab.com/gemseo/dev/gemseo/-/merge_requests/2285).

=== "GEMSEO 6.3"

    ```python
    import numpy as np
    from gemseo.datasets.io_dataset import IODataset
    from gemseo.mlearning.core.calibration import MLAlgoAssessor
    from gemseo.mlearning.regression.quality.mse_measure import MSEMeasure
    x = np.linspace(0.0, 1.0, 11)[:, None]
    dataset = IODataset()
    dataset.add_input_variable("x", x)
    dataset.add_output_variable("y", 1.0 + x**3)
    assessor = MLAlgoAssessor("PolynomialRegressor", dataset, ["degree"], MSEMeasure)
    print([assessor.algo, list(assessor.parameters), assessor.measure.__name__, dict(assessor.measure_options), dict(assessor.transformer)])
    ```

=== "GEMSEO 7"

    ```python
    print("<removed in GEMSEO 7>")
    ```

### `MOERegressor` no longer stores the names of its algorithms

The attributes `cluster_algo`, `classif_algo` and `regress_algo` of `MOERegressor` were removed. The algorithms are now defined by the settings attributes `clusterer_settings`, `classifier_settings` and `regressor_settings`, whose initial values are `KMeans_Settings()`, `KNNClassifier_Settings()` and `LinearRegressor_Settings()`.

bump-gemseo leaves no comment, and reading one of the removed attributes raises `AttributeError: 'MOERegressor' object has no attribute 'cluster_algo'`. To get the name of an algorithm, read the `target_class_name` of the corresponding settings, e.g. `moe.clusterer_settings.target_class_name`, as in the example below, or use the class name of the `clusterer` and `classifier` attributes and of the elements of the `regressors` attribute.

!!! info "After bump-gemseo"

    - **Comment**: none.
    - **Error with GEMSEO 7**: `AttributeError: 'MOERegressor' object has no attribute 'cluster_algo'`
    - **See** issue [#1696](https://gitlab.com/gemseo/dev/gemseo/-/issues/1696), merge request [!2285](https://gitlab.com/gemseo/dev/gemseo/-/merge_requests/2285).

=== "GEMSEO 6.3"

    ```python
    import numpy as np
    from gemseo.datasets.io_dataset import IODataset
    from gemseo.mlearning.regression.algos.moe import MOERegressor
    x = np.linspace(-1.0, 1.0, 20)[:, None]
    dataset = IODataset()
    dataset.add_input_variable("x", x)
    dataset.add_output_variable("y", np.abs(x))
    moe = MOERegressor(dataset)
    print([moe.cluster_algo, moe.classif_algo, moe.regress_algo])
    ```

=== "GEMSEO 7"

    ```python
    import numpy as np
    from gemseo.dataset import IODataset
    from gemseo.machine_learning.regression.model.moe import MOERegressor
    x = np.linspace(-1.0, 1.0, 20)[:, None]
    dataset = IODataset()
    dataset.add_input_variable("x", x)
    dataset.add_output_variable("y", np.abs(x))
    moe = MOERegressor(dataset)
    print([moe.clusterer_settings.target_class_name, moe.classifier_settings.target_class_name, moe.regressor_settings.target_class_name])
    ```

### The quality measures take a `model` argument instead of `algo`

`BaseMLAlgoQuality` was renamed to `BaseMLModelQuality`, and the argument `algo` of its subclasses, e.g. `MSEMeasure`, `R2Measure` or `F1Measure`, was renamed to `model`; the attribute `algo` was renamed to `model` too.

bump-gemseo rewrites the attribute access (`measure.model`) but leaves no comment and does not rename the keyword `algo=` in a call to a subclass, so `MSEMeasure(algo=model)` raises `TypeError: MSEMeasure.__init__() got an unexpected keyword argument 'algo'`. Replace `algo=` by `model=` in these calls, as in the example below.

!!! info "After bump-gemseo"

    - **Comment**: none.
    - **Error with GEMSEO 7**: `TypeError: MSEMeasure.__init__() got an unexpected keyword argument 'algo'`
    - **See** issue [#1696](https://gitlab.com/gemseo/dev/gemseo/-/issues/1696), merge request [!2285](https://gitlab.com/gemseo/dev/gemseo/-/merge_requests/2285).

=== "GEMSEO 6.3"

    ```python
    import numpy as np
    from gemseo.datasets.io_dataset import IODataset
    from gemseo.mlearning.regression.algos.linreg import LinearRegressor
    from gemseo.mlearning.regression.quality.mse_measure import MSEMeasure
    x = np.linspace(0.0, 1.0, 11)[:, None]
    dataset = IODataset()
    dataset.add_input_variable("x", x)
    dataset.add_output_variable("y", x**2)
    model = LinearRegressor(dataset)
    model.learn()
    measure = MSEMeasure(algo=model)
    print([measure.compute_learning_measure(), measure.algo is model])
    ```

=== "GEMSEO 7"

    ```python
    import numpy as np
    from gemseo.dataset import IODataset
    from gemseo.machine_learning.regression.model.linreg import LinearRegressor
    from gemseo.machine_learning.regression.quality.mse_measure import MSEMeasure
    x = np.linspace(0.0, 1.0, 11)[:, None]
    dataset = IODataset()
    dataset.add_input_variable("x", x)
    dataset.add_output_variable("y", x**2)
    model = LinearRegressor(dataset)
    model.learn()
    measure = MSEMeasure(model=model)
    print([measure.compute_learning_measure(), measure.model is model])
    ```

### `MLRegressorQualityViewer` takes a `regressor` argument

The argument `algo` of `MLRegressorQualityViewer` was renamed to `regressor`; the class is now imported from `gemseo.post.machine_learning`.

bump-gemseo leaves no comment on this argument, and `MLRegressorQualityViewer(algo=model)` raises `TypeError: MLRegressorQualityViewer.__init__() got an unexpected keyword argument 'algo'`. Replace `algo=` by `regressor=`, as in the example below.

!!! info "After bump-gemseo"

    - **Comment**: none.
    - **Error with GEMSEO 7**: `TypeError: MLRegressorQualityViewer.__init__() got an unexpected keyword argument 'algo'`

=== "GEMSEO 6.3"

    ```python
    import numpy as np
    from gemseo.datasets.io_dataset import IODataset
    from gemseo.mlearning.regression.algos.linreg import LinearRegressor
    from gemseo.post.mlearning.ml_regressor_quality_viewer import MLRegressorQualityViewer
    x = np.linspace(0.0, 1.0, 11)[:, None]
    dataset = IODataset()
    dataset.add_input_variable("x", x)
    dataset.add_output_variable("y", 1.0 + x**2)
    model = LinearRegressor(dataset)
    model.learn()
    viewer = MLRegressorQualityViewer(algo=model)
    plot = viewer.plot_predictions_vs_observations("y", save=False)
    print(len(plot.dataset))
    ```

=== "GEMSEO 7"

    ```python
    import numpy as np
    from gemseo.dataset import IODataset
    from gemseo.machine_learning.regression.model.linreg import LinearRegressor
    from gemseo.post.machine_learning import MLRegressorQualityViewer
    x = np.linspace(0.0, 1.0, 11)[:, None]
    dataset = IODataset()
    dataset.add_input_variable("x", x)
    dataset.add_output_variable("y", 1.0 + x**2)
    model = LinearRegressor(dataset)
    model.learn()
    viewer = MLRegressorQualityViewer(regressor=model)
    plot = viewer.plot_predictions_vs_observations("y", save=False)
    print(len(plot.dataset))
    ```

### `RegressorChain.add_regressor` takes a settings object

The method `add_algo` of `RegressorChain` was renamed to `add_regressor`, and its arguments `name`, `transformer` and `**parameters` were replaced by a single `settings` argument of type `BaseRegressorSettings`.

bump-gemseo renames the method only and leaves no comment, so a call such as `chain.add_algo("LinearRegressor", fit_intercept=True)` becomes a call to `add_regressor` that raises `TypeError: RegressorChain.add_regressor() got an unexpected keyword argument 'fit_intercept'`. Replace the name and the keyword arguments by an instance of the settings class of the regressor, e.g. `LinearRegressor_Settings(fit_intercept=True)`, as in the example below; a transformer goes in the `transformer` field of these settings.

!!! info "After bump-gemseo"

    - **Comment**: none.
    - **Error with GEMSEO 7**: `TypeError: RegressorChain.add_regressor() got an unexpected keyword argument 'fit_intercept'`
    - **See** issue [#1696](https://gitlab.com/gemseo/dev/gemseo/-/issues/1696), merge request [!2285](https://gitlab.com/gemseo/dev/gemseo/-/merge_requests/2285).

=== "GEMSEO 6.3"

    ```python
    import numpy as np
    from gemseo.datasets.io_dataset import IODataset
    from gemseo.mlearning.regression.algos.regressor_chain import RegressorChain
    x = np.linspace(0.0, 1.0, 11)[:, None]
    dataset = IODataset()
    dataset.add_input_variable("x", x)
    dataset.add_output_variable("y", 1.0 + 2.0 * x)
    chain = RegressorChain(dataset)
    chain.add_algo("LinearRegressor", fit_intercept=True)
    chain.learn()
    print(chain.predict(np.array([0.5])))
    ```

=== "GEMSEO 7"

    ```python
    import numpy as np
    from gemseo.dataset import IODataset
    from gemseo.machine_learning.regression.model.linreg_settings import LinearRegressor_Settings
    from gemseo.machine_learning.regression.model.regressor_chain import RegressorChain
    x = np.linspace(0.0, 1.0, 11)[:, None]
    dataset = IODataset()
    dataset.add_input_variable("x", x)
    dataset.add_output_variable("y", 1.0 + 2.0 * x)
    chain = RegressorChain(dataset)
    chain.add_regressor(LinearRegressor_Settings(fit_intercept=True))
    chain.learn()
    print(chain.predict(np.array([0.5])))
    ```

## Changes flagged by bump-gemseo

bump-gemseo leaves a `# TODO(bump-gemseo)` comment: search for its text below.

### The multi-start of `OTGaussianProcessRegressor` is set with DOE settings

The `multi_start_algo_name` and `multi_start_n_samples` fields of `OTGaussianProcessRegressor_Settings` were removed; use its `multi_start_algo_settings` field instead, whose type is `BaseDOESettings | None`.

bump-gemseo marks these fields with comments such as `OTGaussianProcessRegressor_Settings.multi_start_algo_name was removed; see the GEMSEO 7 changelog.` To migrate, instantiate the settings class of the DOE algorithm, e.g. `OT_MONTE_CARLO_Settings` for `"OT_MONTE_CARLO"`, pass the former `multi_start_n_samples` as its `n_samples` field, and give this object to `multi_start_algo_settings`, as in the example below.

!!! info "After bump-gemseo"

    - **Comment**: `OTGaussianProcessRegressor_Settings.multi_start_algo_name was removed; see the GEMSEO 7 changelog.` / `OTGaussianProcessRegressor_Settings.multi_start_n_samples was removed; see the GEMSEO 7 changelog.`
    - **Error with GEMSEO 7**: `AttributeError: 'OTGaussianProcessRegressor_Settings' object has no attribute 'multi_start_algo_name'`
    - **See** issue [#1716](https://gitlab.com/gemseo/dev/gemseo/-/issues/1716), merge request [!2348](https://gitlab.com/gemseo/dev/gemseo/-/merge_requests/2348).

=== "GEMSEO 6.3"

    ```python
    from gemseo.mlearning.regression.algos.ot_gpr_settings import OTGaussianProcessRegressor_Settings
    settings = OTGaussianProcessRegressor_Settings(multi_start_algo_name="OT_MONTE_CARLO", multi_start_n_samples=7)
    print([str(settings.multi_start_algo_name), settings.multi_start_n_samples])
    ```

=== "GEMSEO 7"

    ```python
    from gemseo.doe import OT_MONTE_CARLO_Settings
    from gemseo.machine_learning.regression.model.ot_gpr_settings import OTGaussianProcessRegressor_Settings
    settings = OTGaussianProcessRegressor_Settings(multi_start_algo_settings=OT_MONTE_CARLO_Settings(n_samples=7))
    print([settings.multi_start_algo_settings.__class__.__name__.removesuffix("_Settings"), settings.multi_start_algo_settings.n_samples])
    ```

### The multi-start sample count moved to the DOE settings

The `multi_start_n_samples` field of `OTGaussianProcessRegressor_Settings` was removed; use the `n_samples` field of the DOE settings given to `multi_start_algo_settings` instead, or `multi_start_algo_settings=None` for no multi-start.

When `multi_start_n_samples` is passed without `multi_start_algo_name`, bump-gemseo cannot know which settings class to create: it marks the line with comments starting with `cannot transform: multi_start_n_samples is passed without multi_start_algo_name` and `use the n_samples field of the multi_start_algo_settings instead`. Left as is, the settings raise a `ValidationError` (extra inputs are not permitted). Choose the DOE settings class explicitly and set its `n_samples` field; the example below uses `OT_OPT_LHS_Settings`, which gives the same algorithm as the GEMSEO 6.3 default.

!!! info "After bump-gemseo"

    - **Comment**: `cannot transform: multi_start_n_samples is passed without multi_start_algo_name, whose default names no Settings class, and multi_start_n_samples has a rule in a Settings class: migrate it by hand` / `use the n_samples field of the multi_start_algo_settings instead, or multi_start_algo_settings=None for no multi-start` / `OTGaussianProcessRegressor_Settings.multi_start_algo_name was removed; see the GEMSEO 7 changelog.`
    - **Error with GEMSEO 7**: `ValidationError: 1 validation error for OTGaussianProcessRegressor_Settings multi_start_n_samples Extra inputs are not permitted [type=extra_forbidden, input_value=6, input_type=int] For further information visit https:/`

=== "GEMSEO 6.3"

    ```python
    from gemseo.mlearning.regression.algos.ot_gpr_settings import OTGaussianProcessRegressor_Settings
    settings = OTGaussianProcessRegressor_Settings(multi_start_n_samples=6)
    print([str(settings.multi_start_algo_name), settings.multi_start_n_samples])
    ```

=== "GEMSEO 7"

    ```python
    from gemseo.doe import OT_OPT_LHS_Settings
    from gemseo.machine_learning.regression.model.ot_gpr_settings import OTGaussianProcessRegressor_Settings
    settings = OTGaussianProcessRegressor_Settings(multi_start_algo_settings=OT_OPT_LHS_Settings(n_samples=6))
    print([settings.multi_start_algo_settings.__class__.__name__.removesuffix("_Settings"), settings.multi_start_algo_settings.n_samples])
    ```

### `PCERegressor` reads the random space from its dataset

The `data` argument of `PCERegressor` can no longer be `None`, and the `probability_space` field of `PCERegressor_Settings` was removed. The probability space is now defined by `data.misc["input_space"]`, where `data` is the first argument of `PCERegressor`; this entry is set automatically when `data` is generated by a DOE algorithm over a `RandomSpace`.

bump-gemseo moves the settings into `PCERegressor_Settings` and marks the line with the comment `PCERegressor_Settings.probability_space was removed; see the GEMSEO 7 changelog.`; left as is, the settings raise a `ValidationError` (extra inputs are not permitted). For a quadrature without training data:

1. create an empty `IODataset` and set `dataset.misc["input_space"]` to your random space;
2. pass this dataset to `PCERegressor` instead of `None`;
3. remove `probability_space` from the settings,

as in the example below.

!!! info "After bump-gemseo"

    - **Comment**: `PCERegressor_Settings.probability_space was removed; see the GEMSEO 7 changelog.`
    - **Error with GEMSEO 7**: `ValidationError: 1 validation error for PCERegressor_Settings probability_space Extra inputs are not permitted [type=extra_forbidden, input_value=Ishigami space: +------+-...-----------------------+, input_type=IshigamiS`
    - **See** issue [#1673](https://gitlab.com/gemseo/dev/gemseo/-/issues/1673), merge request [!2309](https://gitlab.com/gemseo/dev/gemseo/-/merge_requests/2309); issue [#1673](https://gitlab.com/gemseo/dev/gemseo/-/issues/1673), merge request [!2678](https://gitlab.com/gemseo/dev/gemseo/-/merge_requests/2678).

=== "GEMSEO 6.3"

    ```python
    import numpy as np
    from gemseo.mlearning.regression.algos.pce import PCERegressor
    from gemseo.problems.uncertainty.ishigami.ishigami_discipline import IshigamiDiscipline
    from gemseo.problems.uncertainty.ishigami.ishigami_space import IshigamiSpace
    space = IshigamiSpace(IshigamiSpace.UniformDistribution.OPENTURNS)
    model = PCERegressor(None, probability_space=space, discipline=IshigamiDiscipline(), use_quadrature=True, degree=4)
    model.learn()
    print(model.predict(np.array([0.1, 0.2, 0.3])))
    ```

=== "GEMSEO 7"

    ```python
    import numpy as np
    from gemseo.dataset import IODataset
    from gemseo.machine_learning.regression.model.pce import PCERegressor
    from gemseo.machine_learning.regression.model.pce_settings import PCERegressor_Settings
    from gemseo.problem.uncertainty.ishigami import IshigamiDiscipline
    from gemseo.problem.uncertainty.ishigami import IshigamiSpace
    space = IshigamiSpace(IshigamiSpace.UniformDistribution.OPENTURNS)
    dataset = IODataset()
    dataset.misc["input_space"] = space
    model = PCERegressor(dataset, settings=PCERegressor_Settings(discipline=IshigamiDiscipline(), use_quadrature=True, degree=4))
    model.learn()
    print(model.predict(np.array([0.1, 0.2, 0.3])))
    ```

### The RBF kernel can no longer be a callable

The `function` setting of `RBFRegressor_Settings` was renamed to `kernel` and no longer accepts a Python callable. Its value is a member of the `RBF` enumeration or its name: `linear`, `thin_plate_spline` (formerly `thin_plate`), `cubic`, `quintic`, `multiquadric`, `inverse_multiquadric`, `gaussian` or the new `inverse_quadratic`.

bump-gemseo marks the line with the comment `kernel only accepts a member of the RBF enumeration or its name, e.g. RBF.CUBIC or "cubic"; a Python callable is no longer accepted`; left as is, the settings raise a `ValidationError` listing the accepted names. Replace your function by the name of the equivalent kernel, e.g. `"cubic"` for $r^3$, as in the example below. A kernel that is not in this list cannot be used with `RBFRegressor` in GEMSEO 7.

!!! info "After bump-gemseo"

    - **Comment**: `kernel only accepts a member of the RBF enumeration or its name, e.g. RBF.CUBIC or "cubic"; a Python callable is no longer accepted`
    - **Error with GEMSEO 7**: `ValidationError: 1 validation error for RBFRegressor_Settings kernel Input should be 'cubic', 'gaussian', 'inverse_multiquadric', 'inverse_quadratic', 'linear', 'multiquadric', 'quintic' or 'thin_plate_spline' [type=enum`
    - **See** issue [#1861](https://gitlab.com/gemseo/dev/gemseo/-/issues/1861), merge request [!2685](https://gitlab.com/gemseo/dev/gemseo/-/merge_requests/2685).

=== "GEMSEO 6.3"

    ```python
    import numpy as np
    from gemseo.datasets.io_dataset import IODataset
    from gemseo.mlearning.regression.algos.rbf import RBFRegressor
    x = np.linspace(0.0, 1.0, 6)[:, None]
    dataset = IODataset()
    dataset.add_input_variable("x", x)
    dataset.add_output_variable("y", np.sin(3 * x))


    def cubic(self, r):
        return r**3


    model = RBFRegressor(dataset, function=cubic)
    model.learn()
    print([model.predict(x[1]), model.predict(x[4])])
    ```

=== "GEMSEO 7"

    ```python
    import numpy as np
    from gemseo.dataset import IODataset
    from gemseo.machine_learning.regression.model.rbf import RBFRegressor
    from gemseo.machine_learning.regression.model.rbf_settings import RBFRegressor_Settings
    x = np.linspace(0.0, 1.0, 6)[:, None]
    dataset = IODataset()
    dataset.add_input_variable("x", x)
    dataset.add_output_variable("y", np.sin(3 * x))
    model = RBFRegressor(dataset, settings=RBFRegressor_Settings(kernel="cubic"))
    model.learn()
    print([model.predict(x[1]), model.predict(x[4])])
    ```

### `y_average` and `der_function` of the RBF regressors were removed

The attributes `y_average` and `der_function` of `RBFRegressor` and `TPSRegressor` were removed when these regressors moved to SciPy's `RBFInterpolator`; the derivatives of the kernels are now an implementation detail. To know the kernel, use the `RBFRegressor.kernel` property.

bump-gemseo marks each access with a comment such as `RBFRegressor.y_average was removed; see the GEMSEO 7 changelog.`; left as is, the line raises `AttributeError: 'RBFRegressor' object has no attribute 'y_average'`. GEMSEO 7 has no replacement for these values, so remove the code that reads them.

!!! info "After bump-gemseo"

    - **Comment**: `RBFRegressor.y_average was removed; see the GEMSEO 7 changelog.` / `TPSRegressor.y_average was removed; see the GEMSEO 7 changelog.` / `RBFRegressor.der_function was removed; see the GEMSEO 7 changelog.`
    - **Error with GEMSEO 7**: `AttributeError: 'RBFRegressor' object has no attribute 'y_average'`
    - **See** issue [#1861](https://gitlab.com/gemseo/dev/gemseo/-/issues/1861), merge request [!2685](https://gitlab.com/gemseo/dev/gemseo/-/merge_requests/2685).

=== "GEMSEO 6.3"

    ```python
    import numpy as np
    from gemseo.datasets.io_dataset import IODataset
    from gemseo.mlearning.regression.algos.rbf import RBFRegressor
    from gemseo.mlearning.regression.algos.thin_plate_spline import TPSRegressor
    x = np.linspace(0.0, 1.0, 6)[:, None]
    dataset = IODataset()
    dataset.add_input_variable("x", x)
    dataset.add_output_variable("y", np.sin(3 * x))
    rbf = RBFRegressor(dataset)
    rbf.learn()
    tps = TPSRegressor(dataset)
    tps.learn()
    print([rbf.y_average, tps.y_average, rbf.der_function is not None, tps.der_function is not None])
    ```

=== "GEMSEO 7"

    ```python
    print("<removed in GEMSEO 7>")
    ```

### Some settings of `RBFRegressor` and `TPSRegressor` were removed

`RBFRegressor` and `TPSRegressor` are now based on SciPy's `RBFInterpolator` instead of the deprecated `Rbf` class. The `der_function` and `norm` settings of `RBFRegressor_Settings` and `TPSRegressor_Settings` were removed, as the kernels are no longer callables and the distance is always Euclidean. `TPSRegressor_Settings` no longer exposes `function` and `epsilon` either: the kernel is fixed to `thin_plate_spline` and the shape parameter to `1`.

bump-gemseo marks each use of these settings with a comment such as `RBFRegressor_Settings.norm was removed; see the GEMSEO 7 changelog.`; reading one of them raises an `AttributeError`. Delete these arguments and the code that reads them, and keep the other settings, e.g. `smoothing` (formerly `smooth`), as in the example below.

!!! info "After bump-gemseo"

    - **Comment**: `RBFRegressor_Settings.norm was removed; see the GEMSEO 7 changelog.` / `TPSRegressor_Settings.norm was removed; see the GEMSEO 7 changelog.` / `TPSRegressor_Settings.der_function was removed; see the GEMSEO 7 changelog.`
    - **Error with GEMSEO 7**: `AttributeError: 'RBFRegressor_Settings' object has no attribute 'norm'`
    - **GEMSEO 7 cannot give the GEMSEO 6.3 value here**: the distance, the kernel derivatives and the TPS kernel and shape are no longer settings (changelog 1861).
    - **See** issue [#1861](https://gitlab.com/gemseo/dev/gemseo/-/issues/1861), merge request [!2685](https://gitlab.com/gemseo/dev/gemseo/-/merge_requests/2685).

=== "GEMSEO 6.3"

    ```python
    from gemseo.mlearning.regression.algos.rbf_settings import RBFRegressor_Settings
    from gemseo.mlearning.regression.algos.thin_plate_spline_settings import TPSRegressor_Settings
    settings = RBFRegressor_Settings(der_function=None, norm="euclidean", smooth=0.1)
    tps_settings = TPSRegressor_Settings(function="thin_plate", epsilon=None, der_function=None, norm="euclidean", smooth=0.2)
    print([settings.smooth, tps_settings.smooth, settings.norm, tps_settings.norm, tps_settings.der_function is None, settings.der_function is None, str(tps_settings.function), tps_settings.epsilon is None])
    ```

=== "GEMSEO 7"

    ```python
    from gemseo.machine_learning.regression.model.rbf_settings import RBFRegressor_Settings
    from gemseo.machine_learning.regression.model.tps_settings import TPSRegressor_Settings
    settings = RBFRegressor_Settings(smoothing=0.1)
    tps_settings = TPSRegressor_Settings(smoothing=0.2)
    print([settings.smoothing, tps_settings.smoothing, "<removed in GEMSEO 7>"])
    ```

## Removed names

bump-gemseo marks each use with a `# TODO(bump-gemseo)` comment. Unless an entry above gives an equivalent, they have no replacement: remove the code that uses them.

- `RBFRegressor.EUCLIDEAN`
- `RBFRegressor.RBFDerivatives`
- `TPSRegressor.EUCLIDEAN`
- `TPSRegressor.RBFDerivatives`
- `gemseo.mlearning.regression.algos.ot_gpr_settings.DOEAlgorithmName`
- `gemseo.mlearning.regression.algos.rbf.SavedObjectType`
- `gemseo.post.mlearning.ml_regressor_quality_viewer.DatasetPlotOption`
