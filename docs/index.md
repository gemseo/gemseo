<!--
Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
International License. To view a copy of this license, visit
http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
Commons, PO Box 1866, Mountain View, CA 94042, USA.
-->
<h1 style="display: none;">Home</h1>

<div style="text-align: center;"><img src="_static/logo-small.png" style="width: 50%;" alt="GEMSEO logo"/></div>

<p style="text-align: center;">
<a href="https://app.codecov.io/gl/gemseo:dev/gemseo/branch/develop"><img alt="Codecov branch" src="https://img.shields.io/codecov/c/gitlab/gemseo:dev/gemseo/develop" /></a>
<img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/gemseo" />
<a href="https://pypi.org/project/gemseo/"><img alt="PyPI" src="https://img.shields.io/pypi/v/gemseo" /></a>
<a href="https://arc.aiaa.org/doi/10.2514/6.2018-0657"><img alt="Paper" src="https://img.shields.io/badge/DOI-10.2514%2F6.2018--0657-blue" /></a>
<a href="https://www.gnu.org/licenses/lgpl-3.0.en.html"><img alt="PyPI - License" src="https://img.shields.io/pypi/l/gemseo" /></a>
</p>

GEMSEO is an open-source Python software to automate multidisciplinary processes, starting with multidisciplinary design optimization (MDO) ones.

Standing for Generic Engine for Multidisciplinary Scenarios, Exploration and Optimization, GEMSEO offers a catalog of MDO formulations to make this automation possible. Built on top of essentials such as NumPy, SciPy and Matplotlib, it also includes a wide range of algorithms for various fields, namely coupling, design of experiments, linear problems, optimization, machine learning, ordinary differential equations, surrogate modeling, uncertainty quantification, visualization, etc.

GEMSEO can be both easily embedded in simulation platforms and used as a standalone software. The disciplines can wrap Python code, Matlab or Scilab, scripts, Excel spreadsheets and a whole set of executables that can be called from Python.

Its GNU LGPL v3.0 open-source license makes it commercially usable ([see licences][licenses]).

<div style="text-align: center;"><img src="_static/gemseo_schema.png" alt="GEMSEO illustration"/></div>

## Main concepts

<div class="grid cards" markdown>

- ### __Discipline__

    [:octicons-book-24: User guide][the-discipline-a-key-concept]
    [:octicons-play-24: Examples][disciplines-examples]
    [:octicons-gear-24: Types][available-disciplines]

    ---

    Define an input-output discipline to interface a model.

    Features:
    [analytic expressions][how-to-build-an-analytic-discipline],
    [executable][how-to-manually-create-a-discipline-interfacing-an-external-executable],
    [surrogate model][surrogate-models] and
    [much more][interfacing-simulation-software].

- ### __Design space__

    [:octicons-book-24: User guide][how-to-deal-with-design-spaces]
    [:octicons-play-24: Examples][design-space-examples]

    ---

    Define a set of parameters, typically design parameters.

    Features:
    [deterministic parameter space][gemseo.algos.design_space] and
    [uncertain (or mixed) parameter space][gemseo.algos.parameter_space].

- ### __Scenario__

    [:octicons-book-24: User guide][how-to-deal-with-scenarios]
    [:octicons-play-24: Examples][scenario-examples]

    ---

    Define an evaluation process over a design space for a set of disciplines and a given objective.

    Features:
    [DOE scenario][gemseo.scenarios.doe_scenario] and
    [MDO scenario][gemseo.scenarios.mdo_scenario].

- ### __Data persistence__

    [:octicons-book-24: User guide][data-persistence-overview]
    [:octicons-play-24: Examples][cache-examples]

    ---

    Store disciplinary evaluations in a [cache][caching-and-recording-discipline-data]
    either in memory or saved in a file.
    Use a [dataset][gemseo.datasets.dataset] to store many kinds of data
    and make them easy to handle for visualization, display and query purposes.

</div>

## Features

<div class="grid cards" markdown>

- ### __Study analysis__

    [:octicons-book-24: User guide](interface/study_analysis.md)
    [:octicons-play-24: Examples][study-analysis-examples]

    ---

    An intuitive tool to discover MDO without writing any code,
    and define the right MDO problem and process.
    From an Excel workbook,
    specify your disciplines, design space, objective and constraints,
    select an MDO formulation and plot both coupling structure
    ([N2 chart][n2-chart-visualization]
    and MDO process
    ([XDSM][xdsm-visualization]),
    even before wrapping any software.

- ### __Optimization__

    [:octicons-book-24: User guide][optimization-and-doe-framework]
    [:octicons-play-24: Examples][optimization-examples]
    [:octicons-gear-24: Algorithms][available-optimization-algorithms]

    ---

    Define, solve and post-process an optimization problem from an optimization algorithm.

    Based on
    [GCMMA-MMA](https://github.com/arjendeetman/GCMMA-MMA-Python>),
    [NLopt](https://nlopt.readthedocs.io/en/latest/),
    [PDFO](https://www.pdfo.net/),
    [pSeven](https://www.pseven.io/product/pseven/),
    [pymoo](https://pymoo.org/) and
    [SciPy](https://scipy.org/).

- ### __DOE & trade-off__

    [:octicons-book-24: User guide](doe.md)
    [:octicons-play-24: Examples][design-of-experiments-doe]
    [:octicons-gear-24: Algorithms][available-doe-algorithms]

    ---

    Define, solve and post-process a trade-off problem from a DOE (design of experiments) algorithm.

    Based on
    [OpenTURNS](https://openturns.github.io/www/) and
    [pyDOE](https://pythonhosted.org/pyDOE/).

- ### __MDO formulations__ { #overview-mdo-formulations }

    [:octicons-book-24: User guide][mdo-formulations]
    [:octicons-play-24: Examples][mdo-formulation]
    [:octicons-gear-24: Algorithms][available-mdo-formulations]

    ---

    Define the way as the disciplinary coupling is formulated and managed by the optimization or DOE algorithm.

- ### __MDA__

    [:octicons-book-24: User guide][multi-disciplinary-analyses]
    [:octicons-play-24: Examples][multidisciplinary-analysis-mda]
    [:octicons-gear-24: Algorithms][available-mda-algorithms]

    ---

    Find the coupled state of a multidisciplinary system using a multi-disciplinary analysis.

- ### __Linear solvers__

    [:octicons-gear-24: Algorithms][available-linear-solvers]

    ---

    Define and solve a linear problem, typically in the context of an MDA.

    Based on
    [PETSc](https://petsc.org/release/) and
    [SciPy](https://scipy.org/).

- ### __Visualization__

    [:octicons-book-24: User guide][how-to-deal-with-post-processing]
    [:octicons-play-24: Examples][post-process-an-optimization-problem]
    [:octicons-gear-24: Algorithms][available-post-processing-algorithms]

    ---

    Generate graphical representations of optimization histories.

- ### __Surrogate models__ { #overview-surrogate-models }

    [:octicons-play-24: Examples][surrogate-discipline-examples]
    [:octicons-gear-24: Algorithms][surrogate-models-introduction]

    ---

    Replace a discipline by a surrogate one relying on a machine learning regression model.

    Based on
    [OpenTURNS](https://openturns.github.io/www/) and
    [scikit-learn](https://scikit-learn.org/stable/).

- ### __Scalable model__ { #overview-scalable-models }

    Features:
    [scalability][gemseo.problems.mdo.scalable.data_driven.study.process],
    [scalable problem][the-scalable-problem],
    [scalable discipline][the-scalable-problem] and
    [diagonal-based][the-scalable-problem].

    ---

    Use scalable data-driven models to compare MDO formulations and algorithms for different problem dimensions.

    [:octicons-book-24: User guide][scalable-models]
    [:octicons-play-24: Examples][scalable-model]

- ### __Machine learning__

    [:octicons-book-24: User guide][introduction-to-machine-learning]
    [:octicons-play-24: Examples][machine-learning-examples]
    [:octicons-gear-24: Algorithms][algorithms-of-machine-learning]

    ---

    Apply clustering, classification and regression methods from the machine learning community.

    Features:
    [clustering][gemseo.mlearning.clustering.algos],
    [classification][gemseo.mlearning.classification.algos],
    [regression][gemseo.mlearning.regression.algos],
    [quality measures][gemseo.mlearning.core.quality.base_ml_algo_quality] and
    [data transformation][gemseo.mlearning.transformers.pipeline].

- ### __Uncertainty__

    [:octicons-book-24: User guide][introduction-to-uncertainty-quantification-and-management]
    [:octicons-play-24: Examples][uncertainty-examples]
    [:octicons-gear-24: Algorithms][uncertainty-algorithms]

    ---

    Define, propagate, analyze and manage uncertainties.

    Features:
    [distribution][gemseo.uncertainty.distributions],
    [uncertain space][gemseo.algos.parameter_space],
    [empirical and parametric statistics][gemseo.uncertainty.statistics.base_statistics],
    [distribution fitting][gemseo.uncertainty.distributions.base_distribution_fitter] and
    [sensitivity analysis][gemseo.uncertainty.sensitivity].

    Based on
    [OpenTURNS](https://openturns.github.io/www/).

- ### __Ordinary differential equation__

    [:octicons-book-24: User guide][introduction-to-ordinary-differential-equations]
    [:octicons-play-24: Examples][ordinary-differential-equations]
    [:octicons-gear-24: Algorithms][available-ordinary-differential-equations-solvers]

    ---

    Define and solve an ordinary differential equation.

    Based on
    [SciPy](https://scipy.org/) and [PETSc](https://petsc.org/).
