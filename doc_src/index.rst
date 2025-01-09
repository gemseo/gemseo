..
    Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

    This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
    International License. To view a copy of this license, visit
    http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
    Commons, PO Box 1866, Mountain View, CA 94042, USA.

:html_theme.sidebar_secondary.remove: true

####################
GEMSEO documentation
####################

.. raw:: html

   <p>
       <a href="https://www.gnu.org/licenses/lgpl-3.0.en.html"><img alt="PyPI - License" src="https://img.shields.io/pypi/l/gemseo" /></a>
       <a href="https://pypi.org/project/gemseo/"><img alt="PyPI - Python Version" src="https://img.shields.io/pypi/pyversions/gemseo" /></a>
       <a href="https://pypi.org/project/gemseo/"><img alt="PyPI" src="https://img.shields.io/pypi/v/gemseo" /></a>
       <a href="https://app.codecov.io/gl/gemseo:dev/gemseo"><img alt="Codecov branch" src="https://img.shields.io/codecov/c/gitlab/gemseo:dev/gemseo/develop" /></a>
       <a href="https://arc.aiaa.org/doi/10.2514/6.2018-0657"><img alt="DOI" src="https://img.shields.io/badge/DOI-10.2514%2F6.2018--0657-blue" /></a>
   </p>

.. image:: _static/gemseo_schema.png
   :align: left

GEMSEO is an open-source Python software to automate multidisciplinary processes,
starting with multidisciplinary design optimization (MDO) ones.

Standing for Generic Engine for Multidisciplinary Scenarios, Exploration and Optimization,
GEMSEO offers a catalog of MDO formulations to make this automation possible.
Built on top of essentials such as NumPy, SciPy and Matplotlib,
it also includes a wide range of algorithms for various fields, namely
coupling,
design of experiments,
linear problems,
optimization,
machine learning,
ordinary differential equations,
surrogate modeling,
uncertainty quantification,
visualization,
etc.

GEMSEO can be both easily embedded in simulation platforms and used as a standalone software.
The disciplines can wrap Python code, Matlab or Scilab, scripts, Excel spreadsheets
and a whole set of executables that can be called from Python.

Its GNU LGPL v3.0 open-source license makes it commercially usable (`see licences <software/licenses.html>`_).

.. grid:: auto
    :gutter: 1 1 1 1

    .. grid-item::

        .. button-link:: software/installation.html
            :color: secondary
            :shadow:
            :align: center

            :octicon:`download` Installation

    .. grid-item::

        .. button-link:: contributing.html
            :color: secondary
            :shadow:
            :align: center

            :octicon:`pencil` Contributing

    .. grid-item::

        .. button-link:: aboutus.html
            :color: secondary
            :shadow:
            :align: center

            :octicon:`people` About us

    .. grid-item::

        .. button-link:: software/changelog.html
            :color: secondary
            :shadow:
            :align: center

            :octicon:`history` Changelog

    .. grid-item::

        .. button-link:: examples/mdo/plot_gemseo_in_10_minutes.html
            :color: secondary
            :shadow:
            :align: center

            :octicon:`stopwatch` GEMSEO in 10 minutes

    .. grid-item::

        .. button-link:: plugins.html
            :color: secondary
            :shadow:
            :align: center

            :octicon:`plug` Plugins


.. raw:: html

   <h1 class="h1-center">Main concepts<a class="headerlink" href="#main-concepts" title="Link to this heading">#</a></h1>

.. grid:: 1 1 2 2
    :gutter: 2 3 4 4

    .. grid-item-card::
        :text-align: center

        **Discipline**
        ^^^

        Define an input-output discipline to interface a model.

        Features:
        `analytic expressions <disciplines/analytic_discipline.html>`_,
        `executable <disciplines/discipline_interfacing_executable.html>`_,
        `surrogate model <surrogate.html>`_ and
        `much more <interface/software_connection.html>`_.

        +++

        :bdg-link-secondary-line:`Read more <discipline.html>`
        :bdg-link-secondary-line:`Examples <examples/disciplines/index.html>`
        :bdg-link-secondary-line:`Types <algorithms/discipline_algos.html>`

    .. grid-item-card::
        :text-align: center

        **Design space**
        ^^^

        Define a set of parameters, typically design parameters.

        Features:
        `deterministic parameter space <modules/gemseo.algos.design_space.html>`_ and
        `uncertain (or mixed) parameter space <modules/gemseo.algos.parameter_space.html>`_.
        +++

        :bdg-link-secondary-line:`Read more <design_space/design_space.html>`
        :bdg-link-secondary-line:`Examples <examples/design_space/index.html>`

    .. grid-item-card::
        :text-align: center

        **Scenario**
        ^^^

        Define an evaluation process over a design space for a set of disciplines and a given objective.

        Features:
        `DOE scenario <modules/gemseo.scenarios.doe_scenario.html>`_ and
        `MDO scenario <modules/gemseo.scenarios.mdo_scenario.html>`_.
        +++

        :bdg-link-secondary-line:`Read more <scenario.html>`
        :bdg-link-secondary-line:`Examples <examples/scenario/index.html>`

    .. grid-item-card::
        :text-align: center

        **Data persistence**
        ^^^

        Store disciplinary evaluations in a `cache <data_persistence/cache.html>`_
        either in memory or saved in a file.
        Use a `dataset <data_persistence/dataset.html>`_ to store many kinds of data
        and make them easy to handle for visualization, display and query purposes.
        +++

        :bdg-link-secondary-line:`Read more <data_persistence/index.html>`
        :bdg-link-secondary-line:`Examples <examples/cache/index.html>`

.. raw:: html

   <h1 class="h1-center">Features<a class="headerlink" href="#features" title="Link to this heading">#</a></h1>

.. grid:: 1 1 2 2
    :gutter: 2 3 4 4

    .. grid-item-card::
        :text-align: center

        **Study analysis**
        ^^^

        An intuitive tool to discover MDO without writing any code,
        and define the right MDO problem and process.
        From an Excel workbook,
        specify your disciplines, design space, objective and constraints,
        select an MDO formulation and plot both coupling structure
        (`N2 chart <mdo/coupling.html#n2-chart-visualization>`_)
        and MDO process
        (`XDSM <mdo/mdo_formulations.html#xdsm-visualization>`_),
        even before wrapping any software.
        +++

        :bdg-link-secondary-line:`Read more <interface/study_analysis.html>`
        :bdg-link-secondary-line:`Examples <examples/study_analysis/index.html>`

    .. grid-item-card::
        :text-align: center

        **Optimization**
        ^^^

        Define, solve and post-process an optimization problem from an optimization algorithm.

        Based on
        `GCMMA-MMA <https://github.com/arjendeetman/GCMMA-MMA-Python>`_,
        `NLopt <https://nlopt.readthedocs.io/en/latest/>`_,
        `PDFO <https://www.pdfo.net/>`_,
        `pSeven <https://www.pseven.io/product/pseven/>`_,
        `pymoo <https://pymoo.org/>`_ and
        `SciPy <https://scipy.org/>`_.
        +++

        :bdg-link-secondary-line:`Read more <optimization.html>`
        :bdg-link-secondary-line:`Examples <examples/optimization_problem/index.html>`
        :bdg-link-secondary-line:`Algorithms <algorithms/opt_algos.html>`

    .. grid-item-card::
        :text-align: center

        **DOE & trade-off**
        ^^^

        Define, solve and post-process a trade-off problem from a DOE (design of experiments) algorithm.

        Based on
        `OpenTURNS <https://openturns.github.io/www/>`_ and
        `pyDOE <https://pythonhosted.org/pyDOE/>`_.
        +++

        :bdg-link-secondary-line:`Read more <doe.html>`
        :bdg-link-secondary-line:`Examples <examples/doe/index.html>`
        :bdg-link-secondary-line:`Algorithms <algorithms/doe_algos.html>`

    .. grid-item-card::
        :text-align: center

        **MDO formulations**
        ^^^

        Define the way as the disciplinary coupling is formulated and managed by the optimization or DOE algorithm.
        +++

        :bdg-link-secondary-line:`Read more <mdo/mdo_formulations.html>`
        :bdg-link-secondary-line:`Examples <examples/formulations/index.html>`
        :bdg-link-secondary-line:`Algorithms <algorithms/formulation_algos.html>`

    .. grid-item-card::
        :text-align: center

        **MDA**
        ^^^

        Find the coupled state of a multidisciplinary system using a multi-disciplinary analysis.
        +++

        :bdg-link-secondary-line:`Read more <mdo/mda.html>`
        :bdg-link-secondary-line:`Examples <examples/mda/index.html>`
        :bdg-link-secondary-line:`Algorithms <algorithms/mda_algos.html>`

    .. grid-item-card::
        :text-align: center

        **Linear solvers**
        ^^^

        Define and solve a linear problem, typically in the context of an MDA.

        Based on
        `PETSc <https://petsc.org/release/>`_ and
        `SciPy <https://scipy.org/>`_.
        +++

        :bdg-link-secondary-line:`Algorithms <algorithms/linear_solver_algos.html>`

    .. grid-item-card::
        :text-align: center

        **Visualization**
        ^^^

        Generate graphical representations of optimization histories.
        +++

        :bdg-link-secondary-line:`Read more <postprocessing/index.html>`
        :bdg-link-secondary-line:`Examples <examples/post_process/index.html>`
        :bdg-link-secondary-line:`Algorithms <algorithms/post_algos.html>`

    .. grid-item-card::
        :text-align: center

        **Surrogate models**
        ^^^

        Replace a discipline by a surrogate one relying on a machine learning regression model.

        Based on
        `OpenTURNS <https://openturns.github.io/www/>`_ and
        `scikit-learn <https://scikit-learn.org/stable/>`_.
        +++

        :bdg-link-secondary-line:`Read more <surrogate.html>`
        :bdg-link-secondary-line:`Examples <examples/surrogate/index.html>`
        :bdg-link-secondary-line:`Algorithms <algorithms/surrogate_algos.html>`

    .. grid-item-card::
        :text-align: center

        **Scalable models**
        ^^^

        Use scalable data-driven models to compare MDO formulations and algorithms for different problem dimensions.

        Features:
        `scalability <modules/gemseo.problems.mdo.scalable.data_driven.study.process.html>`_,
        `scalable problem <scalable_models/index.html#scalable-mdo-problem>`_,
        `scalable discipline <scalable_models/index.html#scalable-discipline>`_ and
        `diagonal-based <scalable_models/index.html#scalable-diagonal-model>`_.
        +++

        :bdg-link-secondary-line:`Read more <scalable.html>`
        :bdg-link-secondary-line:`Examples <examples/scalable/index.html>`

    .. grid-item-card::
        :text-align: center

        **Machine learning**
        ^^^

        Apply clustering, classification and regression methods from the machine learning community.

        Features:
        `clustering <machine_learning/clustering/clustering_models.html>`_,
        `classification <machine_learning/classification/classification_models.html>`_,
        `regression <machine_learning/regression/regression_models.html>`_,
        `quality measures <machine_learning/quality_measures/quality_measures.html>`_ and
        `data transformation <machine_learning/transform/transformer.html>`_.
        +++

        :bdg-link-secondary-line:`Read more <machine_learning/index.html>`
        :bdg-link-secondary-line:`Examples <examples/mlearning/index.html>`
        :bdg-link-secondary-line:`Algorithms <algorithms/mlearning_algos.html>`

    .. grid-item-card::
        :text-align: center

        **Uncertainty**
        ^^^

        Define, propagate, analyze and manage uncertainties.

        Features:
        `distribution <uncertainty/distribution.html>`_,
        `uncertain space <uncertainty/parameter_space.html>`_,
        `empirical and parametric statistics <uncertainty/statistics.html>`_,
        `distribution fitting <modules/gemseo.uncertainty.distributions.openturns.fitting.html>`_ and
        `sensitivity analysis <uncertainty/sensitivity.html>`_.

        Based on
        `OpenTURNS <https://openturns.github.io/www/>`_.
        +++

        :bdg-link-secondary-line:`Read more <uncertainty/index.html>`
        :bdg-link-secondary-line:`Examples <examples/uncertainty/index.html>`
        :bdg-link-secondary-line:`Algorithms <algorithms/uncertainty_algos.html>`

    .. grid-item-card::
        :text-align: center

        **Ordinary differential equation**
        ^^^

        Define and solve an ordinary differential equation.

        Based on
        `SciPy <https://scipy.org/>`_.
        +++

        :bdg-link-secondary-line:`Read more <ode/index.html>`
        :bdg-link-secondary-line:`Examples <examples/ode/index.html>`
        :bdg-link-secondary-line:`Algorithms <algorithms/ode_algos.html>`

.. toctree::
   :hidden:
   :maxdepth: 2

   User guide <user_guide>
   Examples <examples_and_tutorials>
   API documentation <modules/gemseo>
   Cheat sheets <cheat_sheets/index>
   About us <aboutus>
   Bibliography <zreferences>
   Contributing <contributing>
   Credits <credits>
   FAQ <faq>
   General index <genindex>
   Glossary <glossary>
   Licenses <software/licenses>
   Overview <overview>
   Plugins <plugins>
   Roadmap <roadmap>
   Software <software/index>
   Upgrading <software/upgrading>
