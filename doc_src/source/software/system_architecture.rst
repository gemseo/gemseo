..
   Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

   This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
   International License. To view a copy of this license, visit
   http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
   Commons, PO Box 1866, Mountain View, CA 94042, USA.

..
   Contributors:
          :author:  Francois Gallard, Vincent Gachelin

.. _system_architecure:

Architecture principles
=======================

This page describes the key aspects of |g| architecture.
It describes the main concepts used to build :term:`MDO` :term:`processes<process>`.
Two scientific papers give details on the subject :cite:`gallard2018gemseo`, :cite:`gemseo2019`.

General description
-------------------

|g| is responsible for managing the :term:`MDO` :term:`scenario`,
including the :term:`optimization problem`, that includes the :term:`objective function`,
:term:`constraints`, :term:`coupling variables`, and :term:`design variables`.

|g| is independent of all disciplinary tools and thus can be used with
any business case. Within the :term:`process`, it executes the :term:`disciplines<discipline>` accordingly to the
needs of the :term:`optimization algorithm` or :term:`DOE`, called :term:`driver`.
It is in charge of the of the :term:`optimization problem` and links the mathematical methods
to the :term:`simulation software`.

It can be interfaced with multiple :term:`workflow engines<workflow engine>`, typically in charge of chaining
elementary across multiple machines. |g| triggers
the execution of :term:`disciplines<discipline>` or :term:`chains<chain>` of disciplines when requested by the :term:`driver`.


The interfaces of |g| to an MDO platform and simulation capabilities
--------------------------------------------------------------------------------

|g| must provide generic interfaces, so that it can be integrated within
any :term:`MDO platform` technology.

-  Interfaces with external :term:`workflow engines <workflow engine>`, see :ref:`software_connection`.

-  Interfaces with external :term:`optimization algorithms <optimization algorithm>`, and :term:`DOE` methods, :term:`MDA` solvers, :term:`surrogate model`.

-  Interfaces with the :term:`MDO platform`.


.. figure:: /_images/architecture/components_platform.png
   :scale: 100 %

   The Formulations Engine within its environment


|g| components interactions
---------------------------------------

|g| is split into 3 main component layers:

- The :term:`MDO formulation`, that creates :term:`generic processes<generic process>` and the :term:`optimization problem` from a list of disciplines
- The :term:`generic process`, such as :term:`MDAs<MDA>`, :term:`chains of disciplines<chain>`, MDO :term:`scenarios<scenario>`.
  These classes are actually abstractions of processes.
- The optimization framework contains the :term:`optimization problem` description with
  their :term:`objective function` and :term:`constraints`, and :term:`drivers<driver>` (optimization and :term:`DOE` algorithms) to solve them.

|g| architecture is :term:`modular<modular architecture>`. Many of its packages and classes are independent:

- The Optimization framework, which connects solvers and helps to formulate :term:`optimization problems<optimization problem>`,
  can be used to solve user-defined optimization problem, without any process or :term:`MDO formulation`. :ref:`Here are some examples <sphx_glr_examples_optimization_problem>`.
- Many :term:`generic processes<generic process>`, such as the :term:`MDAs<MDA>` and :term:`chains<chain>`, can be used to solve coupling problems without doing an MDO.
  Again, some examples are provided in the package.
- The optimization results plots can be generated from optimization histories stored in HDF5 files, or by using the Python API to read external data.
  So you may draw the same plots from optimization data generated outside of
  |g|.

The next figure shows the main components of |g| within an :term:`MDO platform`.

.. figure:: /_images/architecture/components_all.png
   :scale: 100 %

   |g| components

At process building step
~~~~~~~~~~~~~~~~~~~~~~~~

When the scenario is instantiated by the user, typically in a script or by a platform, the process is built.
Then the user can configure the overall :term:`process`, before execution, by accessing and changing the objects attributes that define the process.

#. At this stage, the scenario instantiates the :term:`MDO formulation` according to the user's choice.
#. Then, the formulation possibly creates :term:`generic processes<generic process>` such as :term:`MDAs<MDA>` (for :term:`MDF`),
   or MDO subscenarios, for :term:`bi-level` formulations.
#. Once this is performed, the :term:`MDO formulation` creates an :term:`optimization problem`.
#. This :term:`optimization problem` is defined by the :term:`objective function`
   and :term:`constraints`, that eventually points to the :term:`generic processes<generic process>` or directly to the disciplines (for :term:`IDF` for instance).

The next figure shows the components interaction at this step.

.. figure:: /_images/architecture/components_build_process.png
   :scale: 100 %

   Components interaction at the build step of the process


During process execution
~~~~~~~~~~~~~~~~~~~~~~~~

During the process execution :

#. A :term:`driver` is instantiated, it can be either an :term:`optimization algorithm` or a :term:`DOE` algorithm.
#. The driver solves the optimization problem that was created by the MDO formulation at the building step.
#. To this aim, the driver calls the objective function and constraints.
#. These functions point to the generic processes (that aim at solving a coupling problem for a MDA)
   or MDO subscenarios (for :term:`bi-level` scenarios), in order to find an optimum.
#. These calls trigger the :term:`generic process` execution, which themselves execute the :term:`disciplines<discipline>`.

.. figure:: /_images/architecture/components_execute_process.png
   :scale: 100 %

   Components interactions at execution of the process

The :term:`sequence diagram` shows the data exchanges during the execution. Here the generic process may be a :ref:`mda` in the case of :term:`MDF`,
which calls the disciplines. We represent only the :term:`objective function` calls, since the :term:`constraints` are handled in a similar way.
The calls to the objective and its :term:`gradient` are made within a loop, until convergence of the :term:`optimization algorithm` (the :term:`driver`).
The scenario then retrieves the :term:`optimum` from the driver.

.. uml::

    @startuml
    Scenario -> Driver: solve
    Driver -> "Optimization problem": get_objective_and_constraints()
    Driver <- "Optimization problem": objective, constraints

    loop Optimization convergence
        Driver -> "Objective function": call(x)
        "Objective function" -> "Generic process": execute(data)
        loop MDA convergence
            "Generic process" -> "Discipline1": execute(data)
            "Generic process" <- "Discipline1": outputs1
            "Generic process" -> "Discipline2": execute(data)
            "Generic process" <- "Discipline2": outputs2
        end
        "Objective function" <- "Generic process": objective_value
        Driver <- "Objective function": objective_value
        Driver -> "Objective function": gradient(x)
        "Objective function" -> "Generic process": linearize(data)
        "Generic process" -> "Discipline1": linearize(data)
        "Generic process" <- "Discipline1": jacobian1
        "Generic process" -> "Discipline2": linearize(data)
        "Generic process" <- "Discipline2": jacobian2
        "Generic process" <- "Generic process": coupled_derivatives()
        "Objective function" <- "Generic process": coupled_derivatives
        Driver <- "Objective function": objective_gradient
    end

    Scenario <- Driver: optimum

    @enduml


During results analysis
~~~~~~~~~~~~~~~~~~~~~~~

After the execution, convergence plots of the optimization can be generated.
In addition, design space analysis, sensitivity analysis and  constraints plots can be generated.
For a complete overview of |g| post-processing capabilities, see :ref:`post_processing`.
This can be achieved either from disk data (a serialized optimization problem), or from in-memory data after execution.

The user triggers the plots generation from a post-processing factory, or via the scenario, which delegates the execution to the same factory.
The main steps are:

#. The post-processing factory loads an optimization problem from the disk, or in memory from a Scenario.
#. The data is split into a design space data, which contains the design variables names, bounds and types,
#. and a database of all data generated during execution, typically the objective function, constraints, their derivatives,
   and eventually algorithmic data.
#. Once the data is available, the factory loads the appropriate plot generation class for the plots required by the user, and calls the plot generation method.

.. figure:: /_images/architecture/results_analysis.png
   :scale: 100 %

   Components interactions during results analysis

Main classes
------------


The high level classes that are key in the architecture are:

-  :class:`~gemseo.core.mdo_scenario.MDOScenario` builds the process from a set of inputs, several
   disciplines and a formulation. It is one of the main interface class
   for the :term:`MDO user`. The :class:`~gemseo.core.mdo_scenario.MDOScenario` triggers the overall optimization
   process when its ``execute()`` method is called. Through this class, the
   user provides an initial solution, user constraints, and bounds on
   the :term:`design variables`. The user may also generate visualization of the scenario
   execution, such as convergence plots of the algorithm (see :ref:`sellar_mdo`).

-  :class:`~gemseo.core.doe_scenario.DOEScenario` builds the process from a set of inputs, several
   disciplines and a formulation. It is the second main interface class
   for the :term:`MDO user`. The :class:`~gemseo.core.doe_scenario.DOEScenario` triggers the overall trade-off process
   when its ``execute()`` method is called. Through this class, the user
   provides a design space, some outputs to monitor (objective and
   constraints) and a number of samples. The user may also generate
   visualization of the scenario execution, such as convergence plots of
   the algorithm . As the :class:`~gemseo.core.mdo_scenario.MDOScenario`, the
   :class:`~gemseo.core.doe_scenario.DOEScenario` makes the link between all the following classes. It
   is mainly handled by the :term:`MDO integrator`. Both :class:`~gemseo.core.mdo_scenario.MDOScenario` and :class:`~gemseo.core.doe_scenario.DOEScenario`
   inherit from the :class:`~gemseo.core.scenario.Scenario` class that defines common features
   (bounds, constraints, …).

-  :class:`~gemseo.core.discipline.MDODiscipline` represents a wrapped :term:`simulation software` program or a chain of wrapped
   software. It can either be a link to a :term:`discipline` integrated within a :term:`workflow engine`, or can
   be inherited to integrate a :term:`simulation software` directly. Its inputs and outputs are
   represented in a **Grammar** (see :class:`~gemseo.core.grammars.simple_grammar.SimpleGrammar` or :class:`~gemseo.core.grammars.json_grammar.JSONGrammar`).

-  :class:`~gemseo.core.formulation.MDOFormulation` describes the :term:`MDO formulation` (*e.g.* :term:`MDF` and :term:`IDF`)
   used by the  :class:`~gemseo.core.scenario.Scenario` to generate the  :class:`~gemseo.algos.opt_problem.OptimizationProblem`.
   The :term:`MDO user` or the :term:`MDO integrator` may either provide the name of the
   formulation, or a class, or an instance, to the :class:`~gemseo.core.scenario.Scenario`
   (:class:`~gemseo.core.mdo_scenario.MDOScenario` or\ :class:`~gemseo.core.doe_scenario.DOEScenario`) . The :term:`MDO formulations designer` may create, implement,
   test or maintain :term:`MDO formulations <MDO formulation>` with this class.

-  :class:`~gemseo.algos.opt_problem.OptimizationProblem` describes the mathematical functions of the
   optimization problem (:term:`objective function` and :term:`constraints<constraint>`, along with the :term:`design variables`. It is
   generated by the :class:`~gemseo.core.formulation.MDOFormulation`, and solved by the :term:`optimization algorithm`. It has an
   internal database that stores the calls to its functions by the  :term:`optimization algorithm` to
   avoid duplicate computations. It can be stored on disk and analyzed *a
   posteriori* by post-processing available in |g|.

-  :class:`~gemseo.algos.design_space.DesignSpace` is an attribute of the :class:`~gemseo.algos.opt_problem.OptimizationProblem` that describes the :term:`design variables`,
   their bounds, their type (float or integer), and current value. This object can be read from a file.


Two low-level classes at the core of |g| are crucial for the understanding of its basic principles:

-  :class:`~gemseo.core.mdofunctions.mdo_function.MDOFunction` instances (for the :term:`objective function` and the possible constraints) are
   generated by the :class:`~gemseo.core.formulation.MDOFormulation`. Depending on the formulation,
   constraints may be generated as well (*e.g.* consistency constraints in
   :term:`IDF`).

-  :class:`~gemseo.core.mdofunctions.function_generator.MDOFunctionGenerator` is a utility class that handles the
   :class:`~gemseo.core.mdofunctions.mdo_function.MDOFunction` generation for a given :class:`~gemseo.core.discipline.MDODiscipline`.
   It is a key class for the :term:`MDO formulations designer`.


The present figures display the main classes of |g|. They are simplified class diagrams ; only
parts of the subclasses and methods are represented. The present documentation
contains the full classes description in the different sections as well as the full API documentation.


.. uml::

   @startuml
   class Scenario {
   }
   class MDODiscipline {
   }
   class MDOFormulation {
   }
   class OptimizationProblem {
   }
   class DesignSpace {
   }
   class MDOFunctionGenerator {
   }
   class MDOFunction {
   }
   class DriverLibrary {
   }

   MDODiscipline <|- Scenario
   Scenario "1" *-> "n" MDODiscipline
   Scenario "1" *-> "1" MDOFormulation
   MDOFormulation "1" --> "n" OptimizationProblem
   MDOFunctionGenerator "1" --> "n" MDOFunction
   MDOFunctionGenerator "1" *-> "1" MDODiscipline
   Scenario "1" *-> "1" DriverLibrary
   OptimizationProblem "1" *-> "1" DesignSpace
   OptimizationProblem "1" *-> "n" MDOFunction
   MDOFormulation "1" *-> "n" MDOFunctionGenerator
   @end uml


.. uml::

   @startuml
   class MDODiscipline {
   }
   class MDA {
   }
   class MDOChain {
   }
   class MDAJacobi {
   }
   class MDAGaussSeidel {
   }
   class MDAChain {
   }

   MDODiscipline <|- MDA
   MDA "1" *-> "1" MDODiscipline
   MDA <|- MDAJacobi
   MDA <|- MDAGaussSeidel
   MDODiscipline <|- MDOChain
   MDA <|- MDAChain
   MDOChain "1" <-- "1" MDAChain
   MDAChain "1" *-> "n" MDA
   MDOChain "1" *-> "1" MDODiscipline


   @end uml

.. uml::

   @startuml
   class Scenario {
   }
   class DriverLibrary {
   }
   class DOELibrary {
   }
   class OptLibrary {
   }
   class MDOScenario {
   }
   class DOEScenario {
   }

   DOELibrary -up|> DriverLibrary
   OptLibrary -up|> DriverLibrary
   Scenario "1" *-up> "1" DriverLibrary
   DOEScenario "1" *-> "1" DOELibrary
   MDOScenario "1" *-> "1" OptLibrary
   MDOScenario -up|> Scenario
   DOEScenario -up|> Scenario
   @end uml


.. uml::

   @startuml
   class OptimizationProblem {
   }
   class DesignSpace {
   }
   class MDOFunction {
   }
   class Database {
   }
   class MDOFormulation {
   }

   MDOFormulation "1" --up> "1" OptimizationProblem
   OptimizationProblem "1" *-up> "1" DesignSpace
   OptimizationProblem "1" *-up> "n" MDOFunction
   OptimizationProblem "1" *-> "1" Database
   @end uml
