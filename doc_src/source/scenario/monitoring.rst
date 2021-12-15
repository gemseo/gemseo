..
   Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

   This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
   International License. To view a copy of this license, visit
   http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
   Commons, PO Box 1866, Mountain View, CA 94042, USA.

..
   Contributors:
          :author: Francois Gallard

.. _monitoring:

Monitoring the execution of a scenario
======================================

When a scenario is executed (see :ref:`sellar_mdo` for building a scenario), |g| logs the last computed value of the objective
function. But a finer monitoring may be needed, especially in case of crash.
In a situation like this, the current execution status of the :class:`~gemseo.core.discipline.MDODiscipline` is useful as well.

In this page, the different monitoring modes of a |g| scenario are illustrated on the :ref:`Sobieski <sobieski_problem>` MDF test case.

For that, by means of the API function :meth:`~gemseo.api.create_discipline`, we build the :class:`~gemseo.core.discipline.MDODiscipline`:

.. code::

    from gemseo.api import create_discipline

    disciplines = create_discipline(["SobieskiStructure",
                                     "SobieskiPropulsion",
                                     "SobieskiAerodynamics",
                                     "SobieskiMission"])

and by means of the API function :meth:`~gemseo.api.create_discipline`, we build the :class:`~gemseo.core.mdo_scenario.MDOScenario` :

.. code::

    from gemseo.api import create_scenario

    scenario = create_scenario(disciplines,
                               formulation_name="MDF",
                               objective_name="y_4",
                               design_space="design_space.txt")


Basic monitoring using logs
---------------------------

The simplest way to monitor a change in the statuses of the disciplines is to log them in the console or in a file using |g|'s logger.
Use :mod:`~gemseo.api.configure_logger` to configure the logger to log in a file.

The method :meth:`~gemseo.core.scenario.Scenario.xdsmize` of the :class:`~gemseo.core.scenario.Scenario`
can be used to this aim (:code:`monitor=True`).
If the option ``html_output`` is set to ``True``, a self-contained html file will be generated. It may be opened automatically with the option ``open_browser=True``.
If ``json_output`` is ``True``, it will generate a `XDSMjs <https://github.com/OneraHub/XDSMjs>`_ input file :ref:`xdsm`,
and print the statuses in the logs (:code:`print_statuses=True`):

.. code::

    scenario.xdsmize(monitor=True, print_statuses=True, open_browser=False)

This generates outputs such as the following, where the process' hierarchy is represented by a flatten :term:`JSON` structure.

.. code::

    Optimization: |          | 0/5   0% [elapsed: 00:00 left: ?, ? iters/sec]
    {MDOScenario(RUNNING), {MDAChain(RUNNING), [{MDAGaussSeidel(PENDING), [SobieskiStructure(None), SobieskiPropulsion(None), SobieskiAerodynamics(None), ], }, SobieskiMission(None), ], }, }
    {MDOScenario(RUNNING), {MDAChain(RUNNING), [{MDAGaussSeidel(RUNNING), [SobieskiStructure(PENDING), SobieskiPropulsion(None), SobieskiAerodynamics(None), ], }, SobieskiMission(None), ], }, }
    {MDOScenario(RUNNING), {MDAChain(RUNNING), [{MDAGaussSeidel(RUNNING), [SobieskiStructure(RUNNING), SobieskiPropulsion(None), SobieskiAerodynamics(None), ], }, SobieskiMission(None), ], }, }
    {MDOScenario(RUNNING), {MDAChain(RUNNING), [{MDAGaussSeidel(RUNNING), [SobieskiStructure(DONE), SobieskiPropulsion(PENDING), SobieskiAerodynamics(None), ], }, SobieskiMission(None), ], }, }
    {MDOScenario(RUNNING), {MDAChain(RUNNING), [{MDAGaussSeidel(RUNNING), [SobieskiStructure(DONE), SobieskiPropulsion(RUNNING), SobieskiAerodynamics(None), ], }, SobieskiMission(None), ], }, }
    {MDOScenario(RUNNING), {MDAChain(RUNNING), [{MDAGaussSeidel(RUNNING), [SobieskiStructure(DONE), SobieskiPropulsion(DONE), SobieskiAerodynamics(PENDING), ], }, SobieskiMission(None), ], }, }
    {MDOScenario(RUNNING), {MDAChain(RUNNING), [{MDAGaussSeidel(RUNNING), [SobieskiStructure(DONE), SobieskiPropulsion(DONE), SobieskiAerodynamics(RUNNING), ], }, SobieskiMission(None), ], }, }
    {MDOScenario(RUNNING), {MDAChain(RUNNING), [{MDAGaussSeidel(RUNNING), [SobieskiStructure(PENDING), SobieskiPropulsion(DONE), SobieskiAerodynamics(DONE), ], }, SobieskiMission(None), ], }, }
    {MDOScenario(RUNNING), {MDAChain(RUNNING), [{MDAGaussSeidel(RUNNING), [SobieskiStructure(RUNNING), SobieskiPropulsion(DONE), SobieskiAerodynamics(DONE), ], }, SobieskiMission(None), ], }, }
    {MDOScenario(RUNNING), {MDAChain(RUNNING), [{MDAGaussSeidel(RUNNING), [SobieskiStructure(DONE), SobieskiPropulsion(PENDING), SobieskiAerodynamics(DONE), ], }, SobieskiMission(None), ], }, }


Graphical monitoring using `XDSMjs <https://github.com/OneraHub/XDSMjs>`_
-------------------------------------------------------------------------

An :ref:`xdsm` diagram with the status of the :class:`~gemseo.core.discipline.MDODiscipline` can be generated at each status change
of the :class:`~gemseo.core.discipline.MDODiscipline`. See :ref:`xdsm` for setting up the :ref:`XDSM <xdsm>` generation in a web browser.
To trigger this mode in a scenario, use :meth:`~gemseo.core.scenario.Scenario.xdsmize`, with the :code:`monitor` argument set to :code:`True`.
The path to the `XDSMjs <https://github.com/OneraHub/XDSMjs>`_ library must be set to the folder containing the `XDSMjs <https://github.com/OneraHub/XDSMjs>`_ :term:`HTML` files.


.. code::

    scenario.xdsmize(monitor=True, outdir="path_to_xdsmjs")

The following images shows the typical outputs of the process statuses



.. figure:: /_images/monitoring/monitoring_1.png
   :scale: 65 %

   Initial state of the process before execution: the colors represent the type of discipline (scenario, MDA, simple discipline)


.. figure:: /_images/monitoring/monitoring_2.png
   :scale: 65 %

   The process has started:  the colors represent the status of the disciplines : green for RUNNING, blue for PENDING, red for FAILED


.. figure:: /_images/monitoring/monitoring_3.png
   :scale: 65 %

   The process is running, the MDA iterations are ongoing

.. figure:: /_images/monitoring/monitoring_4.png
   :scale: 65 %

   The process is finished and failed, due to the SobieskiMission discipline failure



Monitoring from a external platform using the observer design pattern
---------------------------------------------------------------------

The monitoring interface can be used to generate events every time that the process status changes.
One can observe these events and program a platform to react and display information to the user, or store data in a database.
The observer design pattern is used.

In the following code, we create an :code:`Observer` object that implements an update method.
Then, by means of the API function :meth:`~gemseo.api.monitor_scenario`, we create a :class:`~gemseo.core.monitoring.Monitoring`
and add the observer to the list of the listeners that are notified by |g| monitoring system.

.. code::

    from gemseo.api import monitor_scenario

    class Observer(object):

        def update(self, atom):
            print(atom)

    observer = Observer()
    monitor_scenario(scenario, observer)

The scenario execution generates the following output log:

.. code::

    MDAChain(RUNNING)
    MDAGaussSeidel(RUNNING)
    SobieskiStructure(RUNNING)
    SobieskiStructure(DONE)
    SobieskiPropulsion(RUNNING)
    SobieskiPropulsion(DONE)
    SobieskiAerodynamics(RUNNING)
    SobieskiAerodynamics(DONE)
    SobieskiStructure(RUNNING)
    SobieskiStructure(DONE)
    SobieskiPropulsion(RUNNING)
    SobieskiPropulsion(DONE)
    # ...

More advanced use can be made of this notification system, since the atom has the discipline concerned by the status change as an attribute.
Therefore, one can programmatically track the execution; or the data creation by the discipline's execution, and store it.


Monitoring using a Gantt chart
------------------------------

A Gantt chart can be generated to visualize the process execution.
All discipline's execution and linearization times are recorded and plotted.

To activate the execution times recording,
which are required to plot the Gantt chart,
please enable the time stamps before executing the scenario.

.. code::

   from gemseo.core.discipline import MDODiscipline
   MDODiscipline.activate_time_stamps()

Then, after the scenario execution,
the Gantt chart can be created easily.

.. code::

   from gemseo.post.core.gantt_chart import create_gantt_chart

   create_gantt_chart(save=True, show=False)

This generates the following plot,
here on a Sobieski MDF scenario.


.. figure:: /_images/monitoring/gantt_process.png
   :scale: 65 %

   The Gantt chart: disciplines are sorted by names,
   each discipline has a dedicated row.
   The blue rectangles correspond to the execution time while the red ones represent
   linearization time.
