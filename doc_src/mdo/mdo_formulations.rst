..
   Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

   This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
   International License. To view a copy of this license, visit
   http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
   Commons, PO Box 1866, Mountain View, CA 94042, USA.

..
   Contributors:
          :author: Charlie Vanaret, Francois Gallard, Rémi Lafage

.. _mdo_formulations:

MDO formulations
================

In this section we describe the MDO formulations features of |g|.

Available formulations in |g|
-----------------------------------------

To see which formulations are available in your |g| version, you may have a look in the folder **gemseo.formulations**.
Another possibility is to use the API method :meth:`gemseo.get_available_formulations` to list them:

.. code:: python

    from gemseo import get_available_formulations
    print(get_available_formulations())

This prints the formulations names available in the current configuration.

.. code:: bash

    ['MDF', 'DisciplinaryOpt', 'BiLevel', 'IDF']

These implement the classical formulations:
    - :term:`MDF`
    - :term:`IDF`
    - a simple :term:`disciplinary optimization` formulation for a :term:`weakly coupled problem`
    - a particular :term:`bi-level` formulation from :term:`IRT` Saint exupéry

In the following, general concepts about the formulations are given. The :ref:`mdf_formulation` and :ref:`idf_formulation` text is integrally taken from the paper :cite:`Vanaret2017`.

To see how to setup practical test cases with such formulations, please see :ref:`sellar_mdo` and
:ref:`sphx_glr_examples_mdo_plot_sobieski_use_case.py`.

.. seealso::

   For a review of MDO formulations, see :cite:`MartinsSurvey`.

We use the following notations:

- :math:`N` is the number of disciplines,
- :math:`x=(x_1,x_2,\ldots,x_N)` are the local design variables,
- :math:`z` are the shared design variables,
- :math:`y=(y_1,y_2,\ldots,y_N)` are the coupling variables,
- :math:`f` is the objective,
- :math:`g` are the constraints.

.. _mdf_formulation:

MDF
---

:term:`MDF` is an architecture that guarantees an equilibrium between all
disciplines at each iterate :math:`(x, z)` of the optimization process.
Consequently, should the optimization process be prematurely
interrupted, the best known solution has a physical meaning. MDF generates
the smallest possible optimization problem, in which the coupling
variables are removed from the set of optimization variables and the
residuals removed from the set of constraints:

.. math::

   \begin{aligned}
   & \underset{x,z}{\text{min}}    & & f(x, z, y(x, z)) \\
   & \text{subject to}             & & g(x, z, y(x, z)) \le 0
   \end{aligned}
   \label{eq:mdf-problem}

The coupling variables :math:`y(x, z)` are computed at equilibrium via
an MDA. It amounts to solving a system of (possibly nonlinear) equations
using fixed-point methods (Gauss-Seidel, Jacobi) or root-finding methods
(Newton-Raphson, quasi-Newton). A prerequisite for invoking is the
existence of an equilibrium for any values of the design variables
:math:`(x, z)` encountered during the optimization process.

.. figure:: /_images/mdo_formulations/MDF_process.png
   :scale: 65 %

   A process based on the MDF formulation.


Gradient-based optimization algorithms require the computation of the
total derivatives of :math:`\phi(x, z, y(x, z))`, where
:math:`\phi \in \{f, g\}` and :math:`v \in \{x,
z\}`.

For details on the MDAs and coupled derivatives, see :ref:`mda` and :ref:`jacobian_assembly`.

An example of an MDO study using an MDF formulation can be found in the :ref:`Sellar MDO tutorial <sellar_mdo>`

.. warning::

    Any :class:`.Discipline` that will be placed inside an :class:`.MDF` formulation with strong couplings **must**
    define its default inputs. Otherwise, the execution will fail.

.. _idf_formulation:

IDF
---

:term:`IDF` stands for individual discipline feasible.
This MDO formulation expresses the MDO problem as

.. math::

   \begin{aligned}
   & \underset{x,z,y^t}{\text{min}} & & f(x, z, y^t) \\
   & \text{subject to}     & & g(x, z, y^t) \le 0 \\
   &                       & & h(x, z, y^t) = 0 \\
   &                       & & y_i(x_i, z, y^t_{j \neq i}) - y_i^t = 0,
                               \quad \forall i \in \{1,\ldots, N\}
   \end{aligned}

where :math:`y^t=(y_1^t,y_2^t,\ldots,y_N^t)` are additional optimization variables, called *targets* or *coupling targets*,
used as input coupling variables of the disciplines.
The additional constraints :math:`y_i(x_i, z, y^t_{j \neq i}) - y_i^t = 0, \forall i \in \{1, \ldots, N\}`, called *consistency* constraints,
ensure that the output coupling variables computed by the disciplines :math:`y` coincide with the targets.

The use of coupling targets allows the disciplines to be run in a decoupled way
while the use of consistency constraints guarantees a multidisciplinary feasible solution at convergence of the optimizer.
Thus,
the iterations are less costly than those of MDF, as they do not use an MDA algorithm,
but IF does not allow early stopping with the guarantee of a multidisciplinary feasible solution, unlike MDF.

.. figure:: /_images/mdo_formulations/IDF_process.png
   :scale: 65 %

   A process based on the IDF formulation.

Note that the targets can include either all the couplings or the strong couplings only.
If all couplings,
then all disciplines are executed in parallel,
and all couplings (weak and strong) are set as target variables in the design space.
This maximizes the exploitation of the parallelism but leads to a larger design space,
so usually more iterations by the optimizer.

.. figure:: /_images/mdo_formulations/xdsm_sobieski_idf_all.png
   :scale: 65 %

   The XDSM of the IDF formulation for the Sobieski's SSBJ problem,
   considering all the coupling targets.

If the strong couplings only,
then the coupling graph is analyzed
and the disciplines are chained in sequence and in parallel to solve all weak couplings.
In this case,
the size of the optimization problem is reduced,
so usually leads to less iterations.
The best option depends on the number of strong vs weak couplings,
the availability of gradients,
the availability of CPUs versus the number of disciplines,
so it is very context dependant.

.. figure:: /_images/mdo_formulations/xdsm_sobieski_idf_strong.png
   :scale: 65 %

   The XDSM of the IDF formulation for the Sobieski's SSBJ problem,
   considering the strong coupling targets only.


.. _bilevel_formulation:

Bi level
--------

Bi level formulations are a family of MDO formulations that involve multiple optimization problems to be solved to obtain the solution
of the MDO problem.

In many of them, and in particular in the formulations derived from :term:`BLISS`,
the separation of the optimization problems is made on the :term:`design variables`. The shared
design variables by multiple disciplines are put in a so called system level optimization problem. In so-called disciplinary
optimization problems, only the design variables that have a direct impact on one discipline are used.
Then, the coupling variables may be solved by a :ref:`mda`, as in :term:`BLISS`, :term:`ASO` and :term:`CSSO`,
or by using consistency constraints or a penalty function, like in :term:`CO` or :term:`ATC`.

The next figure shows the decomposition of the bi-level MDO formulation implemented in |g| MDAs,
sub optimization and a main optimization on the shared variables.
It is derived from the BLISS formulation and variants from ONERA :cite:`Blondeau2012`.
This formulation was invented in the MDA-MDO project at IRT Saint Exupery :cite:`gazaix2017towards`, :cite:`Gazaix2019`.


.. figure:: /_images/mdo_formulations/bilevel_process.png
   :scale: 55 %

   A process based on a Bi-level formulation.

.. warning::

    Any :class:`.Discipline` that will be placed inside a :class:`.BiLevel`
    formulation with strong couplings **must** define its default inputs.
    Otherwise, the execution will fail.

.. _xdsm:

XDSM visualization
------------------

|g| allows to visualize a given MDO scenario/formulation as an :term:`XDSM` diagram (see :cite:`Lambe2012`) in a web browser.
The figure below shows an example of such visualization.

.. figure:: /_images/bilevel_ssbj.png
   :scale: 80 %

   An XDSM visualization generated with |g|.

The rendering is handled by the visualization library `XDSMjs <https://github.com/OneraHub/XDSMjs>`_.
|g| provides a utility class :class:`.XDSMizer` to export the given MDO scenario as a suitable
input json file for this visualization library.

Features
^^^^^^^^

XDSM visualization shows:

* dataflow between disciplines (connections between disciplines as list of variables)
* optimization problem display (click on optimizer box)
* workflow animation (top-left contol buttons trigger either automatic or step-by-step mode)

.. only:: html

   Those features are illustrated by the animated gif below.

   .. figure:: /_images/xdsmjs_demo.gif

      |g| XDSM visualization of the Sobiesky example solved with MDF formulation.

Installation
^^^^^^^^^^^^

From |g| v1.4, the manual installation of XDSMjs is not required, since a Python package
is now available. Also, a self contained web page can be generated.

Usage
^^^^^

Then within your Python script, given your ``scenario`` object, you can generate the XDSM json file
with the following code:

.. code:: python

    scenario.xdsmize(show_html=True)


If ``save_html`` (default ``True``), will generate a self contained HTML file, that can be automatically open using the option ``show_html=True``.
If ``save_json`` is True, it will generate a `XDSMjs <https://github.com/OneraHub/XDSMjs>`_ input file :ref:`xdsm` (legacy behavior).
If ``save_pdf=True`` (default ``False``), a LaTex PDF is generated.

You should observe the XDSM diagram related to your MDO scenario.
