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
Another possibility is to use the API method :meth:`gemseo.api.get_available_formulations` to list them:

.. code:: python

    from gemseo.api import get_available_formulations
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

   A process based on the MDF formulation


Gradient-based optimization algorithms require the computation of the
total derivatives of :math:`\phi(x, z, y(x, z))`, where
:math:`\phi \in \{f, g\}` and :math:`v \in \{x,
z\}`.

For details on the MDAs and coupled derivatives, see :ref:`mda` and :ref:`jacobian_assembly`.

An example of an MDO study using an MDF formulation can be found in the :ref:`Sellar MDO tutorial <sellar_mdo>`

.. warning::

    Any :class:`.MDODiscipline` that will be placed inside an :class:`.MDF` formulation with strong couplings **must**
    define its default inputs. Otherwise, the execution will fail.

.. _idf_formulation:

IDF
---

:term:`IDF` handles the disciplines in a decoupled fashion: all disciplinary
analysis are performed independently and possibly in parallel. Coupling
variables :math:`y^t` (called targets) are driven by the optimization
algorithm and are inputs of all disciplinary analyses :math:`y_i(x_i, z,
y_{j \neq i}^t), \forall i \in \{1, \ldots, N\}`. In comparison, handles
the disciplines in a coupled manner: the inputs of the disciplines are
outputs of the other disciplines.

.. math::

   \begin{aligned}
   & \underset{x,z,y^t}{\text{min}} & & f(x, z, y^t) \\
   & \text{subject to}     & & g(x, z, y^t) \le 0 \\
   &                       & & y_i(x_i, z, y^t_{j \neq i}) - y_i^t = 0, \quad \forall i \in \{1,
   \ldots, N\}
   \end{aligned}
   \label{eq:idf-problem}

Additional consistency constraints
:math:`y_i(x_i, z, y^t_{j \neq i}) - y_i^t = 0,
\forall i \in \{1, \ldots, N\}` ensure that the couplings computed by
the disciplinary analysis coincide with the corresponding inputs
:math:`y^t` of the other disciplines. This guarantees an equilibrium
between all disciplines at convergence.

.. figure:: /_images/mdo_formulations/IDF_process.png
   :scale: 65 %

   A process based on the IDF formulation


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

   A process based on a Bi-level formulation

.. warning::

    Any :class:`.MDODiscipline` that will be placed inside a :class:`.BiLevel`
    formulation with strong couplings **must** define its default inputs.
    Otherwise, the execution will fail.

.. _xdsm:

XDSM visualization
------------------

|g| allows to visualize a given MDO scenario/formulation as an :term:`XDSM` diagram (see :cite:`Lambe2012`) in a web browser.
The figure below shows an example of such visualization.

.. figure:: /_images/bilevel_ssbj.png
   :scale: 80 %

   An XDSM visualization generated with |g|

The rendering is handled by the visualization library `XDSMjs <https://github.com/OneraHub/XDSMjs>`_.
|g| provides a utility class :class:`~gemseo.utils.xdsmizer.XDSMizer` to export the given MDO scenario as a suitable
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

      |g| XDSM visualization of the Sobiesky example solved with MDF formulation

Installation
^^^^^^^^^^^^

From |g| v1.4, the manual installation of XDSMjs is not required, since a Python package
is now available. Also, a self contained web page can be generated.

Usage
^^^^^

Then within your Python script, given your ``scenario`` object, you can generate the XDSM json file
with the following code:

.. code:: python

    scenario.xdsmize(open_browser=True)


If html_output (default True), will generate a self contained html file, that can be automatically open using the option open_browser=True.
If outdir is set to Non (default '.'), a temporary file is generated.
If json_output is True, it will generate a `XDSMjs <https://github.com/OneraHub/XDSMjs>`_ input file :ref:`xdsm` (legacy behavior).
If latex_output is set to True (default False), a Latex PDF is generated.

You should observe the XDSM diagram related to your MDO scenario.
