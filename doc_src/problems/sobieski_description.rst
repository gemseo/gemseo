..
   Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

   This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
   International License. To view a copy of this license, visit
   http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
   Commons, PO Box 1866, Mountain View, CA 94042, USA.

..
   Contributors:
          :author: Matthias De Lozzo

Sobieski's SSBJ test case
-------------------------

The Sobieski's SSBJ test case is considered in the different tutorials:

- :ref:`sobieski_doe`
- :ref:`sphx_glr_examples_mdo_plot_sobieski_use_case.py`

.. start_description

Origin of the test case
~~~~~~~~~~~~~~~~~~~~~~~

This test was taken from the reference article by Sobieski, the first
publication on the BLISS formulation. It is based on a 1996 AIAA student
competition organized by the AIAA/United Technologies/Pratt & Whitney
Individual Undergraduate Design Competition. This competition initially
focused on both the technical and economical challenges of a
development.

The formulas used for each are based on semi-empirical and/or analytical
models. Depending on the , some examples can be found in the following
references :cite:`niu,anderson,Raymer`.
As specified by Sobieski *et al.*, additional design and state
variables (not present in the original problem) were introduced in the
disciplinary analyses for testing purposes.

The MDO problem
~~~~~~~~~~~~~~~

The design problem is
to maximize the range ``"y_4"`` of a Super-Sonic Business Jet (SSBJ)
with respect to the design variables ``"x_shared"``, ``"x_1"``, ``"x_2"`` and ``"x_3"``
under the constraints ``"g_1"``, ``"g_2"`` and ``"g_3"``.

The objective and constraints are computed by four disciplines:

1. :class:`~gemseo.problems.sobieski.disciplines.SobieskiStructure` (indexed by 1)
   computes the constraint ``"g_1"`` from ``"x_shared"`` and ``"x_1"``,
2. :class:`~gemseo.problems.sobieski.disciplines.SobieskiAerodynamics` (indexed by 2)
   computes the constraint ``"g_2"`` from ``"x_shared"`` and ``"x_2"``,
3. :class:`~gemseo.problems.sobieski.disciplines.SobieskiPropulsion` (indexed by 3)
   computes the constraint ``"g_3"`` from ``"x_shared"`` and ``"x_3"``,
4. :class:`~gemseo.problems.sobieski.disciplines.SobieskiMission` (indexed by 4)
   computes the constraint ``"y_4"`` from ``"x_shared"``.

``"x_shared"`` denotes the :term:`global design variables <shared design variables>` (a.k.a. shared design variables),
which means that these variables are shared by at least two disciplines (here, all).
``"x_1"``, ``"x_2"`` and ``"x_3"`` denote the :term:`local design variables <local design variables>`,
which means that ``"x_i"`` is used by a single discipline (here, number *i*).

The disciplines
:class:`~gemseo.problems.sobieski.disciplines.SobieskiStructure`,
:class:`~gemseo.problems.sobieski.disciplines.SobieskiAerodynamics`
and :class:`~gemseo.problems.sobieski.disciplines.SobieskiPropulsion`
are strongly coupled to each other
but weakly coupled to :class:`~gemseo.problems.sobieski.disciplines.SobieskiMission`.
The :term:`coupling variable <coupling variables>` ``"y_ij"`` denotes
an output of the discipline no. *i* and an input of the discipline no. *j*.

Input variables
~~~~~~~~~~~~~~~

.. figure:: /tutorials/ssbj/figs/SSBJ.png
   :scale: 100 %

   The planform variables

.. figure:: /tutorials/ssbj/figs/SupersonicAirfoil.png
   :scale: 100 %

   The airfoils variables

.. list-table::
   :widths: 20 20 20 20 20
   :header-rows: 1

   * - Disciplines
     - Variable
     - Description
     - Bounds
     - Notation

   * - All
     - :math:`t/c`
     - Thickness to chord ratio
     - :math:`0.01\leq t/c\leq 0.09`
     - ``"x_shared[0]"``

   * - All
     - :math:`h`
     - Altitude (:math:`ft`)
     - :math:`30000\leq h \leq 60000`
     - ``"x_shared[1]"``

   * - All
     - :math:`M`
     - Mach number
     - :math:`1.4\leq M\leq 1.8`
     - ``"x_shared[2]"``

   * - All
     - :math:`AR=b^2/S_W`
     - Aspect ratio
     - :math:`2.5\leq AR\leq 8.5`
     - ``"x_shared[3]"``

   * - All
     - :math:`\Lambda`
     - Wing sweep (:math:`\deg`)
     - :math:`40\leq\Lambda\leq70`
     - ``"x_shared[4]"``

   * - All
     - :math:`S_W`
     - Wing surface area (:math:`ft^2`)
     - :math:`500\leq S\leq 1500`
     - ``"x_shared[5]"``

   * - Structure
     - :math:`\lambda = {c_{tip}}/{c_{root}}`
     - Wing taper ratio
     - :math:`0.1\leq\lambda\leq0.4`
     - ``"x_1[0]"``

   * - Structure
     - :math:`x`
     - Wingbox x-sectional area (:math:`ft^2`)
     - :math:`0.75\leq x \leq 1.25`
     - ``"x_1[1]"``

   * - Structure
     - :math:`L`
     - Lift from by Aerodynamics (:math:`N`)
     -
     - ``"y_21[0]"``

   * - Structure
     - :math:`W_{E}`
     - Engine mass from Propulsion (:math:`lb`)
     -
     - ``"y_31[0]"``

   * - Aerodynamics
     - :math:`C_f`
     - Skin friction coefficient
     - :math:`0.75\leq C_f\leq 1.25`
     - ``"x_2[0]"``

   * - Aerodynamics
     - :math:`W_T`
     - Total aircraft mass from Structure (:math:`lb`)
     -
     - ``"y_12[0]"``

   * - Aerodynamics
     - :math:`\Delta\alpha_v`
     - Wing twist from Structure
     -
     - ``"y_12[1]"``

   * - Propulsion
     - :math:`ESF`
     - Engine scale factor (ESF) from Propulsion
     -
     - ``"y_32[0]"``

   * - Propulsion
     - :math:`Th`
     - Throttle setting (engine mass flow)
     - :math:`0.1\leq Th\leq 1.25`
     - ``"x_3[0]"``

   * - Propulsion
     - :math:`D`
     - Drag from Aerodynamics (:math:`N`)
     -
     - ``"y_23[0]"``

   * - Mission
     - :math:`L/D`
     - Lift-over-drag ratio from Aerodynamics
     -
     - ``"y_24[0]"``

   * - Mission
     - :math:`W_T`
     - Total aircraft mass from Structure (:math:`lb`)
     -
     - ``"y_14[0]"``

   * - Mission
     - :math:`W_F`
     - Fuel mass from Structure (:math:`lb`)
     -
     - ``"y_14[1]"``

   * - Mission
     - :math:`SFC`
     - Specific fuel consumption (SFC) from Propulsion
     -
     - ``"y_34[1]"``

Output variables
~~~~~~~~~~~~~~~~

.. list-table::
   :widths: 20 20 20 20 20
   :header-rows: 1

   * - Disciplines
     - Variable
     - Description
     - Bounds
     - Notation

   * - Structure
     - :math:`\sigma_1-1.09`
     - Constraint about stress :math:`\sigma_1` on wing section 1
     - :math:`\sigma_1-1.09\leq 0`
     - ``"g_1[0]"``

   * - Structure
     - :math:`\sigma_2-1.09`
     - Constraint about stress :math:`\sigma_2` on wing section 2
     - :math:`\sigma_2-1.09\leq 0`
     - ``"g_1[1]"``

   * - Structure
     - :math:`\sigma_3-1.09`
     - Constraint about stress :math:`\sigma_3` on wing section 3
     - :math:`\sigma_3-1.09\leq 0`
     - ``"g_1[2]"``

   * - Structure
     - :math:`\sigma_4-1.09`
     - Constraint about stress :math:`\sigma_4` on wing section 4
     - :math:`\sigma_4-1.09\leq 0`
     - ``"g_1[3]"``

   * - Structure
     - :math:`\sigma_5-1.09`
     - Constraint about stress :math:`\sigma_5` on wing section 5
     - :math:`\sigma_5-1.09\leq 0`
     - ``"g_1[4]"``

   * - Structure
     - :math:`\Delta\alpha_{v}-1.04`
     - First constraint about wing twist :math:`\Delta\alpha_{v}`
     - :math:`\Delta\alpha_{v}-1.04\leq 0`
     - ``"g_1[5]"``

   * - Structure
     - :math:`0.96-\Delta\alpha_{v}`
     - Second constraint about wing twist :math:`\Delta\alpha_{v}`
     - :math:`0.96-\Delta\alpha_{v}\leq 0`
     - ``"g_1[6]"``

   * - Structure
     - :math:`W_T`
     - Total aircraft mass (:math:`lb`)
     -
     - ``"y_1[0]"``

   * - Structure
     - :math:`W_F`
     - Fuel mass (:math:`lb`)
     -
     - ``"y_1[1]"``

   * - Structure
     - :math:`\Delta\alpha_{v}`
     - Wing twist (:math:`\deg`)
     - :math:`0.96\leq \Delta\alpha_{v}\leq 1.04`
     - ``"y_1[2]"``

   * - Aerodynamics
     - :math:`L`
     - Lift (:math:`lb)
     -
     - ``"y_2[0]"``

   * - Aerodynamics
     - :math:`D`
     - Drag (:math:`lb`)
     -
     - ``"y_2[1]"``

   * - Aerodynamics
     - :math:`L/D`
     - Lift-over-drag ratio
     -
     - ``"y_2[2]"``

   * - Aerodynamics
     - :math:`dp/dx-1.04`
     - Constraint about the pressure gradient :math:`dp/dx`
     - :math:`dp/dx-1.04\leq 0`
     - ``"g_2[0]"``

   * - Propulsion
     - :math:`SFC`
     - Specific fuel consumption (SFC)
     -
     - ``"y_3[0]"``

   * - Propulsion
     - :math:`W_E`
     - Engine mass (:math:`lb`)
     -
     - ``"y_3[1]"``

   * - Propulsion
     - :math:`ESF`
     - Engine scale factor (ESF)
     - :math:`0.5\leq ESF \leq 1.5`
     - ``"y_3[2]"``

   * - Propulsion
     - :math:`ESF-1.5`
     - First constraint about the ESF
     - :math:`ESF-1.5 \leq 0`
     - ``"g_3[0]"``

   * - Propulsion
     - :math:`0.5-ESF`
     - Second constraint about the ESF
     - :math:`0.5-ESF \leq 0`
     - ``"g_3[1]"``

   * - Propulsion
     - :math:`Th-Th_{uA}`
     - Constraint about the throttle :math:`Th`
     - :math:`Th-Th_{uA}\leq 0`
     - ``"g_3[2]"``

   * - Propulsion
     - :math:`T_E-1.02`
     - Constraint about the engine temperature :math:`T_E`
     - :math:`T_E-1.02\leq 0`
     - ``"g_3[3]"``

   * - Mission
     - :math:`R`
     - Range (:math:`nm`)
     -
     - ``"y_4[0]"``

.. end_description

Creation of the disciplines
~~~~~~~~~~~~~~~~~~~~~~~~~~~

To create the disciplines of the Sobieski problem:

.. code::

     from gemseo import  create_discipline

     disciplines = create_discipline(["SobieskiStructure", "SobieskiPropulsion", "SobieskiAerodynamics", "SobieskiMission"])

Reference results
~~~~~~~~~~~~~~~~~

This problem was implemented by Sobieski *et al.* in Matlab and Isight.
Both implementations led to the same results.

As all gradients can be computed, we resort to gradient-based optimization methods.
All Jacobian matrices are coded analytically in |g|.

Reference results using the :ref:`MDF formulation <mdf_formulation>` are presented in the next table.

+-----------------------------------+--------------+---------------+
| Variable                          | Initial      | Optimum       |
+===================================+==============+===============+
| **Range (nm)**                    | **535.79**   | **3963.88**   |
+-----------------------------------+--------------+---------------+
| :math:`\lambda`                   | 0.25         | 0.38757       |
+-----------------------------------+--------------+---------------+
| :math:`x`                         | 1            | 0.75          |
+-----------------------------------+--------------+---------------+
| :math:`C_f`                       | 1            | 0.75          |
+-----------------------------------+--------------+---------------+
| :math:`Th`                        | 0.5          | 0.15624       |
+-----------------------------------+--------------+---------------+
| :math:`t/c`                       | 0.05         | 0.06          |
+-----------------------------------+--------------+---------------+
| :math:`h` :math:`(ft)`)           | 45000        | 60000         |
+-----------------------------------+--------------+---------------+
| :math:`M`                         | 1.6          | 1.4           |
+-----------------------------------+--------------+---------------+
| :math:`AR`                        | 5.5          | 2.5           |
+-----------------------------------+--------------+---------------+
| :math:`\Lambda` :math:`(\deg)`    | 55           | 70            |
+-----------------------------------+--------------+---------------+
| :math:`S_W` :math:`(ft^2)`        | 1000         | 1500          |
+-----------------------------------+--------------+---------------+
