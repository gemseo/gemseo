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
As specified by Sobieski et al., additional design and state
variables (not present in the original problem) were introduced in the
disciplinary analyses for testing purposes.

The MDO problem
~~~~~~~~~~~~~~~

The aim of the problem is to **maximize the range of a Super-Sonic Business under various constraints**.

The problem is built from **three disciplines: structure, aerodynamics and propulsion**.

**A fourth discipline**, not coupled to the other ones, **is required to compute the range**.


Input and output variables
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The next tables display the input variables required by each of the four disciplines.

As they are :term:`shared <shared design variables>` by several disciplines, :term:`global design variables <shared design variables>`, implemented as :math:`x\_shared`, are provided to all disciplines.

:term:`Coupling variables <coupling variables>` are implemented as :math:`y\_ij`, which is an output of discipline :math:`i` and an input of discipline :math:`j`.

Disciplines are listed as follows

    #. Structure,
    #. Aerodynamics,
    #. Propulsion,
    #. Range.

.. figure:: /tutorials/ssbj/figs/SSBJ.png
   :scale: 100 %

   The SSBJ planform variables

.. figure:: /tutorials/ssbj/figs/SupersonicAirfoil.png
   :scale: 100 %

   The airfoils variables

+------------------------------------------+------------------------------------------------+----------------------------------+------------------------+
| **Variable**                             | **Description**                                | **Bounds**                       | **Symbol**             |
+==========================================+================================================+==================================+========================+
| :math:`t/c`                              | Thickness to chord ratio                       | :math:`0.01\leq t/c\leq 0.09`    | :math:`x\_shared[0]`   |
+------------------------------------------+------------------------------------------------+----------------------------------+------------------------+
| :math:`h`                                | Altitude (:math:`ft`)                          | :math:`30000\leq h \leq 60000`   | :math:`x\_shared[1]`   |
+------------------------------------------+------------------------------------------------+----------------------------------+------------------------+
| :math:`M`                                | Mach number                                    | :math:`1.4\leq M\leq 1.8`        | :math:`x\_shared[2]`   |
+------------------------------------------+------------------------------------------------+----------------------------------+------------------------+
| :math:`AR=b^2/S_W`                       | Aspect ratio                                   | :math:`2.5\leq AR\leq 8.5`       | :math:`x\_shared[3]`   |
+------------------------------------------+------------------------------------------------+----------------------------------+------------------------+
| :math:`\Lambda`                          | Wing sweep (:math:`\deg`)                      | :math:`40\leq\Lambda\leq70`      | :math:`x\_shared[4]`   |
+------------------------------------------+------------------------------------------------+----------------------------------+------------------------+
| :math:`S_W`                              | Wing surface area (:math:`ft^2`)               | :math:`500\leq S\leq 1500`       | :math:`x\_shared[5]`   |
+------------------------------------------+------------------------------------------------+----------------------------------+------------------------+
| :math:`\lambda = {c_{tip}}/{c_{root}}`   | Wing taper ratio                               | :math:`0.1\leq\lambda\leq0.4`    | :math:`x\_1[0]`        |
+------------------------------------------+------------------------------------------------+----------------------------------+------------------------+
| :math:`x`                                | Wingbox x-sectional area                       | :math:`0.75\leq x \leq 1.25`     | :math:`x\_1[1]`        |
+------------------------------------------+------------------------------------------------+----------------------------------+------------------------+
| :math:`L`                                | Lift from **aerodynamics** (:math:`N`)         |                                  | :math:`y\_21[0]`       |
+------------------------------------------+------------------------------------------------+----------------------------------+------------------------+
| :math:`W_{E}`                            | Engine mass from **propulsion** (:math:`lb`)   |                                  | :math:`y\_31[0]`       |
+------------------------------------------+------------------------------------------------+----------------------------------+------------------------+
| :math:`C_f`                              | Skin friction coefficient                      | :math:`0.75\leq Cf\leq 1.25`     | :math:`x\_2[0]`        |
+------------------------------------------+------------------------------------------------+----------------------------------+------------------------+
| :math:`W_T`                              | Total aircraft mass from **structure**         |                                  | :math:`y\_12[0]`       |
+------------------------------------------+------------------------------------------------+----------------------------------+------------------------+
| :math:`\Delta\alpha_v`                   | Wing twist from **structure**                  |                                  | :math:`y\_12[1]`       |
+------------------------------------------+------------------------------------------------+----------------------------------+------------------------+
| :math:`ESF`                              | Engine scale factor from  **propulsion**       |                                  | :math:`y\_32[0]`       |
+------------------------------------------+------------------------------------------------+----------------------------------+------------------------+
| :math:`Th`                               | Throttle setting (engine mass flow)            | :math:`0.1\leq T\leq 1.25`       | :math:`x\_3[0]`        |
+------------------------------------------+------------------------------------------------+----------------------------------+------------------------+
| :math:`D`                                | Drag from **aerodynamics** (:math:`N`)         |                                  | :math:`y\_23[0]`       |
+------------------------------------------+------------------------------------------------+----------------------------------+------------------------+
| :math:`L/D`                              | Lift-over-drag ratio from  **aerodynamics**    |                                  | :math:`y\_24[0]`       |
+------------------------------------------+------------------------------------------------+----------------------------------+------------------------+
| :math:`W_T`                              | Total aircraft mass from **structure**         |                                  | :math:`y\_14[0]`       |
+------------------------------------------+------------------------------------------------+----------------------------------+------------------------+
| :math:`W_F`                              | Fuel mass from **structure**                   |                                  | :math:`y\_14[1]`       |
+------------------------------------------+------------------------------------------------+----------------------------------+------------------------+
| :math:`SFC`                              | Specific Fuel Consumption from **propulsion**  |                                  | :math:`y\_34[0]`       |
+------------------------------------------+------------------------------------------------+----------------------------------+------------------------+

Table: Input variables of Sobieski’s problem

+----------------------------+----------------------------------------+--------------------------------------+--------------------------+
| **Variable**               | **Description**                        | **Constraint**                       | **Symbol**               |
+============================+========================================+======================================+==========================+
| :math:`\sigma_1`           | Stress constraints on wing section 1   | :math:`\sigma_1<1.09`                | :math:`g\_1[0]`          |
+----------------------------+----------------------------------------+--------------------------------------+--------------------------+
| :math:`\sigma_2`           | Stress constraints on wing section 2   | :math:`\sigma_2<1.09`                | :math:`g\_1[1]`          |
+----------------------------+----------------------------------------+--------------------------------------+--------------------------+
| :math:`\sigma_3`           | Stress constraints on wing section 3   | :math:`\sigma_3<1.09`                | :math:`g\_1[2]`          |
+----------------------------+----------------------------------------+--------------------------------------+--------------------------+
| :math:`\sigma_4`           | Stress constraints on wing section 4   | :math:`\sigma_4<1.09`                | :math:`g\_1[3]`          |
+----------------------------+----------------------------------------+--------------------------------------+--------------------------+
| :math:`\sigma_5`           | Stress constraints on wing section 5   | :math:`\sigma_5<1.09`                | :math:`g\_1[4]`          |
+----------------------------+----------------------------------------+--------------------------------------+--------------------------+
| :math:`W_T`                | Total aircraft mass (:math:`lb`)       |                                      | :math:`y\_1[0]`          |
+----------------------------+----------------------------------------+--------------------------------------+--------------------------+
| :math:`W_F`                | Fuel mass (:math:`lb`)                 |                                      | :math:`y\_1[1]`          |
+----------------------------+----------------------------------------+--------------------------------------+--------------------------+
| :math:`\Delta\alpha_{v}`   | Wing twist (:math:`\deg`)              | :math:`0.96<\Delta\alpha_{v}<1.04`   | :math:`y\_1[2],g_1[5]`   |
+----------------------------+----------------------------------------+--------------------------------------+--------------------------+
| :math:`L`                  | Lift (:math:`N`)                       |                                      | :math:`y\_2[0]`          |
+----------------------------+----------------------------------------+--------------------------------------+--------------------------+
| :math:`D`                  | Drag (:math:`N`)                       |                                      | :math:`y\_2[1]`          |
+----------------------------+----------------------------------------+--------------------------------------+--------------------------+
| :math:`L/D`                | Lift-over-drag ratio                   |                                      | :math:`y\_2[2]`          |
+----------------------------+----------------------------------------+--------------------------------------+--------------------------+
| :math:`dp/dx`              | Pressure gradient                      | :math:`dp/dx<1.04`                   | :math:`g\_2[0]`          |
+----------------------------+----------------------------------------+--------------------------------------+--------------------------+
| :math:`SFC`                | Specific Fuel Consumption              |                                      | :math:`y\_3[0]`          |
+----------------------------+----------------------------------------+--------------------------------------+--------------------------+
| :math:`W_E`                | Engine mass (:math:`lb`)               |                                      | :math:`y\_3[1]`          |
+----------------------------+----------------------------------------+--------------------------------------+--------------------------+
| :math:`ESF`                | Engine Scale Factor                    | :math:`0.5\leq ESF \leq 1.5`         | :math:`y\_3[2],g_3[0]`   |
+----------------------------+----------------------------------------+--------------------------------------+--------------------------+
| :math:`T_E`                | Engine temperature                     | :math:`T_E\leq 1.02`                 | :math:`g\_3[1]`          |
+----------------------------+----------------------------------------+--------------------------------------+--------------------------+
| :math:`Th`                 | Throttle setting constraint            | :math:`Th\leq Th_{uA}`               | :math:`g\_3[2]`          |
+----------------------------+----------------------------------------+--------------------------------------+--------------------------+
| :math:`R`                  | Range (:math:`nm`)                     |                                      | :math:`y\_4[0]`          |
+----------------------------+----------------------------------------+--------------------------------------+--------------------------+

Table: Output variables of Sobieski’s problem

.. end_description

Creation of the disciplines
~~~~~~~~~~~~~~~~~~~~~~~~~~~

To create the SSBJ disciplines :

.. code::

     from gemseo.api import  create_discipline

     disciplines = create_discipline(["SobieskiStructure",
                                     "SobieskiPropulsion",
                                     "SobieskiAerodynamics",
                                     "SobieskiMission"])

Reference results
~~~~~~~~~~~~~~~~~

This problem was implemented by Sobieski et al. in Matlab and Isight.
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
