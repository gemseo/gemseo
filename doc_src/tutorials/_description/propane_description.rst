..
   Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

   This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
   International License. To view a copy of this license, visit
   http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
   Commons, PO Box 1866, Mountain View, CA 94042, USA.

..
   Contributors:
          :author: Matthias De Lozzo

The Propane combustion problem
------------------------------

The Propane MDO problem can be found in :cite:`Padula1996` and :cite:`TedfordMartins2006`. It represents the
chemical equilibrium reached during the combustion of propane in air. Variables are
assigned to represent each of the ten combustion products as well as the sum of the
products.

The optimization problem is as follows:


.. math::

   \begin{aligned}
   \text{minimize the objective function }& f_2 + f_6 + f_7 + f_9 \\
   \text{with respect to the design variables }&x_{1},\,x_{3},\,x_{6},\,x_{7} \\
   \text{subject to the general constraints }
   & f_2(x) \geq 0\\
   & f_6(x) \geq 0\\
   & f_7(x) \geq 0\\
   & f_9(x) \geq 0\\
   \text{subject to the bound constraints }
   & x_{1} \geq 0\\
   & x_{3} \geq 0\\
   & x_{6} \geq 0\\
   & x_{7} \geq 0\\
   \end{aligned}

where the System Discipline consists of computing the following expressions:

.. math::

   \begin{aligned}
   f_2(x) & = & 2x_1 + x_2 + x_4 + x_7 + x_8 + x_9 + 2x_{10} - R, \\
   f_6(x) & = & K_6x_2^{1/2}x_4^{1/2} - x_1^{1/2}x_6(p/x_{11})^{1/2}, \\
   f_7(x) & = & K_7x_1^{1/2}x_2^{1/2} - x_4^{1/2}x_7(p/x_{11})^{1/2}, \\
   f_9(x) & = & K_9x_1x_3^{1/2} - x_4x_9(p/x_{11})^{1/2}. \\
   \end{aligned}


Discipline 1 computes :math:`(x_{2}, x_{4})` by satisfying the following equations:

.. math::

   \begin{aligned}
   x_1 + x_4 - 3 &=& 0,\\
   K_5x_2x_4 - x_1x_5 &=& 0.\\
   \end{aligned}

Discipline 2 computes :math:`(x_2, x_4)` such that:

.. math::

   \begin{aligned}
   K_8x_1 + x_4x_8(p/x_{11}) &=& 0,\\
   K_{10}x_{1}^{2} - x_4^2x_{10}(p/x_{11}) &=& 0.\\
   \end{aligned}

and Discipline 3 computes :math:`(x_5, x_9, x_{11})` by solving:

.. math::

   \begin{aligned}
   2x_2 + 2x_5 + x_6 + x_7 - 8&=& 0,\\
   2x_3 + x_9 - 4R &=& 0, \\
   x_{11} - \sum_{j=1}^{10} x_j &=& 0. \\
   \end{aligned}

Creation of the disciplines
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The Propane combustion disciplines are available in |g| and can be imported with the following code:

.. code::

     from gemseo.api import  create_discipline

     disciplines = create_discipline(["PropaneComb1",
                                     "PropaneComb2",
                                     "PropaneComb3",
                                     "PropaneReaction"])

A :class:`gemseo.algos.design_space.DesignSpace` file *propane_design_space.txt* is also available in the same folder, which can be read using
the :meth:`gemseo.api.read_design_space` method.

Problem results
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The optimum is :math:`(x1,x3,x6,x7) = (1.378887, 18.426810, 1.094798, 0.931214)`.
The minimum objective value is :math:`0`. At this point,  all the system-level inequality constraints are active.
