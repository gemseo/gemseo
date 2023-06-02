..
   Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

   This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
   International License. To view a copy of this license, visit
   http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
   Commons, PO Box 1866, Mountain View, CA 94042, USA.

.. math::

   \begin{aligned}
   \text{minimize the objective function }&\text{obj}=x_{\text{local}}^2 + x_{\text{shared},2}
   +y_1^2+e^{-y_2} \\
   \text{with respect to the design variables }&x_{\text{shared}},\,x_{\text{local}} \\
   \text{subject to the general constraints }
   & c_1 \leq 0\\
   & c_2 \leq 0\\
   \text{subject to the bound constraints }
   & -10 \leq x_{\text{shared},1} \leq 10\\
   & 0 \leq x_{\text{shared},2} \leq 10\\
   & 0 \leq x_{\text{local}} \leq 10.
   \end{aligned}

where the coupling variables are

.. math::

    \text{Discipline 1: } y_1 = \sqrt{x_{\text{shared},1}^2 + x_{\text{shared},2} +
     x_{local} - 0.2\,y_2},

and

.. math::

    \text{Discipline 2: }y_2 = |y_1| + x_{\text{shared},1} + x_{\text{shared},2}.

and where the general constraints are

.. math::

   c_1 = 3.16 - y_1^2

   c_2 = y_2 - 24.
