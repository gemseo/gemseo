..
   Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

   This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
   International License. To view a copy of this license, visit
   http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
   Commons, PO Box 1866, Mountain View, CA 94042, USA.

.. math::

   \text{OVERALL AIRCRAFT DESIGN} = \left\{
   \begin{aligned}
   &\text{minimize }\text{range}(\text{thick\_airfoils}, \text{thick\_panels}, \text{sweep}) = 8\times10^{11}\times\text{lift}\times\text{mass}/\text{drag} \\
   &\text{with respect to }\text{thick\_airfoils},\,\text{thick\_panels},\,\text{sweep} \\
   &\text{subject to }\\
   & \text{rf}-0.5 = 0\\
   & \text{lift}-0.5 \leq 0
   \end{aligned}\right.

where

.. math::

       \text{AERODYNAMICS} = \left\{
           \begin{aligned}
        &\text{drag}=0.1\times((\text{sweep}/360)^2 + 200 + \text{thick\_airfoils}^2 - \text{thick\_airfoils} - 4\times\text{displ})\\
        &\text{forces}=10\times\text{sweep} + 0.2\times\text{thick\_airfoils}-0.2\times\text{displ}\\
        &\text{lift}=(\text{sweep} + 0.2\times\text{thick\_airfoils}-2\times\text{displ})/3000
           \end{aligned}
           \right.

and

.. math::

       \text{STRUCTURE} = \left\{
           \begin{aligned}
        &\text{mass}=4000\times(\text{sweep}/360)^3 + 200000 + 100\times\text{thick\_panels} + 200\times\text{forces}\\
        &\text{rf}=3\times\text{sweep} - 6\times\text{thick\_panels} + 0.1\times\text{forces} + 55\\
        &\text{displ}=2\times\text{sweep} + 3\times\text{thick\_panels} - 2\times\text{forces}
           \end{aligned}
           \right.
