..
   Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

   This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
   International License. To view a copy of this license, visit
   http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
   Commons, PO Box 1866, Mountain View, CA 94042, USA.

..
   Contributors:
          :author: Matthias De Lozzo

.. _analyticdiscipline:

How to build an analytic discipline?
====================================

A simple :class:`.MDODiscipline` can be created using analytic formulas,
e.g. :math:`y_1=2x^2` and :math:`y_2=5+3x^2z^3`,
thanks to the class  :class:`.AnalyticDiscipline` which is a quick alternative to model a simple analytic MDO problem!

Create the dictionary of analytic outputs
*****************************************

First of all, we have to define the output expressions in a dictionary where keys are output names and values are formula with :code:`string` format:

.. code::

    expressions = {'y_1': '2*x**2', 'y_2': '5+3*x**2+z**3'}

Create and instantiate the discipline
*************************************

Then, we create and instantiate the corresponding :class:`.AnalyticDiscipline` inheriting from :class:`.MDODiscipline`
by means of the API function :meth:`~gemseo.api.create_discipline` with:

- :code:`discipline_name="AnalyticDiscipline"`,
- :code:`name="analytic"`,
- :code:`expressions=expr_dict`.

In practice, we write:

.. code::

    from gemseo.api import create_discipline

    disc = create_discipline("AnalyticDiscipline", name="analytic", expressions=expressions)

.. note::

   |g| takes care of the grammars and :meth:`!MDODiscipline._run` method generation from the :code:`expressions` argument.
   In the background, |g| considers that :code:`x` is a monodimensional float input parameter and :code:`y_1` and :code:`y_2` are monodimensional float output parameters.

Execute the discipline
**********************

Lastly, this discipline can be executed as any other:

.. code::

    from numpy import array
    input_data = {"x": array([2.0]), "z": array([3.0])}

    out = disc.execute(input_data)
    print("y_1 =", out["y_1"])
    print("y_2 =", out["y_2"])

which results in:

.. code::

   y_1 = [ 8.]
   y_2 = [ 44.]

About the analytic jacobian
***************************

The discipline will provide analytic derivatives (Jacobian) automatically using the `sympy library <https://www.sympy.org/fr/>`_,
by means of the :meth:`!AnalyticDiscipline._compute_jacobian` method.

This can be checked easily using :meth:`.MDODiscipline.check_jacobian`:

.. code::

    disc.check_jacobian(input_data,
                             derr_approx=disc.FINITE_DIFFERENCES,
                             step=1e-5, threshold=1e-3)

which results in:

.. code::

      INFO - 10:34:33 : Jacobian:  dp y_2/dp x succeeded!
      INFO - 10:34:33 : Jacobian:  dp y_2/dp z succeeded!
      INFO - 10:34:33 : Jacobian:  dp y_1/dp x succeeded!
      INFO - 10:34:33 : Jacobian:  dp y_1/dp z succeeded!
      INFO - 10:34:33 : Linearization of MDODiscipline: AnalyticDiscipline is correct !
   True
