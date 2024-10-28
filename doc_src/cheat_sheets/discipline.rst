..
   Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

   This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
   International License. To view a copy of this license, visit
   http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
   Commons, PO Box 1866, Mountain View, CA 94042, USA.

==========
Discipline
==========

.. code-block:: python

    from gemseo import create_discipline

Instantiate a discipline from an internal or external module:

.. code-block:: python
   :caption: Instantiate a discipline from an internal or external module

    discipline = create_discipline("Sellar1")

Create a discipline from a Python function:

.. code-block:: python

    def py_func(x=0.0, y=0.0):
        z = x + 2 * y
        return z


    discipline = create_discipline("AutoPyDiscipline", py_func=py_func)

Create an analytic discipline from a dictionary of expressions:

.. code-block:: python

    expressions = {"y_1": "2*x**2", "y_2": "5+3*x**2+z**3"}
    discipline = create_discipline(
        "AnalyticDiscipline", name="my_func", expressions=expressions
    )
