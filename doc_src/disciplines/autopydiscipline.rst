..
   Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

   This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
   International License. To view a copy of this license, visit
   http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
   Commons, PO Box 1866, Mountain View, CA 94042, USA.

..
   Contributors:
          :author: Matthias De Lozzo

.. _autopydiscipline:

Build a discipline from a simple Python function
================================================

Let's consider a simple Python function, e.g.:

.. code::

    def f(x=0., y=0.):
        """A simple Python function"""
        z = x + 2*y
        return z

Then, we can consider the :class:`.AutoPyDiscipline` to convert it into an :class:`.MDODiscipline`.

Create and instantiate the discipline
*************************************

For that, we can use the :meth:`~gemseo.api.create_discipline` API function with :code:`AutoPyDiscipline` as first argument:

.. code::

    from gemseo.api import create_discipline
    from numpy import array

    disc = create_discipline('AutoPyDiscipline', py_func=f)

The original Python function may or may not include default values for input arguments, however, if the resulting
:class:`.AutoPyDiscipline` is going to be placed inside an :class:`.MDF`, a :class:`.BiLevel` formulation
or an :class:`.MDA` with strong couplings, then the Python function **must** assign default values for its input
arguments.

Execute the discipline
**********************

Then, we can execute it easily, even considering default inputs:

.. code::

    print(disc.execute())

which results in:

.. code::

    {'y': array([ 0.]), 'x': array([ 0.]), 'z': array([ 0.])}

or using new inputs:

.. code::

    print(disc.execute({'x': array([1.]), 'y':array([-3.2])}))

which results in:

.. code::

    {'y': array([-3.2]), 'x': array([ 1.]), 'z': array([-5.4])}

Optional arguments
******************

Optional arguments are:

- :code:`py_jac=None`: The Python function to compute the Jacobian which must return a 2D numpy array,
- :code:`use_arrays=False`: if :code:`True`, the function is expected to take arrays as inputs and give outputs as arrays,
- :code:`grammar_type=MDODiscipline.JSON_GRAMMAR_TYPE`: The type of grammar to be used.

Here is an example of Jacobian function, returning a 2D matrix.
The rows of the matrix correspond to the derivatives of the outputs,
the columns correspond to the variables with respect to the outputs are derived.

.. code::

    def dzdxy(x=0., y=0.):
        """Jacobian function of z=f(x,y)"""
        jac = array((1,2))
        jac[0, 0] = 1.
        jac[0, 1] = 2.
        return jac
