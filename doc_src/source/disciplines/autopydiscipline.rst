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

Build a discipline from a simple python function
================================================

Let's consider a simple python function, e.g.:

.. code::

    def f(x=0., y=0.):
        """A simple python function"""
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

- :code:`py_jac=None`: pointer to the jacobian function which must returned a 2D numpy array,
- :code:`use_arrays=False`: if :code:`True`, the function is expected to take arrays as inputs and give outputs as arrays,
- :code:`write_schema=False`: if :code:`True`, write the json schema on the disk.

Here is an example of jacobian function:

.. code::

    def dfdxy(x=0., y=0.):
        """Jacobian function of f"""
        jac = array((2,1))
        jac[0, 0] = 1
        jac[1, 0] = 2
        return jac
