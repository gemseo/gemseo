..
   Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

   This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
   International License. To view a copy of this license, visit
   http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
   Commons, PO Box 1866, Mountain View, CA 94042, USA.

..
   Contributors:
          :author: Matthias De Lozzo

.. _nutshell_design_space:

How to deal with design spaces
==============================

1. How is a design space defined?
*********************************

1.a. What is a design space made of?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A design space is defined by:

- the number of the variables
- the names of the variables
- the sizes of the variables
- the upper bounds of the variables (optional; by default: :math:`-\infty`)
- the lower bounds of the variables (optional; by default: :math:`+\infty`)
- the normalization policies of the variables:

  - bounded float variables are normalizable,
  - bounded interger variables are normalizable,
  - unbounded float variables are not normalizable,
  - unbounded integer variables are not normalizable.

.. note::

   The normalized version of a given variable :math:`x` is either :math:`\frac{x-lb_x}{ub_x-lb_x}` or :math:`\frac{x}{ub_x-lb_x}`.

1.b. How is a design space implemented in |g|?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Design space are implemented in |g| through the :class:`.DesignSpace` class.

1.c. What are the API functions in |g|?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

A design space can be created from the :meth:`~gemseo.api.create_design_space` and :meth:`~gemseo.api.read_design_space` API functions and then, enhanced by methods of the :class:`.DesignSpace` class. It can be exported to a file by means of the :meth:`~gemseo.api.export_design_space`.

2. How to read a design space from a file?
******************************************

Case 1: the file contains a header line
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let's consider the *design_space.txt* file containing the following lines:

.. code::

    name lower_bound value upper_bound type
    x1 -1. 0. 1. float
    x2 5. 6. 8. float

We can read this file by means of the :meth:`~gemseo.api.read_design_space` function API:

.. code::

   from gemseo.api import read_design_space

   design_space = read_design_space('design_space.txt')

and print it:

.. code::

   print(design_space)

which results in:

.. code::

    Design Space:
    +------+-------------+-------+-------------+-------+
    | name | lower_bound | value | upper_bound | type  |
    +------+-------------+-------+-------------+-------+
    | x1   |      -1     |   0   |      1      | float |
    | x2   |      5      |   6   |      8      | float |
    +------+-------------+-------+-------------+-------+

Case 2: the file does not contain a header line
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now, let's consider the *design_space_without_header.txt* file containing the following lines:

.. code::

    x1 -1. 0. 1. float
    x2 5. 6. 8. float

We can read this file by means of the :meth:`~gemseo.api.read_design_space` API function
with the list of labels as optional argument:

.. code::

   from gemseo.api import read_design_space

   design_space = read_design_space(
       "design_space_without_header.txt",
       ["name", "lower_bound", "value", "upper_bound", "type"],
   )

and print it:

.. code::

   print(design_space)

which results in:

.. code::

    Design Space:
    +------+-------------+-------+-------------+-------+
    | name | lower_bound | value | upper_bound | type  |
    +------+-------------+-------+-------------+-------+
    | x1   |      -1     |   0   |      1      | float |
    | x2   |      5      |   6   |      8      | float |
    +------+-------------+-------+-------------+-------+

.. warning::

   - User must provide the following minimal fields in the file defining the design space: :code:`'name'`, :code:`'lower_bound'` and :code:`'upper_bound'`.
   - The inequality :code:`'lower_bound'` <= :code:`'name'` <= :code:`'upper_bound'` must be satisfied.

.. note::

   - Available fields are :code:`'name'`, :code:`'lower_bound'`, :code:`'upper_bound'`, :code:`'value'` and :code:`'type'`.
   - The :code:`'value'` field is optional. By default, it is set at :code:`None`.
   - The :code:`'type'` field is optional. By default, it is set at :code:`float`.
   - Each dimension of a variable must be provided. E.g. when the :code:`'size'` of :code:`'x1'` is 2:

     .. code::

        name lower_bound value upper_bound type
        x1 -1. 0. 1. float
        x1 -3. -1. 1. float
        x2 5. 6. 8. float

.. note::

   - Lower infinite bound is encoded :code:`-inf'` or :code:`'-Inf'`.
   - Upper infinite bound is encoded :code:`'inf'`, :code:`'Inf'`, :code:`'+inf'` or :code:`'+Inf'`.

3. How to create a design space from scratch?
*********************************************

Let's imagine that we want to build a design space with the following requirements:

- *x1* is a one-dimensional unbounded float variable,
- *x2* is a one-dimensional unbounded integer variable,
- *x3* is a two-dimensional unbounded float variable,
- *x4* is a one-dimensional float variable with lower bound equal to 1,
- *x5* is a one-dimensional float variable with upper bound equal to 1,
- *x6* is a one-dimensional unbounded float variable,
- *x7* is a two-dimensional bounded integer variable with lower bound equal to -1, upper bound equal to 1 and current values to (0,1),

We can create this design space from scratch by means of the :meth:`~gemseo.api.create_design_space` API function and the :meth:`.DesignSpace.add_variable` method of the :class:`.DesignSpace` class:

.. code::

    from gemseo.api import create_design_space
    from numpy import ones, array

    design_space = create_design_space()
    design_space.add_variable('x1')
    design_space.add_variable('x2', var_type='integer')
    design_space.add_variable('x3', size=2)
    design_space.add_variable('x4', l_b=ones(1))
    design_space.add_variable('x5', u_b=ones(1))
    design_space.add_variable('x6', value=ones(1))
    design_space.add_variable(
        "x7", size=2, var_type="integer", value=array([0, 1]), l_b=-ones(2), u_b=ones(2)
    )

and print it:

.. code::

    print(design_space)

which results in:

.. code::

    Design Space:
    +------+-------------+-------+-------------+---------+
    | name | lower_bound | value | upper_bound | type    |
    +------+-------------+-------+-------------+---------+
    | x1   |     -inf    |  None |     inf     | float   |
    | x2   |     -inf    |  None |     inf     | integer |
    | x3   |     -inf    |  None |     inf     | float   |
    | x3   |     -inf    |  None |     inf     | float   |
    | x4   |      1      |  None |     inf     | float   |
    | x5   |     -inf    |  None |      1      | float   |
    | x6   |     -inf    |   1   |     inf     | float   |
    | x7   |      -1     |   0   |      1      | integer |
    | x7   |      -1     |   1   |      1      | integer |
    +------+-------------+-------+-------------+---------+

.. note::

   For a variable whose :code:`'size'` is greater than 1, each dimension of this variable is printed (e.g. :code:`'x3'` and :code:`'x7'`).

.. note::

   We can get a list of the variable names with theirs indices by means of the :meth:`.DesignSpace.get_indexed_variables_names` method:

   .. code::

      indexed_variables_names = design_space.get_indexed_variables_names()

   and :code:`print(indexed_variables_names)`:

   .. code::

      ['x1', 'x2', 'x3!0', 'x3!1', 'x4', 'x5', 'x6', 'x7!0', 'x7!1']

   We see that the multidimensional variables have an index (here :code:`'0'` and :code:`'1'`) preceded by a :code:`'!'` separator.

4. How to get information about the design space?
*************************************************

How to get the size of a design variable?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We can get the size of a variable by means of the :meth:`.DesignSpace.get_size` method:

.. code::

   x3_size = design_space.get_size('x3')

and :code:`print(x3_size)` to see the result:

.. code::

   1

How to get the type of a design variable?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We can get the type of a variable by means of the :meth:`.DesignSpace.get_type` method:

.. code::

   x3_type = design_space.get_type('x3')

and :code:`print(x3_type)` to see the result:

.. code::

   ['float']

How to get the size of a lower or upper bound for a given variable?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We can get the lower and upper bounds of a variable by means of the :meth:`.DesignSpace.get_lower_bound` and :meth:`.DesignSpace.get_upper_bound` methods:

.. code::

   x3_lb = design_space.get_lower_bound('x3')
   x3_ub = design_space.get_upper_bound('x3')

and :code:`print(x3_lb, x3_ub)` to see the result:

.. code::

   [-10.], [ 10.]

How to get the size of a lower or upper bound for a set of given variables?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We can get the lower and upper bounds of a set of variables by means of the :meth:`.DesignSpace.get_lower_bounds` and :meth:`.DesignSpace.get_upper_bounds` methods:

.. code::

    x1x3_lb = design_space.get_lower_bounds(['x1', 'x3'])
    x1x3_ub = design_space.get_upper_bounds(['x1', 'x3'])

and :code:`print(x1x3_lb, x1x3_ub)` to see the result:

.. code::

   [-10. -10.], [ 10. 10.]

How to get the current array value of the design parameter vector?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We can get the current value of the design parameters by means of the :meth:`.DesignSpace.get_lower_bounds` and :meth:`.DesignSpace.get_current_x` method:

.. code::

   current_x = design_space.get_current_x

and :code:`print(current_x)` to see the result:

.. code::

   [ 3.  1.  1.  1.]

How to get the current dictionary value of the design parameter vector?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We can get the current value of the design parameters with :code:`dict` format by means of the :meth:`.DesignSpace.get_lower_bounds` and :meth:`.DesignSpace.get_current_x_dict` method:

.. code::

   dict_current_x = design_space.get_current_x_dict

and :code:`print(dict_current_x)` to see the result:

.. code::

   {'x2': array([1.]), 'x3': array([1.]), 'x1': array([3.]), 'x6': array([1.])}

How to get the normalized array value of the design parameter vector?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We can get the normalized current value of the design parameters by means of the :meth:`.DesignSpace.get_lower_bounds` and :meth:`.DesignSpace.get_current_x_normalized` method:

.. code::

   normalized_current_x = design_space.get_current_x_normalized

and :code:`print(normalized_current_x)` to see the result:

.. code::

   [ 0.65  1.    0.55  0.55]

How to get the active bounds at the current design parameter vector or at a given one?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We can get the active bounds by means of the :meth:`.DesignSpace.get_lower_bounds` and :meth:`.DesignSpace.get_active_bounds` method, either at current parameter values:

.. code::

   active_at_current_x = design_space.get_active_bounds()

and :code:`print(active_at_current_x)` to see the result:

.. code::

   ({'x2': array([False], dtype=bool), 'x3': array([False], dtype=bool), 'x1': array([False], dtype=bool), 'x6': array([False], dtype=bool)}, {'x2': array([False], dtype=bool), 'x3': array([False], dtype=bool), 'x1': array([False], dtype=bool), 'x6': array([False], dtype=bool)})

or at a given point:

.. code::

   active_at_given_point = design_space.get_active_bounds(array([1., 10, 1., 1.]))

and :code:`print(active_at_given_point)` to see the result:

.. code..

   ({'x2': array([False], dtype=bool), 'x3': array([False], dtype=bool), 'x1': array([False], dtype=bool), 'x6': array([False], dtype=bool)}, {'x2': array([ True], dtype=bool), 'x3': array([False], dtype=bool), 'x1': array([False], dtype=bool), 'x6': array([False], dtype=bool)})

5. How to modify a design space?
********************************

How to remove a variable from a design space?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let's consider the previous design space and assume that we want to remove the :code:`'x4'` variable.

For that, we can use the :meth:`.DesignSpace.remove_variable` method:

.. code::

   design_space.remove_variable('x4')

and :code:`print(design_space)` to see the result:

.. code::

    Design Space:
    +------+-------------+-------+-------------+---------+
    | name | lower_bound | value | upper_bound | type    |
    +------+-------------+-------+-------------+---------+
    | x1   |     -inf    |  None |     inf     | float   |
    | x2   |     -inf    |  None |     inf     | integer |
    | x3   |     -inf    |  None |     inf     | float   |
    | x3   |     -inf    |  None |     inf     | float   |
    | x5   |     -inf    |  None |      1      | float   |
    | x6   |     -inf    |   1   |     inf     | float   |
    | x7   |      -1     |   0   |      1      | integer |
    | x7   |      -1     |   1   |      1      | integer |
    +------+-------------+-------+-------------+---------+

How to filter the entries of a design space?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We can keep only a subset of variables, e.g. :code:`'x1'`, :code:`'x2'`, :code:`'x3'` and :code:`'x6'`, by means of the :meth:`gemseo.algos.design_space.DesignSpace.filter` method:

.. code::

   design_space.filter(['x1', 'x2', 'x3', 'x6']) # keep the x1, x2, x3 and x6 variables

and :code:`print(design_space)` to see the result:

.. code::

    Design Space:
    +------+-------------+-------+-------------+---------+
    | name | lower_bound | value | upper_bound | type    |
    +------+-------------+-------+-------------+---------+
    | x1   |     -inf    |  None |     inf     | float   |
    | x2   |     -inf    |  None |     inf     | integer |
    | x3   |     -inf    |  None |     inf     | float   |
    | x3   |     -inf    |  None |     inf     | float   |
    | x6   |     -inf    |   1   |     inf     | float   |
    +------+-------------+-------+-------------+---------+

We can also keep only a subset of components for a given variable, e.g. the first component of the variable :code:`'x3'`,  by means of the :meth:`gemseo.algos.design_space.DesignSpace.filter_dim` method:

.. code::

   design_space.filer_dim('x3', [0]) # keep the first dimension of x3

and :code:`print(design_space)` to see the result:

.. code::

    Design Space:
    +------+-------------+-------+-------------+---------+
    | name | lower_bound | value | upper_bound | type    |
    +------+-------------+-------+-------------+---------+
    | x1   |     -inf    |  None |     inf     | float   |
    | x2   |     -inf    |  None |     inf     | integer |
    | x3   |     -inf    |  None |     inf     | float   |
    | x6   |     -inf    |   1   |     inf     | float   |
    +------+-------------+-------+-------------+---------+

How to modify the data values contained in a design space?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We can change the current values and bounds contained in a design space by means of the :meth:`.DesignSpace.set_current_x`, :meth:`.DesignSpace.set_current_variable`, :meth:`.DesignSpace.set_lower_bound` and :meth:`.DesignSpace.set_upper_bound` methods:

.. code::

    design_space.set_current_x(array([1., 1., 1., 1.]))
    design_space.set_current_variable('x1', array([3.]))
    design_space.set_lower_bound('x1', array([-10.]))
    design_space.set_lower_bound('x2', array([-10.]))
    design_space.set_lower_bound('x3', array([-10.]))
    design_space.set_lower_bound('x6', array([-10.]))
    design_space.set_upper_bound('x1', array([10.]))
    design_space.set_upper_bound('x2', array([10.]))
    design_space.set_upper_bound('x3', array([10.]))
    design_space.set_upper_bound('x6', array([10.]))

and :code:`print(design_space)` to see the result:

.. code::

    Design Space:
    +------+-------------+-------+-------------+---------+
    | name | lower_bound | value | upper_bound | type    |
    +------+-------------+-------+-------------+---------+
    | x1   |     -10     |   3   |      10     | float   |
    | x2   |     -10     |   1   |      10     | integer |
    | x3   |     -10     |   1   |      10     | float   |
    | x6   |     -10     |   1   |      10     | float   |
    +------+-------------+-------+-------------+---------+

6. How to (un)normalize a parameter vector?
*******************************************

Let's consider the parameter vector :code:`x_vect = array([1.,10.,1.,1.])`. We can normalize this vector by means of the :meth:`.DesignSpace.normalize_vect`:

.. code::

   normalized_x_vect = design_space.normalize_vect(x_vect)

and :code:`print(normalized_x_vect)`:

.. code::

   [  0.55  1.     0.55   0.55]

Conversely, we can unnormalize this normalized vector by means of the :meth:`.DesignSpace.unnormalize_vect`:

.. code::

   unnormalized_x_vect = design_space.unnormalize_vect(x_vect)

.. code::

   [  1.  10.   1.   1.]

.. note::

   Both methods takes an optional argument denoted :code:`'minus_lb'` which is :code:`True` by default. If :code:`True`, the normalization of the normalizable variables is of the form :code:`(x-lb_x)/(ub_x-lb_x)`. Otherwise, it is of the form :code:`x/(ub_x-lb_x)`. Here, when :code:`minus_lb` is :code:`False`, the normalize parameter vector is:

   .. code::

      [  0.05  0.5  0.05  0.05]

7. How to cast design data?
***************************

How to cast a design point from array to dict?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We can cast a design point from :code:`array` to :code:`dict` by means of the :meth:`.DesignSpace.array_to_dict` method:

.. code::

    array_point = array([1, 2, 3, 4])
    dict_point = design_space.array_to_dict(array_point)

and :code:`print(dict_point)` to see the result:

.. code::

   {'x2': array([2]), 'x3': array([3]), 'x1': array([1]), 'x6': array([4])}

How to cast a design point from dict to array?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We can cast a design point from :code:`dict` to :code:`array` by means of the :meth:`.DesignSpace.dict_to_array` method:

.. code::

   new_array_point = design_space.dict_to_array(dict_point)

and :code:`print(new_array_point)` to see the result:

.. code::

   [1, 2, 3, 4]

.. note::

   - An optional argument denoted :code:`'all_vars'`, which is a boolean and set at :code:`True` by default, indicates if all design variables shall be in the :code:`dict` passed in argument.
   - An optional argument denoted :code:`'all_var_list'`, which is a list of string and set at :code:`None` by default, list all of the variables to consider. If :code:`None`, all design variables are considerd.

How to cast the current value to complex?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We can cast the current value to complex by means of the :meth:`.DesignSpace.to_complex` method:

.. code::

   print(design_space.get_current_x())
   design_space.to_complex()
   print(design_space.get_current_x())

and the successive printed messages are:

.. code::

   [ 3.  1.  1.  1.]
   [ 3.+0.j  1.+0.j  1.+0.j  1.+0.j]

How to cast the right component values of a vector to integer?
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For a given vector where some components should be integer, it is possible to round them by means of the :meth:`.DesignSpace.round_vect` method:

.. code::

   vector = array([1.3, 3.4,3.6, -1.4])
   rounded_vector =  design_space.round_vect(vector)

and :code:`print(rounded_vector)` to see the result:

.. code::

   [ 1.3  3.   3.6 -1.4]

8. How to test if the current value is defined?
***********************************************

We can test if the design parameter set has a current :code:`'value'` by means of the :meth:`.DesignSpace.has_current_x`:

.. code::

   print(design_space.has_current_x())

which results in:

.. code::

   True

.. note::

   The result returned by :meth:`.DesignSpace.has_current_x` is :code:`False` as long as at least one component of one variable is :code:`None`.

9. How to project a point into bounds?
**************************************

Sometimes, components of a design vector are greater than the upper bounds or lower than the upper bounds. For that, it is possible to project the vector into the bounds by means of the :meth:`.DesignSpace.project_into_bounds`:

.. code::

   point = array([1.,3,-15.,23.])
   p_point = design_space.project_into_bounds(point)

and :code:`print(p_point)` to see the result:

.. code::

   [  1.   3. -10.  10.]

10. How to export a design space to a file?
*******************************************

When the design space is created, it is possible to export it by means of the :meth:`~gemseo.api.export_design_space` API function with arguments:

- :code:`design_space`: design space
- :code:`output_file`: output file path
- :code:`export_hdf`: export to a hdf file (True, default) or a txt file (False)
- :code:`fields`: list of fields to export, by default all
- :code:`append`: if :code:`True`, appends the data in the file
- :code:`table_options`: dictionary of options for the :class:`~gemseo.third_party.prettytable.prettytable.PrettyTable`

For example:

.. code::

   from gemseo.api import export_design_space

   export_design_space(design_space, 'new_design_space.txt')
