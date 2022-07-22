..
    Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

    This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
    International License. To view a copy of this license, visit
    http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
    Commons, PO Box 1866, Mountain View, CA 94042, USA.

    Contributors:
          :author: Arthur Piat, François Gallard, Nicolas Roussouly

.. _discipline_matlab:

Build discipline from a MATLAB function
***************************************

As explained in :ref:`software_connection`, |g| can interface any simulation software through
the :class:`.MDODiscipline` class.
When dealing with a MATLAB program given as a function,
a generic interface is
created in order to facilitate this task.
It is described here.

We first start the explanation with a simple example which handles scalar inputs and outputs.
Then, we explain how to use the vector inputs and outputs as well as the ability to
compute and return output Jacobian matrices.

.. note::

    Building and executing a MATLAB discipline requires that a MATLAB
    engine as well as its Python API are installed.
    To make sure that MATLAB works fine through the Python API,
    start a Python interpreter and
    check that there is no error when executing :code:`import matlab`.
    See :ref:`matlab_requirements` for more information.

.. warning::

   MATLAB disciplines cannot be used with Python multiprocessing.


A first simple example with scalar inputs and outputs
=====================================================

Assume that we have a MATLAB file :code:`simple_scalar_func.m` that contains
the following function definition:

.. code::

    function [z1, z2] = simple_scalar_func(x, y)
    z1 = x^2;
    z2 = 3*cos(y);
    end

.. note::

    It is reminded that in MATLAB, the ``.m`` file must have the same
    name than the function, i.e. ``simple_scalar_func`` here.
    |g| raises an error if it is not the case.


Create the discipline instance
------------------------------

A very simple and convenient way that enables to build a discipline from
the previous function is to use the |g| API:

.. code::

    from gemseo.api import create_discipline

    disc = create_discipline("MatlabDiscipline",
                             matlab_fct="simple_scalar_func.m")

where the first parameter allows to select which kind of discipline one
wants to build and the second is the name of the MATLAB function that must be wrapped.

Execute the discipline
----------------------

Executing the previous MATLAB discipline is also straightforward:

.. code::

    from numpy import array

    output = disc.execute({"x": array([2]), "y": array([0.])})
    print(output)

which gives:

.. code::

    {'x': array([2.]), 'y': array([0.]), 'z1': array([4.]), 'z2': array([3.])}


Handling input and output vectors
=================================

If the discipline involves any vector as input and/or output, it is quite the same
as the previous example but one have to be careful with sizes consistency.

Assume for example that the following MATLAB function is defined in file
``simple_vector_func.m``:

.. code::

    function [z1, z2] = simple_vector_func(x, y)
    z1(1) = x(1)^2;
    z1(2) = 2*x(2);
    z2 = 3*cos(y);
    end

Thus, inputs must match the right size when executing the discipline:

.. code::

    from gemseo.api import create_discipline

    disc_vec = create_discipline("MatlabDiscipline",
                                 matlab_fct="simple_vector_func.m")

    output = disc_vec.execute({"x": array([2, 3]), "y": array([0.])})

    print(output)

and the result is:

.. code::

    {'x': array([2., 3.]), 'y': array([0.]), 'z1': array([4., 6.]), 'z2': array([3.])}

.. note::

    If the discipline is executed with inputs that have the wrong size, an error is raised.

.. note::

    It is reminded that in MATLAB, vector indices start from 1, not from 0 as in Python.


Returning Jacobian matrices
===========================

For gradient-based optimization, it is usually convenient to get access to gradients.
If gradients are computed inside the MATLAB function, the |g| discipline can take them into
account: they just need to be returned properly.

.. note::

    Currently, the computation of gradients must be in the same MATLAB function as
    the function itself.

More generally, if the basis function takes an input vector :math:`\bf{x}` and returns an
output vector :math:`\bf{y}`, the total derivatives denoted
:math:`\frac{d\bf{f}}{d\bf{x}}` is called the Jacobian matrix as explained in
:ref:`jacobian_assembly`.

If Jacobian matrices are returned by the MATLAB function, the |g| discipline can take
them into account by prescribing the argument :code:`is_jac_returned_by_func=True`.

Let's take a simple example and assume that the MATLAB file
``jac_fun.m`` contains the following function:

.. code::

    function [ysca, yvec, jac_dysca_dxsca, jac_dysca_dxvec, jac_dyvec_dxsca, jac_dyvec_dxvec] = jac_func(xsca, xvec)

    ysca = xsca + 2*xvec(1) + 3*xvec(2);

    yvec(1) = 4*xsca + 5*xvec(1) + 6*xvec(2);
    yvec(2) = 7*xsca + 8*xvec(1) + 9*xvec(2);

    jac_dysca_dxsca = 4;

    jac_dysca_dxvec = [2, 3];

    jac_dyvec_dxsca = [4; 7];

    jac_dyvec_dxvec = [[5, 6]; [8, 9]];

    end

Create the discipline instance
------------------------------

Building the discipline is still very simple using the API, we just need to add
the boolean argument :code:`is_jac_returned_by_func` in this case:

.. code::

    from gemseo.api import create_discipline

    disc = create_discipline("MatlabDiscipline",
                             matlab_fct="jac_func.m",
                             is_jac_returned_by_func=True)


Executing the discipline
------------------------

We can execute the discipline in the same way as previously:

.. code::

   output = disc.execute({"xsca": array([1]), "xvec": array([2, 3])})

which gives:

.. code::

    {'xsca': array([1.]), 'xvec': array([2., 3.]), 'ysca': array([14.]), 'yvec': array([32., 50.])}

One can see that the Jacobian outputs are not included in the returned values.
Since the argument ``is_jac_returned_by_func`` has been activated, the Jacobian matrices
values are stored in the :attr:`.MDODiscipline.jac` attributes.
Thus printing
:attr:`.MDODiscipline.jac` in a pretty way gives:

.. code::

    Out: ysca / In: xsca
    [[4.]]

    Out: ysca / In: xvec
    [[2. 3.]]

    Out: yvec / In: xsca
    [[4.]
    [7.]]

    Out: yvec / In: xvec
    [[5. 6.]
    [8. 9.]]


Naming convention
-----------------

As one can see, the Jacobian matrices must be added to the outputs in order to be
returned by the MATLAB function.
These outputs must follow a naming convention:
**assuming an input** ``x`` **and output** ``y``, **the corresponding Jacobian must be returned
as** ``jac_dy_dx``.


Jacobian matrix dimension
-------------------------

As explained in the section :ref:`discipline_compute_jacobian`, |g| always manipulates
the Jacobian terms inside 2D arrays even if the Jacobian is reduced to
a scalar value, row-vector or column-vector values.

In order to be consistent with the Jacobian definition, the Jacobian output returned
by the MATLAB function must have the right dimension:

* it is a **scalar** if ``y`` is a scalar and ``x`` is a scalar;
* it is a **row vector** if ``y`` is a scalar and ``x`` is a vector;
* it is a **column vector** if ``y`` is a vector and ``x`` is a scalar;
* it is a **matrix** if ``y`` is a vector and ``x`` is a vector.


Some important optional arguments
=================================

Many others optional parameters can be added when building a MATLAB discipline.
They are all listed in the description of :class:`.MatlabDiscipline` but we give some
information here about the most important ones.

Files location: ``search_file``
-------------------------------

In the previous simple examples, we assumed that the MATLAB ``.m`` file
is located in the current working directory where |g| is executed.

When dealing with more complex programs that have specific location which
could not be changed and/or that contains several files, it is more convenient
to give a directory where the MATLAB function is looked for.

The root directory where a MATLAB function is searched can be prescribed with
the argument ``search_file`` and if the argument ``add_subfold_path`` is set to
``True`` then all the sub-directories will be added to the MATLAB search paths.
An example is:

.. code::

    from gemseo.api import create_discipline

    disc = create_discipline("MatlabDiscipline",
                             matlab_fct="simple_scalar_func.m",
                             search_file="matlab_files",
                             add_subfold_path=True)


Initialize data from a MATLAB file: ``matlab_data_file``
--------------------------------------------------------

It is possible to initialize the input and/or output values of the discipline
from a MATLAB data file with the ``.mat`` extension.
The ``.mat`` file can be passed to the |g| API through the ``matlab_data_file``
argument.
Any input and/or output variables found in this file will be initialized
with the provided value.
An example is:

.. code::

    from gemseo.api import create_discipline

    disc = create_discipline("MatlabDiscipline",
                             matlab_fct="simple_scalar_func.m",
                             matlab_data_file="data_file.mat")


Aliasing input and output names
-------------------------------

The arguments ``input_names`` and ``output_names`` enable to change
the name of the input and/or output variables when using the discipline.
As an example, in the previous simple scalar case, the inputs and outputs are respectively
denoted ``x``, ``y``, ``z1`` and ``z2`` in the MATLAB function:

.. code::

    from gemseo.api import create_discipline

    disc = create_discipline(
        "MatlabDiscipline",
        matlab_fct="simple_scalar_func.m",
        input_names=["in1, in2"],
        output_names=["out1, out2"]
    )

    from numpy import array

    disc.execute({"in1": array([2]), "in2": array([0])})

which gives the following result:

.. code::

    {'in1': array([2.]), 'in2': array([0.]), 'out1': array([4.]), 'out2': array([3.])}


Engine name: ``matlab_engine_name``
-----------------------------------

.. note::

    The current section is mostly for advanced users
    and should not be considered for simple applications.

When building a MATLAB discipline, the MATLAB Python API launches
a MATLAB workspace that will be used in order to execute
the MATLAB function that is wrapped.
MATLAB workspace handling is done through the :class:`.MatlabEngine` class.
Since this class is private, it cannot be imported directly form the module.
An instance of this class is rather obtained through
the function :func:`.get_matlab_engine` which acts like a singleton.
This means that calling :func:`.get_matlab_engine` with the same input argument
(the workspace name), returns exactly the same instance.
Therefore, if one builds two disciplines, they will be executed
in a unique MATLAB workspace.
This is indeed what a MATLAB user do when working
with MATLAB: run MATLAB once and execute any function inside the same environment.

The uniqueness of the :class:`.MatlabEngine` instance depends
more precisely on the workspace name that is passed to the function :func:`.get_matlab_engine`:
when getting two engines, if the names are the same then the instance is unique, otherwise they are not.
Let's see the following simple example with three engines, two based on the same name and
the third based on a different one:

.. code::

    from gemseo.wrappers.matlab.engine.engine import get_matlab_engine

    eng1 = get_matlab_engine("workspace_1")
    eng2 = get_matlab_engine("workspace_1")
    eng3 = get_matlab_engine("workspace_2")

Checking that :code:`eng1 is eng2` equals :code:`True` whereas
:code:`eng1 is eng3` equals :code:`False`.

This ``workspace_name`` string that is passed to the :func:`.get_matlab_engine` can be controlled
with the argument ``matlab_engine_name`` when building the MATLAB discipline from
|g| API.
By default, this argument is set to ``"matlab"`` and should not be changed except
for very specific use.
