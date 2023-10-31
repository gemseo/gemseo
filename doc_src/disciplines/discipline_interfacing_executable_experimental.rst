..
   Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

   This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
   International License. To view a copy of this license, visit
   http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
   Commons, PO Box 1866, Mountain View, CA 94042, USA.

How to manually create a discipline interfacing an external executable (bis)?
*****************************************************************************

.. warning::
    This is an experimental method which uses on-development code.
    Both the API and the behavior may change over time.
    To use with caution.


.. _disciplineexecutable:

Description of the discipline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

We consider an executable ``my_exe`` which can be run with the command line: ``my_exe input_file output_file``.
This page explains how to deal with such an executable, and how to use it as a Discipline.
The ``input_file`` and the ``output_file`` can be formatted in different ways.
We assume it is possible to create the input files and to parse the output files.

A discipline based on an executable is mainly driven by 4 different steps,
defined in the :class:`._BaseDiscFromExe._run` method:
1. Create a unique directory to store the files required by the executable,
2. Write the input files in this directory,
3. Run the executable with the command line,
4. Parse the output files.

Unique directory
----------------

The communication between the executable and its discipline is done via input and output files.
These files may become a source of errors,
since they can be modified by another program while being in use.
To avoid such inconveniences, a unique directory can be created for each run.
This is also compliant with multi-processing features.

The unique directory is set and defined within the :class:`._BaseExecutableRunner`,
that will be detailed below.

Create the inputs
-----------------

This is how the executable obtains inputs from its discipline.
To do so, the input files shall be created from the input data of the discipline.

Run the executable
------------------

The command line for the executable is run using the :class:`._BaseExecutableRunner` class.

The directory wherein the executable is run is defined in this class.
One can choose where and how these directories are created.

Some files might be needed before each execution,
the user can specify these files to be copied within the execution directory.

Parse the outputs
-----------------

The execution should create files which contains the output data to be passed to the discipline.
Those output data are parsed during this step.


A simple example
~~~~~~~~~~~~~~~~

With the global architecture in mind,
let's consider a simple example to illustrate this.

Presentation
------------

An executable :file:`run_discipline.bash`, shown below, is considered.
It computes the float output :math:`c = a^2 + b^2`
from two float inputs : :code:`'a'` and :code:`'b'`.
This executable must use a :file:`inputs.txt` file which looks like:

.. code:: bash
    a=1
    b=2

and the output is written to: :file:`outputs.txt` which looks like ``c=5``.
The executable is run using the shell command :file:`./run_discipline.bash`.
The following script is considered as
the executable :file:`./run_discipline.bash` described in the example:

.. literalinclude:: run_discipline.bash
   :language: bash

Let's make an executable discipline out of this.

Implementation of the discipline
--------------------------------

The construction of this discipline consists in different steps:

1. Instantiate the discipline
    1.1. Instantiate the :class:`._BaseDiscFromExe` using the super constructor,
    1.2. Manage default inputs/outputs within the constructor,
    1.3. Create the :class:`._executable_runner`
2. Implement :meth:`define_inputs` to write inputs into the input file, using the specific format,
3. Implement :meth:`parse_outputs` to read outputs from the formatted file,

The :class:`!MDODiscipline._run` method should not be modified.

For this example, the discipline can be implemented in the following way:

.. literalinclude:: discipline_interfacing_executable_experimental.py
   :language: python

Execution of the discipline
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Now we can run it with default input values:

.. code::

    shell_disc = ShellExecutableDiscipline()
    print(shell_disc.execute())

which results in:

.. parsed-literal::

    Inputs =  {'a': array([ 1.]), 'b': array([ 2.])}
    Running executable
    Outputs =  {'c': array([ 5.])}
    {'a': array([ 1.]), 'c': array([ 5.]), 'b': array([ 2.])}

or run it with new input values:

.. code::

    print(shell_disc.execute({'a': array([2.]), 'b': array([3.])}))

which results in:

.. parsed-literal::

    Inputs =  {'a': array([ 2.]), 'b': array([ 3.])}
    Running executable
    Outputs =  {'c': array([ 13.])}
    {'a': array([ 2.]), 'c': array([ 13.]), 'b': array([ 3.])}
