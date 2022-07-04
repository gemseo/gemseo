..
   Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

   This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
   International License. To view a copy of this license, visit
   http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
   Commons, PO Box 1866, Mountain View, CA 94042, USA.

How to manually create a discipline interfacing an external executable?
***********************************************************************

.. _disciplineexecutable:

Presentation of the problem
~~~~~~~~~~~~~~~~~~~~~~~~~~~

Let's consider a binary software computing the float output :math:`c = a^2 + b^2` from two float inputs : :code:`'a'` and :code:`'b'`.

The inputs are read in the :file:`inputs.txt` file which looks like: ``a=1 b=2`` and the output is written to: :file:`outputs.txt` which looks like ``c=5``.

Then, the executable can be run using the shell command :file:`./run.ksh`:

.. code:: bash

    #!/bin/bash
    set -i

    echo "Parsed inputs.txt file"
    source "inputs.txt"
    echo "a="$a
    echo "b="$b

    echo "executing simulation..."
    c=$(perl -e "print $a*$a+$b*$b")

    echo "Done."
    echo "Computed output : c = a**2+b**2 = "$c
    echo "c="$c>"outputs.txt"

    echo "Wrote output file 'outputs.txt'"

Let's make a discipline out of this from an initial :file:`'inputs.txt'`.

Implementation of the discipline
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The construction of an :class:`.MDODiscipline` consists in three steps:

1. Instantiate the :class:`.MDODiscipline` using the super constructor,
2. Initialize the grammars using the :meth:`.JSONGrammar.update` method,
3. Set the default inputs from the initial :file:`inputs.txt`.

The :class:`!MDODiscipline._run` method consists in three steps:

1. Get the input data from :attr:`!MDODiscipline.local_data` and write the :file:`inputs.txt` file,
2. Run the executable using the :code:`os.system()` command (`https://docs.python.org/3/library/os.html#os.system`_),
3. Get the output values and store them to :attr:`!MDODiscipline.local_data`.

Now you can implement the discipline in the following way:

.. code:: python

    import os
    from gemseo.core.discipline import MDODiscipline

    class ShellExecutableDiscipline(MDODiscipline):

        def __init__(self):
            super(ShellExecutableDiscipline, self).__init__("ShellDisc")
            # Initialize the grammars
            self.input_grammar.update(['a','b'])
            self.output_grammar.update(['c'])
            # Initialize the default inputs
            self.default_inputs=parse_file("inputs.txt")

        def _run(self):
            # Write inputs.txt file
            write_file(self.local_data, 'inputs.txt')

            # Run the executable from the inputs
            os.system('./run.ksh')

            # Parse the outputs.txt file
            outputs = parse_file('outputs.txt')

            # Store the outputs
            self.local_data.update(outputs)

where :code:`parse_file()` and :code:`write_file()` functions are defined by:

.. code:: python

    from numpy import array

    def parse_file(file_path):
        data={}
        with open(file_path) as inf:
            for line in inf.readlines():
                if len(line)==0:
                    continue
                name,value=line.replace("\n","").split("=")
                data[name]=array([float(value)])

        return data

    def write_file(data, file_path):
        with open(file_path, "w") as outf:
            for name,value in data.iteritems():
                outf.write(name+"="+str(value[0])+"\n")

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
