..
   Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

   This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
   International License. To view a copy of this license, visit
   http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
   Commons, PO Box 1866, Mountain View, CA 94042, USA.

..
   Contributors:
          :author: Matthias De Lozzo

.. _disciplinefromscratch:

How to create a discipline from scratch?
****************************************

Creating a discipline from scratch implies to implement a new class inheriting from :class:`.MDODiscipline`.

For example, let's consider a discipline called ``NewDiscipline``,
with two outputs,
``f`` and ``g``,
and two inputs,
``x`` and ``z``,
where ``f=x*z`` and ``f=x*(z+1)^2``.

Overloading the :class:`.MDODiscipline`'s constructor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

First of all, we overload the :class:`.MDODiscipline` constructor.
For that,
we call the :class:`.MDODiscipline` superconstructor:

.. code::

    from gemseo.api import MDODiscipline

    class NewDiscipline(MDODiscipline):

        def __init__(self):
            super(NewDiscipline, self).__init__()
            # TO BE COMPLETED

Setting the input and output grammars
-------------------------------------

Then, we define the :attr:`!MDODiscipline.input_grammar`
and :attr:`!MDODiscipline.output_grammar` created by the superconstructor with :code:`None` value.
We have different ways to do that.

Setting the grammars from data names
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When the variables are float arrays without any particular constraint,
the simplest approach is to apply the :meth:`.JSONGrammar.update` method to a list of variable names:

.. code::

    from gemseo.api import MDODiscipline

    class NewDiscipline(MDODiscipline):

        def __init__(self):
            super(NewDiscipline, self).__init__()
            self.input_grammar.update(['x', 'z'])
            self.output_grammar.update(['f', 'g'])
            # TO BE COMPLETED

Setting the grammars from JSON files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A more complicated approach is to define the grammar into JSON input and output files
with name :code:`'NewDiscipline_inputs.json'` and :code:`'NewDiscipline_outputs.json'`,
put these files in the same directory as the module implementing the :code:`NewDiscipline` and
pass an optional argument to the superconstructor:

.. code::

    from gemseo.api import MDODiscipline

    class NewDiscipline(MDODiscipline):

        def __init__(self):
            super(NewDiscipline, self).__init__(auto_detect_grammar_files=True)
            # TO BE COMPLETED

where the :code:`'NewDiscipline_inputs.json'` file is defined as follows:

.. parsed-literal::

    {
        "name": "NewDiscipline_input",
        "required": ["x","z"],
        "properties": {
            "x": {
                "items": {
                    "type": "number"
                },
                "type": "array"
            },
            "z": {
                "items": {
                    "type": "number"
                },
                "type": "array"
            }
        },
        "$schema": "http://json-schema.org/draft-04/schema",
        "type": "object",
        "id": "#NewDiscipline_input"
    }

and where the :code:`'NewDiscipline_outputs.json'` file is defined as follows:

.. parsed-literal::

    {
        "name": "NewDiscipline_output",
        "required": ["y1","y2"],
        "properties": {
            "y1": {
                "items": {
                    "type": "number"
                },
                "type": "array"
            },
            "y2": {
                "items": {
                    "type": "number"
                },
                "type": "array"
            }
        },
        "$schema": "http://json-schema.org/draft-04/schema",
        "type": "object",
        "id": "#NewDiscipline_output"
    }

Setting the grammars from a dictionary data example
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

An intermediate approach is to apply the :meth:`.JSONGrammar.update_from_data` method
with a :code:`dict` data example:

.. code::

    from gemseo.api import MDODiscipline

    class NewDiscipline(MDODiscipline):

        def __init__(self):
            super(NewDiscipline, self).__init__()
            self.input_grammar.update_from_data({'x': array([0.]), 'z': array([0.])})
            self.output_grammar.update_from_data({'y1': array([0.]), 'y2': array([0.])})
            # TO BE COMPLETED

.. note::

   Variable type is deduced from the values written in the :code:`dict` data example, either :code:`'float`'
   (e.g. :code:`'x'` and :code:`'y'` in :code:`{'x': array([0]), 'z': array([0.])}`) of :code:`'integer'`
   (e.g. :code:`'x'` in :code:`{'x': array([0]), 'z': array([0.])}`).

Checking the grammars
^^^^^^^^^^^^^^^^^^^^^

Lastly, we can verify a grammar by printing it, e.g.:

.. code::

   discipline = NewDiscipline()
   print(discipline.input_grammar)

which results in:

.. parsed-literal::

    Grammar named :NewDiscipline_input, schema = {"required": ["x", "z"], "type": "object", "properties": {"x": {"items": {"type": "number"}, "type": "array"}, "z": {"items": {"type": "number"}, "type": "array"}}}


NumPy arrays
^^^^^^^^^^^^

Discipline inputs and outputs shall be `numpy <http://www.numpy.org/>`_ arrays of real numbers or integers.

The grammars will check this at each execution and prevent any discipline from running with invalid data,
or raise an error if outputs are invalid, which happens sometimes with simulation software...

Setting the default inputs
--------------------------

We also define the default inputs by means of the :attr:`!MDODiscipline.default_inputs` attribute:

.. code::

    from gemseo.api import MDODiscipline
    from numpy import array

    class NewDiscipline(MDODiscipline):

        def __init__(self):
            super(NewDiscipline, self).__init__()
            self.input_grammar.update(['x', 'z'])
            self.output_grammar.update(['f', 'g'])
            self.default_inputs = {'x': array([0.]), 'z': array([0.])}

.. warning::

    An :class:`.MDODiscipline` that will be placed inside an :class:`.MDF`, a :class:`.BiLevel`
    formulation or an :class:`.MDA` with strong couplings **must** define its default inputs.
    Otherwise, the execution will fail.

Overloading the :meth:`!MDODiscipline._run` method
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once the input and output have been declared in the constructor of the discipline,
the abstract :meth:`!MDODiscipline._run` method of :class:`.MDODiscipline` shall be overloaded by
the discipline to define how outputs are computed from inputs.

.. seealso::

   The method is protected (starts with "_") because it shall not be called from outside the discipline.
   External calls that trigger the discipline execution use the :meth:`.MDODiscipline.execute` public method from the base class,
   which provides additional services before and after calling :meth:`!MDODiscipline._run`. These services, such as data checks by the grammars,
   are provided by |g| and the integrator of the discipline does not need to implement them.

Getting the input values from :attr:`!MDODiscipline.local_data` of the discipline
---------------------------------------------------------------------------------

First, the data values shall be retrieved.
For each input declared in the input grammar,
|g| will pass the values as arrays to the :class:`.MDODiscipline` during the execution of the process.
There are different methods to get these values within the :meth:`!MDODiscipline._run` method of the discipline:

- as a dictionary through the :meth:`.MDODiscipline.get_input_data` method, which is also already accessible in the :attr:`!MDODiscipline.local_data` attribute of the :class:`.MDODiscipline`
- or here as a list of values using :meth:`.MDODiscipline.get_inputs_by_name` with the data names passed as a list.

.. code::

        def _run(self):
            x, z = self.get_inputs_by_name(['x', 'z'])
            # TO BE COMPLETED

Computing the output values from the input ones
-----------------------------------------------

Then, we compute the output values from these input ones:

.. code::

        def _run(self):
            x, z = self.get_inputs_by_name(['x', 'z'])
            f = array([x[0]*z[0]])
            g = array([x[0]*(z[0]+1.)^2])
            # TO BE COMPLETED


Storing the output values into :attr:`!MDODiscipline.local_data` of the discipline
----------------------------------------------------------------------------------

Lastly, the computed outputs shall be stored in the :attr:`!MDODiscipline.local_data`,
either directly:

.. code::

        def _run(self):
            x, z = self.get_inputs_by_name(['x', 'z'])
            f = array([x[0]*z[0]])
            g = array([x[0]*(z[0]+1.)^2])
            self.local_data['f'] = f
            self.local_data['g'] = g

or by means of the :meth:`.MDODiscipline.store_local_data` method:

.. code::

        def _run(self):
            x, z = self.get_inputs_by_name(['x', 'z'])
            f = array([x[0]*z[0]])
            g = array([x[0]*(z[0]+1.)^2])
            self.store_local_data(f=f)
            self.store_local_data(g=g)

.. _discipline_compute_jacobian:

Overloading the :meth:`!MDODiscipline._compute_jacobian` method
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :class:`.MDODiscipline` may also provide the derivatives of their outputs with respect to their inputs, i.e. their Jacobians.
This is useful for :term:`gradient-based optimization` or :ref:`mda` based on the :term:`Newton method`.
For a vector of inputs :math:`x` and a vector of outputs :math:`y`, the Jacobian of the discipline is
:math:`\frac{\partial y}{\partial x}`.

The discipline shall provide a method to compute the Jacobian for a given set of inputs.
This is made by overloading the abstract :meth:`!MDODiscipline._compute_jacobian` method of :class:`.MDODiscipline`.
The discipline may have multiple inputs and multiple outputs.
To store the multiple Jacobian matrices associated to all the inputs and outputs,
|g| uses a dictionary of dictionaries structure.
This data structure is sparse and makes easy the access and the iteration over the elements
of the Jacobian.

The method :meth:`!MDODiscipline._init_jacobian` fills the :code:`dict` of :code:`dict` structure
with dense null matrices of the right sizes.
Note that all Jacobians must be 2D matrices, which avoids
ambiguity.

.. code::

    def _compute_jacobian(self, inputs=None, outputs=None):
        """
        Computes the jacobian

        :param inputs: linearization should be performed with respect
            to inputs list. If None, linearization should
            be performed wrt all inputs (Default value = None)
        :param outputs: linearization should be performed on outputs list.
            If None, linearization should be performed
            on all outputs (Default value = None)
        """
        # Initialize all matrices to zeros
        self._init_jacobian(with_zeros=True)
        x, z = self.get_inputs_by_name(['x', 'z'])

        self.jac['y1'] = {}
        self.jac['y1']['x'] = atleast_2d(z)
        self.jac['y1']['z'] = atleast_2d(x)

        self.jac['y2'] = {}
        self.jac['y2']['x'] = atleast_2d(array([(z[0]+1.)^2]))
        self.jac['y2']['z'] = atleast_2d(array([2*x[0]*z[0]*(z[0]+1.)]))
