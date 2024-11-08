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

Creating a discipline from scratch implies to implement a new class inheriting from :class:`.Discipline`.

For example, let's consider a discipline called ``NewDiscipline``,
with two outputs,
``f`` and ``g``,
and two inputs,
``x`` and ``z``,
where ``f=x*z`` and ``f=x*(z+1)^2``.

Overloading the :class:`.Discipline`'s constructor
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

First of all, we overload the :class:`.Discipline` constructor.
For that,
we call the :class:`.Discipline` superconstructor:

.. code::

    from gemseo import Discipline

    class NewDiscipline(Discipline):

        def __init__(self):
            super().__init__()
            # TO BE COMPLETED

Setting the input and output grammars
-------------------------------------

Then, we define the :attr:`!Discipline.input_grammar`
and :attr:`!Discipline.output_grammar` created by the base class constructor.
We have different ways to do that.

Setting the grammars from data names
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

When the variables are float arrays without any particular constraint,
the simplest approach is to apply the :meth:`.BaseGrammar.update_from_names` method to a list of variable names:

.. code::

    from gemseo import Discipline

    class NewDiscipline(Discipline):

        def __init__(self):
            super().__init__()
            self.input_grammar.update_from_names(['x', 'z'])
            self.output_grammar.update_from_names(['f', 'g'])
            # TO BE COMPLETED

Setting the grammars from JSON files
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

A more complicated approach is to define the grammar into JSON input and output files
with name ``'NewDiscipline_inputs.json'`` and ``'NewDiscipline_outputs.json'``,
put these files in the same directory as the module implementing the ``NewDiscipline`` and
set the class attribute ``auto_detect_grammar_files`` to ``True``.

.. code::

    from gemseo import Discipline

    class NewDiscipline(Discipline):

        auto_detect_grammar_files = True

        def __init__(self):
            super().__init__()
            # TO BE COMPLETED

where the ``'NewDiscipline_inputs.json'`` file is defined as follows:

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

and where the ``'NewDiscipline_outputs.json'`` file is defined as follows:

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

An intermediate approach is to apply the :meth:`.BaseGrammar.update_from_data` method
with a ``dict`` data example:

.. code::

    from gemseo import Discipline

    class NewDiscipline(Discipline):

        def __init__(self):
            super().__init__()
            self.input_grammar.update_from_data({'x': array([0.]), 'z': array([0.])})
            self.output_grammar.update_from_data({'y1': array([0.]), 'y2': array([0.])})
            # TO BE COMPLETED

.. note::

   Variable type is deduced from the values written in the ``dict`` data example, either ``'float``'
   (e.g. ``'x'`` and ``'y'`` in ``{'x': array([0]), 'z': array([0.])}``) of ``'integer'``
   (e.g. ``'x'`` in ``{'x': array([0]), 'z': array([0.])}``).

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

We also define the default inputs by means of the :attr:`!Discipline.default_input_data` attribute:

.. code::

    from gemseo import Discipline
    from numpy import array

    class NewDiscipline(Discipline):

        def __init__(self):
            super().__init__()
            self.input_grammar.update_from_names(['x', 'z'])
            self.output_grammar.update_from_names(['f', 'g'])
            self.default_input_data = {'x': array([0.]), 'z': array([0.])}

.. warning::

    An :class:`.Discipline` that will be placed inside an :class:`.MDF`, a :class:`.BiLevel`
    formulation or a :class:`.BaseMDA` with strong couplings **must** define its default inputs.
    Otherwise, the execution will fail.

Overloading the :meth:`!Discipline._run` method
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Once the input and output have been declared in the constructor of the discipline,
the abstract :meth:`!Discipline._run` method of :class:`.Discipline` shall be implemented by
the discipline to define how outputs are computed from inputs.

.. seealso::

   The method is protected (starts with "_") because it shall not be called from outside the discipline.
   External calls that trigger the discipline execution use the :meth:`.Discipline.execute` public method from the base class,
   which provides additional services before and after calling :meth:`!Discipline._run`. These services, such as data checks by the grammars,
   are provided by |g| and the integrator of the discipline does not need to implement them.

Computing the output values from the input ones
-----------------------------------------------

Then, we compute the output values from the input ones passed via the dictionary argument ``input_data``
and return the output data as a dictionary:

.. code::

        def _run(self, input_data):
            x = input_data['x']
            z = input_data['z']
            return {
                'f': array([x[0]*z[0]]),
                'g': array([x[0]*(z[0]+1.)^2]),
            }

.. _discipline_compute_jacobian:

Overloading the :meth:`!Discipline._compute_jacobian` method
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The :class:`.Discipline` may also provide the derivatives of their outputs with respect to their inputs, i.e. their Jacobians.
This is useful for :term:`gradient-based optimization` or :ref:`mda` based on the :term:`Newton method`.
For a vector of inputs :math:`x` and a vector of outputs :math:`y`, the Jacobian of the discipline is
:math:`\frac{\partial y}{\partial x}`.

The discipline shall provide a method to compute the Jacobian for a given set of inputs.
This is made by overloading the abstract :meth:`!Discipline._compute_jacobian` method of :class:`.Discipline`.
The discipline may have multiple inputs and multiple outputs.
To store the multiple Jacobian matrices associated to all the inputs and outputs,
|g| uses a dictionary of dictionaries structure.
This data structure is sparse and makes easy the access and the iteration over the elements
of the Jacobian.

The method :meth:`!Discipline._init_jacobian` fills the ``dict`` of ``dict`` structure
with dense null matrices of the right sizes.
Note that all Jacobians must be 2D matrices, which avoids
ambiguity.

.. code::

    def _compute_jacobian(self, input_names=(), output_names=()):
        # Initialize all matrices to zeros.
        self._init_jacobian(fill_missing_keys=True)

        # Get the inputs from the local data.
        x = self.local_data['x']
        z = self.local_data['z']

        self.jac = {
            'f': {
                'x': atleast_2d(z),
                'z': atleast_2d(x).
            },
            'g': {
                'x': atleast_2d(array([(z[0]+1.)^2])),
                'z': atleast_2d(array([2*x[0]*z[0]*(z[0]+1.)])),
        }
