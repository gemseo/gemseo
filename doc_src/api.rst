..
   Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

   This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
   International License. To view a copy of this license, visit
   http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
   Commons, PO Box 1866, Mountain View, CA 94042, USA.

..
   Contributors:
          :author:  Francois Gallard


.. _api:

API: high level functions to use |g|
================================================

This section describes the Application Programming Interface of |g|.

The basics of APIs
------------------

The aim of the :term:`API` is to provide high level functions that are sufficient to use |g| in most cases,
without requiring a deep knowledge of |g|.

Besides, these functions shall change much less often that the internal classes, which is key for backward compatibility,
which means ensuring that your current scripts using on |g| will be usable with the future versions of |g|.

The API also eases the interface |g| within a platform or other software.

.. seealso::

   To interface a simulation software with |g|, please refer to: :ref:`software_connection`.

An :term:`API` standardizes the programming interface of |g|, and allows to separate inner |g| code from
what users see. A few methods are sufficient to create a scenario, execute it and post process it.

See :ref:`extending-gemseo` to learn how to run |g| with external python
modules.

Logs
----

To configure the logs, and eventually set a logging file, use :meth:`~gemseo.api.configure_logger`.

|g| configuration
-----------------------------

The API function :meth:`~gemseo.api.print_configuration` can print the configuration with successfully loaded modules and failed imports with the reason.

About discipline
----------------

.. tip::

   In order to import easily and instantiate an :class:`.MDODiscipline`,
   |g| provides a :class:`.Factory` mechanism to avoid manual imports.

Get available disciplines
~~~~~~~~~~~~~~~~~~~~~~~~~

The :meth:`~gemseo.api.get_available_disciplines` API function can list the available disciplines:

.. code::

   >>> get_available_disciplines()
   ['RosenMF', 'Struct', 'SobieskiAerodynamics', 'DOEScenario', 'MDOScenario', 'SobieskiMission', 'SobieskiBaseWrapper', 'Sellar1', 'Sellar2', 'Aero', 'MDOChain', 'SobieskiStructure', 'Aerostruct', 'SobieskiPropulsion', 'Scenario', 'AnalyticDiscipline', 'MDOScenarioAdapter', 'SellarSystem', 'ScalableFittedDiscipline', 'PropaneReaction', 'PropaneComb1', 'PropaneComb2', 'PropaneComb3', 'MDOParallelChain']


Create a discipline
~~~~~~~~~~~~~~~~~~~

The :meth:`~gemseo.api.create_discipline` API function can create an :class:`.MDODiscipline`
or a list of :class:`.MDODiscipline` by using its name alone;
specific :code:`**options` can be provided in argument;
e.g.:

.. code::

   >>> from gemseo.api import create_discipline
   >>> disciplines = create_discipline(["SobieskiPropulsion", "SobieskiAerodynamics",
   ...                                  "SobieskiMission", "SobieskiStructure"])

Get discipline schemas for inputs, outputs and options
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- The API function :meth:`~gemseo.api.get_discipline_inputs_schema` can get the JSON schema of the inputs of a :code:`discipline`;
  if the argument :code:`output_json` (default: :code:`False`) is set to :code:`True`,  this method returns a JSON string, otherwise it returns a dictionary;
  e.g.:

.. code::

   >>> get_discipline_inputs_schema(disciplines[0])
   {u'name': u'SobieskiPropulsion_input', 'required': [u'x_3', u'x_shared', u'y_23'], u'id': u'#SobieskiPropulsion_input', u'$schema': u'http://json-schema.org/draft-04/schema', 'type': u'object', 'properties': {u'x_shared': {'items': {'type': u'number'}, 'type': u'array'}, u'y_23': {'items': {'type': u'number'}, 'type': u'array'}, u'x_3': {'items': {'type': u'number'}, 'type': u'array'}}}

- The API function :meth:`~gemseo.api.get_discipline_outputs_schema` can get the JSON schema of the outputs of a :code:`discipline`;
  if the argument :code:`output_json` (default: :code:`False`) is set to :code:`True`,  this method returns a JSON string, otherwise it returns a dictionary;
  e.g.:

.. code::

   >>> get_discipline_outputs_schema(disciplines[0])
   {u'name': u'SobieskiPropulsion_output', 'required': [u'g_3', u'y_3', u'y_31', u'y_32', u'y_34'], u'id': u'#SobieskiPropulsion_output', u'$schema': u'http://json-schema.org/draft-04/schema', 'type': u'object', 'properties': {u'y_31': {'items': {'type': u'number'}, 'type': u'array'}, u'y_32': {'items': {'type': u'number'}, 'type': u'array'}, u'y_3': {'items': {'type': u'number'}, 'type': u'array'}, u'y_34': {'items': {'type': u'number'}, 'type': u'array'}, u'g_3': {'items': {'type': u'number'}, 'type': u'array'}}}


- The API function :meth:`~gemseo.api.get_discipline_options_schema` can get the JSON schema of the options of a :code:`discipline`;
  if the argument :code:`output_json` (default: :code:`False`) is set to :code:`True`,  this method returns a JSON string, otherwise it returns a dictionary;
  e.g.:

.. code::

   >>> get_discipline_options_schema('SobieskiMission')
   {u'$schema': u'http://json-schema.org/draft-04/schema', 'required': ['dtype'], 'type': u'object', u'name': u'MDODiscipline_options', 'properties': {u'linearization_mode': {u'enum': [u'auto', u'direct', u'reverse', u'adjoint'], 'type': u'string'}, u'cache_tolerance': {u'minimum': 0, 'type': u'number', 'description': u'Numerical tolerance on the relative norm of input vectors \n to consider that two sets of inputs are equal, and that the outputs may therefore be returned from the cache without calculations.'}, u'jac_approx_n_processes': {u'minimum': 1, 'type': u'integer', 'description': u'maximum number of processors or threads on \nwhich the jacobian approximation is performed\n by default, 1 means no parallel calculations'}, u'cache_type': {u'enum': [u'HDF5_cache', u'simple_cache'], 'type': u'string', 'description': u'Type of cache to be used.  \nBy default, simple cache stores the last execution inputs and outputs  \nin memory only to avoid computation of the outputs if the inputs are identical.\n To store more executions, use HDF5 caches, which stores data on the disk.\n There is a hashing mechanism which avoids reading on the disk for every calculation.'}, 'dtype': {'type': 'string'}, u'cache_hdf_file': {'type': u'string', 'description': u'Path to the HDF5 file to store the cache data.', u'format': u'uri'}, u'jac_approx_use_threading': {'type': u'boolean', 'description': u'if True, use Threads instead of processes\n to parallelize the execution. \nMultiprocessing will serialize all the disciplines, \nwhile multithreading will share all the memory.\n This is important to note if you want to execute the same\n  discipline multiple times, you shall use multiprocessing'}, u'cache_hdf_node_name': {'type': u'string', 'description': u'Name of the HDF dataset to store the discipline\n data. If None, the discipline name is used.'}, u'jac_approx_type': {u'enum': [u'finite_differences', u'complex_step'], 'type': u'string'}, u'jax_approx_step': {'type': u'number', u'minimum': 0, u'exclusiveMinimum': True, 'description': u'Step for finite differences or complex step for Jacobian approximation'}, u'jac_approx_wait_time': {u'minimum': 0, 'type': u'number', 'description': u'Time waited between two forks of the process or thread when using parallel jacobian approximations (parallel=True)'}}}

- The API function :meth:`~gemseo.api.get_discipline_options_defaults` can get the default values of the JSON schema of the options of a discipline :code:`discipline_name`;
  e.g.:

.. code

   >>> get_discipline_options_defaults('SobieskiMission')
   {'dtype': 'float64'}

Plot coupling structure
~~~~~~~~~~~~~~~~~~~~~~~

The :meth:`~gemseo.api.generate_coupling_graph` API function plots the coupling graph of a set of :class:`.MDODiscipline`:

.. automethod:: gemseo.api.generate_coupling_graph
   :noindex:

The :meth:`~gemseo.api.generate_n2_plot` API function plots the N2 diagram of a set of :class:`.MDODiscipline`:

.. automethod:: gemseo.api.generate_n2_plot
   :noindex:

About surrogate discipline
--------------------------

Similarly, a surrogate discipline can be created. Here are the API functions for that.

Get available surrogate disciplines
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The API function :meth:`~gemseo.api.get_available_surrogates` can list the available surrogate models:

.. code::

   >>> get_available_surrogates()
   ['LinRegSurrogateDiscipline', 'RBFSurrogateDiscipline', 'GPRSurrogateDiscipline']

Get the surrogate schema for options
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The API function :meth:`~gemseo.api.get_surrogate_options_schema` can get the JSON schema of a surrogate;
if the argument :code:`output_json` (default: :code:`False`) is set to :code:`True`,  this method returns a JSON string, otherwise it returns a dictionary;
e.g.:

.. code::

   >>> get_surrogate_options_schema('RBFSurrogateDiscipline', output_json=True)
   '{"required": ["function"], "type": "object", "properties": {"function": {"type": "string", "description": "str or callable, optional\\nThe radial basis function, based on the radius, r, given by the\\n norm\\n:type function: str or callable\\n"}, "input_names": {"description": "list of input names among all inputs in the HDF\\nBy default, takes all inputs in the HDF\\n:type input_names: list(str)\\n"}, "disc_name": {"description": "discipline name\\n:type disc_name: str\\n"}, "train_set": {"description": "sample train set\\n:type train_set: list(int)\\n"}, "epsilon": {"description": "Adjustable constant for gaussian or\\nmultiquadrics functions\\n:type epsilone: float"}, "output_names": {"description": "list of output names among all inputs in the HDF\\nBy default, takes all outputs in the HDF\\n:type output_names: list(str)\\n"}}}'

Create surrogate disciplines
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- The API function :meth:`~gemseo.api.create_surrogate` can create a surrogate discipline.

  - The mandatory arguments are:

    - :code:`surrogate_name`: name of the surrogate model (the class name)
    - :code:`hdf_file_path`: path to the HDF file to be used to train the surrogate
    - :code:`hdf_node_path`: node name in the HDF, by default the original discipline name

  - The optional arguments are:

    - :code:`input_names`: list of input names among all inputs in the HDF. By default, takes all inputs in the HDF (defaut: :code:`None`)
    - :code:`output_names`: list of output names among all outputs in the HDF. By default, takes all outputs in the HDF
    - :code:`disc_name`: surrogate discipline name
    - :code:`**options`: additional options to be passed to the surrogate for its construction

.. seealso::

   See :ref:`surrogates` for more details about the API function :meth:`~gemseo.api.create_surrogate`.

About design space
------------------

Create a design space
~~~~~~~~~~~~~~~~~~~~~

To create a standard :class:`.DesignSpace`, the API function :meth:`~gemseo.api.create_design_space` can be used.

- This function does not take any argument.
- This function returns an instance of :class:`.DesignSpace`.

Read a design space
~~~~~~~~~~~~~~~~~~~

In presence of a design space specified in a CSV file, the API function :meth:`~gemseo.api.read_design_space` can be used.

- Its first argument is the file path of the design space. Its second argument is the list of fields available in the file and is optional.
- By default, the design space reads these information from the file.
- This function returns an instance of :class:`.DesignSpace`.

.. seealso::

   See :ref:`sphx_glr_examples_design_space_plot_create_design_space.py` for more details about the API function :meth:`~gemseo.api.create_design_space`.

   See :ref:`sphx_glr_examples_design_space_plot_load_design_space.py` for more details about the API function :meth:`~gemseo.api.read_design_space`.

Write a design space
~~~~~~~~~~~~~~~~~~~~

To export an instance of :class:`.DesignSpace` into an hdf or txt file,
the :meth:`~gemseo.api.export_design_space` API function can be used:

.. automethod:: gemseo.api.export_design_space
   :noindex:

About MDO formulations
----------------------

Get available formulations
~~~~~~~~~~~~~~~~~~~~~~~~~~

Many API functions allow to discover the :ref:`MDO formulations <mdo_formulations>` and their options.

The API function :meth:`~gemseo.api.get_available_formulations` returns the list of available :ref:`MDO formulations <mdo_formulations>`.

.. code::

   >>> get_available_formulations()
   ['IDF', 'BiLevel', 'MDF', 'DisciplinaryOpt']

Get formulation schemas for (sub-)options
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

For a given :ref:`MDO formulation <mdo_formulations>` named :code:`formulation_name`, e.g. :code:`"MDF"`, we can:

- get its list of option by means of the API function  :meth:`~gemseo.api.get_formulation_options_schema`; if the argument :code:`output_json` (default: :code:`False`) is set to :code:`True`,
  this method returns a JSON string, otherwise it returns a dictionary; e.g.:

.. code::

   >>> get_formulation_options_schema("MDF")
   {'$schema': 'http://json-schema.org/schema#', 'type': 'object', 'properties': {'maximize_objective': {'description': 'If True, the objective function is maximized.', 'type': 'boolean'}, 'grammar_type': {'description': 'The type of the input and output grammars, either :attr:`.MDODiscipline.JSON_GRAMMAR_TYPE` or :attr:`.MDODiscipline.SIMPLE_GRAMMAR_TYPE`.', 'type': 'string'}, 'main_mda_name': {'description': 'The name of the class used for the main MDA, typically the :class:`.MDAChain`, but one can force to use :class:`.MDAGaussSeidel` for instance.', 'type': 'string'}, 'inner_mda_name': {'description': 'The name of the class used for the inner-MDA of the main MDA, if any; typically when the main MDA is an :class:`.MDAChain`.', 'type': 'string'}}, 'required': ['grammar_type', 'inner_mda_name', 'main_mda_name', 'maximize_objective']}

- get its list of sub-options by means of the API function  :meth:`~gemseo.api.get_formulation_sub_options_schema` when the :code:`**options` of :code:`formulation_name` are provided in argument;
  if the argument :code:`output_json` (default: :code:`False`) is set to :code:`True`, this method returns a JSON string, otherwise it returns a dictionary.
- get its list of default option values by means of :meth:`~gemseo.api.get_formulations_options_defaults`; if the argument :code:`output_json` (default: :code:`False`) is set to :code:`True`,
  this method returns a JSON string, otherwise it returns a dictionary.

.. code::

   >>> get_formulations_options_defaults("MDF")
   {'maximize_objective': False, 'grammar_type': 'JSONGrammar', 'main_mda_name': 'MDAChain', 'inner_mda_name': 'MDAJacobi'}

- get its list of default sub-option values by means of :meth:`~gemseo.api.get_formulations_sub_options_defaults` when the :code:`**options` of :code:`formulation_name` are provided in argument;
  if the argument :code:`output_json` (default: :code:`False`) is set to :code:`True`, this method returns a JSON string, otherwise it returns a dictionary.

About scenario
--------------

Get available scenario type
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The API function :meth:`~gemseo.api.get_available_scenario_types` can be used to get the available scenario types (:class:`.MDOScenario` and :class:`.DOEScenario`)

.. code::

   >>> get_available_scenario_types()
   ['MDO', 'DOE']

Get scenario schema for inputs and options
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

- The API function :meth:`~gemseo.api.get_scenario_options_schema` can be used to get the options of a given scenario:
  if the argument :code:`output_json` (default: :code:`False`) is set to :code:`True`, this method returns a JSON string, otherwise it returns a dictionary;
  e.g.:

.. code::

   >>> print(get_scenario_options_schema("MDO"))
   {u'$schema': u'http://json-schema.org/draft-04/schema', 'required': ['name'], 'type': u'object', u'name': u'MDODiscipline_options', 'properties': {u'linearization_mode': {u'enum': [u'auto', u'direct', u'reverse', u'adjoint'], 'type': u'string'}, u'cache_tolerance': {u'minimum': 0, 'type': u'number', 'description': u'Numerical tolerance on the relative norm of input vectors \n to consider that two sets of inputs are equal, and that the outputs may therefore be returned from the cache without calculations.'}, u'jac_approx_n_processes': {u'minimum': 1, 'type': u'integer', 'description': u'maximum number of processors or threads on \nwhich the jacobian approximation is performed\n by default, 1 means no parallel calculations'}, u'cache_type': {u'enum': [u'HDF5_cache', u'simple_cache'], 'type': u'string', 'description': u'Type of cache to be used.  \nBy default, simple cache stores the last execution inputs and outputs  \nin memory only to avoid computation of the outputs if the inputs are identical.\n To store more executions, use HDF5 caches, which stores data on the disk.\n There is a hashing mechanism which avoids reading on the disk for every calculation.'}, u'cache_hdf_file': {'type': u'string', 'description': u'Path to the HDF5 file to store the cache data.', u'format': u'uri'}, u'jac_approx_use_threading': {'type': u'boolean', 'description': u'if True, use Threads instead of processes\n to parallelize the execution. \nMultiprocessing will serialize all the disciplines, \nwhile multithreading will share all the memory.\n This is important to note if you want to execute the same\n  discipline multiple times, you shall use multiprocessing'}, u'cache_hdf_node_name': {'type': u'string', 'description': u'Name of the HDF dataset to store the discipline\n data. If None, the discipline name is used.'}, u'jac_approx_type': {u'enum': [u'finite_differences', u'complex_step'], 'type': u'string'}, u'jac_approx_wait_time': {u'minimum': 0, 'type': u'number', 'description': u'Time waited between two forks of the process or thread when using parallel jacobian approximations (parallel=True)'}, u'jax_approx_step': {'type': u'number', u'minimum': 0, u'exclusiveMinimum': True, 'description': u'Step for finite differences or complex step for Jacobian approximation'}, 'name': {'description': 'scenario name\n'}}}

- The API function :meth:`~gemseo.api.get_scenario_inputs_schema` can be used to get the JSONSchema of the inputs of a :code:`scenario`;
  if the argument :code:`output_json` (default: :code:`False`) is set to :code:`True`, this method returns a JSON string, otherwise it returns a dictionary.

Get scenario differentiation modes
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The API function :meth:`~gemseo.api.get_scenario_differentiation_modes` can be used to get the available differentiation modes of a scenario:

.. code::

   >>> get_scenario_differentiation_modes()
   ['user', 'complex_step', 'finite_differences', 'no_derivatives']

Create a scenario
~~~~~~~~~~~~~~~~~

The API function :meth:`~gemseo.api.create_scenario` can be used to create a scenario:

- The four first arguments are mandatory:

  #. :code:`disciplines`: either a list of :class:`.MDODiscipline` or a single :class:`MDODiscipline`,
  #. :code:`formulation`: the formulation name (available formulations can be listed by using the API function :meth:`gemseo.api.get_available_formulations`),
  #. :code:`objective_name`: the name of the objective function (one of the discipline outputs, which can be listed by using the function :meth:`gemseo.disciplines.utils.get_all_outputs`)
  #. :code:`design_space`: the :class:`.DesignSpace` or the file path of the design space
     the design variables are a subset of all the discipline inputs, which can be obtained by using :meth:`~gemseo.disciplines.utils.get_all_inputs` .

- The other arguments are optional:

  - :code:`name`: scenario name,
  - :code:`scenario_type`: type of scenario, either ``"MDO"`` (default) or ``"DOE"``,
  - :code:`**options`: options passed to the formulation,

- This function returns an instance of :class:`.MDOScenario` or :class:`.DOEScenario`.

.. seealso::

   See :ref:`this part of the Sellar's tutorial <sellar_mdo_create_scenario>` for more details about the API function :meth:`~gemseo.api.create_scenario`.

- The API function :meth:`~gemseo.api.monitor_scenario` can be used to add an observer to a :code:`scenario`;
  the observer must have an "update(atom)" method that  handles the execution status change of atom ; update(atom) is called everytime an atom execution changes;
  if the argument :code:`output_json` (default: :code:`False`) is set to :code:`True`, this method returns a JSON string, otherwise it returns a dictionary.

Monitor a scenario
~~~~~~~~~~~~~~~~~~

To monitor a scenario execution programmatically, ie get a notification when a discipline status is changed,
use :meth:`~gemseo.api.monitor_scenario`. The first argument is the scenario to monitor, and the second is an
observer object, that is notified by its update(atom) method, which takes an
:class:`.AtomicExecSequence` as argument. This method will be called every time
a discipline status changes. The atom represents a discipline's position in the process. One discipline can
have multiple atoms, since one discipline can be used in multiple positions in the MDO formulation.

For more details on monitoring, see :ref:`monitoring`.

About optimization and DOE algorithms
-------------------------------------

Get available algorithms and associated options
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

To execute a scenario, a driver must be selected. Here are the API functions for that.

- The API function :meth:`~gemseo.api.get_available_opt_algorithms` can list the available optimization algorithms:

.. code::

   >>> get_available_opt_algorithms()
   ['NLOPT_SLSQP', 'L-BFGS-B', 'SLSQP', 'NLOPT_COBYLA', 'NLOPT_BFGS', 'NLOPT_NEWUOA', 'TNC', 'P-L-BFGS-B', 'NLOPT_MMA', 'NLOPT_BOBYQA', 'ODD']

- The API function :meth:`~gemseo.api.get_available_doe_algorithms` can list the available DOE algorithms:

.. code::

   >>> get_available_doe_algorithms()
   ['ff2n', 'OT_FACTORIAL', 'OT_FAURE', 'OT_HASELGROVE', 'OT_REVERSE_HALTON', 'OT_HALTON', 'ccdesign', 'OT_SOBOL', 'fullfact', 'OT_FULLFACT', 'OT_AXIAL', 'lhs', 'OT_LHSC', 'OT_MONTE_CARLO', 'OT_RANDOM', 'OT_COMPOSITE', 'CustomDOE', 'pbdesign', 'OT_LHS', 'bbdesign']

- The API function :meth:`~gemseo.api.get_algorithm_options_schema` can list the available options of the algorithm :code:`algorithm_name`;
  if the argument :code:`output_json` (default: :code:`False`) is set to :code:`True`, this method returns a JSON string, otherwise it returns a dictionary;
  e.g.:

.. code::

   >>> get_algorithm_options_schema('OT_HALTON')
   {u'$schema': u'http://json-schema.org/draft-04/schema', 'type': u'object', u'name': u'OPENTURNS_options', 'properties': {u'wait_time_between_samples': {u'minimum': 0, 'type': u'number'}, u'n_processes': {u'minimum': 1, 'type': u'integer'}, u'end': {'type': u'number'}, u'distribution_name': {u'enum': [u'Arcsine', u'Beta', u'Dirichlet', u'Normal', u'TruncatedNormal', u'Triangular', u'Trapezoidal', u'Uniform'], 'description': 'Default value = "Uniform")\n'}, u'eval_jac': {'type': u'boolean'}, u'mu': {'type': u'number'}, u'start': {'type': u'number'}, u'levels': {'items': {u'minItems': 1, 'type': u'number'}, 'type': u'array', 'description': 'Default value = None)\n'}, u'n_samples': {u'minimum': 1, 'type': u'integer'}, u'sigma': {'type': u'number'}, u'centers': {'items': {u'minItems': 1, 'type': u'number'}, 'type': u'array', 'description': 'Default value = None)\n'}}}

Execute an algorithm
~~~~~~~~~~~~~~~~~~~~

We can apply a DOE or optimization algorithm to an :class:`.OptimizationProblem`
by means of the :meth:`~gemseo.api.execute_algo` algorithm:

.. code::

   >>> from gemseo.problems.analytical.rastrigin import Rastrigin
   >>> from gemseo.api import execute_algo
   >>>
   >>> opt_problem = Rastrigin()
   >>> execute_algo(opt_problem, 'SLSQP')
   INFO - 12:59:49 : Optimization problem:
   INFO - 12:59:49 :       Minimize: Rastrigin(x) = 20 + sum(x[i]**2 - 10*cos(2pi*x[i]))
   INFO - 12:59:49 : With respect to:
   INFO - 12:59:49 :     x
   INFO - 12:59:49 : Design Space:
   INFO - 12:59:49 : +------+-------------+-------+-------------+-------+
   INFO - 12:59:49 : | name | lower_bound | value | upper_bound | type  |
   INFO - 12:59:49 : +------+-------------+-------+-------------+-------+
   INFO - 12:59:49 : | x    |     -0.1    |  0.01 |     0.1     | float |
   INFO - 12:59:49 : | x    |     -0.1    |  0.01 |     0.1     | float |
   INFO - 12:59:49 : +------+-------------+-------+-------------+-------+
   INFO - 12:59:49 : Optimization: |          | 0/999   0% [elapsed: 00:00 left: ?, ? iters/sec]
   INFO - 12:59:49 : Optimization: |          | 4/999   0% [elapsed: 00:00 left: 00:00, 1949.25 iters/sec obj:  0.00 ]
   INFO - 12:59:49 : Optimization result:
   INFO - 12:59:49 : Objective value = 1.37852396165e-10
   INFO - 12:59:49 : The result is feasible.
   INFO - 12:59:49 : Status: 0
   INFO - 12:59:49 : Optimizer message: Optimization terminated successfully.
   INFO - 12:59:49 : Number of calls to the objective function by the optimizer: 5
   INFO - 12:59:49 : Constraints values:
   INFO - 12:59:49 :
   INFO - 12:59:49 : Design Space:
   INFO - 12:59:49 : +------+-------------+-----------------------+-------------+-------+
   INFO - 12:59:49 : | name | lower_bound |         value         | upper_bound | type  |
   INFO - 12:59:49 : +------+-------------+-----------------------+-------------+-------+
   INFO - 12:59:49 : | x    |     -0.1    | 5.894250055538119e-07 |     0.1     | float |
   INFO - 12:59:49 : | x    |     -0.1    | 5.894250055538119e-07 |     0.1     | float |
   INFO - 12:59:49 : +------+-------------+-----------------------+-------------+-------+


About MDA
---------

Here are the API functions for :ref:`MDA <mda>`.

- The API function :meth:`~gemseo.api.get_available_mdas` can list the available :ref:`MDAs <mda>`:

.. code::

   >>> get_available_mdas()
   ['MDANewtonRaphson', 'MDAChain', 'MDARoot', 'MDAQuasiNewton', 'MDAGaussSeidel', 'GSNewtonMDA', 'MDASequential', 'MDAJacobi']

- The API function :meth:`~gemseo.api.get_mda_options_schema` can list the available options of an MDA;
  if the argument :code:`output_json` (default: :code:`False`) is set to :code:`True`, this method returns a JSON string, otherwise it returns a dictionary;
  e.g.

.. code::

   >>> get_mda_options_schema('MDAGaussSeidel')
   {'required': ['grammar_type', 'linear_solver_tolerance', 'max_mda_iter', 'tolerance', 'use_lu_fact', 'warm_start'], 'type': 'object', 'properties': {'warm_start': {'type': 'boolean', 'description': 'if True, the second iteration and ongoing\nstart from the previous coupling solution\n:type warm_start: bool\n'}, 'name': {'description': 'the name of the chain\n:type name: str\n'}, 'use_lu_fact': {'type': 'boolean', 'description': 'if True, when using adjoint/forward\ndifferenciation, store a LU factorization of the matrix\nto solve faster multiple RHS problem\n:type use_lu_fact: bool'}, 'grammar_type': {'type': 'string', 'description': 'the type of grammar to use for IO declaration\neither JSON_GRAMMAR_TYPE or SIMPLE_GRAMMAR_TYPE\n:type grammar_type: str\n'}, 'linear_solver_tolerance': {'type': 'number', 'description': 'Tolerance of the linear solver\nin the adjoint equation\n:type linear_solver_tolerance: float\n'}, 'max_mda_iter': {'type': 'integer', 'description': 'maximum number of iterations\n:type max_mda_iter: int\n'}, 'tolerance': {'type': 'number', 'description': 'tolerance of the iterative direct coupling solver,\nnorm of the current residuals divided by initial residuals norm\nshall be lower than the tolerance to stop iterating\n:type tolerance: float\n'}}}

- The API function :meth:`~gemseo.api.create_mda` can create a :ref:`MDA <mda>` called :code:`mda_name`, from a list of :code:`disciplines`
  and additional :code:`**options`.

.. seealso::

   See :ref:`mda` for more details about the API function :meth:`~gemseo.api.get_available_mdas`

About post processing
---------------------

|g| provides various methods to post process the results. Here are the API functions for that.

- The API function :meth:`~gemseo.api.get_available_post_processings` can list the available visualizations
  in the current |g| setup (depending on plugins and availability of dependencies),

.. code::

   >>> get_available_post_processings()
   ['ScatterPlotMatrix', 'VariableInfluence', 'RadarChart', 'ConstraintsHistory', 'SOM', 'Correlations', 'Robustness', 'KMeans', 'ParallelCoordinates', 'GradientSensitivity', 'OptHistoryView', 'BasicHistory', 'ObjConstrHist', 'QuadApprox']

- The API function :meth:`~gemseo.api.get_post_processing_options_schema` can list the available options of the post processing :code:`post_proc_name`; e.g.:
  if the argument :code:`output_json` (default: :code:`False`) is set to :code:`True`, this method returns a JSON string, otherwise it returns a dictionary;
  e.g.:

.. code::

   >>> get_post_processing_options_schema('RadarChart')
   {u'name': u'RadarChart_options', 'required': [u'constraint_names', u'save'], u'$schema': u'http://json-schema.org/draft-04/schema', 'type': u'object', 'properties': {u'save': {'type': u'boolean', 'description': 'if True, exports plot to pdf\n'}, u'iteration': {'type': u'integer', 'description': 'number of iteration to post process\n'}, u'file_path': {'type': u'string', 'description': 'the base paths of the files to export'}, u'contraint_names': {'items': {u'minItems': 1, 'type': u'string'}, 'type': u'array', 'description': 'list of constraints names\n'}, u'show': {'type': u'boolean', 'description': 'if True, display plot\n'}}}


- The API function :meth:`~gemseo.api.execute_post` can generate visualizations of the MDO results.
  For that, it consider the object to post process :code:`to_post_proc`, the post processing :code:`post_name` with its :code:`**options`;
  e.g.:

.. autofunction:: gemseo.api.execute_post
   :noindex:
