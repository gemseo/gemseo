- The `gemseo.formulations.bilevel.BiLevel` and its derived class `gemseo.formulations.bilevel_bcd.BiLevelBCD` now allow
  users to provide instances of `BaseMDA` or `Discipline` to be used as the MDA1 or the MDA2 of the formulation. These
  instances shall be provided via the fields `mda1_instance` and `mda2_instance` of the
  `gemseo.formulations.bilevel_settings.BiLevel_Settings`.
- It is now possible to deactivate the use of the MDA1 and MDA2 of the `gemseo.formulations.bilevel.BiLevel` and its
  derived class `gemseo.formulations.bilevel_bcd.BiLevelBCD` using the `use_mda1` and `use_mda2` of the
  `gemseo.formulations.bilevel_settings.BiLevel_Settings`.
- The `gemseo.formulations.bilevel.BiLevel` and its derived class `gemseo.formulations.bilevel_bcd.BiLevelBCD` now allow
  users to provide instances of `Discipline` to be used as sub-scenarios of the formulation. These instances shall be
  provided via the `disc_as_sub_scenario` field of the `gemseo.formulations.bilevel_settings.BiLevel_Settings`.
- `SimpleGrammar.schema` and `SimplerGrammar.schema` return a JSON-schema-shaped dict, matching `JSONGrammar` and `PydanticGrammar`.
- The `BaseModelDiscipline` allows users to use Pydantic models to handle I/O instead of Python dictionaries. It also
  determines automatically the I/O's from the model and defines the grammar from them: the fields read before being
  assigned as a whole are the inputs and the fields written are the outputs, so that a field computed and then
  re-used as an intermediate value is an output only. This means that different
  disciplines that inherit from `BaseModelDiscipline` can share the same Pydantic model, each of them using only a
  subset of the model's fields for their I/O's.
- `gemseo.discipline.propagate_namespace` propagates a namespace forward along the discipline coupling graph, namespacing every
  input and output affected by the seed variables while leaving untouched the variables that are neither seeds nor
  produced inside the reached set.
- The abstract class `gemseo.space.base.BaseVariableSpace` is the common parent of `DesignSpace` and `RandomSpace`; it
  owns the registry of the variables of a space and everything derived from their names, sizes, types and order, and
  declares the mapping to and from the unit hypercube.
- The class `gemseo.space.random.RandomSpace` describes a space of random variables
  defined by probability distributions.
- `RandomSpace.add_variable(name, *settings)` adds a random variable from the settings of the marginal probability
  distributions of its components, one per component, so an iid random variable of size $d$ is added by repeating its
  settings, e.g. `space.add_variable("x", *[settings] * 3)`.
- A random variable is defined by the settings of the marginal probability distributions of its components only; its
  joint probability distribution, size, type and bounds are derived from these settings and read-only, and its joint
  probability distribution is built on demand.
- Unlike a `DesignSpace`, a `RandomSpace` has no bounds setters, no current value, no normalization and no integer
  management; its only vector mapping is the iso-probabilistic `transform_vect`/`untransform_vect` to and from the unit
  hypercube.
- `BaseVariableSpace.variables` extends the read-only live view of `DesignSpace.variables` to every space of variables,
  hence to a `RandomSpace`; `add_copula` is one more method of the space mutating the registry that this view exposes.
- `BaseVariableSpace.add_variable` is the common entry point to add a variable, whatever the kind of space; its
  arguments depend on this kind, e.g. a size, a data type and bounds for a `DesignSpace`, the settings of the marginal
  probability distributions of the components for a `RandomSpace`.
- A `RandomSpace` reads all its variables through `variables`, which is its only accessor, as for a `DesignSpace`.
- The probabilistic data of a `RandomSpace` is read through `variables` too: `variables.distribution` (joint
  distribution of the space), `variables[name].distribution`, `variables[name].distribution_settings` and the statistics
  of `variables[name].distribution` such as `range` and `support`.
- `space.variables.has_variables_of_type(DataType.INTEGER)` is `False` for a `RandomSpace`, whose random variables
  are always of real type.
- The function `gemseo.create_random_space` creates an empty `RandomSpace`.
- The factory `gemseo.space.factory.random_space_factory` creates `RandomSpace` objects.
- `BaseDOELibrary.sample_space` and the function `gemseo.compute_doe` sample any `BaseVariableSpace`, hence a
  `RandomSpace` as well as a `DesignSpace`, without any conversion.
- The sensitivity analyses accept a `RandomSpace`, which they sample through its iso-probabilistic mapping.
- `EvaluationProblem`, `Database` and `EvaluationScenario` operate on any `BaseVariableSpace`, hence on a `DesignSpace`
  as well as on a `RandomSpace`; `EvaluationProblem.input_space`, `Database.input_space` and
  `EvaluationScenario.input_space` are typed accordingly.
- The features specific to a design space are unavailable for a space that defines no current value: passing
  `normalize_design_space=True` to a driver raises a `ValueError`, writing the database to an HDF file logs a
  warning and omits the input space from the file, and `EvaluationProblem.evaluate_functions` raises a `ValueError`
  when given a normalized input value (its default), so `input_value_is_normalized=False` must be passed.
- `BaseVariableSpace.check` raises a `ValueError` when the space is empty; `DesignSpace.check` also checks the
  consistency of the current value.
- A current value being a notion of the deterministic spaces, `has_current_value` is specific to a `DesignSpace`.
- `BaseVariableSpace.reference_value` is the reference value of a space, as a map from a variable name to a single
  representative value: the current value for a `DesignSpace` and the mean of the probability distributions for a
  `RandomSpace`; it is empty when the space defines no such value.
- The formulation of a scenario seeds the default input values of its top-level disciplines from `reference_value`,
  so these values need not be set discipline by discipline and, for a `RandomSpace`, follow the settings of the
  probability distributions without any further action.
- `BaseVariableSpace.filter_dimensions` keeps a subset of dimensions of a variable in any space of variables. For a
  `RandomSpace`, the random variable is rebuilt from the settings of the marginal probability distributions of the kept
  components and the copula covering it, if any, is removed, so that the random variables it covered become independent.
- The hierarchy of the variables of a space is public, in the `gemseo.space.variable` package. Every kind of variable
  derives from the abstract `BaseVariable`, which defines a variable by a size and a data type; a variable is immutable
  and its fields are exactly what its kind takes as input, anything deriving from them being a read-only property.
  The arrays a variable hands out, e.g. the bounds of a `RealVariable` or the choices of a `DiscreteVariable`,
  are read-only views of what the variable stores: in-place mutation raises
  `ValueError: assignment destination is read-only`, the writeable flag cannot be re-enabled, and reassigning the
  shape, the strides or the data type of such an array, which NumPy allows on a read-only array, changes that array
  only.
  Two abstract classes place a kind in this hierarchy: `BaseDeterministicVariable`, whose variables take a value that a
  space can check and a driver can normalize, and `BaseNumericVariable`, whose components are numbers, hence ordered,
  hence bounded. `RealVariable` and `IntegerVariable` derive from `BaseIntervalVariable`, whose domain is an
  interval defined by a size and bounds supplied by the caller; `DiscreteVariable` derives its size and its bounds from
  its choices; the `gemseo.space.variable.random.RandomVariable` of a `RandomSpace` derives its size, its type and its
  bounds from the settings of the probability distributions of its components.
- The `gemseo.space.variable.factory.deterministic_variable_factory` builds a deterministic variable from its data type.
  A random variable is built from the settings of its probability distributions, not from a data type, and so is out of
  the scope of this factory.
- Passing a read-only member of a variable, e.g. the size of a scalar variable or a bound derived from the choices of a
  discrete variable, raises an error naming that member and what the kind of variable takes as input, e.g.
  `"lower_bound is read-only; the input of a DiscreteVariable is choices."`.
- The MDO formulations operate on any `BaseVariableSpace`, hence on a `RandomSpace` as well as on a `DesignSpace`:
  the space of variables is a type parameter of `BaseMDOFormulation`, and `DisciplinaryOpt`, `MDF`, `BiLevel`,
  `BiLevelBCD` and `IDF` are generic in it, so an `EvaluationScenario` can sample a random space through any of them.
  The sub-scenarios of a `BiLevel` remain `MDOScenario` objects, hence defined over design spaces, whatever the
  system-level space.
- Two settings of `IDF` require more than a mere space of variables and raise a `ValueError` otherwise:
  `normalize_constraints=True` scales the consistency constraints by the bound range of the target coupling variables,
  which must be finite, e.g. a random variable following a normal distribution has no finite range, and
  `start_at_equilibrium=True` stores the equilibrium as the current value of the space, so the input space must be
  a `DesignSpace`.
