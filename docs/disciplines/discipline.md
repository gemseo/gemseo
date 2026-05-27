---
status: draft
description: ""
tags: ['user_guide']
search:
  boost: 2
---

<!--
 Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

 This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
 International License. To view a copy of this license, visit
 http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
 Commons, PO Box 1866, Mountain View, CA 94042, USA.
-->

# The discipline, a key concept

!!! info "Learn More"

    **Tutorials**: *Link to Tutorials*

    **How-to**: *Link to HOWTO*

    **References**: *Link to references*

## How is a discipline defined?

### What is a discipline?

A wrapper, or library wrapper, is a piece of software which translates
the existing API of an existing program or a library, into a
compatible one. Each program is encapsulated within using a dedicated
interface. GEMSEO defines the standardized interface in the
[Discipline][gemseo.core.discipline.discipline.Discipline] interface,
to define input data,
output data and an execution of the integrated software. Thanks to it,
GEMSEO can treat the integrated software independently of their own
implementation and of their own conventions to describe the inputs and
outputs (file formats for instance).

The next figure displays the concept of wrapper in workflow management.

![The wrapper concept](figs/wrapper.png)

A discipline is a set of calculations that:

- produces a dictionary of arrays as outputs
- from a dictionary of arrays as inputs
- using either a Python function, or equations or an external software, or a workflow engine.

### How is a discipline implemented in GEMSEO?

Programmatically speaking, disciplines are implemented in GEMSEO through the [Discipline][gemseo.core.discipline.discipline.Discipline] class. They are defined by three elements:

- the [input_grammar][gemseo.core.discipline.discipline.Discipline.input_grammar] attribute: the set of rules that defines valid input data,
- the [output_grammar][gemseo.core.discipline.discipline.Discipline.output_grammar] attribute: the set of rules that defines valid output data,
- the `Discipline._run()` method: the method to compute the output data from the input data.

#### Input and output description: grammars

The input and output specifications are defined in a grammar, through the [input_grammar][gemseo.core.discipline.discipline.Discipline.input_grammar] and [output_grammar][gemseo.core.discipline.discipline.Discipline.output_grammar] attributes, which can be either a [SimpleGrammar][gemseo.core.grammars.simple.SimpleGrammar] or a [JSONGrammar][gemseo.core.grammars.json.JSONGrammar] (default grammar), or your own which derives from the [BaseGrammar][gemseo.core.grammars.base.BaseGrammar] class.

- [SimpleGrammar][gemseo.core.grammars.simple.SimpleGrammar]: it manipulates the list of required data names,
  and a list of the associated types (string, float, numpy.ndarray or
  any type provided). There is also a dictionary of default values that
  adds default values to the data if they are not provided.

- [JSONGrammar][gemseo.core.grammars.json.JSONGrammar]: a JSON-based grammar.
  You must provide a JSON file that describes the validity of the data.
  This is a much more advanced and much more powerful description. JSON
  is a web standard supported by many languages: [JSON Schema](https://json-schema.org/tools?query=&sortBy=name&sortOrder=ascending&groupBy=languages&licenses=&languages=&drafts=&toolingTypes=&environments=&showObsolete=false).
  Please read [Understanding JSON Schema](http://spacetelescope.github.io/understanding-json-schema/index.html)
  for details on JSON schema. The input and output schemas for the
  disciplines must be files in the same directory as the Python module
  of the discipline, with a naming convention
  `MyDisciplineName_input.json` and `MyDisciplineName_output.json`.

!!! note
    The *grammar* is a very powerful and key concept.
    There are multiple ways of creating grammars in GEMSEO.
    The preferred one for integrating simulation processes is the use of
    a `Pydantic` model,
    but is not detailed here for the sake of simplicity.

!!! warning
    **All the inputs and outputs names of the disciplines in a scenario shall be consistent**.

    - GEMSEO assumes that the data are tagged by their names
    with a global convention in the whole process.
    - What two disciplines call "X" shall be the same "X".
    The coupling variables for instance, are detected thanks to these conventions.

#### Inheritance

The disciplines are all subclasses of [Discipline][gemseo.core.discipline.discipline.Discipline], from which they must inherit.

To be used, if your [Discipline][gemseo.core.discipline.discipline.Discipline] of interest does not exist, you must:

- define a class inheriting from [Discipline][gemseo.core.discipline.discipline.Discipline],
- define the input and output grammars in the constructor,
- implement the `Discipline._run()` method which defines the way in which the output set values are obtained from the input set values.

!!! note
    Typically, when we deal with an interfaced software, the `Discipline._run()` method gets the inputs from the input grammar, calls a software, and writes the outputs to the output grammar.

!!! note
    The JSON grammars are automatically detected when they are in the same folder as your subclass module and named `"CLASSNAME_input.json"` and `"CLASSNAME_output.json"` and the `auto_detect_grammar_files` option is `True`.

### What are the API functions in GEMSEO?

Once a sub-class of [Discipline][gemseo.core.discipline.discipline.Discipline] is defined, an instance of this discipline can be created from the [create_disciplines()][gemseo.create_discipline] high-level function.

Furthermore, many disciplines inheriting from [Discipline][gemseo.core.discipline.discipline.Discipline] are already implemented in GEMSEO. Use the [get_available_disciplines][gemseo.get_available_disciplines] high-level function to discover them:

!!! note
    The available [Discipline][gemseo.core.discipline.discipline.Discipline] can be classified into different categories:

    -   classes implementing MDO problem disciplines:

        - Sobieski's SSBJ problem:
          [SobieskiAerodynamcs][gemseo.problems.mdo.sobieski.disciplines.SobieskiAerodynamics],
          [SobieskiMission][gemseo.problems.mdo.sobieski.disciplines.SobieskiMission],
          [SobieskiStructure][gemseo.problems.mdo.sobieski.disciplines.SobieskiStructure] and
          [SobieskiPropulsion][gemseo.problems.mdo.sobieski.disciplines.SobieskiPropulsion],
        - Sellar problem: [Sellar1][gemseo.problems.mdo.sellar.sellar_1.Sellar1], [Sellar2][gemseo.problems.mdo.sellar.sellar_2.Sellar2] and [SellarSystem][gemseo.problems.mdo.sellar.sellar_system.SellarSystem],
        - Aerostructure problem:
          [Structure][gemseo.problems.mdo.aerostructure.aerostructure.Structure],
          [Aerodynamics][gemseo.problems.mdo.aerostructure.aerostructure.Structure],
          [Mission][gemseo.problems.mdo.aerostructure.aerostructure.Mission],
        - Propane problem:
          [PropaneComb1][gemseo.problems.mdo.propane.propane.PropaneComb1],
          [PropaneComb2][gemseo.problems.mdo.propane.propane.PropaneComb2],
          [PropaneComb3][gemseo.problems.mdo.propane.propane.PropaneComb3] and
          [PropaneReaction][gemseo.problems.mdo.propane.propane.PropaneReaction],

    -   classes implementing special disciplines: [ParallelDisciplineChain][gemseo.core.chains.parallel_chain.ParallelDisciplineChain], [DisciplineChain][gemseo.core.chains.chain.DisciplineChain] and [MDOScenarioAdapter][gemseo.disciplines.scenario_adapters.mdo_scenario_adapter.MDOScenarioAdapter].
    -   classes implementing optimization discipline: [RosenMF][gemseo.problems.optimization.rosen_mf.RosenMF].

## How to instantiate an existing [Discipline][gemseo.core.discipline.discipline.Discipline]?

We can easily instantiate an internal discipline by means of the [create_discipline()][gemseo.create_discipline], e.g.:

``` python
from gemseo import create_discipline

sellar_system = create_discipline("SellarSystem")
```

We can easily instantiate multiple built-in disciplines by means of the [create_discipline()][gemseo.create_discipline] method, using a list of discipline names rather than a single discipline name, e.g.:

``` python
from gemseo import create_discipline

disciplines = create_discipline(["Sellar1", "Sellar2", "SellarSystem"])
```

In this case, `disciplines` is a list of [Discipline][gemseo.core.discipline.discipline.Discipline], where the first one is an instance of [Sellar1][gemseo.problems.mdo.sellar.sellar_1.Sellar1], the second one is an instance of [Sellar2][gemseo.problems.mdo.sellar.sellar_2.Sellar2] and the third one is an instance of [SellarSystem][gemseo.problems.mdo.sellar.sellar_system.SellarSystem].

!!! note
    If the constructor of a discipline has specific arguments, these arguments can be passed into a `dict` to the [create_discipline()][gemseo.create_discipline] method, e.g.:

    ``` python
    from gemseo import create_discipline

    discipline = create_discipline("MyDisciplineWithArguments", **kwargs)
    ```

    where `kwargs = {'arg1_key': arg1_val, 'arg1_key': arg1_val, ...}`.

!!! note
    We can easily instantiate an external discipline by means of the [create_discipline()][gemseo.create_discipline] (see [Extending Gemseo][extend-gemseo-features]):

    ``` python
    from gemseo import create_discipline

    discipline = create_discipline("MyExternalDiscipline")
    ```

## How to set the cache policy?

We can set the cache policy of a discipline by means of the [Discipline.set_cache()][gemseo.core.discipline.discipline.Discipline.set_cache] method, either using the default cache strategy, e.g.:

``` python
sellar_system.set_cache(cache_type=sellar_system.CacheType.SIMPLE)
```

or the HDF5 cache strategy with the discipline name as node name (here `SellarSystem`), e.g.:

``` python
sellar_system.set_cache(
    cache_type=sellar_system.CacheType.HDF5, cache_hdf_file="cached_data.hdf5"
)
```

or the HDF5 cache strategy with a user-defined name as node name (here `node`), e.g.:

``` python
sellar_system.set_cache(
    cache_type=sellar_system.CacheType.HDF5,
    cache_hdf_file="cached_data.hdf5",
    cache_hdf_node_path="node",
)
```

!!! note
    [Click here][caching-and-recording-discipline-data]. to get more information about caching strategies.

!!! note
    The [set_cache()][gemseo.core.discipline.discipline.Discipline.set_cache] method takes an additional argument, named `cache_tolerance`, which represents the tolerance for the approximate cache maximal relative norm difference to consider that two input arrays are equal.

    By default, `cache_tolerance` is equal to zero. We can get its value by means of the [cache.tolerance][gemseo.core.discipline.discipline.Discipline.cache] getter and change its value by means of the eponymous setter.

## How to execute an [Discipline][gemseo.core.discipline.discipline.Discipline]?

We can execute an [Discipline][gemseo.core.discipline.discipline.Discipline], either with its default input values, e.g.:

``` python
sellar_system.execute()
> {'obj': array([ 1.36787944+0.j]), 'y_2': array([ 1.+0.j]), 'y_1': array([ 1.+0.j]), 'c_1': array([ 2.16+0.j]), 'c_2': array([-23.+0.j]), 'x_shared': array([ 1.+0.j,  0.+0.j]), 'x_local': array([ 0.+0.j])}
```

or with user-defined values, defined into a `dict` indexed by input data names with NumPy array values, e.g.:

``` python
import numpy as np

input_data = {'y_1': array([ 2.]), 'x_shared': array([ 1.,  0.]), 'y_2': array([ 1.]), 'x_local': array([ 0.])}
sellar_system.execute(input_data)
> {'obj': array([ 4.36787944+0.j]), 'y_2': array([ 1.]), 'y_1': array([ 2.]), 'c_1': array([-0.84+0.j]), 'c_2': array([-23.+0.j]), 'x_shared': array([ 1.,  0.]), 'x_local': array([ 0.])}
```

## How to get information about an instantiated [Discipline][gemseo.core.discipline.discipline.Discipline]?

### 5.a. How to get input and output data names?

We can get the input and output data names by means of the `input_grammar.names` and `output_grammar.names`, e.g.:

``` python
print(sellar_system.input_grammar.names, sellar_system.output_grammar.names)
> ['y_1', 'x_shared', 'y_2', 'x_local'] ['c_1', 'c_2', 'obj']

```

### 5.b. How to check the validity of input or output data?

We can check the validity of a `dict` of input data (resp. output data) by means of the .io.input_grammar.validate()` (resp. `Discipline.io.output_grammar.validate()`}) methods, e.g.:

``` python
sellar_system.io.input_grammar.validate(sellar_system.default_input_data)
```

does not raise any error while:

``` python
sellar_system.io.input_grammar.validate({"a": array([1.0]), "b": array([1.0, -6.2])})
```

raises the error:

``` text
gemseo.core.grammars.InvalidDataException: Invalid input data for: SellarSystem
```

### How to get the default input values?

We can get the default input data by means of the [default_input_data][gemseo.core.discipline.discipline.Discipline.default_input_data] attribute, e.g.:

``` python
print(sellar_system.default_input_data)
> {'y_0': array([ 1.+0.j]), 'x_shared': array([ 1.+0.j,  0.+0.j]), 'y_1': array([ 1.+0.j]), 'x_local': array([ 0.+0.j])}
```

### How to get input and output data values?

#### All input or output data values as a dictionary

The same result can be obtained with a `dict` format by means of the [get_input_data()][gemseo.core.discipline.discipline.Discipline.get_input_data] and [get_output_data()][gemseo.core.discipline.discipline.Discipline.get_output_data] methods:

``` python
_ = sellar_system.execute()
sellar_system.get_input_data()
> {'x_local': array([ 0.+0.j]), 'x_shared': array([ 1.+0.j,  0.+0.j]), 'y_1': array([ 1.+0.j]), 'y_0': array([ 1.+0.j])}
sellar_system.get_output_data()
> {'c_1': array([ 2.16+0.j]), 'c_2': array([-23.+0.j]), 'obj': array([ 1.36787944+0.j])}
```

## How to read input and output data

The input and output data of the last execution are exposed as two separate
attributes:
[Discipline.input_data][gemseo.core.discipline.discipline.Discipline.input_data]
and
[Discipline.output_data][gemseo.core.discipline.discipline.Discipline.output_data].

``` python
print(sellar_system.input_data)
> {'y_2': array([ 1.+0.j]), 'y_1': array([ 1.+0.j]), 'x_shared': array([ 1.+0.j,  0.+0.j]), 'x_local': array([ 0.+0.j])}
print(sellar_system.output_data)
> {'obj': array([ 1.36787944+0.j]), 'c_1': array([ 2.16+0.j]), 'c_2': array([-23.+0.j])}
```

To update output values, use `Discipline.io.update_output_data()`:

``` python
sellar_system.io.update_output_data({'obj': array([1.]), 'new_variable': 'value'})
print(sellar_system.output_data)
> {'obj': array([ 1.]), 'new_variable': 'value', 'c_1': array([ 2.16+0.j]), 'c_2': array([-23.+0.j])}
```

The legacy `Discipline.local_data` attribute (and `IO.data`) is still
available as a deprecated read-only union of the two stores; accessing it
emits a `DeprecationWarning`. New code should use `input_data` and
`output_data` directly.
