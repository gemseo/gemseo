<!--
 Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

 This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
 International License. To view a copy of this license, visit
 http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
 Commons, PO Box 1866, Mountain View, CA 94042, USA.
-->

# The discipline, a key concept

## How is a discipline defined?

### What is a discipline?

A discipline is a set of calculations that:

- produces a dictionary of arrays as outputs
- from a dictionary of arrays as inputs
- using either a Python function, or equations or an external software, or a workflow engine.

### How is a discipline implemented in GEMSEO?

Programmatically speaking, disciplines are implemented in GEMSEO through the [Discipline][gemseo.core.discipline.discipline.Discipline] class. They are defined by three elements:

- the [input_grammar][gemseo.core.discipline.discipline.Discipline.input_grammar] attribute: the set of rules that defines valid input data,
- the [output_grammar][gemseo.core.discipline.discipline.Discipline.output_grammar] attribute: the set of rules that defines valid output data,
- the `Discipline._run()` method: the method to compute the output data from the input data.

#### Grammar

The input and output specifications are defined in a grammar, through the [input_grammar][gemseo.core.discipline.discipline.Discipline.input_grammar] and [output_grammar][gemseo.core.discipline.discipline.Discipline.output_grammar] attributes, which can be either a [SimpleGrammar][gemseo.core.grammars.simple_grammar.SimpleGrammar] or a [JSONGrammar][gemseo.core.grammars.json_grammar.JSONGrammar] (default grammar), or your own which derives from the [BaseGrammar][gemseo.core.grammars.base_grammar.BaseGrammar] class.

!!! note
    The *grammar* is a very powerful and key concept. There are multiple ways of creating grammars in GEMSEO. The preferred one for integrating simulation processes is the use of a `JSON schema`, but is not detailed here for the sake of simplicity. For more explanations about grammars, see [Interfacing simulation software](../interface/software_connection.md).

!!! warning
    **All the inputs and outputs names of the disciplines in a scenario shall be consistent**.

    -   GEMSEO assumes that the data are tagged by their names with a global convention in the whole process.
    -   What two disciplines call "X" shall be the same "X". The coupling variables for instance, are detected thanks to these conventions.

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

``` python
from gemseo import get_available_disciplines

get_available_disciplines()
> ['RosenMF', 'SobieskiAerodynamics', 'DOEScenario', 'MDOScenario', 'SobieskiMission', 'SobieskiBaseWrapper', 'Sellar1', 'Sellar2', 'MDOChain', 'SobieskiStructure', 'Structure', 'SobieskiPropulsion', 'BaseScenario', 'AnalyticDiscipline', 'MDOScenarioAdapter', 'SellarSystem', 'ScalableFittedDiscipline', 'Aerodynamics', 'Mission', 'PropaneComb1', 'PropaneComb2', 'PropaneComb3', 'PropaneReaction', 'MDOParallelChain']
```

!!! note
    These available [Discipline][gemseo.core.discipline.discipline.Discipline] can be classified into different categories:

    -   classes implementing scenario, a key concept in GEMSEO: [BaseScenario][gemseo.scenarios.base_scenario.BaseScenario] and [DOEScenario][gemseo.scenarios.doe_scenario.DOEScenario], [MDOScenario][gemseo.scenarios.mdo_scenario.MDOScenario],
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

    -   classes implementing special disciplines: [MDOParallelChain][gemseo.core.chains.parallel_chain.MDOParallelChain], [MDOChain][gemseo.core.chains.chain.MDOChain] and [MDOScenarioAdapter][gemseo.disciplines.scenario_adapters.mdo_scenario_adapter.MDOScenarioAdapter].
    -   classes implementing optimization discipline: [RosenMF][gemseo.problems.optimization.rosen_mf.RosenMF].

## How to instantiate an existing [Discipline][gemseo.core.discipline.discipline.Discipline]?

We can easily instantiate an internal discipline by means of the [create_discipline()][gemseo.create_discipline], e.g.:

``` python
from gemseo import create_discipline

sellar_system = create_discipline('SellarSystem')
```

We can easily instantiate multiple built-in disciplines by means of the [create_discipline()][gemseo.create_discipline] method, using a list of discipline names rather than a single discipline name, e.g.:

``` python
from gemseo import create_discipline

disciplines = create_discipline(['Sellar1', 'Sellar2', 'SellarSystem'])
```

In this case, `disciplines` is a list of [Discipline][gemseo.core.discipline.discipline.Discipline], where the first one is an instance of [Sellar1][gemseo.problems.mdo.sellar.sellar_1.Sellar1], the second one is an instance of [Sellar2][gemseo.problems.mdo.sellar.sellar_2.Sellar2] and the third one is an instance of [SellarSystem][gemseo.problems.mdo.sellar.sellar_system.SellarSystem].

!!! note
    If the constructor of a discipline has specific arguments, these arguments can be passed into a `dict` to the [create_discipline()][gemseo.create_discipline] method, e.g.:

    ``` python
    from gemseo import create_discipline

    discipline = create_discipline('MyDisciplineWithArguments', **kwargs)
    ```

    where `kwargs = {'arg1_key': arg1_val, 'arg1_key': arg1_val, ...}`.

!!! note
    We can easily instantiate an external discipline by means of the [create_discipline()][gemseo.create_discipline] (see [Extending Gemseo][extend-gemseo-features]):

    ``` python
    from gemseo import create_discipline

    discipline = create_discipline('MyExternalDiscipline')
    ```

## How to set the cache policy?

We can set the cache policy of a discipline by means of the [Discipline.set_cache()][gemseo.core.discipline.discipline.Discipline.set_cache] method, either using the default cache strategy, e.g.:

``` python
sellar_system.set_cache(cache_type=sellar_system.CacheType.SIMPLE)
```

or the HDF5 cache strategy with the discipline name as node name (here `SellarSystem`), e.g.:

``` python
sellar_system.set_cache(cache_type=sellar_system.CacheType.HDF5, cache_hdf_file='cached_data.hdf5')
```

or the HDF5 cache strategy with a user-defined name as node name (here `node`), e.g.:

``` python
sellar_system.set_cache(cache_type=sellar_system.CacheType.HDF5, cache_hdf_file='cached_data.hdf5', cache_hdf_node_path='node')
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
sellar_system.io.input_grammar.validate({'a': array([1.]), 'b': array([1., -6.2])})
```

raises the error:

``` text
gemseo.core.grammar.InvalidDataException: Invalid input data for: SellarSystem
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

## How to store data in the `local_data` attribute?

We can store data in the [Discipline.local_data][gemseo.core.discipline.discipline.Discipline.local_data] attribute by means of the `Discipline.io.update_output_data()` method whose arguments are the names of the variables to store. We can store either data for variables from input or output grammars, or data for other variables, e.g.:

``` python
print(sellar_system.local_data)
> {'obj': array([ 1.36787944+0.j]), 'y_2': array([ 1.+0.j]), 'y_1': array([ 1.+0.j]), 'c_1': array([ 2.16+0.j]), 'c_2': array([-23.+0.j]), 'x_shared': array([ 1.+0.j,  0.+0.j]), 'x_local': array([ 0.+0.j])}
sellar_system.io.update_output_data({'obj': array([1.]), 'new_variable': 'value'})
> {'obj': array([ 1.]), 'new_variable': 'value', 'y_2': array([ 1.+0.j]), 'y_1': array([ 1.+0.j]), 'c_1': array([ 2.16+0.j]), 'c_2': array([-23.+0.j]), 'x_shared': array([ 1.+0.j,  0.+0.j]), 'x_local': array([ 0.+0.j])}
```
