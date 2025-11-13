<!--
 Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

 This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
 International License. To view a copy of this license, visit
 http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
 Commons, PO Box 1866, Mountain View, CA 94042, USA.
-->

<!--
Contributors:
   :author:  Francois Gallard, Gilberto Ruiz
-->

# Excel wrapper

This section describes how to use an [XLSDiscipline][gemseo.disciplines.wrappers.xls_discipline.XLSDiscipline] with a practical
application using a simple discipline in a [DOEScenario][gemseo.scenarios.doe_scenario.DOEScenario].

## Imports

We start by importing all the necessary dependencies.

``` python
from numpy import array, ones

from gemseo import create_design_space, create_scenario
from gemseo.disciplines.wrappers.xls_discipline import XLSDiscipline
```

## Create an XLSDiscipline

For this simple problem, the Excel book will compute $c=a+b$.

1. Create the Excel file that will compute the outputs (`c`) from the
    inputs (`a`, `b`). Inputs must be specified in the "Inputs" sheet:

    |     | A   | B   |
    |-----|-----|-----|
    | 1   | a   | 3   |
    | 2   | b   | 5   |

    "Inputs" sheet setting $a=3$ and $b=5$.

    !!! warning
         The number of rows is arbitrary, but they must be contiguous (no empty lines) and start at line 1.

    The same applies for the "Outputs" sheet:

    |     | A   | B   |
    |-----|-----|-----|
    | 1   | c   | 8   |

    "Outputs" sheet setting $c=8$.

2. Instantiate the discipline. For this basic implementation we only
    need to provide the path to the Excel file: `my_book.xlsx`.

``` python
xls_discipline = XLSDiscipline('my_book.xlsx')
```

## Instantiate the scenario

The scenario requires a [DesignSpace][gemseo.algos.design_space.DesignSpace]
defining the design variables `a` and `b`:

``` python
design_space = create_design_space()
design_space.add_variable("a", 1, lower_bound=0.0, upper_bound=10.0, value=array([1]))
design_space.add_variable("b", 1, lower_bound=-10.0, upper_bound=10.0, value=array([2]))
```

Create the [DOEScenario][gemseo.scenarios.doe_scenario.DOEScenario]
with the [XLSDiscipline][gemseo.disciplines.wrappers.xls_discipline.XLSDiscipline],
the [DesignSpace][gemseo.algos.design_space.DesignSpace]
and an [MDF][gemseo.formulations.mdf.MDF] formulation:

``` python
scenario = create_scenario(
    xls_discipline,
    "c",
    design_space,
    formulation_name="DisciplinaryOpt",
    scenario_type="DOE",
)
```

## Execute the Scenario

Define the execution options using a dictionary, then execute the
scenario. Here, we use a [CustomDOE][gemseo.algos.doe.custom_doe.custom_doe.CustomDOE] and provide two samples to be
evaluated:

``` python
sample_1 = [1, 2]  # a=1, b=2
sample_2 = [2, 3]  # a=2, b=3
samples = array([sample_1, sample_2])
scenario.execute(algo_name="CustomDOE", samples=samples)
print(scenario.to_dataset().export_to_dataframe())
```

Which prints the results of the computation as follows:

``` bash
design_parameters      functions
                a    b         c
                0    0         0
0               1.0  2.0       3.0
1               2.0  3.0       5.0
```

## Parallel execution considerations

GEMSEO relies on the [xlswings library](https://www.xlwings.org) to
communicate with Excel. This imposes some constraints to our
development. In particular, [we cannot pass xlwings objects between
processes or
threads](https://docs.xlwings.org/en/stable/threading_and_multiprocessing.html).
We have different strategies to comply with this requirement in parallel
execution, depending on whether we are using multiprocessing,
multithreading or both.

In the following, we no longer use the previous discipline to illustrate
these parallel execution considerations but an [XLSDiscipline][gemseo.disciplines.wrappers.xls_discipline.XLSDiscipline] named
`xls_discipline` and strongly coupled to another discipline called
`other_discipline`. The idea is to minimize the objective function `"f"`
computed by this multidisciplinary system over a design space. For
that, we will use the MDF formulation:

``` python
scenario = create_scenario(
    [xls_discipline, other_discipline],
    "f",
    design_space,
    formulation_name="MDF",
    scenario_type='DOE',
)
```

### Multiprocessing

In multiprocessing, we recreate the `xlwings` object in each subprocess
through `__setstate__`. However, the same Excel file cannot be used by
all the subprocesses at the same time. Which means that we need a unique
copy of the original file for each one.

The option `copy_xls_at_setstate` shall be set to `True` whenever an
[XLSDiscipline][gemseo.disciplines.wrappers.xls_discipline.XLSDiscipline] will be used
in a [DiscParallelExecution][gemseo.core.parallel_execution.disc_parallel_execution.DiscParallelExecution] instance implementing multiprocessing.

If we wanted to run the previously defined scenario in parallel, then
the discipline instantiation would be:

``` python
xls_discipline = XLSDiscipline('my_book.xlsx', copy_xls_at_setstate=True)
```

The algo settings would change as well to request the number of
processes to run and the execution call shall be protected:

``` python
if __name__ == '__main__':
    scenario.execute(algo_name="CustomDOE", samples=samples, n_processes=2)
```

### Multithreading

In multithreading, we recreate the `xlwings` object at each call to the
[XLSDiscipline][gemseo.disciplines.wrappers.xls_discipline.XLSDiscipline]. Thus, when instantiating an [XLSDiscipline][gemseo.disciplines.wrappers.xls_discipline.XLSDiscipline] that will
be executed in multithreading, the user must set
`recreate_book_at_run=True`.

!!! warning
      An [MDAJacobi][gemseo.mda.jacobi.MDAJacobi] uses multithreading to accelerate its convergence,
      even if the overall scenario is being run in serial mode. If your
      [XLSDiscipline][gemseo.disciplines.wrappers.xls_discipline.XLSDiscipline] is inside an [MDAJacobi][gemseo.mda.jacobi.MDAJacobi], you must instantiate it
      with `recreate_book_at_run=True`.

Going back to the example scenario, if we want to run it using an
[MDAJacobi][gemseo.mda.jacobi.MDAJacobi] then the [XLSDiscipline][gemseo.disciplines.wrappers.xls_discipline.XLSDiscipline] would be created as follows:

``` python
xls_discipline = XLSDiscipline('my_book.xlsx', copy_xls_at_setstate=True)
```

The scenario creation would specify the MDA:

``` python
scenario = create_scenario(
    [xls_discipline, other_discipline],
    "f",
    design_space,
    formulation_name="MDF",
    main_mda_class="MDAJacobi",
    scenario_type="DOE",
)
```

The scenario execution remains the same:

``` python
scenario.execute(algo_name="CustomDOE", samples=samples)
```

### Multiprocessing & Multithreading

There is one last case to consider, which occurs when the
[XLSDiscipline][gemseo.disciplines.wrappers.xls_discipline.XLSDiscipline]will run in multithreading mode from a subprocess that
was itself created by a multiprocessing instance. A good example of this
particular situation is when a [DOEScenario][gemseo.scenarios.doe_scenario.DOEScenario] runs in parallel with an
[MDAJacobi][gemseo.mda.jacobi.MDAJacobi] that solves the couplings for each sample.

It will be necessary to set both `copy_xls_at_setstate=True` and
`recreate_book_at_run=True`.

In our example, the `XLSDiscipline` instantiation would be:

``` python
xls_discipline = XLSDiscipline('my_book.xlsx', copy_xls_at_setstate=True, recreate_book_at_run=True)
```

The scenario would be created as follows:

``` python
scenario = create_scenario(
    [xls_discipline, other_discipline],
    "f",
    design_space,
    formulation_name="MDF",
    main_mda_class="MDAJacobi",
    scenario_type="DOE",
)
```

The algo options would change as well to request the number of processes
to run: and the execution call shall be protected:

``` python
if __name__ == '__main__':
    scenario.execute(algo_name="CustomDOE", samples=samples, n_processes=2)
```

## What about macros?

The next figure illustrates how a macro can be wrapped to compute
outputs from inputs. You shall pass the name of the macro with the
option `macro_name` at instantiation.

![Example of macro that can be wrapped](figs/xls_macro.png)
