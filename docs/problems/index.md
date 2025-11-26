<!--
 Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

 This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
 International License. To view a copy of this license, visit
 http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
 Commons, PO Box 1866, Mountain View, CA 94042, USA.
-->

# Benchmark problems

In this section, we describe the GEMSEO's benchmark MDO problems.

## Sellar's problem

The Sellar's problem is considered in different examples:

- [GEMSEO in 10 minutes][gemseo-in-10-minutes],
- [A from scratch example][a-from-scratch-example-on-the-sellar-problem].

### Description of the problem

The Sellar problem is defined by analytical functions:

$$
\begin{aligned}
\text{minimize the objective function }&\text{obj}=x_{\text{local}}^2 + x_{\text{shared},2}
+y_1^2+e^{-y_2} \\
\text{with respect to the design variables }&x_{\text{shared}},\,x_{\text{local}} \\
\text{subject to the general constraints }
& c_1 \leq 0\\
& c_2 \leq 0\\
\text{subject to the bound constraints }
& -10 \leq x_{\text{shared},1} \leq 10\\
& 0 \leq x_{\text{shared},2} \leq 10\\
& 0 \leq x_{\text{local}} \leq 10.
\end{aligned}
$$

where the coupling variables are

$$\text{Discipline 1: } y_1 = \sqrt{x_{\text{shared},1}^2 + x_{\text{shared},2} + x_{local} - 0.2\,y_2}$$

and

$$\text{Discipline 2: }y_2 = |y_1| + x_{\text{shared},1} + x_{\text{shared},2}.$$

and where the general constraints are

$$
c_1 = 3.16 - y_1^2
c_2 = y_2 - 24.
$$

The Sellar disciplines are also available with analytic derivatives in GEMSEO,
as well as a [DesignSpace][gemseo.algos.design_space.DesignSpace]:

### Creation of the disciplines

To create the Sellar disciplines, use the function [create_discipline()][gemseo.create_discipline]:

``` python
from gemseo import create_discipline

disciplines = create_discipline(["Sellar1", "Sellar2", "SellarSystem"])
```

### Importation of the design space

The [SellarDesignSpace][gemseo.problems.mdo.sellar.sellar_design_space.SellarDesignSpace] can be imported as follows:

``` python
from gemseo.problems.mdo.sellar.sellar_design_space import SellarDesignSpace

design_space = SellarDesignSpace()
```

Then, you can visualize it with `print(design_space)`:

``` shell
    +----------+-------------+--------+-------------+-------+
    | name     | lower_bound | value  | upper_bound | type  |
    +----------+-------------+--------+-------------+-------+
    | x_local  |      0      | (1+0j) |      10     | float |
    +          +             +        +             +       +
    | x_shared |     -10     | (4+0j) |      10     | float |
    +          +             +        +             +       +
    | x_shared |      0      | (3+0j) |      10     | float |
    +          +             +        +             +       +
    | y_1      |     -100    | (1+0j) |     100     | float |
    +          +             +        +             +       +
    | y_2      |     -100    | (1+0j) |     100     | float |
    +----------+-------------+--------+-------------+-------+
```

!!! info "See Also"
        See [this example][a-from-scratch-example-on-the-sellar-problem] to create the Sellar problem from scratch.

## Sobieski's SSBJ test case

The Sobieski's SSBJ test case is considered in the different examples:

- [DOE example][mdf-based-doe-on-the-sobieski-ssbj-test-case],
- [MDO example][application-sobieskis-super-sonic-business-jet-mdo].

### Origin of the test case

This test was taken from the reference article by Sobieski, the first
publication on the BLISS formulation. It is based on a 1996 AIAA student
competition organized by the AIAA/United Technologies/Pratt & Whitney
Individual Undergraduate Design Competition. This competition initially
focused on both the technical and economical challenges of a
development.

The formulas used for each are based on semi-empirical and/or analytical
models. Depending on the , some examples can be found in the following
references [@niu],[@anderson],[@Raymer].
As specified by Sobieski *et al.*, additional design and state
variables (not present in the original problem) were introduced in the
disciplinary analyses for testing purposes.

### The MDO problem

The design problem is
to maximize the range `"y_4"` of a Super-Sonic Business Jet (SSBJ)
with respect to the design variables `"x_shared"`, `"x_1"`, `"x_2"` and `"x_3"`
under the constraints `"g_1"`, `"g_2"` and `"g_3"`.

The objective and constraints are computed by four disciplines:

1. [SobieskiStructure][gemseo.problems.mdo.sobieski.disciplines.SobieskiStructure] (indexed by 1)
   computes the constraint `"g_1"` from `"x_shared"` and `"x_1"`,
2. [SobieskiAerodynamics][gemseo.problems.mdo.sobieski.disciplines.SobieskiAerodynamics] (indexed by 2)
   computes the constraint `"g_2"` from `"x_shared"` and `"x_2"`,
3. [SobieskiPropulsion][gemseo.problems.mdo.sobieski.disciplines.SobieskiPropulsion] (indexed by 3)
   computes the constraint `"g_3"` from `"x_shared"` and `"x_3"`,
4. [SobieskiMission][gemseo.problems.mdo.sobieski.disciplines.SobieskiMission] (indexed by 4)
   computes the constraint `"y_4"` from `"x_shared"`.

`"x_shared"` denotes the global design variables <shared design variables> (a.k.a. shared design variables),
which means that these variables are shared by at least two disciplines (here, all).
`"x_1"`, `"x_2"` and `"x_3"` denote the local design variables <local design variables>,
which means that `"x_i"` is used by a single discipline (here, number *i*).

The disciplines
[SobieskiStructure][gemseo.problems.mdo.sobieski.disciplines.SobieskiStructure],
[SobieskiAerodynamics][gemseo.problems.mdo.sobieski.disciplines.SobieskiAerodynamics]
and [SobieskiPropulsion][gemseo.problems.mdo.sobieski.disciplines.SobieskiPropulsion]
are strongly coupled to each other
but weakly coupled to [SobieskiMission][gemseo.problems.mdo.sobieski.disciplines.SobieskiMission].
The coupling variable <coupling variables> `"y_ij"` denotes
an output of the discipline no. *i* and an input of the discipline no. *j*.

### Input variables

![The planform variables](../problems/sobieski_figures/SSBJ.png)

![The airfoils variables](../problems/sobieski_figures/SupersonicAirfoil.png)

| Disciplines | Variable                          | Description   |   Bounds                 |    Notation   |
| ----------- | --------------------------------- | ------------- | ------------------------ | ------------- |
| All         |  $t/c$                           | Thickness to chord ratio  | $0.01\leq t/c\leq 0.09$ |   `"x_shared[0]"` |
| All         |  $h$                             |   Altitude ($ft$)  | $30000\leq h \leq 60000$ |  `"x_shared[1]"` |
| All         |  $M$                             |   Mach number      | $1.4\leq M\leq 1.8$      |  `"x_shared[2]"` |
| All         |  $AR=b^2/S_W$                    |   Aspect ratio   |  $2.5\leq AR\leq 8.5$     |  `"x_shared[3]"` |
| All         |  $\Lambda$                       |  Wing sweep ($\deg$)   |   $40\leq\Lambda\leq70$    |  `"x_shared[4]"` |
| All         |  $S_W$                           |   Wing surface area ($ft^2$)  |  $500\leq S\leq 1500$     |  `"x_shared[5]"` |
| Structure   |  $\lambda = {c_{tip}}/{c_{root}}$ |  Wing taper ratio | $0.1\leq\lambda\leq0.4$  | `"x_1[0]"` |
| Structure   |  $x$                             |   Wingbox x-sectional area ($ft^2$)     |    $0.75\leq x \leq 1.25$    | `"x_1[1]"` |
| Structure   |  $L$                             |   Lift from by Aerodynamics ($N$)       |                     |  `"y_21[0]"` |
| Structure   |  $W_{E}$                         |   Engine mass from Propulsion  ($lb$)   |                     |   `"y_31[0]"` |
| Aerodynamics | $C_f$                           |   Skin friction coefficient |   $0.75\leq C_f\leq 1.25$  |  `"x_2[0]"` |
| Aerodynamics | $W_T$                           |   Total aircraft mass from Structure ($lb$)     |                |       `"y_12[0]"` |
| Aerodynamics | $\Delta\alpha_v$                |   Wing twist from Structure                     |      | `"y_12[1]"` |
| Propulsion   | $ESF$                           |   Engine scale factor (ESF) from Propulsion     |      |                   `"y_32[0]"` |
| Propulsion   | $Th$                            |   Throttle setting (engine mass flow) | $0.1\leq Th\leq 1.25$  |    `"x_3[0]"` |
| Propulsion   | $D$                             |   Drag from Aerodynamics ($N$)                      |          |    `"y_23[0]"` |
| Mission      | $L/D$                           |   Lift-over-drag ratio from Aerodynamics            |          |    `"y_24[0]"` |
| Mission      | $W_T$                           |   Total aircraft mass from Structure ($lb$)         |          |    `"y_14[0]"` |
| Mission      | $W_F$                           |   Fuel mass from Structure ($lb$)                   |          |    `"y_14[1]"` |
| Mission      | $SFC$                           |   Specific fuel consumption (SFC) from Propulsion   |          |    `"y_34[1]"` |

### Output variables

| Disciplines | Variable | Description | Bounds | Notation |
|-------------|----------|-------------|--------|----------|
| Structure | $\sigma_1-1.09$ | Constraint about stress $\sigma_1$ on wing section 1 | $\sigma_1-1.09\leq 0$ | `"g_1[0]"` |
| Structure | $\sigma_2-1.09$ | Constraint about stress $\sigma_2$ on wing section 2 | $\sigma_2-1.09\leq 0$ | `"g_1[1]"` |
| Structure | $\sigma_3-1.09$ | Constraint about stress $\sigma_3$ on wing section 3 | $\sigma_3-1.09\leq 0$ | `"g_1[2]"` |
| Structure | $\sigma_4-1.09$ | Constraint about stress $\sigma_4$ on wing section 4 | $\sigma_4-1.09\leq 0$ | `"g_1[3]"` |
| Structure | $\sigma_5-1.09$ | Constraint about stress $\sigma_5$ on wing section 5 | $\sigma_5-1.09\leq 0$ | `"g_1[4]"` |
| Structure | $\Delta\alpha_{v}-1.04$ | First constraint about wing twist $\Delta\alpha_{v}$ | $\Delta\alpha_{v}-1.04\leq 0$ | `"g_1[5]"` |
| Structure | $0.96-\Delta\alpha_{v}$ | Second constraint about wing twist $\Delta\alpha_{v}$ | $0.96-\Delta\alpha_{v}\leq 0$ | `"g_1[6]"` |
| Structure | $W_T$ | Total aircraft mass ($lb$) |  | `"y_1[0]"` |
| Structure | $W_F$ | Fuel mass ($lb$) |  | `"y_1[1]"` |
| Structure | $\Delta\alpha_{v}$ | Wing twist ($\deg$) | $0.96\leq \Delta\alpha_{v}\leq 1.04$ | `"y_1[2]"` |
| Aerodynamics | $L$ | Lift ($lb$) |  | `"y_2[0]"` |
| Aerodynamics | $D$ | Drag ($lb$) |  | `"y_2[1]"` |
| Aerodynamics | $L/D$ | Lift-over-drag ratio |  | `"y_2[2]"` |
| Aerodynamics | $dp/dx-1.04$ | Constraint about the pressure gradient $dp/dx$ | $dp/dx-1.04\leq 0$ | `"g_2[0]"` |
| Propulsion | $SFC$ | Specific fuel consumption (SFC) |  | `"y_3[0]"` |
| Propulsion | $W_E$ | Engine mass ($lb$) |  | `"y_3[1]"` |
| Propulsion | $ESF$ | Engine scale factor (ESF) | $0.5\leq ESF \leq 1.5$ | `"y_3[2]"` |
| Propulsion | $ESF-1.5$ | First constraint about the ESF | $ESF-1.5 \leq 0$ | `"g_3[0]"` |
| Propulsion | $0.5-ESF$ | Second constraint about the ESF | $0.5-ESF \leq 0$ | `"g_3[1]"` |
| Propulsion | $Th-Th_{uA}$ | Constraint about the throttle $Th$ | $Th-Th_{uA}\leq 0$ | `"g_3[2]"` |
| Propulsion | $T_E-1.02$ | Constraint about the engine temperature $T_E$ | $T_E-1.02\leq 0$ | `"g_3[3]"` |
| Mission | $R$ | Range ($nm$) |  | `"y_4[0]"` |

### Creation of the disciplines

To create the disciplines of the Sobieski problem:

``` python
from gemseo import  create_discipline

disciplines = create_discipline(
  ["SobieskiStructure", "SobieskiPropulsion", "SobieskiAerodynamics", "SobieskiMission"]
  )
```

### Reference results

This problem was implemented by Sobieski *et al.* in Matlab and Isight.
Both implementations led to the same results.

As all gradients can be computed, we resort to gradient-based optimization methods.
All Jacobian matrices are coded analytically in GEMSEO.

Reference results using the [MDF formulation][the-mdf-formulation] are presented in the next table.

| Variable                          | Initial      | Optimum       |
| --------------------------------- | ------------ | ------------- |
| **Range (nm)**                    | **535.79**   | **3963.88**   |
| $\lambda$                         | 0.25         | 0.38757       |
| $x$                               | 1            | 0.75          |
| $C_f$                             | 1            | 0.75          |
| $Th$                              | 0.5          | 0.15624       |
| $t/c$                             | 0.05         | 0.06          |
| $h (ft)$                          | 45000        | 60000         |
| $M$                               | 1.6          | 1.4           |
| $AR$                              | 5.5          | 2.5           |
| $\Lambda (\deg)$                  | 55           | 70            |
| $S_W (ft^2)$                      | 1000         | 1500          |

## The Propane combustion problem

The Propane MDO problem can be found in [@Padula1996] and [@TedfordMartins2006]. It represents the
chemical equilibrium reached during the combustion of propane in air. Variables are
assigned to represent each of the ten combustion products as well as the sum of the
products.

The optimization problem is as follows:

$$
\begin{aligned}
\text{minimize the objective function }& f_2 + f_6 + f_7 + f_9 \\
\text{with respect to the design variables }&x_{1},\,x_{3},\,x_{6},\,x_{7} \\
\text{subject to the general constraints }
& f_2(x) \geq 0\\
& f_6(x) \geq 0\\
& f_7(x) \geq 0\\
& f_9(x) \geq 0\\
\text{subject to the bound constraints }
& x_{1} \geq 0\\
& x_{3} \geq 0\\
& x_{6} \geq 0\\
& x_{7} \geq 0\\
\end{aligned}
$$

where the System Discipline consists of computing the following expressions:

$$
\begin{aligned}
f_2(x) & = & 2x_1 + x_2 + x_4 + x_7 + x_8 + x_9 + 2x_{10} - R, \\
f_6(x) & = & K_6x_2^{1/2}x_4^{1/2} - x_1^{1/2}x_6(p/x_{11})^{1/2}, \\
f_7(x) & = & K_7x_1^{1/2}x_2^{1/2} - x_4^{1/2}x_7(p/x_{11})^{1/2}, \\
f_9(x) & = & K_9x_1x_3^{1/2} - x_4x_9(p/x_{11})^{1/2}. \\
\end{aligned}
$$

Discipline 1 computes $(x_{2}, x_{4})$ by satisfying the following equations:

$$
\begin{aligned}
x_1 + x_4 - 3 &=& 0,\\
K_5x_2x_4 - x_1x_5 &=& 0.\\
\end{aligned}
$$
Discipline 2 computes $(x_2, x_4)$ such that:

$$\begin{aligned}
K_8x_1 + x_4x_8(p/x_{11}) &=& 0,\\
K_{10}x_{1}^{2} - x_4^2x_{10}(p/x_{11}) &=& 0.\\
\end{aligned}
$$

and Discipline 3 computes $(x_5, x_9, x_{11})$ by solving:

$$
\begin{aligned}
2x_2 + 2x_5 + x_6 + x_7 - 8&=& 0,\\
2x_3 + x_9 - 4R &=& 0, \\
x_{11} - \sum_{j=1}^{10} x_j &=& 0. \\
\end{aligned}
$$

### Creation of the disciplines

The Propane combustion disciplines are available in GEMSEO and can be imported with the following code:

``` python
from gemseo import  create_discipline

disciplines = create_discipline(["PropaneComb1", "PropaneComb2", "PropaneComb3", "PropaneReaction"])
```

A [DesignSpace][gemseo.algos.design_space.DesignSpace] file *propane_design_space.csv* is also available in the same folder,
which can be read using the [read_design_space()][gemseo.read_design_space] function.

### Problem results

The optimum is $(x1,x3,x6,x7) = (1.378887, 18.426810, 1.094798, 0.931214)$.
The minimum objective value is $0$. At this point,  all the system-level inequality constraints are active.

## Aerostructure problem

The Aerostructure problem is considered in [this example][scalable-problem].

### Description of the problem

The Aerostructure problem is defined by analytical functions:

$$\text{OVERALL AIRCRAFT DESIGN} = \left\{
\begin{aligned}
&\text{minimize }\text{range}(\text{thick_airfoils}, \text{thick_panels}, \text{sweep}) = 8\times10^{11}\times\text{lift}\times\text{mass}/\text{drag} \\
&\text{with respect to }\text{thick_airfoils},\,\text{thick_panels},\,\text{sweep} \\
&\text{subject to }\\
& \text{rf}-0.5 = 0\\
& \text{lift}-0.5 \leq 0
\end{aligned}\right.$$

where

$$\text{AERODYNAMICS} = \left\{
    \begin{aligned}
&\text{drag}=0.1\times((\text{sweep}/360)^2 + 200 + \text{thick_airfoils}^2 - \text{thick_airfoils} - 4\times\text{displ})\\
&\text{forces}=10\times\text{sweep} + 0.2\times\text{thick_airfoils}-0.2\times\text{displ}\\
&\text{lift}=(\text{sweep} + 0.2\times\text{thick_airfoils}-2\times\text{displ})/3000
    \end{aligned}
    \right.$$

and

$$\text{STRUCTURE} = \left\{
    \begin{aligned}
&\text{mass}=4000\times(\text{sweep}/360)^3 + 200000 + 100\times\text{thick_panels} + 200\times\text{forces}\\
&\text{rf}=3\times\text{sweep} - 6\times\text{thick_panels} + 0.1\times\text{forces} + 55\\
&\text{displ}=2\times\text{sweep} + 3\times\text{thick_panels} - 2\times\text{forces}
    \end{aligned}
    \right.$$

The Aerostructure disciplines are also available with analytic derivatives in the classes
[Mission][gemseo.problems.mdo.aerostructure.aerostructure.Mission],
[Aerodynamics][gemseo.problems.mdo.aerostructure.aerostructure.Aerodynamics],
[Structure][gemseo.problems.mdo.aerostructure.aerostructure.Structure],
and
[AerostructureDesignSpace][gemseo.problems.mdo.aerostructure.aerostructure_design_space.AerostructureDesignSpace].

### Creation of the disciplines

To create the aerostructure disciplines, use the function [create_discipline()][gemseo.create_discipline]:

``` python
from gemseo import create_discipline

disciplines = create_discipline(["Aerodynamics", "Structure", "Mission"])
```

### Importation of the design space

The
[AerostructureDesignSpace][gemseo.problems.mdo.aerostructure.aerostructure_design_space.AerostructureDesignSpace]
class can be imported as follows:

``` python
from gemseo.problems.aerostructure.aerostructure_design_space import AerostructureDesignSpace
design_space = AerostructureDesignSpace()
```

Then, you can visualize it with `print(design_space)`:

``` shell
    +----------------+-------------+-------------+-------------+-------+
    | name           | lower_bound |    value    | upper_bound | type  |
    +----------------+-------------+-------------+-------------+-------+
    | thick_airfoils |      5      |   (15+0j)   |      25     | float |
    | thick_panels   |      1      |    (3+0j)   |      20     | float |
    | sweep          |      10     |   (25+0j)   |      35     | float |
    | drag           |     100     |   (340+0j)  |     1000    | float |
    | forces         |    -1000    |   (400+0j)  |     1000    | float |
    | lift           |     0.1     |   (0.5+0j)  |      1      | float |
    | mass           |    100000   | (100000+0j) |    500000   | float |
    | displ          |    -1000    |  (-700+0j)  |     1000    | float |
    | rf             |    -1000    |      0j     |     1000    | float |
    +----------------+-------------+-------------+-------------+-------+
```
