<!--
 Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

 This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
 International License. To view a copy of this license, visit
 http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
 Commons, PO Box 1866, Mountain View, CA 94042, USA.
-->

# About us

## Context

The potential of numerical optimization techniques to assist the designers and automate shape design improvements is foreseen since the 80s, but is still not widely used as of today, in particular to address multidisciplinary design problems.

Several reasons explain that, among which:

- The difficulty to use state of the art numerical optimization method in an industrial context.
- The difficulty to create fully automated design optimization process involving multiple physics, components, disciplines. That domain is called MDO (Multidisciplinary Design Optimization).
- The limitation of current numerical optimization processes technologies (PIDO, Process Integration and Design Automation) in terms of adaptability to a wide range of use cases, and in terms of maintenance in industrial environments.

These technical challenges motivated the development of GEMSEO.

GEMSEO is a scientific software for engineers and researchers used to automatically explore design spaces and find optimal multidisciplinary solutions.

GEMSEO aims at reducing the cost and setup time needed to develop and maintain automated simulation processes.

GEMSEO relies on a disruptive approach based on MDO formulations.

An MDO formulation, or architecture, is a simulation process template, or generic strategy. This enables to generate automatically the MDO process and facilitate its reconfiguration.

## History

GEMS (Generic Engine for MDO Scenarios) was created in 2015 by François Gallard
within the Multidisciplinary Design Optimization (MDO) Competence Center of [IRT Saint Exupéry](https://www.irt-saintexupery.com/).

![IRT Saint Exupéry](_static/contributors/irt.png){ width=25%, align=right }
This team,
under the leadership of Anne Gazaix,
is dedicated to the development of process automation technologies encompassing a wide range of disciplines,
and their usage in a range of applications.

GEMS became GEMSEO in 2021 when [IRT Saint Exupéry](https://www.irt-saintexupery.com/)
and its partners decided to release it as an open source project
with the aim to make it widely collaborative.
![IRT Saint Exupéry](_static/contributors/pia.png){ width=12.5%, align=left }
GEMSEO is the acronym for Generic Engine for Multi-disciplinary Scenarios, Exploration and Optimization.

GEMSEO is the result of successive projects carried by [IRT Saint Exupéry](https://www.irt-saintexupery.com/),
funded by both [Investments for the Future Programme](https://www.gouvernement.fr/le-programme-d-investissements-d-avenir) (french acronym: PIA)
and companies and involving both academic and industrial partners.

<div class="md-grid" style="display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 10px;">
<a href="https://www.airbus.com/" target="_blank" width="12.5%"><img src="../_static/contributors/airbus.png" alt="Airbus"/></a>
<a href="https://www.altran.com/" target="_blank" width="12.5%"><img src="../_static/contributors/altran.png" alt="Altran"/></a>
<a href="https://www.capgemini.com/" target="_blank" width="12.5%"><img src="../_static/contributors/capgemini.png" alt="Capgemini"/></a>
<a href="https://www.cenaero.be/" target="_blank" width="12.5%"><img src="../_static/contributors/cenaero.jpg" alt="CENAERO"/></a>
<a href="https://www.cerfacs.fr/en/" target="_blank" width="12.5%"><img src="../_static/contributors/cerfacs.png" alt="CERFACS"/></a>
<a href="https://expleogroup.com/" target="_blank" width="12.5%"><img src="../_static/contributors/expleo.png" alt="Expleo"/></a>
<a href="https://ica.cnrs.fr/home/" target="_blank" width="12.5%"><img src="../_static/contributors/ica.jpg" alt="Institut Clément Ader"/></a>
<a href="https://www.insa-toulouse.fr/en" target="_blank" width="12.5%"><img src="../_static/contributors/insa.jpg" alt="INSA Toulouse"/></a>
<a href="https://www.isae-supaero.fr/en" target="_blank" width="12.5%"><img src="../_static/contributors/isae_supaero.png" alt="ISAE-SUPAERO"/></a>
<a href="https://www.liebherr.com" target="_blank" width="12.5%"><img src="../_static/contributors/liebherr.jpg" alt="Liebherr"/></a>
<a href="https://www.madeleine-project.eu" target="_blank" width="12.5%"><img src="../_static/contributors/madeleine.png" alt="MADELEINE project"/></a>
<a href="https://www.stelia-aerospace.com/en" target="_blank" width="12.5%"><img src="../_static/contributors/stelia.jpg" alt="STELIA Aerospace"/></a>
</div>
*All the logos belong to their owners and cannot be reused without their consent.*

### Key contributing project

In 2015, [IRT Saint Exupéry](https://www.irt-saintexupery.com/) launched the MDA-MDO project (2015-2019), with the contributions of the following members: [Airbus](https://www.airbus.com/), [Altran](https://www.altran.com/), [Capgemini](https://www.capgemini.com/), [Cerfacs](https://www.cerfacs.fr/en/), [ISAE-SUPAERO](https://www.isae-supaero.fr/en/) & [ICA](https://ica.cnrs.fr/home/), and in collaboration with [ONERA](https://www.onera.fr/en). This team developed the core elements of GEMSEO by introducing a new paradigm merging dataflow and workflow strategies in order to make design process automation and reconfiguration possible. They added various algorithms (MDA, design of experiments, optimization, MDO formulations, etc.) to apply MDO on several test cases, from state-of-the art to industry-oriented ones.

### Maturation

[IRT Saint Exupéry](https://www.irt-saintexupery.com/) has brought GEMSEO in several projects to robustify its methods and extend its capabilities.

The [MADELEINE project](https://www.madeleine-project.eu/) (2017-2021), funded by the [European Union's Horizon 2020 research and innovation program](https://ec.europa.eu/programmes/horizon2020/en) under grant agreement No 769025, has improved the scalable data-driven modelling and has developed parallelism capabilities.

The VITAL project (2019-2021) carried by [IRT Saint Exupéry](https://www.irt-saintexupery.com/), with the contributions of the following members: [Airbus](https://www.airbus.com/) and [STELIA](https://www.stelia-aerospace.com/en/), contributes to GEMSEO by creating a package for uncertainty quantification and adaptive learning capabilities.

The R-EVOL project (2020-2024) carried by [IRT Saint Exupéry](https://www.irt-saintexupery.com/) implements a machine learning package for surrogate modelling, develops a framework for robust and reliable MDO under uncertainty and makes MDO techniques more efficient thanks to advanced numerical techniques and surrogate-based algorithms. with the contributions of following members: [Airbus](https://www.airbus.com/), [Expleo](https://expleogroup.com/), [Altran](https://www.altran.com/), [Capgemini](https://www.capgemini.com/), [Cerfacs](https://www.cerfacs.fr/en/), [Cenaero](http://www.cenaero.be/) and [INSA Toulouse](https://www.insa-toulouse.fr/en).

### Open source

Since 2021, GEMSEO is open source, under the [LGPL v3 license](https://www.gnu.org/licenses/lgpl-3.0.en.html). The project is hosted on [gitlab](https://gitlab.com/gemseo).

### Roadmap { #about-us-roadmap}

[IRT Saint Exupéry](https://www.irt-saintexupery.com/) and its partners choose to make their roadmap public in order to elicit contributions. [Discover it!](roadmap.md)

## Citation

If you produce communications (scientific papers, conferences, reports) about work using GEMSEO, thank you for citing us :

- Gallard, F., Vanaret, C., Guénot, D, *et al.*, [GEMS: A Python Library for Automation of Multidisciplinary Design Optimization Process Generation](https://arc.aiaa.org/doi/10.2514/6.2018-0657). In : 2018 AIAA/ASCE/AHS/ASC Structures, Structural Dynamics, and Materials Conference. 2018. p. 0657.

Bibtex entry:

    \@inproceedings{gemseo_paper,
    title={GEMS: A Python Library for Automation of Multidisciplinary Design Optimization Process Generation},
    author={Gallard, F. and Vanaret, C. and Guénot, D. and Gachelin, V. and Lafage, R. and Pauwels, B. and Barjhoux, P.-J. and Gazaix, A.},
    booktitle={2018 AIAA/ASCE/AHS/ASC Structures, Structural Dynamics, and Materials Conference},
    year={2018}
    }

## References

Here are some references about GEMSEO and its capabilities:

- Gallard, F., Vanaret, C., Guénot, D, *et al.*, [GEMS: A Python Library for Automation of Multidisciplinary Design Optimization Process Generation](https://arc.aiaa.org/doi/10.2514/6.2018-0657), In : 2018 AIAA/ASCE/AHS/ASC Structures, Structural Dynamics, and Materials Conference
- Gallard, F., Barjhoux, P. J., Olivanti, R., *et al.*, [GEMS, a Generic Engine for MDO Scenarios: Key Features In Application](https://doi.org/10.2514/6.2019-2991), In : AIAA Aviation 2019 Forum
- Gazaix, A., Gallard, F., Gachelin *et al.*, [Towards the Industrialization of New MDO Methodologies and Tools for Aircraft Design](https://doi.org/10.2514/6.2017-3149), In : 18th AIAA/ISSMO Multidisciplinary Analysis and Optimization Conference, 2017
- Gazaix, A., Gallard, F., Ambert, *et al.*, [Industrial Application of an Advanced Bi-level MDO Formulation to an Aircraft Engine Pylon Optimization](https://doi.org/10.2514/6.2019-3109), In : AIAA Aviation 2019 Forum
- Druot, T., Beleville, M., Roches, P., *et al.*, [A Multidisciplinary Airplane Research Integrated Library With Applications To Partial Turboelectric Propulsion.](https://doi.org/10.2514/6.2019-3243), In : AIAA Aviation 2019 Forum
- Barjhoux, P. J., Diouane, Y., Grihon, S., *et al.* [A bi-level methodology for solving large-scale mixed categorical structural optimization.](https://doi.org/10.1007/s00158-020-02491-w), In : Structural and Multidisciplinary Optimization, 2020
- Guénot, D., Gallard, F., Brezillon, J., *et al.* [Aerodynamic optimisation of a parametrised engine pylon on a mission path using the adjoint method](https://doi.org/10.1080/10618562.2019.1683163), In : International Journal of Computational Fluid Dynamics, 2019
- Olivanti, R., Gallard F., Brezillon, J, *et al.* [Comparison of Generic Multi-Fidelity Approaches for Bound-Constrained Nonlinear Optimization Applied to Adjoint-Based CFD Applications](https://doi.org/10.2514/6.2019-3102), In : AIAA Aviation 2019 Forum
- Vanaret, C., Gallard, F., Martins, J. R. [On the consequence of the "No Free Lunch" Theorem for Optimization on the Choice of an Appropriate MDO Architecture](https://arc.aiaa.org/doi/10.2514/6.2017-3148), 18th AIAA/ISSMO Multidisciplinary Analysis and Optimization Conference, Denver, CO, USA, 2017

!!! info "See Also"

    Find (almost) all the publications about GEMSEO [on this page](publications.md).

## Artwork

This is the logo of GEMSEO:

![image](_static/logo-small.png){ width=50% }

High quality PNG and SVG logos are available:

- [PNG format](_static/logo/gemseo_logo_transparent.png)
- [SVG format](_static/logo/gemseo_logo_transparent.svg)

## Authors

--8<-- "_docs/authors.md"

## Contributing

Anyone can contribute to the development of GEMSEO. The types of contributions are multiple:

- improving the documentation,
- declaring a bug, solving a bug,
- answering questions,
- proposing a new algorithm,
- suggesting a new feature,
- etc.

!!! info "See Also"

    Find more information on how to contribute to GEMSEO [by clicking here](contributing.md).
