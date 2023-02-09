..
   Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

   This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
   International License. To view a copy of this license, visit
   http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
   Commons, PO Box 1866, Mountain View, CA 94042, USA.

About us
========

Context
-------

The potential of numerical optimization techniques to assist the designers and automate shape design improvements is foreseen since the 80s, but is still not widely used as of today, in particular to address multidisciplinary design problems.

Several reasons explain that, among which:

- The difficulty to use state of the art numerical optimization method in an industrial context.
- The difficulty to create fully automated design optimization process involving multiple physics, components, disciplines. That domain is called MDO (Multidisciplinary Design Optimization).
- The limitation of current numerical optimization processes technologies (PIDO, Process Integration and Design Automation) in terms of adaptability to a wide range of use cases, and in terms of maintenance in industrial environments.

These technical challenges motivated the development of |g|.

|g| is a scientific software for engineers and researchers used to automatically explore design spaces and find optimal multidisciplinary solutions.

|g| aims at reducing the cost and setup time needed to develop and maintain automated simulation processes.

|g| relies on a disruptive approach based on MDO formulations.

An MDO formulation, or architecture, is a simulation process template, or generic strategy.
This enables to generate automatically the MDO process and facilitate its reconfiguration.

History
-------

.. raw :: html

    <p>
    <div class="containter">
      <div class="row">
         <div class="col-8">

|g| was initially created in 2015 by François Gallard
within the Multidisciplinary Design Optimization (MDO) Competence Center
of `IRT Saint Exupéry <https://www.irt-saintexupery.com/>`_.
This team, under the leadership of Anne Gazaix, is dedicated to the development of process automation technologies
encompassing a wide range of disciplines, and their usage in a range of applications.

.. raw :: html

         </div>
         <div class="col-4">
            <a href='https://www.irt-saintexupery.com/' target="_blank"><img src='_static/contributors/irt.png'style="float: right;" /></a>
         </div>
      </div>
    </div>
    </p>

Originally known as GEMS (Generic Engine for MDO Scenarios),
GEMS became |g| in 2021 when
`IRT Saint Exupéry <https://www.irt-saintexupery.com/>`_ and its partners decided
to release it as an open source project
with the aim to make it widely collaborative.
|g| is the acronym for Generic Engine for Multi-disciplinary Scenarios, Exploration and Optimization.

.. raw :: html

    <p>
    <div class="containter">
      <div class="row">
         <div class="col-10">

|g| is the result of successive projects carried by `IRT Saint Exupéry <https://www.irt-saintexupery.com/>`_,
funded by both `Investments for the Future Programme <https://www.gouvernement.fr/le-programme-d-investissements-d-avenir>`_ (french acronym: PIA) and companies
and involving both academic and industrial partners.


.. raw :: html

         </div>
         <div class="col-2">
            <a href='https://www.gouvernement.fr/le-programme-d-investissements-d-avenir' target="_blank"><img src="_static/contributors/pia.png" title="Programm of Investments for the Future"/></a>
         </div>
      </div>
    </div>
    </p>

.. raw :: html

   <div class="container">
      <div class="row" style="margin-bottom:30px;">
        <div class="col my-auto text-center"><a href='https://www.airbus.com/' target="_blank"><img src='_static/contributors/airbus.png'/></a></div>
        <div class="col my-auto text-center"><a href='https://www.altran.com/' target="_blank"><img src='_static/contributors/altran.png' style="max-height: 75px;"/></a></div>
        <div class="col my-auto text-center"><a href='https://www.capgemini.com/' target="_blank"><img src='_static/contributors/capgemini.png'/></a></div>
        <div class="col my-auto text-center"><a href='http://www.cenaero.be/' target="_blank"><img src='_static/contributors/cenaero.jpg' style="max-height: 50px;"/></a></div>
      </div>
      <div class="row" style="margin-bottom:30px;">
        <div class="col my-auto text-center"><a href='https://www.cerfacs.fr/en/' target="_blank"><img src='_static/contributors/cerfacs.png'/></a></div>
        <div class="col my-auto text-center"><a href='https://expleogroup.com/' target="_blank"><img src='_static/contributors/expleo.png'/></a></div>
        <div class="col my-auto text-center"><a href='https://ica.cnrs.fr/home/' target="_blank"><img src='_static/contributors/ica.jpg' style="max-height: 50px;"/></a></div>
        <div class="col my-auto text-center"><a href='https://www.insa-toulouse.fr/en/index.html' target="_blank"><img src='_static/contributors/insa.jpg'/></a></div>
      </div>
      <div class="row" style="margin-bottom:30px;">
        <div class="col my-auto text-center"><a href='https://www.isae-supaero.fr/en/' target="_blank"><img src='_static/contributors/isae_supaero.png' style="max-height: 50px;"/></a></div>
        <div class="col my-auto text-center"><a href='https://www.liebherr.com/' target="_blank"><img src='_static/contributors/liebherr.jpg'/></a></div>
        <div class="col my-auto text-center"><a href='https://www.madeleine-project.eu/'><img src="_static/contributors/madeleine.png"/></a></div>
        <div class="col my-auto text-center"><a href='https://www.stelia-aerospace.com/en/' target="_blank"><img src='_static/contributors/stelia.jpg' style="max-height: 75px;"/></a></div>
      </div>
   </div>

*All the logos belong to their owners and cannot be reused without their consent.*

Key contributing project
************************

In 2015, `IRT Saint Exupéry <https://www.irt-saintexupery.com/>`_ launched the MDA-MDO project (2015-2019),
with the contributions of the following members:
`Airbus <https://www.airbus.com/>`_,
`Altran <https://www.altran.com/>`_,
`Capgemini <https://www.capgemini.com/>`_,
`Cerfacs <https://www.cerfacs.fr/en/>`_,
`ISAE-SUPAERO <https://www.isae-supaero.fr/en/>`_ & `ICA <https://institut-clement-ader.org/>`_,
and in collaboration with `ONERA <https://www.onera.fr/en>`_.
This team developed the core elements of |g| by introducing a new paradigm merging dataflow and workflow strategies in order to make design process automation and reconfiguration possible.
They added various algorithms (MDA, design of experiments, optimization, MDO formulations, etc.) to apply MDO on several test cases, from state-of-the art to industry-oriented ones.

Maturation
**********

`IRT Saint Exupéry <https://www.irt-saintexupery.com/>`_ has brought |g| in several projects to robustify its methods and extend its capabilities.

The `MADELEINE project <https://www.madeleine-project.eu/>`_ (2017-2021),
funded by the `European Union's Horizon 2020 research and innovation program <https://ec.europa.eu/programmes/horizon2020/en>`_ under grant agreement No 769025,
has improved the scalable data-driven modelling and has developed parallelism capabilities.

The VITAL project (2019-2021) carried by `IRT Saint Exupéry <https://www.irt-saintexupery.com/>`_,
with the contributions of the following members: `Airbus <https://www.airbus.com/>`_ and `STELIA <https://www.stelia-aerospace.com/en/>`_,
contributes to |g| by creating a package for uncertainty quantification and adaptive learning capabilities.

The R-EVOL project (2020-2024) carried by `IRT Saint Exupéry <https://www.irt-saintexupery.com/>`_ implements a machine learning package for surrogate modelling,
develops a framework for robust and reliable MDO under uncertainty
and makes MDO techniques more efficient thanks to advanced numerical techniques and surrogate-based algorithms.
with the contributions of following members:
`Airbus <https://www.airbus.com/>`_,
`Expleo <https://expleogroup.com/>`_,
`Altran <https://www.altran.com/>`_,
`Capgemini <https://www.capgemini.com/>`_,
`Cerfacs <https://www.cerfacs.fr/en/>`_,
`Cenaero <http://www.cenaero.be/>`_
and `INSA Toulouse <https://www.insa-toulouse.fr/en/index.html>`_.

Open source
***********

Since 2021, |g| is open source, under the `LGPL v3 license <https://www.gnu.org/licenses/lgpl-3.0.en.html>`_.
The project is hosted on `gitlab <https://gitlab.com/gemseo>`_.

Roadmap
*******

`IRT Saint Exupéry <https://www.irt-saintexupery.com/>`_ and its partners
choose to make their roadmap public in order to elicit contributions.
:ref:`Discover it! <roadmap>`

Citation
--------

If you produce communications (scientific papers, conferences, reports) about work using |g|, thank you for citing us :

    - Gallard, F., Vanaret, C., Guénot, D, et al. `GEMS: A Python Library for Automation of Multidisciplinary Design Optimization Process Generation <https://arc.aiaa.org/doi/10.2514/6.2018-0657>`_. In : 2018 AIAA/ASCE/AHS/ASC Structures, Structural Dynamics, and Materials Conference. 2018. p. 0657.

    Bibtex entry::

        @inproceedings{gemseo_paper,
        title={GEMS: A Python Library for Automation of Multidisciplinary Design Optimization Process Generation},
        author={Gallard, F. and Vanaret, C. and Guénot, D. and Gachelin, V. and Lafage, R. and Pauwels, B. and Barjhoux, P.-J. and Gazaix, A.},
        booktitle={2018 AIAA/ASCE/AHS/ASC Structures, Structural Dynamics, and Materials Conference},
        year={2018}
        }

References
----------

Here are some references about |g| and its capabilities:

- Gallard, F., Vanaret, C., Guénot, D, et al. `GEMS: A Python Library for Automation of Multidisciplinary Design Optimization Process Generation <https://arc.aiaa.org/doi/10.2514/6.2018-0657>`_, In : 2018 AIAA/ASCE/AHS/ASC Structures, Structural Dynamics, and Materials Conference
- Gallard, F., Barjhoux, P. J., Olivanti, R., et al. `GEMS, a Generic Engine for MDO Scenarios: Key Features In Application <https://doi.org/10.2514/6.2019-2991>`_, In : AIAA Aviation 2019 Forum
- Gazaix, A., Gallard, F., Gachelin et al., `Towards the Industrialization of New MDO Methodologies and Tools for Aircraft Design <https://doi.org/10.2514/6.2017-3149>`_, In : 18th AIAA/ISSMO Multidisciplinary Analysis and Optimization Conference, 2017
- Gazaix, A., Gallard, F., Ambert, et al., `Industrial Application of an Advanced Bi-level MDO Formulation to an Aircraft Engine Pylon Optimization <https://doi.org/10.2514/6.2019-3109>`_, In : AIAA Aviation 2019 Forum
- Druot, T., Beleville, M., Roches, P., et al. `A Multidisciplinary Airplane Research Integrated Library With Applications To Partial Turboelectric Propulsion. <https://doi.org/10.2514/6.2019-3243>`_, In : AIAA Aviation 2019 Forum
- Barjhoux, P. J., Diouane, Y., Grihon, S., et al.  `A bi-level methodology for solving large-scale mixed categorical structural optimization. <https://doi.org/10.1007/s00158-020-02491-w>`_, In : Structural and Multidisciplinary Optimization, 2020
- Guénot, D., Gallard, F., Brezillon, J., et al. `Aerodynamic optimisation of a parametrised engine pylon on a mission path using the adjoint method <https://doi.org/10.1080/10618562.2019.1683163>`_, In : International Journal of Computational Fluid Dynamics, 2019
- Olivanti, R., Gallard F., Brezillon, J, et al. `Comparison of Generic Multi-Fidelity Approaches for Bound-Constrained Nonlinear Optimization Applied to Adjoint-Based CFD Applications <https://doi.org/10.2514/6.2019-3102>`_, In : AIAA Aviation 2019 Forum
- Vanaret, C., Gallard, F., Martins, J. R. `On the consequence of the "No Free Lunch" Theorem for Optimization on the Choice of an Appropriate MDO Architecture <https://arc.aiaa.org/doi/10.2514/6.2017-3148>`_, 18th AIAA/ISSMO Multidisciplinary Analysis and Optimization Conference, Denver, CO, USA, 2017


Artwork
-------

This is the logo of |g|:

.. image:: _static/logo-small.png
   :align: center

High quality PNG and SVG logos are available:

- `PNG format <_static/logo/gemseo_logo_transparent.png>`_
- `SVG format <_static/logo/gemseo_logo_transparent.svg>`_


Authors
-------

.. include:: authors.rst

Contributing
------------

Anyone can contribute to the development of |g|.
The types of contributions are multiple:

- improving the documentation,
- declaring a bug, solving a bug,
- answering questions,
- proposing a new algorithm,
- suggesting a new feature,
- etc.

.. seealso::

   Find more information on how to contribute to |g| :ref:`by clicking here <contributing>`.
