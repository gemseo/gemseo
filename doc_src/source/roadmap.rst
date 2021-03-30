..
   Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

   This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
   International License. To view a copy of this license, visit
   http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
   Commons, PO Box 1866, Mountain View, CA 94042, USA.

.. _roadmap:

Roadmap
=======

In order to elicit contributions (read :ref:`contributing`),
we outline below the key elements of our roadmap,
which is currently under development.
Some modules and packages produced within the framework of this roadmap are already present in the current |g| version,
as beta version capabilities.

-  **Multi-fidelity MDO processes**

   They allow the use of multiple levels of precision for a given discipline,
   in order to accelerate the computational time.
   Basic refinement strategies will be distributed,
   together with switch criteria between the levels.
   Cooperation is welcome on other strategies.

-  **Optimisation algorithms**

   Global optimization algorithms,
   based on deterministic as well as heuristic approaches,
   are being interfaced with GEMSEO.
   Surrogate-based optimization capabilities are also under development.
   A connection with the Cuter/Cutest problem test suite will be published.

-  **Machine learning**

   |g| proposes a machine learning (ML) package to easily interface any ML algorithm,
   for surrogate-based modeling in the presence of expensive or noisy codes.
   We are gradually moving towards adaptive learning methods using infill criteria,
   for surrogate-based optimization, MDO and uncertainty quantification.

-  **MDO formulations**

   The contributors are welcome to make use of |g| concepts flexibility and develop new MDO formulations.
   The multi-level formulations family seems in particular very promising
   and the |g| team would be happy to support an active collaborative community on the subject.

-  **MPI**

   An extension of the GEMS process classes (MDODiscipline, MDAs, MDOChain etc), with the adjoint capabilities, to MPI communications will be released.
   We are open to collaborate on real applications using these features, especially high fidelity HPC ones.

-  **Uncertainty quantification and management**

   |g| already provides `OpenTURNS-based <https://openturns.github.io/www/>`_ statistical and probabilistic capabilities
   to describe uncertainty sources, propagate them in a multidisciplinary system and quantify the resulting output uncertainty.
   We are currently applying these basic functionalities to connect |g| to different UQ&M fields,
   such as sensitivity analysis, robust optimization and robust MDO.

-  **Time-dependent MDO**

   ODE solvers with adjoint capabilities are being interfaced with GEMSEO.
   The Petsc TSAdjoint library is targeted; and we are open to collaboration on that subject.

-  **Scalable data-driven models**

   The contributors are welcome to evaluate and further develop this methodology.
   Possible extensions could be to include cross-effects in the learning stage
   or to make machine learning models scalable.

-  **Benchmark MDO problems**

   Our roadmap also aims at enriching the list of existing MDO benchmarking problems with new ones
   and further assess comparative advantages of the MDO formulations.
   We are happy to share these elements with the MDO community
   and are open to collaborate on the creation of benchmark MDO problems.
   In addition, a general framework to compare drivers, formulations, or any algorithm is under development.

-  **Interoperability**

   Finally we are convinced that interoperability between platforms and methodologies is key.
   One objective is to interoperate |g|
   with `OpenMDAO <https://github.com/OpenMDAO/OpenMDAO>`_ and `WhatsOpt <https://github.com/OneraHub/WhatsOpt>`_.
   The support of the MDO community is welcome.
