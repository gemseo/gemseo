..
   Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

   This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
   International License. To view a copy of this license, visit
   http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
   Commons, PO Box 1866, Mountain View, CA 94042, USA.

..
   Contributors:
          :author:  Francois Gallard

.. _job_schedulers:

Interface with HPC job schedulers (SLURM, LSF, PBS, etc)
********************************************************

This section describes how to send any discipline, or sub process such as an MDA to a HPC
using the job scheduler interfaces.

The method to be used is :meth:`~gemseo.api.wrapp_discipline_in_job_scheduler` to wrapp any discipline.

This feature is extensible through plugins using
:class:`~gemseo.wrappers.job_schedulers.schedulers_factory.SchedulersFactory`.

.. currentmodule:: gemseo.api
.. autofunction:: wrapp_discipline_in_job_scheduler
