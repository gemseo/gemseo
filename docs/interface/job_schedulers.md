<!--
 Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

 This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
 International License. To view a copy of this license, visit
 http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
 Commons, PO Box 1866, Mountain View, CA 94042, USA.
-->

<!--
Contributors:
         :author:  Francois Gallard
-->

# Interface with HPC job schedulers (SLURM, LSF, PBS, etc)

This section describes how to send any discipline, or sub process such
as an MDA to a HPC using the job scheduler interfaces.

The method to be used is [wrap_discipline_in_job_scheduler][gemseo.wrap_discipline_in_job_scheduler] to wrap any
discipline.

This feature is extensible through plugins using
[JobSchedulerDisciplineWrapperFactory][gemseo.disciplines.wrappers.job_schedulers.factory.JobSchedulerDisciplineWrapperFactory].
