<!--
 Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

 This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
 International License. To view a copy of this license, visit
 http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
 Commons, PO Box 1866, Mountain View, CA 94042, USA.
-->

# Remote process execution { #needs-remote-process-execution }

## The problem { #needs-the-problem }

In practice,
the simulation tools involved in a multidisciplinary process
rarely all run on the same machine.
Some require a Linux workstation,
others a Windows server,
and the most demanding ones need to be submitted
to an HPC cluster managed by a job scheduler (SLURM, PBS, LSF, etc.).
The computing infrastructure is **heterogeneous and distributed**.

Manually orchestrating file transfers, job submissions, and result retrieval
across multiple machines and operating systems
adds significant complexity
on top of the already challenging MDO problem.

## GEMSEO's answer: infrastructure disciplines { #needs-gemseos-answer-infrastructure-disciplines }

GEMSEO provides specialized disciplines
that handle the communication with remote computing resources,
so that the MDO process definition remains independent
of the underlying infrastructure.

<!-- ![Distributed computing architecture](../../assets/images/user_guide/distributed_computing.png){: style="display:block; margin:auto; max-width:80%" } -->

*A GEMSEO process distributing discipline evaluations across heterogeneous infrastructure.*

The available infrastructure disciplines include:

- **`SSHDiscipline`**: executes a discipline on a remote machine via SSH.
- **`JobSchedulerDiscipline`**: submits a discipline evaluation as a job to an HPC scheduler (SLURM, PBS, LSF).
- **`HTTPDiscipline`**: exposes a discipline as an HTTP service, or calls a remote discipline served over HTTP.
- **`RetryDiscipline`**: adds resilience to infrastructure failures by automatically retrying failed evaluations.

These disciplines are **composable**:
for instance, a `RetryDiscipline` can wrap an `SSHDiscipline`,
which itself wraps a user discipline.
From the MDO process perspective,
nothing changes---the composed discipline
has the same inputs and outputs as the original one.
