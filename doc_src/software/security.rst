..
   Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

   This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
   International License. To view a copy of this license, visit
   http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
   Commons, PO Box 1866, Mountain View, CA 94042, USA.

Security information
====================

Static Application Security Testing
-----------------------------------

We use the ``gitlab.com`` integrated
`SAST analyzers <https://docs.gitlab.com/ee/user/application_security/sast/index.html>`_
to scan for vulnerabilities in |g| source code.

Dependency scanning
-------------------

Our ``gitlab.com`` projects are mirrored on ``github.com`` and
scanned there for dependency vulnerabilities by
`Dependabot <https://docs.github.com/en/code-security/dependabot/dependabot-alerts>`_.
