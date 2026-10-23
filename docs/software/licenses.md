---
description: "The licenses of the GEMSEO source code, documentation and examples, the third-party code it ships, and the rules for using the GEMSEO trademark."
tags: ['reference']
search:
  boost: 1
---

<!--
 Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

 This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
 International License. To view a copy of this license, visit
 http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
 Commons, PO Box 1866, Mountain View, CA 94042, USA.
-->

# Licenses

GEMSEO uses different licenses for its code, documentation and examples,
and keeps its trademark separate.

| Content | License | SPDX identifier |
| ------- | ------- | --------------- |
| Source code, tests, utility scripts of the documentation | [GNU LGPL v3.0](https://www.gnu.org/licenses/lgpl-3.0.en.html) | `LGPL-3.0-only` |
| Documentation | [CC BY-SA 4.0 International](https://creativecommons.org/licenses/by-sa/4.0/) | `CC-BY-SA-4.0` |
| Examples, tutorials and code snippets | [BSD Zero-Clause](https://opensource.org/licenses/0BSD) | `0BSD` |
| Third-party code distributed with GEMSEO | Its own license | See [below](#third-party-code) |
| GEMSEO name and logo | Trademark, not licensed | See [below](#trademark) |

## Source code

The source code and the tests of GEMSEO are licensed under the
[GNU LGPL v3.0](https://www.gnu.org/licenses/lgpl-3.0.en.html) license.
Its text is in the **`LICENSE.txt`** file at the root of the repository,
and in the metadata of the installed package.
The GNU LGPL v3.0 is a set of additional permissions on top of the
[GNU GPL v3.0](https://www.gnu.org/licenses/gpl-3.0.en.html),
whose text is in the **`LICENSES/GPLv3.txt`** file of the repository.

## Documentation

The GEMSEO documentation is licensed under the
[CC BY-SA 4.0 International](https://creativecommons.org/licenses/by-sa/4.0/) license,
with the following exceptions:

- the utility Python and shell scripts of the **`docs`** folder
  are licensed under the [GNU LGPL v3.0](https://www.gnu.org/licenses/lgpl-3.0.en.html) license,
- the examples, the tutorials and the code snippets
  are licensed under the [BSD Zero-Clause](https://opensource.org/licenses/0BSD) license,
  so that you can copy them into your own code without any condition.

## Third-party code { #third-party-code }

GEMSEO distributes some third-party code, such as JavaScript libraries and fonts,
each under its own license;
the [credits][credits] list them with their licenses.
The **`LICENSES`** folder of the repository contains the texts of these licenses
that GEMSEO does not use for its own content:
**`Apachev2.0.txt`** (Apache License 2.0) and **`OFL.txt`** (SIL Open Font License 1.1).

## GEMSEO trademark and logo usage { #trademark }

GEMSEO is a registered European trade and service mark of IRT Antoine de Saint Exupéry, a Scientific Cooperation Foundation created by the decree of 21 March 2013, under French SIREN number 793 007 048 and APE code: 7490B having its registered office at B612 Building, 3 rue Tarfaya, CS 34436, 31405 Toulouse CEDEX, France ("IRT Saint Exupéry").

**The trademark and logo GEMSEO are not part of the GNU LGPL v3.0, nor the CC-BY-SA 4.0 license.**

You may only use the GEMSEO mark in accordance with the following rules:

### General Usage

The GEMSEO mark may only be used to refer to the GEMSEO products or services, or to give credit to IRT Saint Exupéry in relation with the GEMSEO products or services. When used on a document or web page, an attribution statement should be included at or near the mention of the GEMSEO mark as follows: **"GEMSEO" is a registered mark of IRT Antoine de Saint Exupéry** and it should point to the following URL: [gemseo.org](https://gemseo.org). You must always use the GEMSEO mark when referring to the GEMSEO products or services. The GEMSEO mark must be used unaltered, without additional or missing elements, with its original colors and original typography.

### Prominence { #trademark-prominence }

You must not visualize the GEMSEO mark more prominently than your own marks in relation to your products or services. It has to be clear for an average viewer that the GEMSEO mark is not owned by you. You must not use the GEMSEO trademark as your own.

### Scientific communications, publications and conferences

You may use the GEMSEO mark without the prior written authorization of IRT Saint Exupéry in non-commercial scientific communications, publications and conferences.

### Commercial use

You may use the GEMSEO mark for the promotion of your products or services in trade shows, events and associated documentation provided you comply with the [prominence rule](#trademark-prominence). You may not use the GEMSEO mark in any other way in relation to your products or services except if expressly authorized by IRT Saint Exupéry and subject to the execution of a specific licensing agreement and to commercial conditions to be negotiated. Should you wish to use the GEMSEO mark in relation to your products or services, you should send a written request to us and provide a full description of how you intend to use the GEMSEO mark with your products or services. Your use of the GEMSEO mark must never adversely affect, whether directly or indirectly the reputation of IRT Saint Exupéry, IRT Saint Exupéry's products or services, nor the fulfilment by IRT Saint Exupéry of its GEMSEO related activities.

If you have further questions, do not hesitate to contact us: [contact@gemseo.org](mailto:contact@gemseo.org).

## Dependencies { #licenses-dependencies }

GEMSEO depends on third-party packages, each under its own license;
the [credits][credits] list them with their licenses.
