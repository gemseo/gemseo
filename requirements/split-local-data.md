<!--
 Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

 This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
 International License. To view a copy of this license, visit
 http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
 Commons, PO Box 1866, Mountain View, CA 94042, USA.
-->

# Split discipline local data into input and output stores

## Context

`gemseo.core.discipline.io.IO` currently stores both inputs and outputs of the
last discipline execution in a single `DisciplineData` dict exposed as
`IO.data` (mirrored on `BaseDiscipline.local_data`). Inputs and outputs share
the same namespace; only the grammars distinguish which keys are which. This
mixing forces filtering by grammar every time callers need a side
(`get_input_data`, `get_output_data`, `_store_cache`, auto-coupling deepcopy),
and makes it easy to silently overwrite an input with an output of the same
name (auto-coupled variables).

## Goal

Split `IO.data` into two separate stores — one for input data, one for output
data — so the input/output side of every value is explicit and there is no
ambiguity at any execution stage.

## Drivers

1. **Clarity / API** (primary): make explicit which keys are inputs vs.
   outputs at any stage of the execution lifecycle (init, run, finalize,
   cache load).
2. **Performance** (secondary, follow-up): once the data is split, remove
   per-call filtering by grammar and reduce dict copies during execution and
   MDA coupling.

## Backward compatibility

- Keep `BaseDiscipline.local_data` and `IO.data` as read-only / merged
  views with a `DeprecationWarning`, returning the union of the new input
  and output stores. Existing user code reading `discipline.local_data[name]`
  must keep working.
- Internal call sites in `gemseo` are migrated to the new attributes.

## Out of scope

- Performance optimizations (left for a follow-up once the split lands).
- Changes to grammar API, cache API, or `_run()` contract.
- Removal of `local_data`/`io.data` (only deprecation in this iteration).
