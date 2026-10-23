<!--
 Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com

 This work is licensed under the Creative Commons Attribution-ShareAlike 4.0
 International License. To view a copy of this license, visit
 http://creativecommons.org/licenses/by-sa/4.0/ or send a letter to Creative
 Commons, PO Box 1866, Mountain View, CA 94042, USA.
-->

# SPDD Analysis: Catalog Variables in the DesignSpace

> **GitLab issue [#1791](https://gitlab.com/gemseo/dev/gemseo/-/work_items/1791)**
> — "Extend the variable hierarchy with a catalog-backed catalog variable".
>
> This story **builds on** the variable class hierarchy of issue
> [#1845](https://gitlab.com/gemseo/dev/gemseo/-/work_items/1845) and on the discrete
> variable of issue
> [#1885](https://gitlab.com/gemseo/dev/gemseo/-/work_items/1885), and merges
> **after** both. Everything below assumes the post-1885 code: `BaseVariable` /
> `ContinuousVariable` / `IntegerVariable` / `DiscreteVariable` / `VariableFactory`
> in `src/gemseo/space/_variable/`, and the
> `DesignSpace.add_variable(..., variable=...)` escape hatch.
>
> It delivers the first of the two kinds that
> [#1885](https://gitlab.com/gemseo/dev/gemseo/-/work_items/1885) deferred
> (`spdd/analysis/GGQPA-1885-202608181109-[Analysis]-discrete-variables.md:439`,
> *"Catalog (unordered) variables and catalog-backed variables"*): the two turn
> out to be **one** kind, because the catalogue is what supplies the categories.
>
> It introduces **two** concepts, not one: a `Catalog` value object owning the table
> and its normalization, and a `CatalogVariable` whose value is a row position in
> that catalogue.
>
> Scope is deliberately **the data model only**. A catalog variable becomes
> declarable, checkable, persistable and viewable. The discipline that turns
> `(catalogue, position)` into one coupling variable per catalogue column is a
> separate story, as is making the kind *solvable*. See
> [Out of Scope](#out-of-scope).

## Original Business Requirement

> bon, nous allons créer un nouveau fichier SPDD analysis pour créer une variable
> catégorielle. On va s'arreter à l'ajout de cette variable. Aucune suite (pas
> d'algo, pas de driver ....). Je veux juste pouvoir ajouter une telle variable dans
> mon design space. Ce qu'elle fait: on lui donne un tableau pandas (catalogue). La
> variable est donc l'index de ce tableau. Chaque colonne du tableau correspond à
> une propriété du catalogue. Plus tard, on va créer une discipline qui, à partir du
> catalogue et de l'index, donnera une valeur à la variable de couplage
> correspondante au nom de la colonne.

### Clarifications obtained from the requester

The requirement above is a short feature statement; the following points were
settled with the requester before this analysis and are treated as requirements.

On the variable:

- The value of the variable is the **row position** `0…n-1`, not the index label.
  The label is a human-readable tag used to talk about a row; the design vector
  stays numeric.
- The kind inherits **`BaseVariable` directly**, as a sibling of `DiscreteVariable`
  rather than a subclass of it or of `IntegerVariable`.
- The class is named `CatalogVariable`, its data type `catalog`, its field
  `catalog`.
- The class stays **private** (`gemseo.space._variable`), as `DiscreteVariable` is
  today. — **Overtaken by GGQPA-1845**, which made the hierarchy public before this
  story landed, removing the consistency argument behind the clarification; the kind
  ships public (see [Key Design Decisions](#key-design-decisions)).
- The **tabular view gains nothing**: the variable renders as a bounded integer
  whose type is `catalog`.
- The **index labels are rendered in the domain error message**, and nowhere else.
- **HDF serializes the catalogue** and must round-trip to an equal space. **CSV
  export is refused**: a table cannot be written into one space-delimited cell.
- On the façade, the only new member is `has_catalog_variables`.

On the catalogue:

- The **variable owns its catalogue**, and the catalogue is grouped into a
  **dedicated `Catalog` object** rather than left as a raw table inside the variable.
  The requester's reason: the input goes through a chain of preprocessing steps, and
  that chain is about the table, not about the variable.
- The catalogue is **read-only once the variable is built**. This is a hard
  requirement — *"pour éviter des bêtises"* — not a convention: nothing a caller
  does after construction may change the domain.
- The catalogue accepts a **`DataFrame` or a mapping** `column name → sequence`.
- With a mapping, every sequence must hold the **same number of elements**, and the
  labels — when supplied — must hold that same number. A ragged mapping is
  **rejected**.
- A **scalar** given for a column is **broadcast** over the rows, not rejected.
- **Strings are stripped** of leading and trailing whitespace, in the property
  columns **and in the index labels**, and the stripped values are what the
  catalogue stores.
- An **empty column** is **dropped with a log warning** naming it, not rejected. A
  column is empty when it holds no element, or when every element is **blank** —
  `NaN`, `None` or, after stripping, the empty string.
- A column of **zeros is not empty**: `0` is a value.
- The catalogue must be **non-empty in both dimensions** once normalized: at least
  one row and at least one column.
- A mapping holding **only scalars** yields a **one-row catalogue**. This case is
  allowed, and it is **degenerate**: the variable then has a single admissible value,
  so there is nothing to choose and no combinatorics. Accepted as is for now, to be
  treated in a second step.
- **Duplicate index labels** and **non-numeric columns** are accepted.

## Domain Concept Identification

The design-space code lives in `src/gemseo/space/`: the façade and its collaborators
in `space/design/`, the variable hierarchy one level up in `space/_variable/`, where
it is shared with `ParameterSpace` and re-exported by `gemseo/enum/__init__.py`.

### Existing Concepts (from codebase)

!!! note "Paths in this section are pre-#1845"

    This survey was written against the private variable package,
    `src/gemseo/space/_variable/`, and its line numbers are those of that code.
    GGQPA-1845 made the package public before this story landed:
    `_variable/` is now `variable/`, and `_base.py`, `_discrete.py` and `_factory.py`
    are now `base.py`, `discrete.py` and `factory.py`. The line numbers quoted below
    have drifted with the move; the paths are kept as surveyed so that the reasoning
    still reads against the code it was drawn from. Every path *outside* this section
    names the code as it is today.

- **`BaseVariable`** (`space/_variable/_base.py:102`): abstract Pydantic model,
  `frozen=True`, fields `size`, `type`, `lower_bound`, `upper_bound`, plus the
  `component_type` `ClassVar` (`:124`) holding the NumPy type of the components,
  constrained to `ComponentDType = type[int64 | float64]` (`:74`). Its single
  `@model_validator(mode="after")` converts each bound (`__convert_bound`, `:156`)
  and checks it (`__check_bound`, `:188`); `__convert_bound` **broadcasts a scalar
  bound over `size`** and **freezes** the resulting array with
  `setflags(write=False)`, writing it back through `self.__dict__[...] =` to bypass
  the frozen model. The polymorphic interface a new kind must satisfy is **one**
  `@abstractmethod`, `compute_normalization_mask` (`:283`), plus the overridable
  hooks `check_finite_bound_components` (`:296`),
  `find_components_outside_domain` (`:313`), `compute_default_value` (`:266`), its
  per-component helper `compute_default_component_value` (`:242`), `cast` (`:230`)
  and the two wording hooks `_get_out_of_domain_message` (`:326`) and
  `_get_out_of_domain_component_message` (`:351`). `model_copy` (`:382`) carries a
  bound over **only** when the caller had set it, which is what lets a
  derived-bounds kind survive `filter_components`. `__setstate__` (`:417`) refreezes
  the bound arrays, because NumPy loses the writeable flag across pickling.
  `__eq__` (`:425`) compares **every field declared by either kind** — it walks the
  union of both `model_fields` — so a field added by a subclass takes part in the
  comparison with no edit, provided that comparison reduces to a boolean.
- **A variable is not hashable in practice.** Despite `frozen=True`, every existing
  kind holds NumPy arrays as fields, so the Pydantic-generated `__hash__` raises
  `TypeError: unhashable type: 'numpy.ndarray'` — verified on `ContinuousVariable`
  and `DiscreteVariable`. Nothing in the codebase hashes a variable, so a new
  non-hashable field changes nothing.
- **`DataType`** (`space/_variable/_base.py:94`): `StrEnum{FLOAT, INTEGER,
  DISCRETE}`. The module-level `TYPE_MAP` sits one level up, in
  `space/_variable/__init__.py:32`, and is derived rather than written by hand: a
  comprehension over the hard-coded tuple `(ContinuousVariable, IntegerVariable,
  DiscreteVariable)` reading each class's `component_type`. Adding a kind therefore
  means editing that tuple and `__all__` in the same file. `DataType` is publicly
  re-exported as `gemseo.enum.DesignVariableType` and aliased as
  `DesignSpace.DesignVariableType` (`space/design/__init__.py:126`), with
  `VARIABLE_TYPES_TO_DTYPES = TYPE_MAP` (`:129`).
- **`DiscreteVariable`** (`space/_variable/_discrete.py:66`): the closest sibling and
  the template for this story. It pins `type` to `DataType.DISCRETE`,
  `component_type` to `float64` and `size` to `Literal[1]` with a
  `field_validator("size", mode="before")` giving a readable message (`:87`); it
  carries the domain as a `choices` field, **deduplicated, sorted and frozen** in
  its own `model_validator` (`:116-140`), **derives its bounds** from the extremes
  (`__derive_bounds`, `:199`) and **rejects explicitly-set bounds**
  (`__check_bounds_are_not_set`, `:142`); it returns an all-`False` normalization
  mask (`:210`), uses strict equality for membership (`:215`), tolerates `None` as
  "not set" (`:217`), takes its default value from the first choice (`:225`),
  renders an elided representation of its domain (`_format_choices`, `:232`, cut off
  at `_MAX_FORMATTED_VALUES = 6`, `:55`), overrides both wording hooks (`:254`,
  `:267`), and refreezes `choices` in `__setstate__` (`:276`). Its `ChoicesType`
  (`:58`) accepts an array, a list or a tuple — the precedent for accepting more
  than one input shape.
- **`VariableFactory`** (`space/_variable/_factory.py`): `BaseFactory[BaseVariable]`
  scanning `gemseo.space._variable`. `create(data_type, *args, **kwargs)` resolves a
  `DataType` to its class through the cached `_data_type_to_class_name` map, built by
  reading each discovered class's `model_fields["type"].default` and raising if two
  classes pin the same type. A new kind **self-registers by existing as a module in
  that package** — no factory edit. It cannot be plugged in from outside, though:
  `DataType` is a closed `StrEnum`, so a new kind also needs a new `DataType` member
  and a new `TYPE_MAP` entry. The factory skips abstract classes and filters on the
  package, so a non-variable helper class living in the same package is ignored.
- **`Variables`** (`space/design/_variables.py`): ordered, versioned
  `MutableMapping[str, BaseVariable]`; every mutation bumps `version`.
  `__setitem__` (`:171`) recomputes the normalization mask and reindexes;
  `__reindex` (`:190`) maintains a cached `__discrete_variable_count` (`:98`, `:205`)
  so that `has_discrete_variable` (`:302`) is O(1). `get_integer_mask` (`:279`) and
  `has_integer_variable` (`:294`) test `isinstance(variable, IntegerVariable)`, so a
  kind that is not an `IntegerVariable` is excluded from both masks **with no hook to
  implement**. `filter_components` (`:237`) rebuilds an entry with
  `variable.model_copy(update=...)`, which preserves the kind and every field of that
  kind.
- **`Bounds`** (`space/design/_bounds.py`): concatenates the per-variable bounds into
  `full_lower_bound` / `full_upper_bound` and hands out read-only views;
  `set_lower_bound` / `set_upper_bound` do not mutate but rebuild the variable with
  `model_copy`, which is how a derived-bounds kind rejects them.
- **`Normalizer`** / **`IntegerRounder`** (`space/design/_normalizer.py`,
  `_integer_rounder.py`): both `RegistryDerivedData`, caching aggregate masks keyed
  on `Variables.version`. Normalization is applied only where the normalization mask
  is `True`, rounding only where the integer mask is `True`.
  `space/design/_normalizer.py:46` carries the package's precedent for a module-level
  `LOGGER = logging.getLogger(__name__)` and a `LOGGER.warning` (`:214`).
- **Membership checks** (`space/design/_checking.py`, free functions):
  `check_addable_value` (`:69`) validates a value before it is stored and phrases the
  failure through `variable._get_out_of_domain_message` (`:136`); `check_membership`
  (`:169`) falls back from the vectorized full-array path
  (`_check_membership_array`, `:252`) to the per-variable path
  (`_check_membership_dict`, `:343`) **whenever the space holds a discrete
  variable** (`:201`), since derived bounds cannot express a finite domain;
  `_check_index_in_domain` (`:292`) and `check_domain` (`:319`) call
  `find_components_outside_domain` and phrase the failure through
  `_get_out_of_domain_component_message` (`:315`).
- **`Value`** (`space/design/_value.py`): owns the current values. `set_variable`
  (`:278`), `to_complex` (`:325`), `initialize_missing` (`:337`, delegating to
  `variable.compute_default_value()` at `:351`), `check_value` (`:353`).
- **I/O** (`space/design/_io.py`): `to_hdf` (`:87`) writes one group per variable,
  and for a discrete variable **deletes and recreates** the `choices` dataset first
  so that a changed length or a changed kind cannot leave a stale dataset behind
  (`:117-127`). `from_hdf` (`:156`) reads the optional `choices` dataset (`:183`),
  cross-validates it against the declared type (`:187-201`) and rebuilds through
  `VARIABLE_FACTORY.create(var_type, choices=choices)` (`:212`) precisely because the
  bounds are derived (`:207`). `_to_dataframe` (`:235`) and `to_csv` (`:275`) speak
  the same five fields plus a `choices` column, joined with `_CHOICES_SEPARATOR` in
  one space-free cell (`_format_choices_cell`, `:218`); `to_csv` raises if the
  caller's `delimiter` is the separator itself (`:295`).
- **`View`** (`space/design/_view.py`): `get_pretty_table` (`:57`) renders one row per
  scalar component over `_TABLE_NAMES`, and appends a `choices` column **only** when
  the space holds a discrete variable (`:115`), through
  `variable._format_choices()` (`:52`).
- **`DesignSpace`** (`space/design/__init__.py`): the façade. `add_variable` (`:313`)
  takes either a bound-shaped signature or a ready-made `variable`, and **rejects the
  type string of a kind whose domain the signature cannot express** — today
  `DataType.DISCRETE` (`:357`). `has_discrete_variables` (`:426`) and `get_choices`
  (`:1042`) are the kind-specific façade accessors. `filter_dimensions` (`:280`),
  `extend` (`:1450`), `add_variables_from` (`:1494`) / `_add_variable_from` (`:1504`)
  and `to_scalar_variables` (`:1518`) share the frozen variable object rather than
  destructuring it, so a kind-specific field is not dropped.
- **`TYPE_MAP` consumers outside the space package**:
  `core/problem/database.py:1102` and `doe/core/base_doe_library.py:209` both index
  `VARIABLE_TYPES_TO_DTYPES` by a variable type — this is why one `DataType` member
  maps to exactly one NumPy dtype.
- **`ParameterSpace`** (`space/parameter.py`): `extract_deterministic_space` already
  shares the variable object as is, with a comment naming the discrete case, so a
  new kind survives that path unchanged.
- **pandas is a hard runtime dependency** (`pyproject.toml`, `pandas >=2.2,<=2.3.3`),
  and `Dataset` (`dataset/dataset.py:91`) already subclasses `DataFrame`, so a
  DataFrame-shaped user input is idiomatic here. `pandas.isna` is the cross-dtype
  missing-value primitive. The only existing label-to-integer encoding in the
  codebase is `pandas.factorize` in `problem/dataset/iris.py:53`, which stores the
  reverse map in `dataset.misc["labels"]` — the same position/label split this story
  formalizes. `util/discipline.py` `VariableRenamer` (`from_csv`,
  `from_spreadsheet`, `__from_dataframe`, `:405`) is the closest existing precedent
  for a class built from a table.
- **`util/pydantic_ndarray.py`** (`NDArrayPydantic`): the pydantic/NumPy bridge used
  by every bound and by `choices`; its validator ignores string length, so a labels
  array is expressible.
- **Documentation** (`docs/user_guide/concepts/design_space.md`): `:32` and `:51-70`
  enumerate the three available types; `:135-203` is the discrete section, including
  the *"not yet solvable"* admonition (`:166`) and the storage note (`:193`);
  `:222` states that integer and discrete variables cannot be normalized.

### New Concepts Required

- **`Catalog`** (`space/catalog.py`) — a frozen value object owning the
  table: the index labels, the property columns, their normalization and their
  immutability. It is the concept the requirement's *"tableau pandas (catalogue)"*
  names, and it is where the whole input pipeline lives. It is **not** a variable, and
  it lives in its own module beside the variable package rather than inside it, so the
  `VariableFactory`, which walks `gemseo.space.variable`, never sees it.
- **The normalization pipeline** — the ordered chain that turns a user input into a
  catalogue: convert, check the kinds, strip, check dimensionality, check the row
  count, drop blank columns, check rectangularity, broadcast scalars, check
  non-emptiness, default the labels and freeze. The order is part of the contract, not
  an implementation detail (see [Key Business Rules](#key-business-rules)).
- **Blankness** — the property that makes a column droppable: no element, or every
  element missing (`NaN`, `None`) or, after stripping, an empty string. Distinct from
  falsiness, which would also swallow `0` and `False`.
- **Frozen columnar storage** — how immutability is achieved. A `DataFrame` has no
  equivalent of `ndarray.setflags(write=False)`, so the catalogue is stored as frozen
  NumPy arrays — the labels plus one array per column — and a `DataFrame` is **rebuilt
  from them on access**. This is the same mechanism the hierarchy already trusts for
  bounds (`variable/base.py:142`) and for `choices` (`variable/discrete.py:137`),
  applied one level up.
- **Position encoding** — the variable's value is the **row position** in the
  catalogue, an integer in `0…n-1`. The catalogue is what makes the position
  meaningful, and the variable owns it, so no map has to be maintained by the user.
- **`CatalogVariable`** (`space/variable/catalog.py`) — a `BaseVariable`
  subclass pinning `type` to a new `DataType.CATALOG` member, pinning its
  `component_type` `ClassVar` to `int64`, **scalar by construction** (`size` pinned
  to 1), and holding one `Catalog`. Its bounds are **derived, not supplied**: `0` and
  `n-1`. It implements the polymorphic interface with a row-position domain instead
  of an interval domain, and is otherwise thin.
- **`DataType.CATALOG`** — a fourth enum member, serialized as `"catalog"`.
  It is the discriminator that lets `VariableFactory`, HDF and every external reader
  of `variable.type` recognise the kind, exactly as the three existing members do.
- **`VariablesView.has_catalog_variables`** — the public predicate, symmetric
  with `has_discrete_variables` (`variables_view.py:60`), backed by an O(1) flag cached
  in `Variables.__reindex` (`_variables.py:188`) the way the discrete flag already is.
  It lives on the read-only view, not on `DesignSpace`, which carries neither
  predicate; it is what the checking layer and the I/O layer query.
- **An HDF layout for a table** — a per-variable sub-group holding the labels and one
  dataset per column. This is the first non-vector payload the design-space HDF
  format has had to carry, and it is the `Catalog`'s responsibility, not the
  variable's.
- **A refusal path in CSV export** — the first case where a design space is
  representable in HDF but not in CSV.

### Conceptual Relationships

- **The variable owns one catalogue; the catalogue owns the table.** The variable
  answers questions about the *domain* (how many alternatives, is this position
  admissible, what are the bounds); the catalogue answers questions about the
  *table* (what are the labels, what are the columns, give me a DataFrame). Neither
  reaches into the other's business, and the split is what keeps
  `CatalogVariable` the size of its siblings.
- **The catalogue is a value, not an entity.** It has no identity, no name and no
  owner beyond the variable that holds it; two catalogues built from equal inputs are
  equal. That is what makes it safe to share, copy and serialize with the variable.
- **The row count is the sole source of truth for the domain**, and the bounds
  `[0, n-1]` are a *consequence* of it, never an independent constraint. Unlike a
  discrete variable, whose set leaves gaps inside its interval, a catalog
  variable's domain is exactly the integers of its derived interval — so the domain
  check reduces to "integer, and within the bounds".
- **`Variables` stores a `CatalogVariable` exactly as it stores any other kind**;
  the registry, its ordering, its versioning and its index ranges are unaffected.
- **`Bounds` and every bounds consumer keep working unchanged**, precisely because
  the bounds are derived rather than absent: `full_lower_bound` and
  `full_upper_bound` stay finite and meaningful, and nothing in the aggregate-array
  machinery learns about catalog variables.
- **`Normalizer` and `IntegerRounder` need no new branch**: the kind reports an
  all-`False` normalization policy, and it falls out of the integer mask for free
  because `Variables.get_integer_mask` selects on
  `isinstance(variable, IntegerVariable)` (`_variables.py:292`). Not being rounded is
  a *consequence* of the chosen parent, and is recorded as a known limitation.
- **`Value` derives the default from the catalogue**, not from the bounds: position
  `0`, the first row. The midpoint rule of `compute_default_component_value` would
  land on an arbitrary row, so the override happens at the whole-variable level,
  `compute_default_value`, which is what `Value.initialize_missing` calls
  (`_value.py:351`).
- **The catalogue columns are the future coupling variables.** This story names,
  normalizes and preserves them, but reads nothing from them: they exist so that the
  discipline of the follow-up story has something to emit. Their dtypes are therefore
  not constrained here — and the `Catalog` is precisely the object that discipline
  will be handed.
- **The index labels are documentation, not data.** They are normalized and stored,
  they appear in the domain error message, and they are used nowhere else in this
  story — no view column, no label-based declaration.
- **`ParameterSpace`** keeps its deterministic-versus-random split untouched. Random
  variables remain continuous; a catalog random variable would need a discrete
  distribution over the catalogue rows and is a separate story.

### Key Business Rules

On the pipeline — **the order is part of the contract**, because these rules
interact:

1. **Convert.** A `DataFrame` yields its index as labels and its columns as columns,
   its column names converted to strings, two names colliding once converted being
   rejected here. A mapping yields its keys as column names and its values as columns,
   with labels taken from an optional argument and defaulting to the positions
   themselves. A `str`, `bytes`, `bytearray` or zero-dimensional array given as the
   labels is **one** label, not a sequence of its characters, bytes or its single
   element.
2. **Check the kinds.** A column given as an unordered collection — a `set`, a mapping
   view — is refused, since the order of the rows it would define is arbitrary, and so
   is a column given as a nested table, which is iterable over its *column names*
   rather than over values.
3. **Strip.** Every string is stripped of leading and trailing whitespace — in the
   property columns **and in the index labels** — and the stripped value is what the
   catalogue stores. So `" alu "` is stored, rendered and looked up as `"alu"`. `str`,
   `bytes` and `bytearray` are all strings here; a `bytearray` is normalized to `bytes`.
4. **Check dimensionality.** Every column, and the labels, must be one-dimensional.
   This runs *after* stripping, because stripping a table column of equal-length
   sequences turns it from an opaque one-dimensional object array into a nested list
   that reveals its true shape.
5. **Check the row count.** If every sized column is empty and no label is supplied,
   the input describes columns without any row and is rejected — before the blank
   columns are dropped, which would otherwise swallow every such column and report a
   missing *column* instead. With labels supplied, the length disagreement is reported
   instead.
6. **Drop blank columns.** A column is blank when it holds no element, or when every
   element is blank: missing (`NaN`, `None`) or, *after* stripping, an empty string.
   A blank column is discarded and a **log warning names it**. Stripping precedes
   this test, which is what makes a column of whitespace-only strings droppable.
7. **Check rectangularity.** Every remaining sequence column, and the labels, must
   hold the same number of elements. That number is the row count `n`. A mismatch is
   rejected with a message naming the offending columns and their lengths.
   Scalar columns take no part: they have no length. When **every** column is a
   scalar and no labels are supplied, nothing carries a length, and the row count is
   **1**.
8. **Broadcast scalars.** A scalar given for a column is repeated over the `n` rows.
9. **Check non-emptiness.** At least one column must remain — a column-less catalogue
   could feed no coupling variable; the absence of a row is already caught at steps 1
   and 5.
10. **Default the labels and freeze.** Absent labels become the stringified positions,
    and every array — labels and columns — is frozen with `setflags(write=False)`. A
    column mixing a string and a missing value is frozen as `object`, so that NumPy
    does not promote both to a string dtype and turn the missing cell into the literal
    text `"nan"`.

A different order would break the contract: dropping after the rectangular check
would make a zero-length column look ragged; testing blankness before stripping
would keep a column of whitespace; broadcasting before the rectangular check would
let a scalar decide the row count; checking dimensionality before stripping would miss
a nested column; counting the rows after dropping would report a missing column for an
input that supplied columns without any row.

On the catalogue:

- **Frozen at construction**: the catalogue cannot change afterwards. Neither
  mutating the object the caller passed in, nor mutating the `DataFrame` handed back
  by the accessor, may alter it. Changing a catalogue means removing the variable and
  adding a new one. This keeps the cache-invalidation story identical to today's.
- **The stored catalogue is the normalized one**, not the input: stripped, minus its
  blank columns, with its scalars expanded. That is what the accessor returns, what
  HDF writes, and what equality compares — so a round-trip stays exact even though
  the input is not reproduced verbatim.
- **Blankness is about the absence of a value, not its magnitude**: a column of
  **zeros is kept**, and so is a column of `False`. `0`, `False`, `""`, `None` and
  `NaN` are all falsy in Python, so the test must be blankness and never falsiness.
- **A partly-blank column is kept as is**: one missing element is an absent property
  of one alternative, which is information; only a column blank *everywhere* carries
  nothing. No imputation, no default, no rejection.
- **Duplicate labels and non-numeric columns are accepted**: the position, not the
  label, is the value, so duplicates cannot make the domain ambiguous; and no column
  is read in this story.
- **Two catalogues are equal** iff their labels, their column names and order, and
  their values are equal, a missing value counting as equal to a missing value. Their
  dtypes do not take part, so a catalogue also **hashes** on its column names and its
  labels alone.
- **A one-row catalogue is legal**, however it arises — a one-row table, a
  one-element mapping, or a mapping of scalars only. It makes the variable
  **degenerate**: bounds `[0, 0]`, one admissible value, nothing to explore. The
  data model accepts it deliberately rather than guessing that the user made a
  mistake; what a space and a driver should do with a variable that has no choice is
  deferred.

On the variable:

- **Domain**: a value is admissible **iff** it is an integer in `0…n-1`, where `n` is
  the number of catalogue rows.
- **Scalarity**: a catalog variable has `size == 1`. A vector of catalog
  choices is declared as several catalog variables.
- **Derived bounds are read-only**: `lower_bound` is `0` and `upper_bound` is `n-1`.
  Supplying either explicitly is an error, and `set_lower_bound` / `set_upper_bound`
  fail with a message naming the catalogue as the domain.
- **Default current value**: position `0`, the first row. Deterministic, trivially
  testable, and admissible by construction.
- **Every value is an array**: a scalar accepted for convenience at the call site is
  promoted to a shape-`(1,)` array before storage, per the GEMSEO-wide convention.
- **Round-trip fidelity in HDF**: a space serialized and reloaded must be **equal**
  to the original, catalogue included. A lossy round-trip is a defect.
- **Explicit refusal in CSV**: `to_csv` on a space holding a catalog variable
  fails with a message naming the variable, rather than writing a file that cannot be
  read back.
- **Additive public surface**: every existing `add_variable` call, every existing
  serialized file and every existing behavior of the three current kinds is
  unchanged.

## Use Cases

### UC-1: Declare a design space mixing all four kinds

```python
from pandas import DataFrame

from gemseo.space import DesignSpace
from gemseo.space.variable import CatalogVariable
from gemseo.space.variable import DiscreteVariable

catalog = DataFrame(
    {"mass": [2.7, 7.8, 4.5], "cost": [10.0, 5.0, 50.0]},
    index=["aluminium", "steel", "titanium"],
)

design_space = DesignSpace()
design_space.add_variable("thickness", lower_bound=1.0, upper_bound=5.0)
design_space.add_variable("n_stringers", type_="integer", lower_bound=2, upper_bound=8)
design_space.add_variable("rib_pitch", variable=DiscreteVariable(choices=[0.4, 0.47]))
design_space.add_variable("material", variable=CatalogVariable(catalog=catalog))
```

`material` has `size == 1`, `type == "catalog"`, bounds `[0, 2]` and current
value `0`, i.e. `"aluminium"`. The `catalog` field accepts the table directly and
builds the `Catalog` itself, so the common case needs no extra import. Declaring the
variable through the bound-shaped signature —
`add_variable("material", type_="catalog", ...)` — fails, as it already does for
a discrete variable, because the signature cannot carry a catalogue: `add_variable`
tests the `_BOUNDED_TYPES` allowlist (`design/__init__.py:96`, read at `:373`) rather
than naming the kinds it refuses.

### UC-2: The input goes through the normalization pipeline

```python
material = CatalogVariable(
    catalog=Catalog(
        columns={
            "mass": [2.7, 7.8, 4.5],
            "supplier": [" ACME ", "Foundry", "  "],
            "note": ["", "   ", ""],
            "stock": 0,
        },
        labels=[" aluminium", "steel ", "titanium"],
    )
)
```

The `catalog` field also accepts a bare `DataFrame` or a bare mapping, which it wraps
into a `Catalog` itself; an explicit `Catalog` is what supplying labels alongside a
mapping requires, since the labels belong to the catalogue and not to the variable.

The mapping is accepted, and the resulting catalogue holds three rows and three
columns: `supplier` is stripped to `["ACME", "Foundry", ""]`, `note` is **dropped**
because every element is blank once stripped — with a log warning naming it — `stock`
is broadcast to `[0, 0, 0]` and **kept**, because `0` is a value, and the labels are
stripped to `["aluminium", "steel", "titanium"]`. A ragged mapping, or labels of the
wrong length, would have been rejected instead.

### UC-3: The catalogue cannot be changed after the fact

```python
catalog.loc["steel", "mass"] = 999.0           # the caller's own table
material.catalog.to_dataframe().loc["steel", "mass"] = 999.0  # the table handed back
material.catalog.name_to_column["mass"][1] = 999.0            # the column handed out
```

None of the three statements changes the variable, and the third raises: the first
because the catalogue copied and froze the table at construction, the second because
the accessor hands back a table rebuilt from that frozen storage, and the third
because every column is handed out as a read-only view of a frozen array. The domain,
the derived bounds and the current value are unaffected.

### UC-4: An inadmissible position is rejected, naming the items

```python
design_space.set_current_variable("material", array([5]))
```

fails with a message naming the items of the catalogue, elided beyond six entries as
the discrete kind already elides its choices (`variable/discrete.py:233`):

```text
The variable material is a catalog variable with the items
[aluminium, steel, titanium]; got material[0] = 5.
```

The same wording serves a non-integer position, so a value produced by a solver that
is unaware of the kind fails with a message that explains the domain.

### UC-5: Persistence round-trips the catalogue, and CSV refuses

```python
design_space.to_hdf("space.h5")
assert DesignSpace.from_hdf("space.h5") == design_space  # catalogue included

design_space.to_csv("space.csv")  # fails, naming 'material'
```

HDF stores the catalogue as a sub-group of the variable's group; the reader rebuilds
the variable through the factory with a catalogue, never with bounds, exactly as it
already does for `choices` (`_io.py:207-212`). CSV export fails with an actionable
message rather than producing a file whose `material` row cannot be read back.

### UC-6: Reading a space that contains catalog variables

```python
print(design_space)
design_space.variables.has_catalog_variables
design_space.variables["material"].catalog.to_dataframe()
```

The tabular view renders `material` as a bounded integer of type `catalog`, with
no additional column, so the rendering of every existing space is unchanged. The
predicate lives on the read-only `variables` view, not on `DesignSpace`, which carries
no `has_discrete_variables` either; it is what the checking and I/O layers query, and
what a user checks before handing the space to a driver. The same view is how a
catalogue is read back — from a space just built, or from one reloaded from HDF.

## Strategic Approach

### Solution Direction

- **Two classes, one of which is not a variable.** `Catalog` owns the table, the
  pipeline and the immutability; `CatalogVariable` owns the domain and stays as
  thin as its siblings. As built, `CatalogVariable` lives in the (now public)
  `space/variable/` package and `Catalog` in its own module beside it,
  `space/catalog.py`, so the factory — which scans the variable package and filters on
  `BaseVariable` subclasses — cannot see it on either count.
- **Add one variable subclass, plus the two registration lines.**
  `CatalogVariable` implements the 1845 polymorphic interface, so `Bounds`,
  `Normalizer`, `IntegerRounder` and the aggregate-mask machinery need **no new
  branch**. The unavoidable non-polymorphic edits are the new `DataType` member
  (`variable/base.py:78`) and the `(ContinuousVariable, IntegerVariable,
  DiscreteVariable)` tuple that builds `TYPE_MAP` (`variable/__init__.py:34`). Every
  `if variable.type == …` needed beyond those is a design smell to remove, and is the
  measure of whether this story is designed correctly.
- **Make the catalogue immutable by construction, not by convention.** Decompose the
  table into frozen NumPy arrays and rebuild a `DataFrame` on access. This reuses the
  mechanism the hierarchy already trusts, keeps the stored state in the array regime
  that `__eq__`, `model_copy` and `__setstate__` already handle, and removes the
  mutable-shared-state failure mode rather than documenting it away.
- **Make the pipeline an explicit, ordered contract.** Six rules that interact
  pairwise are a defect factory when they are implemented as scattered checks. Write
  the order down, in the docstring and in the analysis, and test each step's
  interaction with the next.
- **Derive the bounds from the row count.** `[0, n-1]` are the true bounds of the
  domain, so deriving them keeps every existing bounds consumer correct and honest —
  the same decision, and for the same reason, as deriving a discrete variable's
  bounds from its sorted choices.
- **Reduce the domain check to integrality plus the bounds.** A catalog
  variable's domain is exactly the integers of its derived interval, so the existing
  bounds comparison plus an integrality check is complete. What the kind adds is the
  *wording*, through the two hooks 1885 already made polymorphic.
- **Extend the two extension points 1885 opened, rather than adding new ones.** The
  `add_variable` type-string rejection (`design/__init__.py:357`) and the
  per-variable membership fallback (`_checking.py:201`) both already exist for the
  discrete kind and both need to admit a second kind; neither needs a new shape.
- **Let the HDF layout fall out of the storage.** The frozen storage *is* a labels
  array plus one array per column, which is exactly what a sub-group of datasets
  holds, so serialization is a direct write with no flattening step, and
  deserialization reconstructs the same frozen storage. The read and write belong to
  `Catalog`, so `_io.py` gains a delegation, not a table serializer.
- **Refuse CSV instead of degrading it.** The CSV format of a design space is one
  line per component with a space delimiter; a table has no cell-shaped
  representation there. Refusing names the limitation instead of hiding it in a file
  that cannot be read back.

### Key Design Decisions

- **A dedicated `Catalog` object versus the table inside the variable.** The
  requester first chose to keep the table in the variable, and reversed that once the
  pipeline had grown to six ordered rules plus freezing, rebuilding, equality and
  table I/O. Three arguments carried the reversal: the pipeline is about the table,
  not about the variable, so it does not belong in a `BaseVariable` subclass
  alongside domain logic; the follow-up discipline needs to read columns and will be
  handed a `Catalog` rather than reaching into a variable; and, decisively, a frozen
  Pydantic `Catalog` **removes the largest technical risk of the original design** —
  a `DataFrame`-typed field needed `arbitrary_types_allowed` and forced a specialized
  `__eq__`, because `df1 == df2` yields a `DataFrame` instead of a bool, whereas
  `BaseVariable.__eq__` (`variable/base.py:415`) composes natively with a field whose own
  `__eq__` returns a bool. → **Decided — a frozen Pydantic `Catalog`** in
  its own public module `gemseo/space/catalog.py`, holding the labels, the columns
  and the pipeline.
  The cost is a second concept to document, serialize and version.
- **Row position versus the index label as the value**: the label reads more
  naturally at the call site, but a label-valued variable would need a per-instance,
  non-numeric `component_type`, which `ComponentDType = type[int64 | float64]`
  (`variable/base.py:75`) and the one-dtype-per-`DataType` shape of `TYPE_MAP` forbid — a
  constraint the 1885 analysis already recorded as wider than a single story. The
  1885 analysis also rejected an index convention for *discrete* variables, but on a
  ground that does not apply here: it pushed the index-to-value map onto the user,
  whereas here the variable owns the catalogue and the mapping is never the user's
  to maintain. → **Decided — the value is the row position, `component_type =
  int64`.**
- **`BaseVariable` versus `IntegerVariable` or `DiscreteVariable` as the parent**:
  inheriting `IntegerVariable` would give integrality checking and rounding for free
  — `get_integer_mask` selects on `isinstance` (`_variables.py:292`) — but it would
  also inherit the optional integer normalization and error wording that names the
  variable "of type integer", and it would make every `isinstance(…,
  IntegerVariable)` test in the codebase silently true for a catalog variable.
  Inheriting `DiscreteVariable` would reuse the finite-domain machinery but forces
  re-pinning `type`, `component_type` and `choices`, and abstracts prematurely over
  two different notions. → **Decided — inherit `BaseVariable` directly** (per the
  requester). The cost is explicit: the kind is absent from the integer mask, so it
  is not rounded; recorded as a known limitation.
- **Frozen columnar storage versus storing the DataFrame**: storing a deep copy
  defends only against the caller's handle — the table reachable through the object
  stays mutable. Freezing the internal blocks (`df._mgr.blocks`) reaches into private
  pandas API that Copy-on-Write can defeat by replacing the block, so the guarantee
  would be version-dependent. Returning a fresh deep copy on every access is safe but
  copies a potentially large table on every read and keeps mutable state inside a
  frozen model. → **Decided — decompose into frozen NumPy arrays at construction; the
  `DataFrame` is input and output, never state.** The costs to record: a rebuild per
  access, and the pandas metadata a plain column decomposition does not carry (see
  Ambiguities).
- **Accepting a `DataFrame` or a mapping**: `ChoicesType` (`variable/discrete.py:59`) already
  accepts an array, a list or a tuple, so a concept that accepted only one input
  shape would be the odd one out; and a mapping `column name → sequence` is the
  natural literal form once the storage is columnar. → **Decided — accept both**,
  with the same frozen storage either way. The cost is the validation the `DataFrame`
  path gets for free: a mapping can be **ragged**, so the column lengths — and the
  labels' — must be checked against each other before the row count is derived, and a
  mismatch must name the offending columns and their lengths rather than surface as a
  NumPy broadcast error. Truncating to the shortest column, or padding the others,
  would silently invent or discard alternatives. → **A ragged mapping is rejected.**
- **A scalar column: broadcast versus rejected.** Rejecting is the symmetric reading
  of the ragged rule, but a scalar is unambiguous — a property constant over the
  catalogue — and `BaseVariable.__convert_bound` (`variable/base.py:142`) already broadcasts a
  scalar bound over `size`, so rejecting here would contradict a convention the
  hierarchy already has. → **Decided — broadcast to the row count** (per the
  requester).
- **A mapping of scalars only: one row versus rejected.** Nothing in such a mapping
  carries a length, so the row count has to be chosen rather than derived. Rejecting
  would be defensible — a catalogue of one alternative offers no choice — but it
  would reject a description that is perfectly well-formed, and the resulting
  variable is *degenerate*, not *invalid*: it has one admissible value, bounds
  `[0, 0]`, and no combinatorics. → **Decided — a one-row catalogue** (per the
  requester), and more generally **a degenerate catalog variable is allowed and
  left as is**. The consequence to state plainly: such a variable still occupies a
  component of the design vector and is still reported as a design variable, so a
  driver would carry a dimension with a single admissible value. Whether a
  single-row catalogue should warn, be pinned as a constant, or be excluded from the
  design vector is **a second-step question**, tracked out of scope.
- **An empty column: dropped with a warning versus rejected.** Rejecting is stricter
  but punishes a user for a column that carries nothing, typically produced upstream
  by a filter or a merge; silently dropping it would hide a real upstream defect. →
  **Decided — dropped, with a log warning naming every dropped column** (per the
  requester). Three consequences to accept: the catalogue the variable owns is the
  cleaned one, which is also what HDF stores, so round-trip equality still holds; the
  warning is the only trace that a column disappeared, so it must name each one and
  not report a count; and the catalogue is the **first module of `space/` outside
  `design/` to log**, so it owns a module-level `LOGGER = logging.getLogger(__name__)`,
  as `space/design/_normalizer.py:46` does.
- **What counts as blank.** A zero-length column is the obvious case, but it can only
  occur on the mapping path. Counting an **all-blank** column as empty is what makes
  the rule useful for a `DataFrame`, where a column always has the table's length: an
  all-`NaN` column is the residue a filter or a left join leaves behind, and an
  all-`""` column is the residue a text merge leaves behind. → **Decided — blank
  means missing (`NaN`, `None`, `NaT`) or, after stripping, the empty string; a
  column of zeros is a value and is kept** (per the requester). This is the line the
  rule must not cross: `0`, `False`, `""`, `None` and `NaN` are all falsy in Python,
  so the test must be blankness and never falsiness. Detection needs a primitive that
  spans dtypes, since `isnan` does not accept an `object` array; `pandas.isna` covers
  the missing values uniformly and is already a hard dependency, with the
  empty-string test applied on top of it.
- **Stripping normalizes the stored values, not just the blankness test.** Testing on
  a stripped copy while storing the original would make `" alu "` a label that
  renders with invisible padding and that no lookup finds, and it would leave two
  catalogues unequal for a difference nobody can see. → **Decided — strings are
  stripped in the property columns and in the index labels, and the stripped values
  are stored** (per the requester). The consequence to accept: the stored catalogue
  is not the input catalogue, so `catalog.loc[" alu "]` raises where
  `catalog.loc["alu"]` succeeds. Stripping also subsumes the whitespace-only case,
  which is why it precedes the blankness test.
- **HDF serialization of the catalogue**: a single 2-D dataset would be simpler but
  cannot hold columns of different dtypes, which the accepted validation explicitly
  allows. → **Decided — a sub-group per variable: one dataset for the labels, one
  dataset per column**, with `to_hdf` deleting a stale sub-group before writing, as
  it already deletes a stale `choices` dataset (`_io.py:117-127`), so that an append
  which changes the catalogue's shape or the variable's kind cannot leave wreckage
  behind. `from_hdf` cross-validates the presence of the sub-group against the
  declared type and rebuilds through the factory with a catalogue, never with bounds
  (`:207-212`). The read and write live on `Catalog`.
- **CSV export refused versus a sidecar file versus a labels-only column**: a sidecar
  file per catalogued variable would round-trip faithfully but silently turns a
  one-file format into a multi-file one; a labels-only column would lose the
  properties, so `from_csv` could not rebuild an equal space, which is a silent
  defect. → **Decided — `to_csv` raises**, naming the offending variable. HDF remains
  the faithful format.
- **No new view column and no catalogue accessor on the façade**: adding a `labels`
  column would mirror the `choices` column, but the labels are not the value and
  would suggest they are; and a `get_catalog` accessor on `DesignSpace` would be
  surface with no consumer in this story — the symmetric `get_choices` the analysis
  cited has since been dropped from the façade, which now hands the variable object
  back through `variables` instead. →
  **Decided — the view is unchanged, and `has_catalog_variables` is the only new
  façade member.** The gap the analysis recorded — a space reloaded from HDF exposing
  its catalogues nowhere public — closed on its own: `VariablesView` is public and
  hands back the variable, so the catalogue is reachable as
  `design_space.variables[name].catalog` and its table as `.catalog.to_dataframe()`.
- **Labels rendered in the domain error message, and nowhere else**: a message that
  said only "not in `[0, 2]`" would be correct and useless, since the user thinks in
  materials, not positions. → **Decided — the two wording hooks render the items,
  elided beyond six entries** as `_format_choices` does (`variable/discrete.py:233`).
- **Class stays private** — *reversed during implementation.* The analysis assumed
  the hierarchy still lived in `gemseo.space._variable`, so it kept the new kind
  private for consistency with `DiscreteVariable` and recorded the visibility as a
  gap. GGQPA-1845 made the package public before this story landed, which removed the
  consistency argument along with the gap. → **Decided — public**:
  `CatalogVariable` is exported from `gemseo.space.variable`, and `Catalog` is a
  public module of its own, `gemseo.space.catalog`, lazily re-exported as
  `gemseo.space.Catalog`. It sits beside the variable package rather than inside it,
  so that the factory, which walks `gemseo.space.variable`, never sees it, and so that
  the dependency runs one way only.

### Alternatives Considered

- **The table inside the variable, with no `Catalog`**: the original decision,
  reversed above. Kept here because its failure mode is instructive — a
  `DataFrame`-typed field on a frozen Pydantic model forces `arbitrary_types_allowed`
  and a specialized `__eq__`, and buries a table-processing pipeline in a variable
  class.
- **A label-valued variable** carrying `StringArray` / `NDArrayPydantic[str_]`
  components: the most natural reading of "catalog". Rejected — `ComponentDType`
  admits only `int64` and `float64`, `TYPE_MAP` maps one dtype per `DataType` member
  and is read by two consumers outside the space package
  (`core/problem/database.py:1102`, `doe/core/base_doe_library.py:209`), and
  `get_common_dtype` would silently coerce a string array. The design vector must
  stay numeric.
- **Reusing `DiscreteVariable` with `choices = range(n)`** plus a catalogue field:
  one fewer class. Rejected — it pins `float64`, so positions would render as `0.0`,
  and it makes `choices` a redundant second encoding of the row count, which can
  disagree with the catalogue.
- **Reusing `IntegerVariable`** with bounds `[0, n-1]` and a catalogue field:
  rejected for the inheritance reasons above.
- **A shared external catalogue registry**, with variables referring to a catalogue
  by key: rejected — it moves the domain out of the variable, reintroduces the
  user-maintained map that 1885 rejected, and makes a design space no longer
  self-contained. Note that `Catalog` is a *value object*, not a registry: it is
  owned by the variable, not referenced by name.
- **`Catalog` as a plain frozen dataclass, or as a `DataFrame` subclass**: a
  dataclass would not compose with Pydantic validation, and subclassing `DataFrame`
  — as `Dataset` does (`dataset/dataset.py:91`) — would inherit every mutating method
  the immutability requirement exists to forbid.
- **Storing the DataFrame with its internal blocks frozen** through `df._mgr`, or
  **deep-copying on every access**: both rejected above.
- **Rejecting a scalar column**, **rejecting an empty column**, **testing blankness
  before stripping**, **stripping only for the test**, and **rejecting a scalar-only
  mapping**: all rejected above.
- **Treating a degenerate variable specially** — warning on construction, pinning it
  as a constant, or keeping it out of the design vector: rejected for this story
  (requester's call), to be revisited in a second step.
- **Delivering the catalogue discipline in the same story**: rejected — it needs a
  grammar, a policy for non-numeric and partly-blank columns, and a differentiation
  story, and it would double the review surface of a data-model change.

### Out of Scope

Explicitly not in this story, each tracked separately:

- **The catalogue-to-coupling discipline**: the discipline that reads
  `(catalogue, position)` and emits one output per catalogue column, along with the
  policy for non-numeric and partly-blank columns at the discipline boundary.
- **Solving.** No encoder, no relaxation, no working design space, no capability
  flag, no driver guard, no rounding.
- **A label-based API**: declaring or reading back a value as a label, and any
  label-to-position conversion helper.
- **Vectors of catalog variables** — declare several scalar ones.
- **Handling the degenerate one-row catalogue** — a variable with a single
  admissible value is accepted and left as is; deciding whether it should warn, be
  treated as a constant, or be dropped from the design vector is a second step.
- **Sharing one `Catalog` between several variables**, or naming catalogues.
- **Catalog random variables** in `ParameterSpace`.
- **Exporting the variable hierarchy publicly**, for this kind or any other.

## Risk & Gap Analysis

### Requirement Ambiguities

- **Which pandas metadata the frozen storage may drop.** A column decomposition
  carries labels, column names and per-column dtypes; it does not carry a
  `MultiIndex` on rows or columns, an index name, or a pandas extension dtype
  (`Catalog`, nullable `Int64`). Each must either be preserved, dropped
  deliberately and documented, or rejected at construction. This analysis assumes a
  flat index and a flat column axis, and rejects anything else with an explicit
  message. **Resolved in implementation**: a `MultiIndex` on either axis is rejected,
  naming the offending axis; an extension array is read positionally as a sequence of
  values, so its dtype is not preserved but its missing values are, through the
  `object` storage described below; a column name that is not a string is converted to
  one, and two names colliding once converted are rejected before the collision could
  silently drop a column; the index name is dropped.
- **Non-numeric columns.** Accepted at construction per the requester, but nothing
  reads them in this story, so the policy is deferred to the discipline. The visible
  consequence here is HDF: an `object`-dtype column has no natural dataset type. This
  analysis assumes string-like columns are stored as string datasets and any other
  object column is rejected **at serialization time**, not at construction.
  **Resolved in implementation**, and with one more failure mode than the analysis
  foresaw: the rejection lives in `Catalog.check_hdf_writable`, which runs before the
  file is opened, and it covers not only an `object` column and an unstorable dtype but
  also a column name, a cell or a label that UTF-8 cannot encode — a lone surrogate
  survives in a Python string and fails only at encoding time.
- **Duplicate index labels.** Accepted per the requester, and harmless for the domain
  since the position is the value. Two consequences to accept knowingly: the error
  message can list the same label twice, and any future label-to-position helper will
  have to define which position a repeated label resolves to.
- **What the default labels of a mapping are.** With no labels supplied, this
  analysis uses the positions themselves, as pandas does with a `RangeIndex`. The
  consequence is that a label and a position then render identically in the domain
  error message; confirm that no distinct placeholder is wanted.
- **Stripping non-string objects.** Stripping is defined for strings. A column of
  `bytes`, or of objects with a `strip` method, is left untouched by this analysis;
  confirm that only `str` values are stripped. **Resolved in implementation**: `str`,
  `bytes` and `bytearray` are stripped — a byte string is as much a string here as a
  unicode one — and everything else is untouched. A `bytearray` is normalized to
  `bytes` on the way out, upstream of any broadcast or freeze, because NumPy reads a
  `bytearray` through the buffer protocol and would otherwise explode it into one row
  per byte.
- **A single-column catalogue with no properties.** The rule "at least one column"
  forbids a pure label set. If a user wants an unordered choice with no properties,
  they must add a dummy column. Confirm that this is intended rather than relaxing
  the rule to zero columns.
- **Whether `Catalog` is part of the public story.** It stays in the private package
  with the variable, but it is the object the follow-up discipline will consume, so
  it will need a public home sooner than the variable kinds do. **Resolved**: the
  hierarchy became public with GGQPA-1845 before this story landed, so `Catalog` got
  its public home immediately — its own module `gemseo.space.catalog`, lazily
  re-exported as `gemseo.space.Catalog`, beside the public `gemseo.space.variable`
  package rather than inside it.
- **No public path to the catalogue of a reloaded space.** With no `get_catalog` on
  the façade, the catalogue is reachable only from the variable object, which the
  caller holds only when they built it. A space reloaded from HDF exposes its
  catalogues nowhere public — `_variables` is private. The 1885 analysis raised the
  same need for the discrete kind and answered it with a façade accessor,
  `get_choices`, which has since been dropped. The requester declined the symmetric
  accessor for this story. **Resolved without one**: `VariablesView` is public and
  hands the variable object back, so a reloaded space exposes its catalogues as
  `design_space.variables[name].catalog`, and their tables as
  `.catalog.to_dataframe()` — which is what the follow-up discipline story will
  consume.

### Edge Cases

- **A one-row catalogue** yields bounds `[0, 0]` and a single admissible value. It
  must be accepted, and the derived bounds must not be mistaken for an unset bound.
  The variable is then **degenerate** — frozen at position `0`, no combinatorics —
  and still counts as a design variable and still occupies a design-vector
  component. Reached three ways, all of which must behave identically: a one-row
  table, a one-element mapping, and a mapping of scalars only.
- **A mapping of scalars only with labels supplied**: the labels then carry the
  length, so the row count comes from them and every scalar is broadcast to it. A
  scalar-only mapping is a one-row catalogue **only** when no labels fix a different
  count.
- **A scalar broadcast over a one-row catalogue** is indistinguishable from a
  one-element sequence, and must produce the same catalogue.
- **A ragged mapping**: columns of unequal length, or labels whose length does not
  match the columns'. Rejected with a message naming the lengths.
- **A blank scalar column**: a scalar that is itself blank — `None`, `NaN` or a
  whitespace-only string — makes an entirely blank column, so it is dropped like any
  other, which means the blankness test must look at scalars too and not only at
  sequences.
- **An all-`NaN` column in a `DataFrame`**: the column has the table's length but
  every element is missing, so it is dropped with a warning while the other columns
  and every row survive. This is the main case the dropping rule exists for.
- **An all-`""` or all-whitespace column** is dropped, because stripping precedes the
  test; a column mixing `""`, whitespace and `NaN` is blank throughout and is dropped
  too.
- **An all-zero column, and an all-`False` column, are kept.** This pair is the
  regression test that keeps a future refactor from replacing the blankness test with
  a truthiness test.
- **A partly-blank column** is kept untouched, `NaN` included, so a property can be
  absent for some alternatives. The follow-up discipline will decide what it emits
  for a missing property; nothing here forces that choice.
- **An `object` column of `None`** must be detected as blank even though `isnan`
  cannot be applied to it — the reason detection goes through `pandas.isna`.
- **Every column blank**: each is dropped, each is named in the warning, and the
  "at least one column" rule then fires. The warning and the failure must tell a
  consistent story.
- **A label that becomes empty after stripping** — `" "` in the labels — is a label,
  not a column, so nothing drops it; it must be accepted and rendered as an empty
  string in the error message.
- **`filter_components` and `filter_dimensions`** on a scalar kind: the only valid
  request is the single component `0`, and a duplicated or out-of-range index must
  fail with a readable message rather than a Pydantic error naming the internal
  `size` field — the reason `DiscreteVariable` carries a `field_validator("size",
  mode="before")` (`variable/discrete.py:90`).
- **`model_copy`** and the caller-set-bounds carry-over rule (`variable/base.py:370`): a copy
  must not re-supply the derived bounds, or the "bounds are not settable" check
  would trip on every copy — the subtlety the base already handles for the discrete
  kind.
- **Pickle and deepcopy**: `__setstate__` must refreeze **every** catalogue array, as
  `DiscreteVariable` refreezes `choices` (`variable/discrete.py:276`), because NumPy loses
  the writeable flag across pickling and Pydantic restores the model without
  re-validating it. With a nested `Catalog`, the refreezing belongs to the
  catalogue's own `__setstate__`, and the variable must not assume the base class
  handled it.
- **Mutating the caller's table** after construction, and **mutating the table
  returned by the accessor**: both must leave the variable's domain, bounds and
  current value intact. These are the two failure modes the storage decision exists
  to remove, and both deserve an explicit test.
- **An HDF append** into a file that already holds the same space, where the
  catalogue has changed shape, or where the variable has changed kind: the stale
  sub-group must be deleted before writing, or the reload silently mixes two
  catalogues.
- **`extend`, `add_variables_from`, `rename_variable`, `to_scalar_variables`**: all
  share the frozen variable object, so the catalogue follows without those sites
  learning about it — except the multi-component branch of `to_scalar_variables`,
  which rebuilds from `(size, type, lower_bound, upper_bound)` and is unreachable for
  a scalar kind. Worth an explicit test rather than an assumption.
- **`to_complex`** on a space holding a catalog variable: the position becomes a
  complex number with a zero imaginary part, as it already does for an integer
  variable. Nothing to add, worth asserting.
- **A value of `None`** for the current value: every variable always has an entry in
  `Value`, and the domain hooks must tolerate `None` as "not set", the way
  `DiscreteVariable.find_components_outside_domain` does (`variable/discrete.py:216`).

The implementation surfaced a second family of edge cases, each now covered by a test:

- **A string given as the labels.** `labels="alu"` is one label, not a nine-character
  sequence, and `labels=""` is one label naming one row, not an absence of labels; the
  same holds for `bytes`, `bytearray` and a zero-dimensional array. Without this rule a
  bare `len` either explodes the string or raises on the array. The rule is the one
  `_is_scalar` already applies to a column, so a value that is one scalar as a column is
  one label as a label.
- **A column that is an unordered collection**, e.g. a `set` or a mapping's keys/items
  view: sized and iterable, so it would be read as a sequence whose row order is
  arbitrary. Rejected, pointing the caller at an ordered sequence.
- **A column that is itself a `DataFrame`**: iterable over its *column names*, so it
  would be read, silently, as a column holding those names. Rejected.
- **Column names that collide once converted to strings**, e.g. the integer `1` and the
  string `"1"`, or two literally equal names: pandas keeps both, and a dict keyed by the
  stringified name would keep only the last. Rejected at conversion time, before the
  collision could drop a column.
- **A column mixing a string and a missing value**: left to infer the dtype, NumPy
  promotes both to a fixed-width string dtype and freezes the missing cell as the
  literal text `"nan"`. Such a column is stored as `object`, which then makes equality
  a value-by-value comparison, since `array_equal(..., equal_nan=True)` raises on an
  object pair and a missing value may never reach a bare `==` — `pd.NA == 1` is itself
  missing, and `bool` raises on it.
- **A dimensionality check that must run after stripping**: a `DataFrame` column of
  equal-length sequences is an opaque one-dimensional object array until stripping turns
  it into a nested list, which is when its true shape becomes visible.
- **Columns describing no row at all**, e.g. `{"mass": []}`: every such column is blank,
  so dropping first would report a missing *column* for an input that did supply one.
  The row count is checked before the blank columns are dropped, and reports the length
  disagreement instead when labels name rows those empty columns contradict.
- **A non-numeric or non-finite value passed to the domain hook**: a label reaches
  `find_components_outside_domain` as a string array, where `.real` raises, and an
  infinity reaches `mod`, which emits a spurious `RuntimeWarning`. Both are out of
  domain and both return early, before the arithmetic.
- **A catalogue used as a dictionary key, or rendered**: pydantic derives a frozen
  model's hash and repr from the field values, so the former raises on the mapping of
  arrays and the latter prints every cell — some 13 kB for a thousand rows — into a
  traceback or a debugger view. Both are overridden.
- **A string that UTF-8 cannot encode**, a lone surrogate: legal in a Python string,
  and only fails when the catalogue is written. Caught by `check_hdf_writable`, before
  the file is opened, for a column name, a cell and a label alike.

### Technical Risks

- **The blankness test is a NumPy trap.** Comparing a numeric array to `""` does not
  reliably broadcast — NumPy can return the scalar `False` with an "elementwise
  comparison failed" warning instead of a per-element mask — so the empty-string test
  must go through pandas, or be restricted to string and `object` columns, and never
  be written as a bare `array == ""`. A catalogue with one numeric and one text
  column catches the regression.
- **The pipeline's ordering is the main correctness risk.** Rules that each change
  what the next one sees; every reordering produces a plausible-looking but wrong
  result. Mitigation: the order is written into the `Catalog` docstring and covered by
  one test per adjacent pair — blank-before-rectangular, strip-before-blank,
  rectangular-before-broadcast. It grew from six rules to **ten steps** during
  implementation, each addition an ordering constraint of its own: kind-checking before
  stripping, dimensionality after stripping, and the row count before the blank columns
  are dropped.
- **Nesting a Pydantic model inside a frozen Pydantic model** is a smaller risk than
  the `DataFrame` field it replaces, but it is not free: `model_copy(deep=...)`,
  `__setstate__`, and `BaseVariable.__eq__` walking `model_fields` all now traverse a
  nested model. The mitigation is the same as for the rest of the hierarchy: add the
  kind to `ALL_KINDS` in `tests/space/variable/utils.py` early and let the shared
  cross-kind matrix run against it.
- **HDF write of an `object`-dtype column** has no natural dataset type. Mitigated by
  deciding the policy at serialization time (see Ambiguities) and by testing a
  string-column catalogue explicitly. The risk materialized once more than expected: an
  `object` column is now a *normal* outcome of the pipeline, since that is how a column
  mixing a string and a missing value is stored, so the rejection is reachable from a
  plain mapping of Python values and not only from an exotic dtype.
- **A catalog variable is not solvable.** Like a discrete variable, it falls out
  of every normalization mask and, because it does not subclass `IntegerVariable`,
  out of the integer mask too. A DOE therefore raises *"some components of the design
  space are unbounded"* (`doe/core/base_doe_library.py:419-439`) and an optimizer
  fails only when it stores a non-integral optimum. This is the same known limitation
  the discrete kind ships with, already documented at
  `docs/user_guide/concepts/design_space.md:166-203`; the user-guide section this
  story adds must carry the same admonition.
- **Not being rounded** is a consequence of the chosen parent: a driver that produced
  `1.4` for a catalog variable would be rejected by the domain check rather than
  rounded to `1`. This is the honest behavior while the kind is unsolvable, but the
  solving story will have to revisit it.
- **The catalogue rebuild cost** is paid on every access to the `DataFrame` view. It
  is invisible for a catalogue of tens of rows and would matter for one of millions;
  no caching is proposed, and the accessor's cost should be documented rather than
  optimized speculatively.
- **A documentation defect sits in the file this story must edit**:
  `docs/user_guide/concepts/design_space.md:68-70` ends *"it is declared by passing a
  variable object to `add_variable()`:"* with no code block following, so the colon
  dangles into the next heading. Introduced by `a2acc46b56`. Out of scope as
  specified, but adjacent to the new section and worth fixing in the same MR.

### Acceptance Criteria Coverage

| AC# | Description | Addressable? | Notes |
|-----|-------------|--------------|-------|
| 1 | A catalog variable is declared from a pandas catalogue and added to a design space | Yes | Through `add_variable(name, variable=CatalogVariable(catalog=...))`, the path 1885 opened. |
| 2 | The value of the variable is the row position `0…n-1`, stored as `int64` | Yes | `component_type = int64`; `TYPE_MAP` gains one entry. |
| 3 | The catalogue is grouped into a dedicated `Catalog` object | Yes | `Catalog(BaseModel, frozen=True)` in `gemseo/space/catalog.py`, a public module of its own; ignored by the variable factory, which walks `gemseo.space.variable` and filters on `BaseVariable`. |
| 4 | Each catalogue column is preserved as a named property | Yes | Column names and values are part of the frozen storage; nothing reads them in this story. |
| 5 | The variable is scalar | Yes | `size` pinned to 1, with a readable message for any other size. |
| 6 | The bounds are derived from the row count and cannot be set | Yes | `[0, n-1]`; explicit bounds rejected at construction, both setters fail. |
| 7 | The catalogue cannot be modified after construction | Yes | Frozen columnar storage; the accessor rebuilds a `DataFrame`, so neither the caller's table nor the returned one is the catalogue's state. |
| 8 | A `DataFrame` and a mapping are both accepted | Yes | Same frozen storage either way; labels optional on the mapping path. |
| 9 | A ragged mapping is rejected | Yes | Every sequence column, and the labels, must hold the same number of elements; the message names the mismatching lengths. |
| 10 | A scalar column is broadcast over the rows | Yes | Consistent with the scalar-bound broadcast the base class already performs (`variable/base.py:142`). |
| 11 | Strings are stripped, in the columns and in the labels | Yes | The stripped values are what the catalogue stores, so lookups and messages use them. |
| 12 | A blank column is dropped with a log warning | Yes | Blank means no element, or every element missing (`pandas.isna`) or empty after stripping. Dropped after the row-count check and before the rectangular check; the warning names every dropped column. The catalogue module owns the logger. |
| 13 | A column of zeros, and a partly-blank column, are kept | Yes | The test is blankness, never falsiness; no imputation. |
| 14 | An empty catalogue is rejected | Yes | At least one row and at least one column, checked after dropping and broadcasting. |
| 14b | A mapping of scalars only yields a one-row catalogue | Yes | Nothing carries a length, so the row count is 1; with labels supplied, the labels fix it instead. |
| 14c | A degenerate variable is accepted as is | Yes | One admissible value, bounds `[0, 0]`, no warning and no special treatment; it still occupies a design-vector component. |
| 15 | The pipeline order is a documented contract | Yes | Convert → check kinds → strip → check dimensionality → check row count → drop → rectangular → broadcast → non-empty → default labels and freeze, with one test per adjacent pair. Ten steps as built, six as analysed. |
| 16 | An inadmissible value is rejected with a message naming the items | Yes | The two wording hooks, with the elision rule of `_format_choices`. |
| 17 | The default current value is the first row | Yes | `compute_default_value` returns `[0]`. |
| 18 | A catalog variable is never normalized | Yes | All-`False` normalization mask, as for the discrete kind. |
| 19 | HDF round-trips the space, catalogue included | Yes | A sub-group per variable; stale sub-group deleted before writing; read and write owned by `Catalog`, which refuses an unwritable or unencodable catalogue before the file is opened. |
| 20 | CSV export fails explicitly | Yes | `to_csv` raises, naming the variable. |
| 21 | The tabular view is unchanged | Yes | No new column; the variable renders as a bounded integer of type `catalog`. |
| 22 | `has_catalog_variables` reports the presence of the kind | Yes | O(1), from a count cached in `Variables.__reindex`. |
| 23 | Existing behavior of the three current kinds is unchanged | Yes | The only shared edits are additive: a `DataType` member, a `TYPE_MAP` entry, and one extra arm in the `add_variable` rejection and the membership fallback. |
| 24 | The kind is usable by an algorithm or a driver | **No** | Out of scope, and a known limitation: a DOE raises "unbounded components" and an optimizer fails at storage time. Tracked with the discrete kind's solving story. |
| 25 | A catalogue column feeds a coupling variable | **No** | Out of scope: that is the follow-up discipline story, which will consume the `Catalog`. |
| 26 | A value can be declared or read back as a label | **No** | Out of scope; the labels appear only in the error message. |
| 27 | The catalogue of a reloaded space is reachable publicly | Yes | Not through a `get_catalog` on the façade, which the requester declined, but through the public `VariablesView`: `design_space.variables[name].catalog`. |
| 28 | `Catalog` and the variable kinds are importable from a public module | Yes | GGQPA-1845 made the hierarchy public before this story landed: `CatalogVariable` from `gemseo.space.variable`, `Catalog` from `gemseo.space.catalog`, re-exported as `gemseo.space.Catalog`. |
