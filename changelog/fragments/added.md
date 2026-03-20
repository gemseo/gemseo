- The `gemseo.formulations.bilevel.BiLevel` and its derived class `gemseo.formulations.bilevel_bcd.BiLevelBCD` now allow
  users to provide instances of `BaseMDA` or `Discipline` to be used as the MDA1 or the MDA2 of the formulation. These
  instances shall be provided via the fields `mda1_instance` and `mda2_instance` of the
  `gemseo.formulations.bilevel_settings.BiLevel_Settings`.
- It is now possible to deactivate the use of the MDA1 and MDA2 of the `gemseo.formulations.bilevel.BiLevel` and its
  derived class `gemseo.formulations.bilevel_bcd.BiLevelBCD` using the `use_mda1` and `use_mda2` of the
  `gemseo.formulations.bilevel_settings.BiLevel_Settings`.
- The `gemseo.formulations.bilevel.BiLevel` and its derived class `gemseo.formulations.bilevel_bcd.BiLevelBCD` now allow
  users to provide instances of `Discipline` to be used as sub-scenarios of the formulation. These instances shall be
  provided via the `disc_as_sub_scenario` field of the `gemseo.formulations.bilevel_settings.BiLevel_Settings`.
