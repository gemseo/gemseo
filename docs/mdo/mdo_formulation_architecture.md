---
status: draft
description: ""
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

# Architecture of the MDO formulations

## Class diagram

```mermaid
classDiagram
    class BaseFormulation {
        +differentiated_input_names_substitute
        +design_space
        +disciplines
        +optimization_problem
        +Settings
        +variable_sizes
        #settings
        +add_constraint()
        +get_default_sub_option_values()
        +add_observable()
        +get_optim_variable_names()
        +get_top_level_disciplines()
        +get_sub_options_grammar()
        +get_sub_scenarios()
        +get_x_mask_x_swap_order()
        +get_x_names_of_disc()
        +mask_x_swap_order()
        +unmask_x_swap_order()
        #build_objective()
        #get_dv_indices()
        #init_before_design_space_and_objective()
        #remove_unused_variables()
        #remove_sub_scenario_dv_from_ds()
        #set_default_input_values_from_design_space()
        #update_design_space()
        -disciplines
        -check_disciplines()
        -unmask_x_swap_order_if_one_sample()
        -unmask_x_swap_order_if_several_samples()
    }

    BaseFormulation <|-- BaseMDOFormulation
    class BaseMDOFormulation {
        +add_observable()
        +add_constraint()
    }

    BaseMDOFormulation <|-- MDF
    class MDF {
        +Settings
        +mda
        +get_default_sub_option_values()
        +get_sub_options_grammar()
        +get_top_level_disciplines()
        #init_before_design_space_and_objective()
        #remove_couplings_from_ds()
        #update_design_space()
        -check_mda()
    }

    BaseMDOFormulation <|-- IDF
    class IDF {
        +Settings
        +all_couplings
        +coupling_structure
        +get_top_level_disciplines
        +normalize_constraints
        #build_consistency_constraints()
        #compute_equilibrium()
        #get_normalization_factor
        #init_before_design_space_and_objective()
        #process_discipline
        #update_design_space()
        #update_top_level_disciplines()
        -coupling_structure
    }

    BaseMDOFormulation <|-- DisciplinaryOpt
    class DisciplinaryOpt {
        +Settings
        +get_top_level_disciplines()
        #init_before_design_space_and_objective()
        #update_design_space()
        -top_level_disciplines
    }

    BaseMDOFormulation <|-- BiLevel
    class BiLevel {
        +CHAIN_NAME
        +DEFAULT_SCENARIO_RESULT_CLASS_NAME
        +LEVELS
        +MDA1_RESIDUAL_NAMESPACE
        +MDA2_RESIDUAL_NAMESPACE
        +SUBSCENARIOS_LEVEL
        +SYSTEM_LEVEL
        +Settings
        +chain
        +coupling_structure
        +mda1
        +mda2
        +scenario_adapters
        +add_constraint()
        +get_default_sub_option_values()
        +get_sub_options_grammar()
        +get_top_level_disciplines()
        #add_sub_level_constraint()
        #add_system_level_constraint()
        #create_mdas()
        #create_inner_chain()
        #create_sub_scenarios_chain()
        #compute_adapter_inputs()
        #compute_adapter_outputs()
        #create_scenario_adapters()
        #get_mda1_outputs()
        #get_mda2_inputs()
        #get_variable_names_to_warm_start()
        #init_before_design_space_and_objective()
        #remove_couplings_from_ds()
        #scenario_computes_outputs()
        #store_optimal_local_design_values()
        #update_design_space()
        #mda1
        #mda2
        #scenario_adapters
        -mda_factory
    }

    BiLevel <|-- BilevelBCD
    class BilevelBCD {
        +Settings
        +bcd_mda
        #bcd_mda
        #create_sub_scenarios_chain()
    }
```
