# Copyright 2021 IRT Saint-Exupéry, https://www.irt-saintexupery.com
#
# This program is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License version 3 as published by the Free Software Foundation.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program; if not, write to the Free Software Foundation,
# Inc., 51 Franklin Street, Fifth Floor, Boston, MA  02110-1301, USA.

# Contributors:
#    INITIAL AUTHORS - initial API and implementation and/or initial
#                         documentation
#        :author: Francois Gallard
#    OTHER AUTHORS   - MACROSCOPIC CHANGES

import re
from pathlib import Path

import jinja2

from gemseo.algos.doe.doe_factory import DOEFactory
from gemseo.algos.opt.opt_factory import OptimizersFactory
from gemseo.api import (
    get_algorithm_options_schema,
    get_available_doe_algorithms,
    get_available_formulations,
    get_available_mdas,
    get_available_opt_algorithms,
    get_available_post_processings,
    get_available_surrogates,
    get_formulation_options_schema,
    get_mda_options_schema,
    get_post_processing_options_schema,
)
from gemseo.formulations.formulations_factory import MDOFormulationsFactory
from gemseo.mda.mda_factory import MDAFactory
from gemseo.mlearning.api import (
    get_classification_models,
    get_clustering_models,
    get_regression_models,
)
from gemseo.mlearning.classification.factory import ClassificationModelFactory
from gemseo.mlearning.cluster.factory import ClusteringModelFactory
from gemseo.mlearning.regression.factory import RegressionModelFactory
from gemseo.post.post_factory import PostFactory

GEN_OPTS_PATH = None


def get_template(template_path):
    """Generate jinja environment.

    :param template_path: path to the template
    """
    env = jinja2.Environment(loader=jinja2.FileSystemLoader(GEN_OPTS_PATH))
    template = env.get_template(template_path)
    return template


def get_options_schemas(feature_name, feature_api_opts_pt):
    """Get a dictionary of options schema for a feature (algo, doe, mda...)

    :param feature_name: name of the feature to get
    :param feature_api_opts_pt: pointer to the API function that retrieves
        the options schema for the feature, for instance : get_algorithm_options_schema
        for algos
    """
    descr = "description"
    obj_type = "type"
    all_data = feature_api_opts_pt(feature_name, output_json=False)
    all_options = all_data["properties"]
    for opt_name, opt_schema in list(all_options.items()):
        if descr in opt_schema:
            opt_schema[descr] = opt_schema[descr].replace("\n", " ")
        elif "anyOf" in opt_schema:
            if descr in opt_schema["anyOf"][0]:
                opt_schema[descr] = opt_schema["anyOf"][0][descr].replace("\n", " ")
        else:
            print(
                Warning(
                    "Missing description for option {} of algo {}".format(
                        opt_name, feature_name
                    )
                )
            )
            opt_schema[descr] = ""
        if obj_type in opt_schema:
            opt_schema[obj_type] = opt_schema[obj_type].replace("\n", " ")
        elif "anyOf" in opt_schema:
            var_types = []
            for sub_opt_schema in opt_schema["anyOf"]:
                if obj_type in sub_opt_schema:
                    var_types.append(sub_opt_schema[obj_type].replace("\n", " "))
            opt_schema[obj_type] = "Union[{}]".format(",".join(var_types))
        else:
            print(
                Warning(
                    "Missing object type for option {} of algo {}".format(
                        opt_name, feature_name
                    )
                )
            )
            opt_schema[obj_type] = ""

    return all_options


def get_value_from_schema(feature_name, feature_api_opts_pt, keyword):
    all_data = feature_api_opts_pt(feature_name, output_json=False)
    if keyword in list(all_data.keys()):
        value = all_data[keyword].replace("\n", " ")
    else:
        value = None
    return value


def render_template(template_name, **template_keywords):
    template = get_template(template_name)
    doc = template.render(**template_keywords)

    out_path = Path(GEN_OPTS_PATH).parent / template_name.replace(
        "_template.tmpl", ".rst"
    )

    with open(out_path, "w", encoding="utf-8") as outf:
        outf.write(doc)


def split_param_type(options_dict):
    for algo in options_dict.keys():
        for option in options_dict[algo].keys():
            tmp = re.split(
                r":type ([\*\w]+): (.*?)", options_dict[algo][option]["description"]
            )
            options_dict[algo][option]["description"] = tmp[0]
            if len(tmp) == 4:
                options_dict[algo][option]["ptype"] = tmp[3].strip()
            else:
                options_dict[algo][option]["ptype"] = "Unknown"
            if options_dict[algo][option]["type"] == "":
                options_dict[algo][option]["type"] = "Unknown"
            # sphinx uses "ptype" to display the type of an argument.
            # The latter is read from the docstrings.
            # If it is "Unknown", we use the type found in the grammar
            # which is often less meaningful than the ones we can find in docstrings.
            if options_dict[algo][option]["ptype"] == "Unknown":
                options_dict[algo][option]["ptype"] = options_dict[algo][option]["type"]


def main(gen_opts_path):
    global GEN_OPTS_PATH
    GEN_OPTS_PATH = gen_opts_path

    # Get algorithms sorted by names
    doe_algos = sorted(get_available_doe_algorithms())
    opt_algos = sorted(get_available_opt_algorithms())
    mda_algos = sorted(get_available_mdas())
    formulation_algos = sorted(get_available_formulations())
    surrogate_algos = sorted(get_available_surrogates())
    post_algos = sorted(get_available_post_processings())
    regression_algos = sorted(get_regression_models())
    clustering_algos = sorted(get_clustering_models())
    classification_algos = sorted(get_classification_models())

    opt_options = {
        algo: get_options_schemas(algo, get_algorithm_options_schema)
        for algo in opt_algos
    }
    split_param_type(opt_options)
    opt_descriptions = {
        algo: OptimizersFactory().create(algo).lib_dict[algo].get("description")
        for algo in opt_algos
    }
    opt_url = {
        algo: OptimizersFactory().create(algo).lib_dict[algo].get("website")
        for algo in opt_algos
    }
    render_template(
        "opt_algos_template.tmpl",
        opt_algos=opt_algos,
        opt_options=opt_options,
        opt_descriptions=opt_descriptions,
        opt_url=opt_url,
    )

    doe_options = {
        algo: get_options_schemas(algo, get_algorithm_options_schema)
        for algo in doe_algos
    }
    split_param_type(doe_options)
    doe_descriptions = {
        algo: DOEFactory().create(algo).lib_dict[algo].get("description")
        for algo in doe_algos
    }
    doe_url = {
        algo: DOEFactory().create(algo).lib_dict[algo].get("website")
        for algo in doe_algos
    }
    render_template(
        "doe_algos_template.tmpl",
        doe_algos=doe_algos,
        doe_options=doe_options,
        doe_descriptions=doe_descriptions,
        doe_url=doe_url,
    )

    formulation_options = {
        algo: get_options_schemas(algo, get_formulation_options_schema)
        for algo in formulation_algos
    }
    split_param_type(formulation_options)
    formulation_descriptions = {
        algo: MDOFormulationsFactory().factory.get_class(algo).__doc__
        for algo in formulation_algos
    }
    formulation_modules = {
        algo: MDOFormulationsFactory().factory.get_class(algo).__module__
        for algo in formulation_algos
    }
    render_template(
        "formulation_algos_template.tmpl",
        formulation_algos=formulation_algos,
        formulation_options=formulation_options,
        formulation_descriptions=formulation_descriptions,
        formulation_modules=formulation_modules,
    )

    post_options = {
        algo: get_options_schemas(algo, get_post_processing_options_schema)
        for algo in post_algos
    }
    split_param_type(post_options)
    post_descriptions = {
        algo: PostFactory().factory.get_class(algo).__doc__ for algo in post_algos
    }
    post_modules = {
        algo: PostFactory().factory.get_class(algo).__module__ for algo in post_algos
    }
    render_template(
        "post_algos_template.tmpl",
        post_algos=post_algos,
        post_options=post_options,
        post_descriptions=post_descriptions,
        post_modules=post_modules,
    )

    mda_options = {
        algo: get_options_schemas(algo, get_mda_options_schema) for algo in mda_algos
    }
    split_param_type(mda_options)
    mda_descriptions = {
        algo: MDAFactory().factory.get_class(algo).__doc__ for algo in mda_algos
    }
    mda_modules = {
        algo: MDAFactory().factory.get_class(algo).__module__ for algo in mda_algos
    }
    render_template(
        "mda_algos_template.tmpl",
        mda_algos=mda_algos,
        mda_options=mda_options,
        mda_descriptions=mda_descriptions,
        mda_modules=mda_modules,
    )

    surrogate_modules = {
        algo: RegressionModelFactory().factory.get_class(algo).__module__
        for algo in surrogate_algos
    }
    render_template(
        "surrogate_algos_template.tmpl",
        surrogate_algos=surrogate_algos,
        surrogate_modules=surrogate_modules,
    )

    regression_modules = {
        algo: RegressionModelFactory().factory.get_class(algo).__module__
        for algo in regression_algos
    }
    render_template(
        "regression_algos_template.tmpl",
        regression_algos=regression_algos,
        regression_modules=regression_modules,
    )

    clustering_modules = {
        algo: ClusteringModelFactory().factory.get_class(algo).__module__
        for algo in clustering_algos
    }
    render_template(
        "clustering_algos_template.tmpl",
        clustering_algos=clustering_algos,
        clustering_modules=clustering_modules,
    )

    classification_modules = {
        algo: ClassificationModelFactory().factory.get_class(algo).__module__
        for algo in classification_algos
    }
    render_template(
        "classification_algos_template.tmpl",
        classification_algos=classification_algos,
        classification_modules=classification_modules,
    )
