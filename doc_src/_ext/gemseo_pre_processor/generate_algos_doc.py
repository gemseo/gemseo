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
from __future__ import annotations

import inspect
import re
from pathlib import Path
from typing import Any
from typing import Callable

import jinja2
from gemseo.algos.doe.doe_factory import DOEFactory
from gemseo.algos.driver_factory import DriverFactory
from gemseo.algos.driver_lib import DriverLib
from gemseo.algos.linear_solvers.linear_solvers_factory import LinearSolversFactory
from gemseo.algos.opt.opt_factory import OptimizersFactory
from gemseo.api import _get_schema
from gemseo.api import get_algorithm_features
from gemseo.core.factory import Factory
from gemseo.formulations.formulations_factory import MDOFormulationsFactory
from gemseo.mda.mda_factory import MDAFactory
from gemseo.mlearning.classification.factory import ClassificationModelFactory
from gemseo.mlearning.cluster.factory import ClusteringModelFactory
from gemseo.mlearning.qual_measure.quality_measure import MLQualityMeasureFactory
from gemseo.mlearning.regression.factory import RegressionModelFactory
from gemseo.post.post_factory import PostFactory
from gemseo.uncertainty.distributions.factory import DistributionFactory
from gemseo.uncertainty.sensitivity.factory import SensitivityAnalysisFactory
from gemseo.utils.source_parsing import get_options_doc

GEN_OPTS_PATH = None


def get_options_schemas(
    feature_name: str,
    feature_api_opts_pt: Callable[[str, bool, bool], str | dict[str, Any]],
) -> dict[str, dict[str, str] | list[dict[str, str]]]:
    """Get the options schema for an algorithm, e.g. DOE, MDA, ...

    Args:
        feature_name: The name of the algorithm to get.
        feature_api_opts_pt: The pointer to the API function
            that retrieves the options schema for the feature,
            for instance: ``get_algorithm_options_schema`` for drivers.

    Returns:
        The options schema.
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


def update_options_from_rest_docstring(
    algo: str, options: dict[str, dict[str, dict[str, str]]]
) -> None:
    """Update options from reST docstring.

    Args:
        algo: The name of the algorithms.
        options: The reST docstring of a function.
    """
    for option_name, option in options.items():
        try:
            tmp = re.split(r":type ([*\w]+): (.*?)", option["description"])
        except KeyError:
            print(
                "ERROR: failed to detect description for {} of algorithm {}".format(
                    option_name, algo
                )
            )
            tmp = [""] * 4

        option["description"] = tmp[0]
        option["default"] = "Unknown"

        if len(tmp) == 4:
            option["ptype"] = tmp[3].strip()
        else:
            option["ptype"] = "Unknown"

        if option["type"] == "":
            option["type"] = "Unknown"

        # sphinx uses "ptype" to display the type of an argument.
        # The latter is read from the docstrings.
        # If it is "Unknown", we use the type found in the grammar
        # which is often less meaningful than the ones we can find in docstrings.
        if option["ptype"] == "Unknown":
            option["ptype"] = option["type"]


class AlgoOptionsDoc:
    """Generator of the reST documentation of an algorithm from a Jinja2 template."""

    TEMPLATE = "algos_options.tmpl"
    ENV = jinja2.Environment(loader=jinja2.FileSystemLoader(Path(__file__).parent))

    def __init__(
        self,
        algo_type: str,
        long_algo_type: str,
        algo_factory: Any | Factory,
        template: str | None = None,
        user_guide_anchor: str = "",
    ) -> None:
        """
        Args:
            algo_type: The name of the algorithm type, e.g. "formulation",
                to be used internally (e.g. HTML anchors).
            long_algo_type: The long name of the algorithm type, e.g. "MDO formulation",
                to be used externally (e.g. HTML rendering).
            algo_factory: The factory of algorithms.
            template: The name of the template file located in the same directory
                as the current file.
                If None, :attr:`AlgoOptionsDoc.TEMPLATE` will be used.
            user_guide_anchor: The anchor of the section of the user guide
                about these algorithms.
        """
        if template is None:
            self.template = self.TEMPLATE
        else:
            self.template = template

        self.algo_type = algo_type
        self.long_algo_type = long_algo_type
        if isinstance(algo_factory, Factory):
            self.factory = algo_factory
        else:
            self.factory = algo_factory.factory

        self.algos_names = self.factory.classes
        self.get_class = self.factory.get_class
        self.get_library_name = self.factory.get_library_name
        self.__get_options_schema = self.__default_options_schema_getter

        def _get_options_schema(
            algo: str,
        ) -> dict[str, dict[str, str] | list[dict[str, str]]]:
            schema = get_options_schemas(algo, self.__get_options_schema)
            update_options_from_rest_docstring(algo, schema)
            return schema

        self.get_options_schema = _get_options_schema
        self.get_website = None
        self.get_description = None
        self.get_features = None
        self.user_guide_anchor = user_guide_anchor

    def get_module(self, algo_name: str) -> str:
        """Return the module path of an algorithm.

        Args:
            algo_name: The name of the algorithm.

        Returns:
            The module path of the algorithm.
        """
        return self.get_class(algo_name).__module__

    @property
    def libraries(self) -> dict[str, str]:
        """The names of the libraries related to the algorithms."""
        return {
            algo: self.get_library_name(self.get_class(algo).__name__)
            for algo in self.algos_names
        }

    @property
    def options(self) -> dict[str, str]:
        """The options of the different algorithms."""
        return {algo: self.get_options_schema(algo) for algo in self.algos_names}

    @property
    def features(self) -> dict[str, str] | None:
        """The features, if any."""
        if self.get_features is not None:
            return {algo: self.get_features(algo) for algo in self.algos_names}

    @property
    def websites(self) -> dict[str, str] | None:
        """The websites to get more details about the different algorithms, if any."""
        if self.get_website is None:
            return None
        else:
            return {algo: self.get_website(algo) for algo in self.algos_names}

    @property
    def descriptions(self) -> dict[str, str] | None:
        """The descriptions of the different algorithms, if any."""
        if self.get_description is None:
            return None
        else:
            return {algo: self.get_description(algo) for algo in self.algos_names}

    @property
    def modules(self) -> dict[str, str]:
        """The modules paths for the different algorithms."""
        return {algo: self.get_module(algo) for algo in self.algos_names}

    def __default_options_schema_getter(
        self, algo_type: str, output_json: bool = False, pretty_print: bool = False
    ) -> str | dict[str, Any]:
        """Get the options schema from the algorithm factory."""
        grammar = self.algo_factory.factory.get_options_grammar(algo_type)
        return _get_schema(grammar, output_json, pretty_print)

    def to_rst(
        self,
        template_file_name: str | None = None,
        output_file_name: str | None = None,
    ) -> None:
        """Convert options documentation into an rST file.

        Args:
            template_file_name: The name of the template file
                located in the same directory as the current file.
                If None, :attr:`AlgoOptionsDoc.TEMPLATE` will be used.
            output_file_name: The name of the rST output file
                to be stored in the directory :attr:`GEN_OPTS_PATH`.
                If None, the name will be "{algo_type}_algos.rst".
        """
        if template_file_name is None:
            template_file_name = self.template

        if output_file_name is None:
            output_file_name = f"{self.algo_type}_algos.rst"

        template = self.ENV.get_template(template_file_name)
        doc = template.render(
            algo_type=self.algo_type,
            long_algo_type=self.long_algo_type,
            algos=sorted(self.algos_names),
            modules=self.modules,
            options=self.options,
            websites=self.websites,
            descriptions=self.descriptions,
            features=self.features,
            libraries=self.libraries,
            user_guide_anchor=self.user_guide_anchor,
        )
        output_file_path = Path(GEN_OPTS_PATH).parent / output_file_name
        with open(output_file_path, "w", encoding="utf-8") as outf:
            outf.write(doc)

    @staticmethod
    def get_options_schema_from_method(
        method: Callable[[Any], Any],
    ) -> dict[str, dict[str, str]]:
        parameters = inspect.signature(method).parameters
        defaults = {
            p.name: p.default for p in parameters.values() if p.default is not p.empty
        }
        types = method.__annotations__
        types.pop("return", None)
        names = {
            p.name: f"**{p.name}" if "**" in str(p) else p.name
            for p in parameters.values()
        }
        descriptions = get_options_doc(method)
        return {
            names[name]: {
                "ptype": type_,
                "default": defaults.get(name, ""),
                "description": descriptions.get(name, ""),
            }
            for name, type_ in types.items()
        }


class DriverOptionsDoc(AlgoOptionsDoc):
    """Generator of the reST documentation of a driver from a Jinja2 template."""

    def __init__(
        self,
        algo_type: str,
        long_algo_type: str,
        algo_factory: Any | Factory,
        template: str | None = None,
        user_guide_anchor: str = "",
    ) -> None:
        super().__init__(
            algo_type,
            long_algo_type,
            algo_factory,
            template=template,
            user_guide_anchor=user_guide_anchor,
        )
        self.algos_names = algo_factory.algorithms
        self.get_description = self.__default_description_getter(algo_factory)
        self.get_website = self.__default_website_getter(algo_factory)
        self.get_class = self.__default_class_getter(algo_factory)
        self.get_options_schema = self.__default_options_schema_getter(algo_factory)
        if algo_type == "opt":
            self.get_features = get_algorithm_features

    def __default_options_schema_getter(
        self,
        algo_factory: DriverFactory,
    ) -> Callable[[str], dict[dict[str, str]]]:
        """Return the default algorithm description getter from a driver factory."""

        def get_options_schema(algo: str) -> dict[dict[str, str]]:
            """Return the options schema.

            Args:
                algo: The name of the algorithm.

            Returns:
                The options schema.
            """
            options_schema = self.get_options_schema_from_method(
                self.get_class(algo)._get_options
            )
            algo_lib = algo_factory.create(algo)
            options_grammar = algo_lib.init_options_grammar(algo)
            return {
                k: v for k, v in options_schema.items() if k in options_grammar.names
            }

        return get_options_schema

    @staticmethod
    def __default_description_getter(
        algo_factory: DriverFactory,
    ) -> Callable[[str], str]:
        """Return the default algorithm description getter from a driver factory."""

        def get_description(algo: str) -> str:
            """Return the description of an algorithm.

            Args:
                algo: The name of the algorithm.

            Returns:
                The description of the algorithm.
            """
            return algo_factory.create(algo).descriptions[algo].description

        return get_description

    @staticmethod
    def __default_website_getter(
        algo_factory: DriverFactory,
    ) -> Callable[[str], str]:
        """Return the default algorithm website getter from a driver factory."""

        def get_website(algo: str) -> str:
            """Return the website associated with an algorithm.

            Args:
                algo: The name of the algorithm.

            Returns:
                The website associated with the algorithm.
            """
            return algo_factory.create(algo).descriptions[algo].website

        return get_website

    @staticmethod
    def __default_class_getter(
        algo_factory: DriverFactory,
    ) -> Callable[[str], DriverLib]:
        """Return the default algorithm class getter from a driver factory."""

        def get_class(algo: str) -> DriverLib:
            """Return the driver library associated with an algorithm.

            Args:
                algo: The name of the algorithm.

            Returns:
                The driver library associated with the algorithm.
            """
            return algo_factory.factory.get_class(
                algo_factory.algo_names_to_libraries[algo]
            )

        return get_class


class OptPostProcessorAlgoOptionsDoc(AlgoOptionsDoc):
    """Generator of the reST documentation of a post-processor from a Jinja2 template."""

    def __init__(
        self,
        algo_type: str,
        long_algo_type: str,
        algo_factory: Any | Factory,
        template: str | None = None,
        user_guide_anchor: str = "",
    ) -> None:
        super().__init__(
            algo_type,
            long_algo_type,
            algo_factory,
            template=template,
            user_guide_anchor=user_guide_anchor,
        )

        def get_options_schema(algo):
            klass = self.get_class(algo)
            schema = self.get_options_schema_from_method(klass.execute)
            schema.update(self.get_options_schema_from_method(klass._run))
            schema.update(self.get_options_schema_from_method(klass._plot))
            return schema

        self.get_options_schema = get_options_schema


class InitOptionsDoc(AlgoOptionsDoc):
    """Generator of the reST documentation of an __init__ from a Jinja2 template."""

    def __init__(
        self,
        algo_type: str,
        long_algo_type: str,
        algo_factory: Any | Factory,
        template: str | None = None,
        user_guide_anchor: str = "",
    ) -> None:
        super().__init__(
            algo_type,
            long_algo_type,
            algo_factory,
            template=template,
            user_guide_anchor=user_guide_anchor,
        )
        self.get_options_schema = lambda algo: self.get_options_schema_from_method(
            self.get_class(algo).__init__
        )


def main(gen_opts_path: str | Path) -> None:
    global GEN_OPTS_PATH
    GEN_OPTS_PATH = gen_opts_path

    algos_options_docs = [
        InitOptionsDoc("clustering", "Clustering", ClusteringModelFactory()),
        InitOptionsDoc(
            "classification", "Classification", ClassificationModelFactory()
        ),
        InitOptionsDoc("ml_quality", "Quality measure", MLQualityMeasureFactory()),
        InitOptionsDoc("mda", "MDA", MDAFactory()),
        InitOptionsDoc("formulation", "MDO Formulation", MDOFormulationsFactory()),
        OptPostProcessorAlgoOptionsDoc("post", "Post-processing", PostFactory()),
        DriverOptionsDoc("doe", "DOE", DOEFactory(), user_guide_anchor="doe"),
        DriverOptionsDoc("opt", "Optimization", OptimizersFactory()),
        DriverOptionsDoc("linear_solver", "Linear solver", LinearSolversFactory()),
        InitOptionsDoc(
            "distribution", "Probability distribution", DistributionFactory()
        ),
        InitOptionsDoc(
            "sensitivity", "Sensitivity analysis", SensitivityAnalysisFactory()
        ),
    ]
    for algos_options_doc in algos_options_docs:
        algos_options_doc.to_rst()

    options_doc = InitOptionsDoc("regression", "Regression", RegressionModelFactory())
    options_doc.to_rst()
    options_doc.to_rst("surrogate_algos_template.tmpl", "surrogate_algos.rst")
