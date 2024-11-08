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
from typing import TYPE_CHECKING
from typing import Any
from typing import Callable
from typing import NamedTuple

import jinja2
from pydantic_core import PydanticUndefined

from gemseo import _get_schema
from gemseo.algos.doe.factory import DOELibraryFactory
from gemseo.algos.linear_solvers.factory import LinearSolverLibraryFactory
from gemseo.algos.ode.factory import ODESolverLibraryFactory
from gemseo.algos.opt.factory import OptimizationLibraryFactory
from gemseo.disciplines.factory import DisciplineFactory
from gemseo.formulations.factory import MDOFormulationFactory
from gemseo.mda.factory import MDAFactory
from gemseo.mlearning.classification.algos.factory import ClassifierFactory
from gemseo.mlearning.clustering.algos.factory import ClustererFactory
from gemseo.mlearning.core.quality.factory import MLAlgoQualityFactory
from gemseo.mlearning.regression.algos.factory import RegressorFactory
from gemseo.post.factory import PostFactory
from gemseo.uncertainty.distributions.factory import DistributionFactory
from gemseo.uncertainty.sensitivity.factory import SensitivityAnalysisFactory
from gemseo.utils.source_parsing import get_options_doc

if TYPE_CHECKING:
    from pydantic import BaseModel
    from pydantic.fields import FieldInfo

    from gemseo.algos.base_algo_factory import BaseAlgoFactory
    from gemseo.algos.base_driver_library import BaseDriverLibrary
    from gemseo.core.base_factory import BaseFactory

GEN_OPTS_PATH = None


class AlgorithmFeatures(NamedTuple):
    """The features of an algorithm."""

    algorithm_name: str
    library_name: str
    root_package_name: str
    handle_equality_constraints: bool
    handle_inequality_constraints: bool
    handle_float_variables: bool
    handle_integer_variables: bool
    handle_multiobjective: bool
    require_gradient: bool


def get_algorithm_features(
    algorithm_name: str,
) -> AlgorithmFeatures:
    """Return the features of an optimization algorithm.

    Args:
        algorithm_name: The name of the optimization algorithm.

    Returns:
        The features of the optimization algorithm.

    Raises:
        ValueError: When the optimization algorithm does not exist.
    """
    from gemseo.algos.opt.factory import OptimizationLibraryFactory

    factory = OptimizationLibraryFactory()
    if not factory.is_available(algorithm_name):
        msg = f"{algorithm_name} is not the name of an optimization algorithm."
        raise ValueError(msg)

    driver = factory.create(algorithm_name)
    description = driver.ALGORITHM_INFOS[algorithm_name]
    return AlgorithmFeatures(
        algorithm_name=description.algorithm_name,
        library_name=description.library_name,
        root_package_name=factory.get_library_name(driver.__class__.__name__),
        handle_equality_constraints=description.handle_equality_constraints,
        handle_inequality_constraints=description.handle_inequality_constraints,
        handle_float_variables=True,
        handle_integer_variables=description.handle_integer_variables,
        handle_multiobjective=description.handle_multiobjective,
        require_gradient=description.require_gradient,
    )


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
    for _opt_name, opt_schema in list(all_options.items()):
        if descr in opt_schema:
            opt_schema[descr] = opt_schema[descr].replace("\n", " ")
        elif "anyOf" in opt_schema:
            if descr in opt_schema["anyOf"][0]:
                opt_schema[descr] = opt_schema["anyOf"][0][descr].replace("\n", " ")
        else:
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
    for option in options.values():
        try:
            tmp = re.split(r":type ([*\w]+): (.*?)", option["description"])
        except KeyError:
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
        algo_factory: Any | BaseFactory,
        template: str | None = None,
        user_guide_anchor: str = "",
        use_pydantic_model: bool = True,
        pydantic_model_module_path: str = "",
    ) -> None:
        """Args:
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
        use_pydantic_model: Whether the classes use Pydantic settings.
        pydantic_model_module_path: The path to the module including the Pydantic model.
        """
        if template is None:
            self.template = self.TEMPLATE
        else:
            self.template = template

        self.algo_type = algo_type
        self.long_algo_type = long_algo_type
        self.factory = algo_factory
        if hasattr(self.factory, "class_names"):
            self.algos_names = self.factory.class_names
        else:
            self.algos_names = self.factory.libraries
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
        self.use_pydantic_model = use_pydantic_model
        self.pydantic_model_module_path = pydantic_model_module_path

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
    def pydantic_model_imports(self) -> dict[str, str]:
        """The pydantic model imports for the different algorithms."""
        if not self.pydantic_model_module_path:
            return dict.fromkeys(self.algos_names, "")

        pattern = f"from {self.pydantic_model_module_path} import " + "{}"
        print(self.get_pydantic_model_class_name(self.algos_names[0]))
        return {
            algo: pattern.format(self.get_pydantic_model_class_name(algo))
            for algo in self.algos_names
        }

    @property
    def features(self) -> dict[str, str] | None:
        """The features, if any."""
        if self.get_features is not None:
            return {algo: self.get_features(algo) for algo in self.algos_names}
        return None

    @property
    def websites(self) -> dict[str, str] | None:
        """The websites to get more details about the different algorithms, if any."""
        if self.get_website is None:
            return None
        return {algo: self.get_website(algo) for algo in self.algos_names}

    @property
    def descriptions(self) -> dict[str, str] | None:
        """The descriptions of the different algorithms, if any."""
        if self.get_description is None:
            return None
        return {algo: self.get_description(algo) for algo in self.algos_names}

    @property
    def modules(self) -> dict[str, str]:
        """The modules paths for the different algorithms."""
        return {algo: self.get_module(algo) for algo in self.algos_names}

    def __default_options_schema_getter(
        self, algo_type: str, output_json: bool = False, pretty_print: bool = False
    ) -> str | dict[str, Any]:
        """Get the options schema from the algorithm factory."""
        grammar = self.algo_factory.get_options_grammar(algo_type)
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
            use_pydantic_model=self.use_pydantic_model,
            pydantic_model_imports=self.pydantic_model_imports,
        )
        output_file_path = Path(GEN_OPTS_PATH).parent / output_file_name
        with Path(output_file_path).open("w", encoding="utf-8") as outf:
            outf.write(doc)

    @classmethod
    def get_options_schema_from_pydantic_model(
        cls,
        model: BaseModel,
    ) -> dict[str, dict[str, str]]:
        return {
            name: {
                "ptype": field.annotation,
                "description": field.description,
                **(
                    {}
                    if field.is_required()
                    else {"default": cls.__get_pydantic_default(field)}
                ),
            }
            for name, field in model.model_fields.items()
        }

    @staticmethod
    def __get_pydantic_default(field: FieldInfo) -> Any:
        default = field.default
        if default is PydanticUndefined:
            default_factory = field.default_factory
            if default_factory is dict:
                return {}
            if default_factory is list:
                return []

        return default

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
                "description": descriptions.get(name, ""),
                **({"default": defaults[name]} if name in defaults else {}),
            }
            for name, type_ in types.items()
        }


class DriverOptionsDoc(AlgoOptionsDoc):
    """Generator of the reST documentation of a driver from a Jinja2 template."""

    def __init__(
        self,
        algo_type: str,
        long_algo_type: str,
        algo_factory: Any | BaseFactory,
        template: str | None = None,
        user_guide_anchor: str = "",
        use_pydantic_model: bool = True,
        pydantic_model_module_path: str = "",
    ) -> None:
        super().__init__(
            algo_type,
            long_algo_type,
            algo_factory,
            template=template,
            user_guide_anchor=user_guide_anchor,
            use_pydantic_model=use_pydantic_model,
            pydantic_model_module_path=pydantic_model_module_path,
        )
        self.algos_names = algo_factory.algorithms
        self.get_description = self.__default_description_getter(algo_factory)
        self.get_website = self.__default_website_getter(algo_factory)
        self.get_class = self.__default_class_getter(algo_factory)

        def f(algo):
            klass = self.get_class(algo)
            return klass.ALGORITHM_INFOS[algo].Settings.__name__

        self.get_pydantic_model_class_name = f

        def get_options_schema(algo):
            klass = self.get_class(algo)
            schema = self.get_options_schema_from_pydantic_model(
                klass.ALGORITHM_INFOS[algo].Settings
            )
            return schema

        self.get_options_schema = get_options_schema

        if algo_type == "opt":
            self.get_features = get_algorithm_features

    def __default_options_schema_getter(
        self,
        algo_factory: BaseAlgoFactory,
    ) -> Callable[[str], dict[dict[str, str]]]:
        """Return the default algorithm description getter from a driver factory."""

        def get_options_schema(algo: str) -> dict[str, str]:
            """Return the options schema.

            Args:
                algo: The name of the algorithm.

            Returns:
                The options schema.
            """
            return self.get_class(algo).ALGORITHM_INFOS[algo].Settings.model_fields

        return get_options_schema

    @staticmethod
    def __default_description_getter(
        algo_factory: BaseAlgoFactory,
    ) -> Callable[[str], str]:
        """Return the default algorithm description getter from a driver factory."""

        def get_description(algo: str) -> str:
            """Return the description of an algorithm.

            Args:
                algo: The name of the algorithm.

            Returns:
                The description of the algorithm.
            """
            return algo_factory.create(algo).ALGORITHM_INFOS[algo].description

        return get_description

    @staticmethod
    def __default_website_getter(
        algo_factory: BaseAlgoFactory,
    ) -> Callable[[str], str]:
        """Return the default algorithm website getter from a driver factory."""

        def get_website(algo: str) -> str:
            """Return the website associated with an algorithm.

            Args:
                algo: The name of the algorithm.

            Returns:
                The website associated with the algorithm.
            """
            return algo_factory.create(algo).ALGORITHM_INFOS[algo].website

        return get_website

    @staticmethod
    def __default_class_getter(
        algo_factory: BaseAlgoFactory,
    ) -> Callable[[str], BaseDriverLibrary]:
        """Return the default algorithm class getter from a driver factory."""

        def get_class(algo: str) -> BaseDriverLibrary:
            """Return the driver library associated with an algorithm.

            Args:
                algo: The name of the algorithm.

            Returns:
                The driver library associated with the algorithm.
            """
            return algo_factory.get_class(algo_factory.algo_names_to_libraries[algo])

        return get_class


class BasePostAlgoOptionsDoc(AlgoOptionsDoc):
    """Generator of the reST documentation of a post-processor from a Jinja2
    template."""

    def __init__(
        self,
        algo_type: str,
        long_algo_type: str,
        algo_factory: Any | BaseFactory,
        template: str | None = None,
        user_guide_anchor: str = "",
        use_pydantic_model: bool = True,
        pydantic_model_module_path: str = "",
    ) -> None:
        super().__init__(
            algo_type,
            long_algo_type,
            algo_factory,
            template=template,
            user_guide_anchor=user_guide_anchor,
            use_pydantic_model=use_pydantic_model,
            pydantic_model_module_path=pydantic_model_module_path,
        )

        def f(algo):
            klass = self.get_class(algo)
            return klass.Settings.__name__

        self.get_pydantic_model_class_name = f

        def get_options_schema(algo):
            klass = self.get_class(algo)
            schema = self.get_options_schema_from_pydantic_model(klass.Settings)
            return schema

        self.get_options_schema = get_options_schema


class InitOptionsDoc(AlgoOptionsDoc):
    """Generator of the reST documentation of an __init__ from a Jinja2 template."""

    def __init__(
        self,
        algo_type: str,
        long_algo_type: str,
        algo_factory: Any | BaseFactory,
        template: str | None = None,
        user_guide_anchor: str = "",
        use_pydantic_model: bool = True,
        pydantic_model_module_path: str = "",
    ) -> None:
        super().__init__(
            algo_type,
            long_algo_type,
            algo_factory,
            template=template,
            user_guide_anchor=user_guide_anchor,
            use_pydantic_model=use_pydantic_model,
            pydantic_model_module_path=pydantic_model_module_path,
        )
        self.get_options_schema = lambda algo: self.get_options_schema_from_method(
            self.get_class(algo).__init__
        )


def main(gen_opts_path: str | Path) -> None:
    global GEN_OPTS_PATH
    GEN_OPTS_PATH = gen_opts_path

    algos_options_docs = [
        BasePostAlgoOptionsDoc(
            "clustering",
            "Clustering algorithms",
            ClustererFactory(),
            pydantic_model_module_path="gemseo.settings.mlearning",
        ),
        BasePostAlgoOptionsDoc(
            "classification",
            "Classification algorithms",
            ClassifierFactory(),
            pydantic_model_module_path="gemseo.settings.mlearning",
        ),
        InitOptionsDoc(
            "ml_quality",
            "Quality measures",
            MLAlgoQualityFactory(),
            use_pydantic_model=False,
        ),
        BasePostAlgoOptionsDoc(
            "mda",
            "MDA algorithms",
            MDAFactory(),
            pydantic_model_module_path="gemseo.settings.mda",
        ),
        BasePostAlgoOptionsDoc(
            "formulation",
            "MDO formulations",
            MDOFormulationFactory(),
            pydantic_model_module_path="gemseo.settings.formulations",
        ),
        BasePostAlgoOptionsDoc(
            "post",
            "Post-processing algorithms",
            PostFactory(),
            pydantic_model_module_path="gemseo.settings.post",
        ),
        DriverOptionsDoc(
            "doe",
            "DOE algorithms",
            DOELibraryFactory(),
            user_guide_anchor="doe",
            pydantic_model_module_path="gemseo.settings.doe",
        ),
        DriverOptionsDoc(
            "opt",
            "Optimization algorithms",
            OptimizationLibraryFactory(),
            pydantic_model_module_path="gemseo.settings.opt",
        ),
        DriverOptionsDoc(
            "linear_solver",
            "Linear solvers",
            LinearSolverLibraryFactory(),
            pydantic_model_module_path="gemseo.settings.linear_solvers",
        ),
        DriverOptionsDoc(
            "ode",
            "Ordinary differential equations solvers",
            ODESolverLibraryFactory(),
            pydantic_model_module_path="gemseo.settings.ode",
        ),
        InitOptionsDoc(
            "distribution",
            "Probability distributions",
            DistributionFactory(),
            use_pydantic_model=False,
        ),
        InitOptionsDoc(
            "sensitivity",
            "Sensitivity analysis algorithms",
            SensitivityAnalysisFactory(),
            use_pydantic_model=False,
        ),
        InitOptionsDoc(
            "discipline", "Disciplines", DisciplineFactory(), use_pydantic_model=False
        ),
    ]
    for algos_options_doc in algos_options_docs:
        algos_options_doc.to_rst()

    options_doc = BasePostAlgoOptionsDoc(
        "regression",
        "Regression algorithms",
        RegressorFactory(),
        pydantic_model_module_path="gemseo.settings.mlearning",
    )
    options_doc.to_rst()
    options_doc.to_rst("surrogate_algos_template.tmpl", "surrogate_algos.rst")
