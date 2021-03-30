# -*- coding: utf-8 -*-
# Copyright 2021 IRT Saint Exupéry, https://www.irt-saintexupery.com
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
#                           documentation
#        :author: Matthias De Lozzo
#        :author: Syver Doving Agdestein
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
"""
Machine learning API
--------------------

The machine learning API provides methods for creating new and loading
existing machine learning models. It also provides methods for listing
available models and options.
"""
from __future__ import absolute_import, division, unicode_literals

from future import standard_library

from gemseo.api import _get_schema

standard_library.install_aliases()

# pylint: disable=import-outside-toplevel


def get_mlearning_models():
    """Get available machine learning algorithms.

    See also
    --------
    import_mlearning_model
    create_mlearning_model
    get_mlearning_options
    import_mlearning_model
    """
    from gemseo.mlearning.core.factory import MLAlgoFactory

    factory = MLAlgoFactory()
    return factory.models


def get_regression_models():
    """Get available regression models.

    See also
    --------
    create_regression_model
    get_regression_options
    import_regression_model
    """
    from gemseo.mlearning.regression.factory import RegressionModelFactory

    factory = RegressionModelFactory()
    return factory.models


def get_classification_models():
    """Get available classification models.

    See also
    --------
    create_classification_model
    get_classification_options
    import_classification_model
    """
    from gemseo.mlearning.classification.factory import ClassificationModelFactory

    factory = ClassificationModelFactory()
    return factory.models


def get_clustering_models():
    """Get available clustering models.

    See also
    --------
    create_clustering_model
    get_clustering_options
    import_clustering_model
    """
    from gemseo.mlearning.cluster.factory import ClusteringModelFactory

    factory = ClusteringModelFactory()
    return factory.models


def create_mlearning_model(name, data, transformer=None, **parameters):
    """Create machine learning algorithm from a learning data set.

    :param str name: name of the machine learning algorithm.
    :param Dataset data: learning data set.
    :param dict(str) transformer: transformation strategy for data groups.
        If None, do not transform data. Default: None.
    :param parameters: machine learning algorithm parameters.

    See also
    --------
    get_mlearning_models
    get_mlearning_options
    import_mlearning_model
    """
    from gemseo.mlearning.core.factory import MLAlgoFactory

    factory = MLAlgoFactory()
    return factory.create(name, data=data, transformer=transformer, **parameters)


def create_regression_model(name, data, transformer=None, **parameters):
    """Create a regression model from a learning data set.

    :param str name: name of the regression model.
    :param Dataset data: learning data set.
    :param dict(str) transformer: transformation strategy for data groups.
        If None, do not transform data. Default: None.
    :param parameters: regression model parameters.

    See also
    --------
    get_regression_models
    get_regression_options
    import_regression_model
    """
    from gemseo.mlearning.regression.factory import RegressionModelFactory

    factory = RegressionModelFactory()
    return factory.create(name, data=data, transformer=transformer, **parameters)


def create_classification_model(name, data, transformer=None, **parameters):
    """Create a classification model from a learning data set.

    :param str name: name of the classification model.
    :param Dataset data: learning data set.
    :param dict(str) transformer: transformation strategy for data groups.
        If None, do not transform data. Default: None.
    :param parameters: classification model parameters.

    See also
    --------
    get_classification_models
    get_classification_options
    import_classification_model
    """
    from gemseo.mlearning.classification.factory import ClassificationModelFactory

    factory = ClassificationModelFactory()
    return factory.create(name, data=data, transformer=transformer, **parameters)


def create_clustering_model(name, data, transformer=None, **parameters):
    """Create a clustering model from a learning data set.

    :param str name: name of the clustering model.
    :param Dataset data: learning data set.
    :param dict(str) transformer: transformation strategy for data groups.
        If None, do not transform data. Default: None.
    :param parameters: clustering model parameters.

    See also
    --------
    get_clustering_models
    get_clustering_options
    import_clustering_model
    """
    from gemseo.mlearning.cluster.factory import ClusteringModelFactory

    factory = ClusteringModelFactory()
    return factory.create(name, data=data, transformer=transformer, **parameters)


def import_mlearning_model(directory):
    """Import a machine learning algorithm from a directory.

    :param str directory: directory name.

    See also
    --------
    create_mlearning_model
    get_mlearning_models
    get_mlearning_options
    """
    from gemseo.mlearning.core.factory import MLAlgoFactory

    factory = MLAlgoFactory()
    return factory.load(directory)


def import_regression_model(directory):
    """Import a regression model from a directory.

    :param str directory: directory name.

    See also
    --------
    create_regression_model
    get_regression_models
    get_regression_options
    """
    from gemseo.mlearning.regression.factory import RegressionModelFactory

    factory = RegressionModelFactory()
    return factory.load(directory)


def import_classification_model(directory):
    """Import a classification model from a directory.

    :param str directory: directory name.

    See also
    --------
    create_classification_model
    get_classification_models
    get_classification_options
    """
    from gemseo.mlearning.classification.factory import ClassificationModelFactory

    factory = ClassificationModelFactory()
    return factory.load(directory)


def import_clustering_model(directory):
    """Import a clustering model from a directory.

    :param str directory: directory name.

    See also
    --------
    create_clustering_model
    get_clustering_models
    get_clustering_options
    """
    from gemseo.mlearning.cluster.factory import ClusteringModelFactory

    factory = ClusteringModelFactory()
    return factory.load(directory)


def get_mlearning_options(model_name, output_json=False, pretty_print=True):
    """
    Lists the available options for a machine learning algorithm.

    :param str model_name: Name of the machine learning algorithm.
    :param bool output_json: Apply json format for the schema.
    :param bool pretty_print: Print the schema in a pretty table.
    :returns: Option schema (string) of the machine learning algorithm.

    See also
    --------
    create_mlearning_model
    get_mlearning_models
    import_mlearning_model
    """
    from gemseo.mlearning.core.factory import MLAlgoFactory

    factory = MLAlgoFactory().factory
    grammar = factory.get_options_grammar(model_name)
    return _get_schema(grammar, output_json, pretty_print)


def get_regression_options(model_name, output_json=False, pretty_print=True):
    """
    Lists the available options for a regression model.

    :param str model_name: Name of the regression model.
    :param bool output_json: Apply json format for the schema.
    :param bool pretty_print: Print the schema in a pretty table.
    :returns: Option schema (string) of the regression model.

    See also
    --------
    create_regression_model
    get_regression_models
    import_regression_model
    """
    from gemseo.mlearning.regression.factory import RegressionModelFactory

    factory = RegressionModelFactory().factory
    grammar = factory.get_options_grammar(model_name)
    return _get_schema(grammar, output_json, pretty_print)


def get_classification_options(model_name, output_json=False, pretty_print=True):
    """
    Lists the available options for a classification model.

    :param str model_name: Name of the classification model.
    :param bool output_json: Apply json format for the schema.
    :param bool pretty_print: Print the schema in a pretty table.
    :returns: Option schema (string) of the classification model.

    See also
    --------
    create_classification_model
    get_classification_models
    import_classification_model
    """
    from gemseo.mlearning.classification.factory import ClassificationModelFactory

    factory = ClassificationModelFactory().factory
    grammar = factory.get_options_grammar(model_name)
    return _get_schema(grammar, output_json, pretty_print)


def get_clustering_options(model_name, output_json=False, pretty_print=True):
    """
    Lists the available options for clustering model.

    :param str model_name: Name of the clustering model.
    :param bool output_json: Apply json format for the schema.
    :param bool pretty_print: Print the schema in a pretty table.
    :returns: Option schema (string) of the clustering model.

    See also
    --------
    create_clustering_model
    get_clustering_models
    import_clustering_model
    """
    from gemseo.mlearning.cluster.factory import ClusteringModelFactory

    factory = ClusteringModelFactory().factory
    grammar = factory.get_options_grammar(model_name)
    return _get_schema(grammar, output_json, pretty_print)
