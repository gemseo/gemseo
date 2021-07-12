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
#                         documentation
#        :author: Francois Gallard, Matthias De Lozzo
#    OTHER AUTHORS   - MACROSCOPIC CHANGES
from __future__ import division, unicode_literals

import sys
import unittest
from os import remove
from os.path import dirname, exists

import pytest
from numpy import arange, array, eye, float64, zeros
from numpy.linalg import norm

from gemseo.api import create_scenario
from gemseo.caches.cache_factory import CacheFactory
from gemseo.caches.hdf5_cache import HDF5Cache, HDF5FileSingleton
from gemseo.core.cache import hash_data_dict, to_real
from gemseo.core.chain import MDOParallelChain
from gemseo.problems.sellar.sellar import Sellar1, SellarSystem
from gemseo.problems.sellar.sellar_design_space import SellarDesignSpace
from gemseo.utils.py23_compat import long, xrange

DIRNAME = dirname(__file__)


@pytest.mark.usefixtures("tmp_wd")
class TestCaches(unittest.TestCase):
    @staticmethod
    def get_caches(h5_node="DummyCache"):
        factory = CacheFactory()
        h5_cache_path = "dummy.h5"
        if exists(h5_cache_path):
            remove(h5_cache_path)
        h_cache = factory.create(
            "HDF5Cache", hdf_file_path=h5_cache_path, hdf_node_path=h5_node
        )
        s_cache = factory.create("SimpleCache", tolerance=0.0)
        m_cache = factory.create("MemoryFullCache")
        m_cache_loc = factory.create("MemoryFullCache", is_memory_shared=False)
        return s_cache, h_cache, m_cache, m_cache_loc

    def test_jac_and_outputs_caching(self):
        s_cache, h_cache, m_cache, m_cache_loc = self.get_caches()

        for cache in [s_cache, h_cache, m_cache, m_cache_loc]:
            assert cache.get_length() == 0
            assert not cache
            assert cache.get_last_cached_inputs() is None
            assert cache.get_last_cached_outputs() is None
            input_data = {"i": arange(3)}
            data_out = {"o": arange(4)}
            cache.cache_outputs(input_data, input_data.keys(), data_out)
            out, jac = cache.get_outputs(input_data, input_data.keys())
            assert out is not None
            assert jac is None
            self.assert_items_equal(out, data_out)

            jac = {"o": {"i": eye(4, 3)}}
            cache.cache_jacobian(input_data, input_data.keys(), jac)

            out, jac_l = cache.get_outputs(input_data, input_data.keys())
            assert jac_l is not None
            assert norm(jac_l["o"]["i"].flatten() - eye(4, 3).flatten()) == 0.0
            assert out is not None

            cache.tolerance = 1e-6
            out_t, jac_t = cache.get_outputs(input_data, input_data.keys())
            assert jac_t is not None
            assert norm(jac_t["o"]["i"].flatten() - eye(4, 3).flatten()) == 0.0
            assert out_t is not None

            cache.tolerance = 0.0

            # Cache jac again, to check it still works
            cache.cache_jacobian(input_data, input_data.keys(), jac)

        assert s_cache.get_length() == 1

        for cache in [s_cache, h_cache, m_cache, m_cache_loc]:

            assert cache.get_last_cached_inputs() is not None
            assert cache.get_last_cached_outputs() is not None

            cache.clear()
            assert cache.get_length() == 0

    def test_hdf_cache_read(self):
        outf = "dummy.h5"
        if exists(outf):
            remove(outf)
        cache = HDF5Cache(outf, "DummyCache")

        self.assertRaises(
            ValueError,
            cache.export_to_ggobi,
            "out.ggobi",
            inputs_names=["i", "j"],
            outputs_names=["o"],
        )

        n = 10
        for i in xrange(1, n + 1):
            input_data = {"i": i * arange(3), "j": array([1.0])}
            data_out = {"o": i * arange(4)}
            cache.cache_outputs(input_data, input_data.keys(), data_out)
        # Cache it again, just to check
        cache.cache_outputs(input_data, input_data.keys(), data_out)

        assert len(cache) == n
        cache_read = HDF5Cache(outf, "DummyCache")

        self.assertEqual(cache_read.get_length(), n)

        cache.export_to_ggobi(
            "out1.ggobi",
            inputs_names=["i", "j"],
            outputs_names=["o"],
        )

        input_data = {"i": zeros(3), "k": zeros(3)}
        data_out = {"o": arange(4), "t": zeros(3)}
        cache.cache_outputs(input_data, input_data.keys(), data_out, data_out.keys())

        assert len(cache) == n + 1
        exp_ggobi = "out2.ggobi"
        cache.export_to_ggobi(
            exp_ggobi, inputs_names=["i", "k"], outputs_names=["o", "t"]
        )
        assert exists(exp_ggobi)

        self.assertRaises(
            ValueError,
            cache.export_to_ggobi,
            "out3.ggobi",
            inputs_names=["izgz"],
            outputs_names=["o"],
        )

    def test_merge_caches(self):
        c1 = HDF5Cache("out11.h5", "DummyCache")
        c2 = HDF5Cache("out21.h5", "DummyCache")

        keys = ["i"]
        input_data = {"i": 10 * arange(3)}
        c1.cache_outputs(input_data, keys, {"o": arange(4)})

        input_data2 = {"i": 2 * arange(3)}
        c2.cache_outputs(input_data2, keys, {"o": 2 * arange(4)})
        c2.cache_jacobian(input_data2, keys, {"o": {"i": 3 * arange(4)}})

        c1.merge(c2)
        assert c1.get_length() == 2

        outs, jacs = c1.get_outputs(input_data2, keys)
        assert (outs["o"] == 2 * arange(4)).all()
        assert (jacs["o"]["i"] == 3 * arange(4)).all()

    def test_collision(self):
        c1 = HDF5Cache("out7.h5", "DummyCache")
        input_data = {"i": arange(3)}
        output_data = {"o": arange(3)}
        c1.cache_outputs(input_data, input_data.keys(), output_data)

        # Fake a collision of hashes
        c1.cache_outputs(input_data, input_data.keys(), output_data)
        hash_0 = next(iter(c1._hashes.keys()))
        groups = c1._hashes.pop(hash_0)
        input_data2 = {"i": 2 * arange(3)}
        hash_1 = hash_data_dict(input_data2, input_data2.keys())
        c1._hashes[hash_1] = groups
        output_data2 = {"o": 2.0 * arange(3)}
        c1.cache_outputs(input_data2, input_data2.keys(), output_data2)

        c1_outs = c1.get_outputs(input_data2, input_data2.keys())
        assert (c1_outs[0]["o"] == output_data2["o"]).all()

    def test_hash_data_dict(self):
        input_data = {"i": 10 * arange(3)}
        hash_0 = hash_data_dict(input_data)
        assert isinstance(hash_0, long)
        assert hash_0 == hash_data_dict(input_data)
        assert hash_0 == hash_data_dict({"i": 10 * arange(3), "t": None})
        assert hash_0 == hash_data_dict({"i": 10 * arange(3), "j": None})
        hash_data_dict({"i": 10 * arange(3)})
        # Discontiguous array
        hash_data_dict({"i": arange(10)[::3]})

    def test_to_real(self):
        assert to_real(1.0j * arange(3)).dtype == float64
        assert to_real(1.0 * arange(3)).dtype == float64
        assert (to_real(1.0j * arange(3) + arange(3)) == 1.0 * arange(3)).all()

    def test_write_data(self):
        out_f = "out2.h5"
        if exists(out_f):
            remove(out_f)
        file_sing = HDF5FileSingleton(out_f)
        input_data = {"i": arange(3)}
        file_sing.write_data(input_data, input_data.keys(), "group", 1, "node", None)

    #             h5file = h5py.File(join(tmp_out, "out51.h5"), "a")
    #             file_sing.write_data(input_data, input_data.keys(), "group2",
    #                                  2, "node", h5file)
    #             assert not file_sing._has_group(3, "group", "node", h5file)
    #             file_sing.write_data(input_data, input_data.keys(), "group2",
    #                                  2, "node", h5file)
    #             self.assertRaises((RuntimeError, IOError), file_sing.write_data,
    #                               input_data, input_data.keys(), "group2",
    #                               2, "node", h5file)

    def test_read_hashes(self):
        if exists("out1.h5"):
            remove("out1.h5")
        file_sing = HDF5FileSingleton("out1.h5")
        input_data = {"i": arange(3)}
        file_sing.write_data(input_data, input_data.keys(), "group", 1, "node", None)
        assert file_sing.read_hashes({}, "unknown") == 0
        hashes_dict = {}
        file_sing.read_hashes(hashes_dict, "unknown")
        n_0 = len(hashes_dict)
        file_sing.read_hashes(hashes_dict, "unknown")
        assert n_0 == len(hashes_dict)
        hashes_dict = {}
        hashes_dict[977299934065931519957167197057685376965897664534] = []
        file_sing.read_hashes(hashes_dict, "node")

    def test_read_group(self):
        cache = HDF5Cache("out3.h5", "node")
        cache.cache_outputs(
            {"x": arange(3), "y": arange(3)}, ["x", "y"], {"f": array([1])}, ["f"]
        )
        input_data = {"x": arange(3), "y": arange(2)}
        cache._read_group([1], input_data)

    def test_get_all_data(self):
        s_cache, h_cache, m_cache, m_cache_loc = self.get_caches()
        inputs = {"x": arange(3), "y": arange(3)}
        outputs = {"f": array([1])}
        for cache in [s_cache, h_cache, m_cache, m_cache_loc]:

            cache.cache_outputs(inputs, ["x", "y"], outputs, ["f"])
            all_data = cache.get_all_data()
            assert len(all_data) == 1
            self.assert_items_equal(all_data[1]["inputs"], inputs)
            self.assert_items_equal(all_data[1]["outputs"], outputs)
            assert all_data[1]["jacobian"] is None

            assert cache.get_outputs(inputs)[0]["f"][0] == outputs["f"][0]

        jac = {"f": {"x": eye(1, 3), "y": eye(1, 3)}}
        h_cache.cache_jacobian({"x": arange(3), "y": arange(3)}, ["x", "y"], jac)

    def test_all_data(self):
        _, h_cache, m_cache, m_cache_loc = self.get_caches()
        for cache in [h_cache, m_cache, m_cache_loc]:
            cache.cache_outputs(
                {"x": arange(3), "y": arange(3)}, ["x", "y"], {"f": array([1])}, ["f"]
            )
            jac = {"f": {"x": eye(1, 3), "y": eye(1, 3)}}
            cache.cache_jacobian({"x": arange(3), "y": arange(3)}, ["x", "y"], jac)
            cache.cache_outputs(
                {"x": arange(3) * 2, "y": arange(3)},
                ["x", "y"],
                {"f": array([2])},
                ["f"],
            )
            all_data = cache.get_all_data(True)
            data = next(all_data)
            self.assert_items_equal(data["inputs"], {"x": arange(3), "y": arange(3)})
            self.assert_items_equal(data["outputs"], {"f": array([1])})
            data = next(all_data)
            self.assert_items_equal(
                data["inputs"], {"x": arange(3) * 2, "y": arange(3)}
            )
            self.assert_items_equal(data["outputs"], {"f": array([2])})

    def assert_items_equal(self, data1, data2):
        assert len(data1) == len(data2)
        for key, val in data1.items():
            assert key in data2
            assert (val == data2[key]).all()

    def test_addition(self):
        cache1 = CacheFactory().create("MemoryFullCache")
        cache1.cache_outputs(
            {"x": arange(3), "y": arange(3)}, ["x", "y"], {"f": array([1])}, ["f"]
        )
        cache2 = CacheFactory().create("MemoryFullCache")
        cache2.cache_outputs(
            {"x": arange(3), "y": arange(3)}, ["x", "y"], {"f": array([1])}, ["f"]
        )
        cache3 = cache1 + cache2
        assert cache3.get_length() == 1
        cache2.cache_outputs(
            {"x": arange(3) * 2, "y": arange(3)},
            ["x", "y"],
            {"f": array([1]) * 2},
            ["f"],
        )
        cache3 = cache1 + cache2
        assert cache3.get_length() == 2
        all_data = cache3.get_all_data(True)
        data = next(all_data)
        self.assert_items_equal(data["inputs"], {"x": arange(3), "y": arange(3)})
        self.assert_items_equal(data["outputs"], {"f": array([1])})
        data = next(all_data)
        self.assert_items_equal(data["inputs"], {"x": arange(3) * 2, "y": arange(3)})
        self.assert_items_equal(data["outputs"], {"f": array([2])})

    def test_duplicate_from_scratch(self):
        _, h_cache, m_cache, _ = self.get_caches()
        h_cache._duplicate_from_scratch()
        m_cache._duplicate_from_scratch()

    def test_multithreading(self):
        caches1 = self.get_caches()
        caches2 = self.get_caches()

        caches1 = caches1[2:]
        caches2 = caches2[2:]

        for c_1, c_2 in zip(caches1, caches2):
            s_s = SellarSystem()
            s_1 = Sellar1()
            s_1.cache = c_1
            s_s.cache = c_2
            assert len(c_1) == 0
            assert len(c_2) == 0
            par = MDOParallelChain([s_1, s_s], use_threading=True)
            ds = SellarDesignSpace("float64")
            scen = create_scenario(
                par, "DisciplinaryOpt", "obj", ds, scenario_type="DOE"
            )

            options = {"algo": "fullfact", "n_samples": 10, "n_processes": 4}
            scen.execute(options)

            nexec_1 = s_1.n_calls
            nexec_2 = s_s.n_calls

            scen = create_scenario(
                par, "DisciplinaryOpt", "obj", ds, scenario_type="DOE"
            )
            scen.execute(options)

            assert nexec_1 == s_1.n_calls
            assert nexec_2 == s_s.n_calls

    def test_copy(self):
        _, _, m_cache, _ = self.get_caches()
        input_data = {"i": arange(3)}
        data_out = {"o": arange(4)}
        m_cache.cache_outputs(input_data, input_data.keys(), data_out)
        input_data = {"i": arange(4)}
        data_out = {"o": arange(5)}
        m_cache.cache_outputs(input_data, input_data.keys(), data_out)
        copy = m_cache.copy
        assert copy.get_length() == 2
        input_data = {"i": arange(5)}
        data_out = {"o": arange(6)}
        copy.cache_outputs(input_data, input_data.keys(), data_out)
        assert copy.get_length() == 3
        assert m_cache.get_length() == 2

    def test_hdf5singleton(self):
        node = "node"
        file_path = "singleton.h5"
        if exists(file_path):
            remove(file_path)
        CacheFactory().create("HDF5Cache", hdf_file_path=file_path, hdf_node_path=node)
        singleton = HDF5FileSingleton(file_path)
        data = {"x": array([0.0])}
        singleton.write_data(data, ["x"], HDF5Cache.INPUTS_GROUP, 1, node)

        self.assertRaises(
            RuntimeError,
            singleton.write_data,
            data,
            ["x"],
            HDF5Cache.INPUTS_GROUP,
            1,
            node,
        )

    def test_cache_max_length(self):
        """Tests the maximum length getter."""
        s_cache, h_cache, m_cache, m_cache_loc = self.get_caches()
        assert s_cache.max_length == 1
        assert h_cache.max_length == sys.maxsize
        assert m_cache.max_length == sys.maxsize
        assert m_cache_loc.max_length == sys.maxsize
