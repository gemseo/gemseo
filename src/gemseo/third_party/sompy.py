# -*- coding: utf-8 -*-
#  A modified version by Francois Gallard, from :
# Vahid Moosavi 2015 02 04 10:08 pm
# sevamoo@gmail.com
# Chair For Computer Aided Architectural Design, ETH  Zurich
# Future Cities Lab
# www.vahidmoosavi.com
#         Apache License
#                            Version 2.0, January 2004
#                         http://www.apache.org/licenses/
#
#    TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION
#
#    1. Definitions.
#
#       "License" shall mean the terms and conditions for use, reproduction,
#       and distribution as defined by Sections 1 through 9 of this document.
#
#       "Licensor" shall mean the copyright owner or entity authorized by
#       the copyright owner that is granting the License.
#
#       "Legal Entity" shall mean the union of the acting entity and all
#       other entities that control, are controlled by, or are under common
#       control with that entity. For the purposes of this definition,
#       "control" means (i) the power, direct or indirect, to cause the
#       direction or management of such entity, whether by contract or
#       otherwise, or (ii) ownership of fifty percent (50%) or more of the
#       outstanding shares, or (iii) beneficial ownership of such entity.
#
#       "You" (or "Your") shall mean an individual or Legal Entity
#       exercising permissions granted by this License.
#
#       "Source" form shall mean the preferred form for making modifications,
#       including but not limited to software source code, documentation
#       source, and configuration files.
#
#       "Object" form shall mean any form resulting from mechanical
#       transformation or translation of a Source form, including but
#       not limited to compiled object code, generated documentation,
#       and conversions to other media types.
#
#       "Work" shall mean the work of authorship, whether in Source or
#       Object form, made available under the License, as indicated by a
#       copyright notice that is included in or attached to the work
#       (an example is provided in the Appendix below).
#
#       "Derivative Works" shall mean any work, whether in Source or Object
#       form, that is based on (or derived from) the Work and for which the
#       editorial revisions, annotations, elaborations, or other modifications
#       represent, as a whole, an original work of authorship. For the purposes
#       of this License, Derivative Works shall not include works that remain
#       separable from, or merely link (or bind by name) to the interfaces of,
#       the Work and Derivative Works thereof.
#
#       "Contribution" shall mean any work of authorship, including
#       the original version of the Work and any modifications or additions
#       to that Work or Derivative Works thereof, that is intentionally
#       submitted to Licensor for inclusion in the Work by the copyright owner
#       or by an individual or Legal Entity authorized to submit on behalf of
#       the copyright owner. For the purposes of this definition, "submitted"
#       means any form of electronic, verbal, or written communication sent
#       to the Licensor or its representatives, including but not limited to
#       communication on electronic mailing lists, source code control systems,
#       and issue tracking systems that are managed by, or on behalf of, the
#       Licensor for the purpose of discussing and improving the Work, but
#       excluding communication that is conspicuously marked or otherwise
#       designated in writing by the copyright owner as "Not a Contribution."
#
#       "Contributor" shall mean Licensor and any individual or Legal Entity
#       on behalf of whom a Contribution has been received by Licensor and
#       subsequently incorporated within the Work.
#
#    2. Grant of Copyright License. Subject to the terms and conditions of
#       this License, each Contributor hereby grants to You a perpetual,
#       worldwide, non-exclusive, no-charge, royalty-free, irrevocable
#       copyright license to reproduce, prepare Derivative Works of,
#       publicly display, publicly perform, sublicense, and distribute the
#       Work and such Derivative Works in Source or Object form.
#
#    3. Grant of Patent License. Subject to the terms and conditions of
#       this License, each Contributor hereby grants to You a perpetual,
#       worldwide, non-exclusive, no-charge, royalty-free, irrevocable
#       (except as stated in this section) patent license to make, have made,
#       use, offer to sell, sell, import, and otherwise transfer the Work,
#       where such license applies only to those patent claims licensable
#       by such Contributor that are necessarily infringed by their
#       Contribution(s) alone or by combination of their Contribution(s)
#       with the Work to which such Contribution(s) was submitted. If You
#       institute patent litigation against any entity (including a
#       cross-claim or counterclaim in a lawsuit) alleging that the Work
#       or a Contribution incorporated within the Work constitutes direct
#       or contributory patent infringement, then any patent licenses
#       granted to You under this License for that Work shall terminate
#       as of the date such litigation is filed.
#
#    4. Redistribution. You may reproduce and distribute copies of the
#       Work or Derivative Works thereof in any medium, with or without
#       modifications, and in Source or Object form, provided that You
#       meet the following conditions:
#
#       (a) You must give any other recipients of the Work or
#           Derivative Works a copy of this License; and
#
#       (b) You must cause any modified files to carry prominent notices
#           stating that You changed the files; and
#
#       (c) You must retain, in the Source form of any Derivative Works
#           that You distribute, all copyright, patent, trademark, and
#           attribution notices from the Source form of the Work,
#           excluding those notices that do not pertain to any part of
#           the Derivative Works; and
#
#       (d) If the Work includes a "NOTICE" text file as part of its
#           distribution, then any Derivative Works that You distribute must
#           include a readable copy of the attribution notices contained
#           within such NOTICE file, excluding those notices that do not
#           pertain to any part of the Derivative Works, in at least one
#           of the following places: within a NOTICE text file distributed
#           as part of the Derivative Works; within the Source form or
#           documentation, if provided along with the Derivative Works; or,
#           within a display generated by the Derivative Works, if and
#           wherever such third-party notices normally appear. The contents
#           of the NOTICE file are for informational purposes only and
#           do not modify the License. You may add Your own attribution
#           notices within Derivative Works that You distribute, alongside
#           or as an addendum to the NOTICE text from the Work, provided
#           that such additional attribution notices cannot be construed
#           as modifying the License.
#
#       You may add Your own copyright statement to Your modifications and
#       may provide additional or different license terms and conditions
#       for use, reproduction, or distribution of Your modifications, or
#       for any such Derivative Works as a whole, provided Your use,
#       reproduction, and distribution of the Work otherwise complies with
#       the conditions stated in this License.
#
#    5. Submission of Contributions. Unless You explicitly state otherwise,
#       any Contribution intentionally submitted for inclusion in the Work
#       by You to the Licensor shall be under the terms and conditions of
#       this License, without any additional terms or conditions.
#       Notwithstanding the above, nothing herein shall supersede or modify
#       the terms of any separate license agreement you may have executed
#       with Licensor regarding such Contributions.
#
#    6. Trademarks. This License does not grant permission to use the trade
#       names, trademarks, service marks, or product names of the Licensor,
#       except as required for reasonable and customary use in describing the
#       origin of the Work and reproducing the content of the NOTICE file.
#
#    7. Disclaimer of Warranty. Unless required by applicable law or
#       agreed to in writing, Licensor provides the Work (and each
#       Contributor provides its Contributions) on an "AS IS" BASIS,
#       WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
#       implied, including, without limitation, any warranties or conditions
#       of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
#       PARTICULAR PURPOSE. You are solely responsible for determining the
#       appropriateness of using or redistributing the Work and assume any
#       risks associated with Your exercise of permissions under this License.
#
#    8. Limitation of Liability. In no event and under no legal theory,
#       whether in tort (including negligence), contract, or otherwise,
#       unless required by applicable law (such as deliberate and grossly
#       negligent acts) or agreed to in writing, shall any Contributor be
#       liable to You for damages, including any direct, indirect, special,
#       incidental, or consequential damages of any character arising as a
#       result of this License or out of the use or inability to use the
#       Work (including but not limited to damages for loss of goodwill,
#       work stoppage, computer failure or malfunction, or any and all
#       other commercial damages or losses), even if such Contributor
#       has been advised of the possibility of such damages.
#
#    9. Accepting Warranty or Additional Liability. While redistributing
#       the Work or Derivative Works thereof, You may choose to offer,
#       and charge a fee for, acceptance of support, warranty, indemnity,
#       or other liability obligations and/or rights consistent with this
#       License. However, in accepting such obligations, You may act only
#       on Your own behalf and on Your sole responsibility, not on behalf
#       of any other Contributor, and only if You agree to indemnify,
#       defend, and hold each Contributor harmless for any liability
#       incurred by, or claims asserted against, such Contributor by reason
#       of your accepting any such warranty or additional liability.
#
#    END OF TERMS AND CONDITIONS
#
#    APPENDIX: How to apply the Apache License to your work.
#
#       To apply the Apache License to your work, attach the following
#       boilerplate notice, with the fields enclosed by brackets "{}"
#       replaced with your own identifying information. (Don't include
#       the brackets!)  The text should be enclosed in the appropriate
#       comment syntax for the file format. We also recommend that a
#       file or class name and description of purpose be included on the
#       same "printed page" as the copyright notice for easier
#       identification within third-party archives.
#
#    Copyright {yyyy} {name of copyright owner}
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
#
# This code was modified by Francois Gallard, IRT Saint Exupery
# for integration within |g|
"""
Self Organizing Maps
********************
"""
import itertools
import logging
import os
import tempfile
from timeit import default_timer as timer

import matplotlib
import numpy as np
from matplotlib import cm
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from scipy.sparse import csr_matrix
from sklearn import neighbors

try:
    from sklearn.externals.joblib import Parallel, delayed, dump, load
except ImportError:
    from joblib.parallel import Parallel, delayed

    dump = None
    load = None




try:
    from sklearn.decomposition import RandomizedPCA
except ImportError:
    from sklearn.decomposition import PCA as RandomizedPCA


LOGGER = logging.getLogger(__name__)


class SOM(object):
    """ """

    def __init__(
        self,
        name,
        Data,
        mapsize=None,
        norm_method="var",
        initmethod="pca",
        neigh="Guassian",
    ):
        """
        name and data, neigh== Bubble or Guassian
        """
        self.name = name
        self.data_raw = Data
        if norm_method == "var":
            Data = normalize(Data, method=norm_method)
            self.data = Data

        else:
            self.data = Data
        self.dim = Data.shape[1]
        self.dlen = Data.shape[0]
        self.set_topology(mapsize=mapsize)
        self.set_algorithm(initmethod=initmethod)
        self.calc_map_dist()
        self.neigh = neigh

        # Slow for large data sets
        # self.set_data_labels()

    # set SOM topology
    def set_topology(
        self, mapsize=None, mapshape="planar", lattice="rect", mask=None, compname=None
    ):
        """all_mapshapes = ['planar','toroid','cylinder']
        all_lattices = ['hexa','rect']

        :param mapsize: Default value = None)
        :param mapshape: Default value = 'planar')
        :param lattice: Default value = 'rect')
        :param mask: Default value = None)
        :param compname: Default value = None)

        """
        self.mapshape = mapshape
        self.lattice = lattice

        # to set mask
        if mask is None:
            self.mask = np.ones([1, self.dim])
        else:
            self.mask = mask

        # to set map size
        if mapsize is None:
            tmp = int(round(np.sqrt(self.dlen)))
            self.nnodes = tmp
            self.mapsize = [int(3.0 / 5 * self.nnodes), int(2.0 / 5 * self.nnodes)]
        else:
            if len(mapsize) == 2:
                if np.min(mapsize) == 1:
                    self.mapsize = [1, np.max(mapsize)]
                else:
                    self.mapsize = mapsize
            elif len(mapsize) == 1:
                s = int(mapsize[0] / 2)
                self.mapsize = [1, mapsize[0]]
                LOGGER.debug("input was considered as the numbers of nodes")
                LOGGER.debug("map size is [{0},{1}]".format(s, s))
            self.nnodes = self.mapsize[0] * self.mapsize[1]

        # to set discipline names
        if compname is None:
            try:
                cc = list()
                for i in range(0, self.dim):
                    cc.append("Variable-" + str(i + 1))
                    self.compname = np.asarray(cc)[np.newaxis, :]
            except Exception:
                pass
                LOGGER.debug("no data yet: plesae first set trainign data to the SOM")
        else:
            try:
                dim = getattr(self, "dim")
                if len(compname) == dim:
                    self.compname = np.asarray(compname)[np.newaxis, :]
                else:
                    LOGGER.debug("compname should have the same size")
            except Exception:
                pass
                LOGGER.debug("no data yet: plesae first set trainign data to the SOM")

    # Set labels of the training data
    # it should be in the format of a list of strings
    def set_data_labels(self, dlabel=None):
        """

        :param dlabel: Default value = None)

        """
        if dlabel is None:
            try:
                dlen = getattr(self, "dlen")
                cc = list()
                for i in range(0, dlen):
                    cc.append("dlabel-" + str(i))
                    self.dlabel = np.asarray(cc)[:, np.newaxis]
            except Exception:
                pass
                LOGGER.debug("no data yet: plesae first set trainign data to the SOM")
        else:
            try:
                dlen = getattr(self, "dlen")
                if dlabel.shape == (1, dlen):
                    self.dlabel = dlabel.T  # [:,np.newaxis]
                elif dlabel.shape == (dlen, 1):
                    self.dlabel = dlabel
                elif dlabel.shape == (dlen,):
                    self.dlabel = dlabel[:, np.newaxis]
                else:
                    LOGGER.debug("wrong lable format")
            except Exception:
                pass
                LOGGER.debug("no data yet: plesae first set trainign data to the SOM")

    # calculating the grid distance, which will be called during the training steps
    # currently just works for planar grids
    def calc_map_dist(self):
        """ """
        cd = getattr(self, "nnodes")
        UD2 = np.zeros((cd, cd))
        for i in range(cd):
            UD2[i, :] = grid_dist(self, i).reshape(1, cd)
        self.UD2 = UD2

    def set_algorithm(
        self,
        initmethod="pca",
        algtype="batch",
        neighborhoodmethod="gaussian",
        alfatype="inv",
        alfaini=0.5,
        alfafinal=0.005,
    ):
        """initmethod = ['random', 'pca']
        algos = ['seq','batch']
        all_neigh = ['gaussian','manhatan','bubble','cut_gaussian','epanechicov' ]
        alfa_types = ['linear','inv','power']

        :param initmethod: Default value = 'pca')
        :param algtype: Default value = 'batch')
        :param neighborhoodmethod: Default value = 'gaussian')
        :param alfatype: Default value = 'inv')
        :param alfaini: Default value = .5)
        :param alfafinal: Default value = .005)

        """
        self.initmethod = initmethod
        self.algtype = algtype
        self.alfaini = alfaini
        self.alfafinal = alfafinal
        self.neigh = neighborhoodmethod

    ###################################
    # visualize map
    def view_map(
        self,
        what="codebook",
        which_dim="all",
        pack="Yes",
        text_size=2.8,
        save="No",
        save_dir="empty",
        grid="No",
        text="Yes",
        cmap="None",
        COL_SiZe=6,
    ):
        """

        :param what: Default value = 'codebook')
        :param which_dim: Default value = 'all')
        :param pack: Default value = 'Yes')
        :param text_size: Default value = 2.8)
        :param save: Default value = 'No')
        :param save_dir: Default value = 'empty')
        :param grid: Default value = 'No')
        :param text: Default value = 'Yes')
        :param cmap: Default value = 'None')
        :param COL_SiZe: Default value = 6)

        """

        mapsize = getattr(self, "mapsize")
        if np.min(mapsize) > 1:
            if pack == "No":
                view_2d(self, text_size, which_dim=which_dim, what=what)
            else:
                #         		LOGGER.debug('hi')
                view_2d_Pack(
                    self,
                    text_size,
                    which_dim=which_dim,
                    what=what,
                    save=save,
                    save_dir=save_dir,
                    grid=grid,
                    text=text,
                    CMAP=cmap,
                    col_sz=COL_SiZe,
                )

        elif np.min(mapsize) == 1:
            view_1d(self, text_size, which_dim=which_dim, what=what)

    ##########################################################################
    # Initialize map codebook: Weight vectors of SOM
    def init_map(self):
        """ """
        if getattr(self, "initmethod") == "random":
            # It produces random values in the range of min- max of each
            # dimension based on a uniform distribution
            mn = np.tile(
                np.min(getattr(self, "data"), axis=0), (getattr(self, "nnodes"), 1)
            )
            mx = np.tile(
                np.max(getattr(self, "data"), axis=0), (getattr(self, "nnodes"), 1)
            )
            setattr(
                self,
                "codebook",
                mn
                + (mx - mn)
                * (np.random.rand(getattr(self, "nnodes"), getattr(self, "dim"))),
            )
        elif getattr(self, "initmethod") == "pca":
            # it is based on two largest eigenvalues of correlation matrix
            codebooktmp = lininit(self)
            setattr(self, "codebook", codebooktmp)
        else:
            LOGGER.debug("please select a corect initialization method")
            LOGGER.debug(
                "set a correct one in SOM. current SOM.initmethod:  ",
                getattr(self, "initmethod"),
            )
            LOGGER.debug("possible init methods:'random', 'pca'")

    # Main loop of training
    def train(self, trainlen=None, n_job=1, shared_memory="no", verbose="on"):
        """

        :param trainlen: Default value = None)
        :param n_job: Default value = 1)
        :param shared_memory: Default value = 'no')
        :param verbose: Default value = 'on')

        """
        # LOGGER.debug('data len is %d and data dimension is %d' % (dlen+str( dim)))
        # LOGGER.debug( 'map size is %d, %d' %(mapsize[0], mapsize[1]))
        # LOGGER.debug('array size in log10 scale' +str( mem))
        # LOGGER.debug('nomber of jobs in parallel: '+str( n_job))
        #######################################
        # initialization
        LOGGER.debug(
            "initialization method = %s, initializing.." % getattr(self, "initmethod")
        )
        t0 = timer()
        self.init_map()
        LOGGER.debug("initialization done in %f seconds" % round(timer() - t0))

        batchtrain(self, njob=n_job, phase="rough", verbose=verbose)
        batchtrain(self, njob=n_job, phase="finetune", verbose=verbose)
        err = np.mean(getattr(self, "bmu")[1])
        ts = round(timer() - t0, 3)
        LOGGER.debug("Total time elapsed: %f secodns" % ts)
        LOGGER.debug("final quantization error: %f" % err)
        ts = round(timer() - t0, 3)
        LOGGER.debug("Total time elapsed: %f secodns" % ts)
        LOGGER.debug("final quantization error: %f" % err)

    # to project a data set to a trained SOM and find the index of bmu
    # It is based on nearest neighborhood search module of scikitlearn, but it
    # is not that fast.
    def project_data(self, data):
        """

        :param data:

        """
        codebook = getattr(self, "codebook")
        data_raw = getattr(self, "data_raw")
        clf = neighbors.KNeighborsClassifier(n_neighbors=1)
        labels = np.arange(0, codebook.shape[0])
        clf.fit(codebook, labels)

        # the codebook values are all normalized
        # we can normalize the input data based on mean and std of original
        # data
        data = normalize_by(data_raw, data)
        # data = normalize(data, method='var')
        # plt.hist(data[:,2])
        Predicted_labels = clf.predict(data)
        return Predicted_labels

    def predict_by(self, data, Target, K=5, wt="distance"):
        """‘uniform’

        :param data: param Target:
        :param K: Default value = 5)
        :param wt: Default value = 'distance')
        :param Target:

        """
        # here it is assumed that Target is the last column in the codebook
        # and data has dim-1 columns
        codebook = getattr(self, "codebook")
        data_raw = getattr(self, "data_raw")
        dim = codebook.shape[1]
        ind = np.arange(0, dim)
        indX = ind[ind != Target]
        X = codebook[:, indX]
        Y = codebook[:, Target]
        n_neighbors = K
        clf = neighbors.KNeighborsRegressor(n_neighbors, weights=wt)
        clf.fit(X, Y)
        # the codebook values are all normalized
        # we can normalize the input data based on mean and std of original
        # data
        dimdata = data.shape[1]
        if dimdata == dim:
            data[:, Target] = 0
            data = normalize_by(data_raw, data)
            data = data[:, indX]
        elif dimdata == dim - 1:
            data = normalize_by(data_raw[:, indX], data)
            # data = normalize(data, method='var')
        Predicted_values = clf.predict(data)
        Predicted_values = denormalize_by(data_raw[:, Target], Predicted_values)
        return Predicted_values

    def predict(self, X_test, K=5, wt="distance"):
        """‘uniform’

        :param X_test: param K:  (Default value = 5)
        :param wt: Default value = 'distance')
        :param K:  (Default value = 5)

        """
        # Similar to SKlearn we assume that we have X_tr, Y_tr and X_test
        # here it is assumed that Target is the last column in the codebook
        # and data has dim-1 columns
        codebook = getattr(self, "codebook")
        data_raw = getattr(self, "data_raw")
        Target = data_raw.shape[1] - 1
        X_train = codebook[:, :Target]
        Y_train = codebook[:, Target]
        n_neighbors = K
        clf = neighbors.KNeighborsRegressor(n_neighbors, weights=wt)
        clf.fit(X_train, Y_train)
        # the codebook values are all normalized
        # we can normalize the input data based on mean and std of original
        # data
        X_test = normalize_by(data_raw[:, :Target], X_test)
        Predicted_values = clf.predict(X_test)
        Predicted_values = denormalize_by(data_raw[:, Target], Predicted_values)
        return Predicted_values

    def find_K_nodes(self, data, K=5):
        """

        :param data: param K:  (Default value = 5)
        :param K:  (Default value = 5)

        """
        from sklearn.neighbors import NearestNeighbors

        # we find the k most similar nodes to the input vector
        codebook = getattr(self, "codebook")
        neigh = NearestNeighbors(n_neighbors=K)
        neigh.fit(codebook)
        data_raw = getattr(self, "data_raw")
        # the codebook values are all normalized
        # we can normalize the input data based on mean and std of original
        # data
        data = normalize_by(data_raw, data)
        return neigh.kneighbors(data)

    def ind_to_xy(self, bm_ind):
        """

        :param bm_ind:

        """
        msize = getattr(self, "mapsize")
        rows = msize[0]
        cols = msize[1]
        # bmu should be an integer between 0 to no_nodes
        out = np.zeros((bm_ind.shape[0], 3))
        out[:, 2] = bm_ind
        out[:, 0] = rows - 1 - bm_ind / cols
        out[:, 0] = bm_ind / cols
        out[:, 1] = bm_ind % cols
        return out.astype(int)

    def cluster(self, method="Kmeans", n_clusters=8):
        """

        :param method: Default value = 'Kmeans')
        :param n_clusters: Default value = 8)

        """
        import sklearn.cluster as clust

        km = clust.KMeans(n_clusters=n_clusters)
        labels = km.fit_predict(
            denormalize_by(self.data_raw, self.codebook)
        )
        setattr(self, "cluster_labels", labels)
        return labels

    def hit_map(self, data=None):
        """

        :param data: Default value = None)

        """
        # First Step: show the hitmap of all the training data

        #     	LOGGER.debug('None')
        data_tr = getattr(self, "data_raw")
        proj = self.project_data(data_tr)
        msz = getattr(self, "mapsize")
        coord = self.ind_to_xy(proj)

        # this is not an appropriate way, but it works
        coord[:, 1] = msz[0] - coord[:, 1]
        ###############################
        fig = plt.figure(figsize=(msz[1] / 5, msz[0] / 5))
        ax = fig.add_subplot(111)
        ax.xaxis.set_ticks([i for i in range(0, msz[1])])
        ax.yaxis.set_ticks([i for i in range(0, msz[0])])
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        ax.grid(True, linestyle="-", linewidth=0.5)
        a = plt.hist2d(
            coord[:, 1],
            coord[:, 0],
            bins=(msz[1], msz[0]),
            alpha=0.0,
            norm=LogNorm(),
            cmap=cm.get_cmap("jet"),
        )
        # clbar  = plt.colorbar()
        x = np.arange(0.5, msz[1] + 0.5, 1)
        y = np.arange(0.5, msz[0] + 0.5, 1)
        X, Y = np.meshgrid(x, y)
        area = a[0].T * 12
        plt.scatter(
            X,
            Y,
            s=area,
            alpha=0.2,
            c="b",
            marker="o",
            cmap="jet",
            linewidths=3,
            edgecolor="r",
        )
        plt.scatter(
            X,
            Y,
            s=area,
            alpha=0.9,
            c="None",
            marker="o",
            cmap="jet",
            linewidths=3,
            edgecolor="r",
        )
        plt.xlim(0, msz[1])
        plt.ylim(0, msz[0])

        if data is not None:
            proj = self.project_data(data)
            msz = getattr(self, "mapsize")
            coord = self.ind_to_xy(proj)
            a = plt.hist2d(
                coord[:, 1],
                coord[:, 0],
                bins=(msz[1], msz[0]),
                alpha=0.0,
                norm=LogNorm(),
                cmap=cm.get_cmap("jet"),
            )
            # clbar  = plt.colorbar()
            x = np.arange(0.5, msz[1] + 0.5, 1)
            y = np.arange(0.5, msz[0] + 0.5, 1)
            X, Y = np.meshgrid(x, y)
            area = a[0].T * 50
            plt.scatter(
                X,
                Y,
                s=area,
                alpha=0.2,
                c="b",
                marker="o",
                cmap="jet",
                linewidths=3,
                edgecolor="r",
            )
            plt.scatter(
                X,
                Y,
                s=area,
                alpha=0.9,
                c="None",
                marker="o",
                cmap="jet",
                linewidths=3,
                edgecolor="r",
            )
            plt.xlim(0, msz[1])
            plt.ylim(0, msz[0])

        plt.show()

    def hit_map_cluster_number(self, data=None):
        """

        :param data: Default value = None)

        """
        if hasattr(self, "cluster_labels"):
            codebook = getattr(self, "cluster_labels")
        #     		LOGGER.debug('yesyy')
        else:
            LOGGER.debug("clustering based on default parameters...")
            codebook = self.cluster()
        msz = getattr(self, "mapsize")
        fig = plt.figure(figsize=(msz[1] / 2.5, msz[0] / 2.5))
        ax = fig.add_subplot(111)
        #     	ax.xaxis.set_ticklabels([])
        #     	ax.yaxis.set_ticklabels([])
        #     	ax.grid(True,linestyle='-', linewidth=.5)

        if data is None:
            data_tr = getattr(self, "data_raw")
            proj = self.project_data(data_tr)
            cents = self.ind_to_xy(np.arange(0, msz[0] * msz[1]))
            for i, txt in enumerate(codebook):
                ax.annotate(txt, (cents[i, 1], cents[i, 0]), size=10, va="center")

        if data is not None:
            proj = self.project_data(data)
            cents = self.ind_to_xy(proj)
            #     		cents[:,1] = cents[:,1]+.2
            #     		LOGGER.debug(cents.shape)
            label = codebook[proj]
            for i, txt in enumerate(label):
                ax.annotate(txt, (cents[i, 1], cents[i, 0]), size=10, va="center")

        plt.imshow(codebook.reshape(msz[0], msz[1])[::], alpha=0.5)
        #     	plt.pcolor(codebook.reshape(msz[0],msz[1])[::-1],alpha=.5,cmap='jet')
        plt.show()
        return cents

    def predict_Probability(self, data, Target, K=5):
        """

        :param data: param Target:
        :param K: Default value = 5)
        :param Target:

        """
        # here it is assumed that Target is the last column in the codebook
        # #and data has dim-1 columns
        codebook = getattr(self, "codebook")
        data_raw = getattr(self, "data_raw")
        dim = codebook.shape[1]
        ind = np.arange(0, dim)
        indX = ind[ind != Target]
        X = codebook[:, indX]
        Y = codebook[:, Target]
        n_neighbors = K
        clf = neighbors.KNeighborsRegressor(n_neighbors, weights="distance")
        clf.fit(X, Y)
        # the codebook values are all normalized
        # we can normalize the input data based on mean and std of original
        # data
        dimdata = data.shape[1]
        if dimdata == dim:
            data[:, Target] = 0
            data = normalize_by(data_raw, data)
            data = data[:, indX]
        elif dimdata == dim - 1:
            data = normalize_by(data_raw[:, indX], data)
            # data = normalize(data, method='var')
        weights, ind = clf.kneighbors(data, n_neighbors=K)
        weights = 1.0 / weights
        sum_ = np.sum(weights, axis=1)
        weights = weights / sum_[:, np.newaxis]
        labels = np.sign(codebook[ind, Target])
        labels[labels >= 0] = 1

        # for positives
        pos_prob = labels.copy()
        pos_prob[pos_prob < 0] = 0
        pos_prob = pos_prob * weights
        pos_prob = np.sum(pos_prob, axis=1)[:, np.newaxis]

        # for negatives
        neg_prob = labels.copy()
        neg_prob[neg_prob > 0] = 0
        neg_prob = neg_prob * weights * -1
        neg_prob = np.sum(neg_prob, axis=1)[:, np.newaxis]

        # Predicted_values = clf.predict(data)
        # Predicted_values = denormalize_by(data_raw[:,Target], Predicted_values)
        return np.concatenate((pos_prob, neg_prob), axis=1)

    def node_Activation(self, data, wt="distance", Target=None):
        """‘uniform’

        :param data: param wt:  (Default value = 'distance')
        :param Target: Default value = None)
        :param wt:  (Default value = 'distance')

        """

        if Target is None:
            codebook = getattr(self, "codebook")
            data_raw = getattr(self, "data_raw")
            clf = neighbors.KNeighborsClassifier(n_neighbors=getattr(self, "nnodes"))
            labels = np.arange(0, codebook.shape[0])
            clf.fit(codebook, labels)
            # the codebook values are all normalized
            # we can normalize the input data based on mean and std of original
            # data
            data = normalize_by(data_raw, data)
            weights, ind = clf.kneighbors(data)

            # Softmax function
            weights = 1.0 / weights
            S_ = np.sum(np.exp(weights), axis=1)[:, np.newaxis]
            weights = np.exp(weights) / S_

        return weights, ind

        #

    def para_bmu_find(self, x, y, njb=1):
        """

        :param x: param y:
        :param njb: Default value = 1)
        :param y:

        """
        dlen = x.shape[0]
        y_2 = np.einsum("ij,ij->i", y, y)
        # here it finds BMUs for chunk of data in parallel
        b = Parallel(n_jobs=njb, pre_dispatch="3*n_jobs")(
            delayed(chunk_based_bmu_find)(
                self, x[i * dlen // njb : min((i + 1) * dlen // njb, dlen)], y, y_2
            )
            for i in range(njb)
        )

        # LOGGER.debug('bmu finding: %f seconds ' %round(timer() - t_temp+str( 3)))
        bmu = np.asarray(list(itertools.chain(*b))).T
        # LOGGER.debug('bmu to array: %f seconds' %round(timer() - t1+str( 3)))
        del b
        return bmu

    # First finds the Voronoi set of each node. It needs to calculate a smaller matrix. Super fast comparing to classic batch training algorithm
    # it is based on the implemented algorithm in som toolbox for Matlab by
    # Helsinky university
    def update_codebook_voronoi(self, training_data, bmu, H, radius):
        """

        :param training_data: param bmu:
        :param H: param radius:
        :param bmu:
        :param radius:

        """
        # bmu has shape of 2,dlen, where first row has bmuinds
        # we construct ud2 from precomputed UD2: ud2 = UD2[bmu[0,:]]
        nnodes = getattr(self, "nnodes")
        dlen = getattr(self, "dlen")
        inds = bmu[0].astype(int)
        #         LOGGER.debug('bmu'+str( bmu[0]))
        #         fig = plt.hist(bmu[0],bins=100)
        #         plt.show()
        row = inds
        col = np.arange(dlen)
        val = np.tile(1, dlen)
        P = csr_matrix((val, (row, col)), shape=(nnodes, dlen))
        S = P.dot(training_data)
        # assert( S.shape == (nnodes, dim))
        # assert( H.shape == (nnodes, nnodes))

        # H has nnodes*nnodes and S has nnodes*dim  ---> Nominator has nnodes*dim
        # LOGGER.debug(Nom)
        Nom = H.T.dot(S)
        # assert( Nom.shape == (nnodes, dim))
        nV = P.sum(axis=1).reshape(1, nnodes)
        #         LOGGER.debug('nV'+str( nV))
        #         LOGGER.debug('H')
        #         LOGGER.debug( H)
        # assert(nV.shape == (1, nnodes))
        Denom = nV.dot(H.T).reshape(nnodes, 1)
        #         LOGGER.debug('Denom')
        #         LOGGER.debug( Denom)
        # assert( Denom.shape == (nnodes, 1))
        New_Codebook = np.divide(Nom, Denom)
        #         LOGGER.debug('codebook')
        #         LOGGER.debug(New_Codebook.sum(axis=1))
        Nom = None
        Denom = None
        # assert (New_Codebook.shape == (nnodes,dim))
        # setattr(som, 'codebook', New_Codebook)
        return np.around(New_Codebook, decimals=6)


# we will call this function in parallel for different number of jobs


def chunk_based_bmu_find(self, x, y, y_2):
    """

    :param x: param y:
    :param y_2:
    :param y:

    """
    dlen = x.shape[0]
    nnodes = y.shape[0]
    bmu = np.empty((dlen, 2))
    # it seems that smal batches for large dlen is really faster:
    # that is because of ddata in loops and n_jobs. for large data it slows
    # down due to memory needs in parallel
    blen = min(50, dlen)
    i0 = 0
    d = None
    while i0 + 1 <= dlen:
        Low = i0
        High = min(dlen, i0 + blen)
        i0 += blen
        ddata = x[Low : High + 1]
        d = np.dot(y, ddata.T)
        d *= -2
        d += y_2.reshape(nnodes, 1)
        bmu[Low : High + 1, 0] = np.argmin(d, axis=0)
        bmu[Low : High + 1, 1] = np.min(d, axis=0)
        del ddata
        d = None
    return bmu


# Batch training which is called for rought training as well as finetuning


def batchtrain(self, njob=1, phase=None, shared_memory="no", verbose="on"):
    """

    :param njob: Default value = 1)
    :param phase: Default value = None)
    :param shared_memory: Default value = 'no')
    :param verbose: Default value = 'on')

    """
    nnodes = getattr(self, "nnodes")
    dlen = getattr(self, "dlen")
    mapsize = getattr(self, "mapsize")

    #############################################
    # seting the parameters
    initmethod = getattr(self, "initmethod")
    mn = np.min(mapsize)
    if mn == 1:
        mpd = float(nnodes * 10) / float(dlen)
    else:
        mpd = float(nnodes) / float(dlen)

    ms = max(mapsize[0], mapsize[1])
    if mn == 1:
        ms /= 5.0
    # Based on somtoolbox, Matlab
    # case 'train',    sTrain.trainlen = ceil(50*mpd);
    # case 'rough',    sTrain.trainlen = ceil(10*mpd);
    # case 'finetune', sTrain.trainlen = ceil(40*mpd);
    if phase == "rough":
        # training length
        trainlen = int(np.ceil(30 * mpd))
        # radius for updating
        if initmethod == "random":
            radiusin = max(1, np.ceil(ms / 3.0))
            radiusfin = max(1, radiusin / 6.0)
        #         	radiusin = max(1, np.ceil(ms/1.))
        #         	radiusfin = max(1, radiusin/2.)
        elif initmethod == "pca":
            radiusin = max(1, np.ceil(ms / 8.0))
            radiusfin = max(1, radiusin / 4.0)
    elif phase == "finetune":
        # train lening length
        trainlen = int(np.ceil(60 * mpd))
        # radius for updating
        if initmethod == "random":
            trainlen = int(np.ceil(50 * mpd))
            radiusin = max(1, ms / 12.0)  # from radius fin in rough training
            radiusfin = max(1, radiusin / 25.0)

        # radiusin = max(1, ms/2.) #from radius fin in rough training
        #             radiusfin = max(1, radiusin/2.)
        elif initmethod == "pca":
            radiusin = max(1, np.ceil(ms / 8.0) / 4)
            radiusfin = 1  # max(1, ms/128)

    radius = np.linspace(radiusin, radiusfin, trainlen)
    ##################################################

    UD2 = getattr(self, "UD2")
    New_Codebook_V = getattr(self, "codebook")

    # LOGGER.debug('data is in shared memory?'+str( shared_memory))
    if shared_memory == "yes":
        data = getattr(self, "data")
        Data_folder = tempfile.mkdtemp()
        data_name = os.path.join(Data_folder, "data")
        dump(data, data_name)
        data = load(data_name, mmap_mode="r")
    else:
        data = getattr(self, "data")
    # x_2 is part of euclidean distance (x-y)^2 = x^2 +y^2 - 2xy that we use for each data row in bmu finding.
    # Since it is a fixed value we can skip it during bmu finding for each
    # data point, but later we need it calculate quantification error
    x_2 = np.einsum("ij,ij->i", data, data)
    if verbose == "on":
        LOGGER.debug("%s training..." % phase)
        LOGGER.debug(
            "radius_ini: %f , radius_final: %f, trainlen: %d"
            % (radiusin, radiusfin, trainlen)
        )
    neigh_func = getattr(self, "neigh")
    for i in range(trainlen):
        if neigh_func == "Guassian":
            # in case of Guassian neighborhood
            H = np.exp(-1.0 * UD2 / (2.0 * radius[i] ** 2)).reshape(nnodes, nnodes)
        if neigh_func == "Bubble":
            # in case of Bubble function
            #         	LOGGER.debug(radius[i]+str( UD2.shape))
            #         	LOGGER.debug(UD2)
            H = (
                l(radius[i], np.sqrt(UD2.flatten())).reshape(nnodes, nnodes)
                + 0.000000000001
            )
        #         	LOGGER.debug(H)
        t1 = timer()
        bmu = None
        bmu = self.para_bmu_find(data, New_Codebook_V, njb=njob)
        New_Codebook_V = self.update_codebook_voronoi(data, bmu, H, radius)
        # print 'updating nodes: ', round (timer()- t2, 3)
        if verbose == "on":
            LOGGER.debug(
                "epoch: %d ---> elapsed time:  %f, quantization error: %f "
                % (i + 1, round(timer() - t1, 3), np.mean(np.sqrt(bmu[1] + x_2)))
            )
    setattr(self, "codebook", New_Codebook_V)
    bmu[1] = np.sqrt(bmu[1] + x_2)
    setattr(self, "bmu", bmu)


def grid_dist(self, bmu_ind):
    """som and bmu_ind
    depending on the lattice "hexa" or "rect" we have different grid distance
    functions.
    bmu_ind is a number between 0 and number of nodes-1. depending on the map size
    bmu_coord will be calculated and then distance matrix in the map will be returned

    :param bmu_ind:

    """
    try:
        lattice = getattr(self, "lattice")
    except Exception:
        lattice = "hexa"
        LOGGER.debug("lattice not found! Lattice as hexa was set")

    if lattice == "rect":
        return rect_dist(self, bmu_ind)
    elif lattice == "hexa":
        try:
            msize = getattr(self, "mapsize")
            rows = msize[0]
            cols = msize[1]
        except Exception:
            rows = 0.0
            cols = 0.0
            pass

        # needs to be implemented
        LOGGER.debug("to be implemented" + str(rows) + str(cols))
        return np.zeros((rows, cols))


def rect_dist(self, bmu):
    """

    :param bmu:

    """
    # the way we consider the list of nodes in a planar grid is that node0 is on top left corner,
    # nodemapsz[1]-1 is top right corner and then it goes to the second row.
    # no. of rows is map_size[0] and no. of cols is map_size[1]
    try:
        msize = getattr(self, "mapsize")
        rows = msize[0]
        cols = msize[1]
    except Exception:
        pass

    # bmu should be an integer between 0 to no_nodes
    if 0 <= bmu <= (rows * cols):
        c_bmu = int(bmu % cols)
        r_bmu = int(bmu / cols)
    else:
        LOGGER.debug("wrong bmu")

    # calculating the grid distance
    if np.logical_and(rows > 0, cols > 0):
        r, c = np.arange(0, rows, 1)[:, np.newaxis], np.arange(0, cols, 1)
        dist2 = (r - r_bmu) ** 2 + (c - c_bmu) ** 2
        return dist2.ravel()
    else:
        LOGGER.debug("please consider the above mentioned errors")
        return np.zeros((rows, cols)).ravel()


def view_2d(self, text_size, which_dim="all", what="codebook"):
    """

    :param text_size: param which_dim:  (Default value = 'all')
    :param what: Default value = 'codebook')
    :param which_dim:  (Default value = 'all')

    """
    msz0, msz1 = getattr(self, "mapsize")
    if what == "codebook":
        if hasattr(self, "codebook"):
            codebook = getattr(self, "codebook")
            data_raw = getattr(self, "data_raw")
            codebook = denormalize_by(data_raw, codebook)
        else:
            LOGGER.debug("first initialize codebook")
        if which_dim == "all":
            dim = getattr(self, "dim")
            indtoshow = np.arange(0, dim).T
            ratio = float(dim) / float(dim)
            ratio = np.max((0.35, ratio))
            sH, sV = 16, 16 * ratio * 1
            plt.figure(figsize=(sH, sV))
        elif isinstance(which_dim, int):
            dim = 1
            indtoshow = np.zeros(1)
            indtoshow[0] = int(which_dim)
            sH, sV = 6, 6
            plt.figure(figsize=(sH, sV))
        elif isinstance(which_dim, list):
            max_dim = codebook.shape[1]
            dim = len(which_dim)
            ratio = float(dim) / float(max_dim)
            # print max_dim, dim, ratio
            ratio = np.max((0.35, ratio))
            indtoshow = np.asarray(which_dim).T
            sH, sV = 16, 16 * ratio * 1
            plt.figure(figsize=(sH, sV))

        no_row_in_plot = dim / 6 + 1  # 6 is arbitrarily selected
        if no_row_in_plot <= 1:
            no_col_in_plot = dim
        else:
            no_col_in_plot = 6

        axisNum = 0
        compname = getattr(self, "compname")
        norm = matplotlib.colors.normalize(
            vmin=np.mean(codebook.flatten()) - 1 * np.std(codebook.flatten()),
            vmax=np.mean(codebook.flatten()) + 1 * np.std(codebook.flatten()),
            clip=True,
        )
        while axisNum < dim:
            axisNum += 1
            ax = plt.subplot(no_row_in_plot, no_col_in_plot, axisNum)
            ind = int(indtoshow[axisNum - 1])
            mp = codebook[:, ind].reshape(msz0, msz1)
            pl = plt.pcolor(mp[::-1], norm=norm)
            #             pl = plt.imshow(mp[::-1])
            plt.title(compname[0][ind])
            font = {"size": text_size * sH / no_col_in_plot}
            plt.rc("font", **font)
            plt.axis("off")
            plt.axis([0, msz0, 0, msz1])
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            plt.colorbar(pl)
        plt.show()


def view_2d_Pack(
    self,
    text_size,
    which_dim="all",
    what="codebook",
    save="No",
    grid="Yes",
    save_dir="empty",
    text="Yes",
    CMAP="None",
    col_sz=None,
):
    """

    :param text_size: param which_dim:  (Default value = 'all')
    :param what: Default value = 'codebook')
    :param save: Default value = 'No')
    :param grid: Default value = 'Yes')
    :param save_dir: Default value = 'empty')
    :param text: Default value = 'Yes')
    :param CMAP: Default value = 'None')
    :param col_sz: Default value = None)
    :param which_dim:  (Default value = 'all')

    """
    msz0, msz1 = getattr(self, "mapsize")
    if CMAP == "None":
        CMAP = cm.get_cmap("jet")
    if what == "codebook":
        if hasattr(self, "codebook"):
            codebook = getattr(self, "codebook")
            data_raw = getattr(self, "data_raw")
            codebook = denormalize_by(data_raw, codebook)
        else:
            LOGGER.debug("first initialize codebook")
        if which_dim == "all":
            dim = getattr(self, "dim")
            indtoshow = np.arange(0, dim).T
            ratio = float(dim) / float(dim)
            ratio = np.max((0.35, ratio))
        #             sH, sV = 16, 16 * ratio * 1
        #             plt.figure(figsize=(sH,sV))
        elif isinstance(which_dim, int):
            dim = 1
            indtoshow = np.zeros(1)
            indtoshow[0] = int(which_dim)
        #             sH, sV = 6, 6
        #             plt.figure(figsize=(sH,sV))
        elif isinstance(which_dim, list):
            max_dim = codebook.shape[1]
            dim = len(which_dim)
            ratio = float(dim) / float(max_dim)
            # print max_dim, dim, ratio
            ratio = np.max((0.35, ratio))
            indtoshow = np.asarray(which_dim).T
        #             sH, sV = 16, 16 * ratio * 1
        #             plt.figure(figsize=(sH,sV))

        #         plt.figure(figsize=(7,7))
        no_row_in_plot = dim / col_sz + 1  # 6 is arbitrarily selected
        if no_row_in_plot <= 1:
            no_col_in_plot = dim
        else:
            no_col_in_plot = col_sz

        axisNum = 0
        compname = getattr(self, "compname")
        h = 0.1
        w = 0.1
        fig = plt.figure(
            figsize=(no_col_in_plot * 2.5 * (1 + w), no_row_in_plot * 2.5 * (1 + h))
        )
        #         LOGGER.debug(no_row_in_plot+str( no_col_in_plot))
        #         norm = matplotlib.colors.Normalize(vmin=np.median(codebook.flatten()) - 1.5 * np.std(
        #             codebook.flatten()), vmax=np.median(codebook.flatten()) + 1.5 * np.std(codebook.flatten()), clip=False)
        #
        #         DD = pd.Series(data=codebook.flatten()).describe(
        #             percentiles=[.03, .05, .1, .25, .3, .4, .5, .6, .7, .8, .9, .95, .97])
        # #         norm = matplotlib.colors.Normalize(
        #             vmin=DD.ix['3%'], vmax=DD.ix['97%'], clip=False)

        while axisNum < dim:
            axisNum += 1

            ax = fig.add_subplot(no_row_in_plot, no_col_in_plot, axisNum)
            ind = int(indtoshow[axisNum - 1])
            mp = codebook[:, ind].reshape(msz0, msz1)

            if grid == "Yes":
                pl = plt.pcolor(mp[::-1])
            elif grid == "No":
                plt.imshow(mp[::-1], cmap=CMAP)
                #             	plt.pcolor(mp[::-1])
                plt.axis("off")

            if text == "Yes":
                plt.title(compname[0][ind])
                font = {"size": text_size}
                plt.rc("font", **font)
            plt.axis([0, msz0, 0, msz1])
            ax.set_yticklabels([])
            ax.set_xticklabels([])
            ax.xaxis.set_ticks([i for i in range(0, msz1)])
            ax.yaxis.set_ticks([i for i in range(0, msz0)])
            ax.xaxis.set_ticklabels([])
            ax.yaxis.set_ticklabels([])
            ax.grid(True, linestyle="-", linewidth=0.5, color="k")
        #             plt.grid()
        #             plt.colorbar(pl)
        #         plt.tight_layout()
        plt.subplots_adjust(hspace=h, wspace=w)
    if what == "cluster":
        if hasattr(self, "cluster_labels"):
            codebook = getattr(self, "cluster_labels")
        else:
            LOGGER.debug("clustering based on default parameters...")
            codebook = self.cluster()
        h = 0.2
        w = 0.001
        fig = plt.figure(figsize=(msz0 / 2, msz1 / 2))

        ax = fig.add_subplot(1, 1, 1)
        mp = codebook[:].reshape(msz0, msz1)
        if grid == "Yes":
            pl = plt.pcolor(mp[::-1])
        elif grid == "No":
            plt.imshow(mp[::-1])
            #             plt.pcolor(mp[::-1])
            plt.axis("off")

        if text == "Yes":
            plt.title("clusters")
            font = {"size": text_size}
            plt.rc("font", **font)
        plt.axis([0, msz0, 0, msz1])
        ax.set_yticklabels([])
        ax.set_xticklabels([])
        ax.xaxis.set_ticks([i for i in range(0, msz1)])
        ax.yaxis.set_ticks([i for i in range(0, msz0)])
        ax.xaxis.set_ticklabels([])
        ax.yaxis.set_ticklabels([])
        ax.grid(True, linestyle="-", linewidth=0.5, color="k")
        plt.subplots_adjust(hspace=h, wspace=w)

    if save == "Yes":

        if save_dir != "empty":
            #         		LOGGER.debug(save_dir)
            fig.savefig(save_dir, bbox_inches="tight", transparent=False, dpi=400)
        else:
            #         		LOGGER.debug(save_dir)
            add = "/Users/itadmin/Desktop/SOM.png"
            fig.savefig(add, bbox_inches="tight", transparent=False, dpi=400)

        plt.close(fig)


def view_1d(self, text_size, which_dim="all", what="codebook"):
    """

    :param text_size: param which_dim:  (Default value = 'all')
    :param what: Default value = 'codebook')
    :param which_dim:  (Default value = 'all')

    """
    #     msz0, msz1 = getattr(self, 'mapsize')
    if what == "codebook":
        if hasattr(self, "codebook"):
            codebook = getattr(self, "codebook")
            data_raw = getattr(self, "data_raw")
            codebook = denormalize_by(data_raw, codebook)
        else:
            LOGGER.debug("first initialize codebook")
        if which_dim == "all":
            dim = getattr(self, "dim")
            indtoshow = np.arange(0, dim).T
            ratio = float(dim) / float(dim)
            ratio = np.max((0.35, ratio))
            sH, sV = 16, 16 * ratio * 1
            plt.figure(figsize=(sH, sV))
        elif isinstance(which_dim, int):
            dim = 1
            indtoshow = np.zeros(1)
            indtoshow[0] = int(which_dim)
            sH, sV = 6, 6
            plt.figure(figsize=(sH, sV))
        elif isinstance(which_dim, list):
            max_dim = codebook.shape[1]
            dim = len(which_dim)
            ratio = float(dim) / float(max_dim)
            # print max_dim, dim, ratio
            ratio = np.max((0.35, ratio))
            indtoshow = np.asarray(which_dim).T
            sH, sV = 16, 16 * ratio * 1
            plt.figure(figsize=(sH, sV))

        no_row_in_plot = dim / 6 + 1  # 6 is arbitrarily selected
        if no_row_in_plot <= 1:
            no_col_in_plot = dim
        else:
            no_col_in_plot = 6

        axisNum = 0
        compname = getattr(self, "compname")
        while axisNum < dim:
            axisNum += 1
            plt.subplot(no_row_in_plot, no_col_in_plot, axisNum)
            ind = int(indtoshow[axisNum - 1])
            mp = codebook[:, ind]
            plt.plot(mp, "-k", linewidth=0.8)
            # pl = plt.pcolor(mp[::-1])
            plt.title(compname[0][ind])
            font = {"size": text_size * sH / no_col_in_plot}
            plt.rc("font", **font)
            # plt.axis('off')
            # plt.axis([0, msz0, 0, msz1])
            # ax.set_yticklabels([])
            # ax.set_xticklabels([])
            # plt.colorbar(pl)
        plt.show()


def lininit(self):
    """ """
    # X = UsigmaWT
    # XTX = Wsigma^2WT
    # T = XW = Usigma #Transformed by W EigenVector, can be calculated by
    # multiplication PC matrix by eigenval too
    # Furthe, we can get lower ranks by using just few of the eigen vevtors
    # T(2) = U(2)sigma(2) = XW(2) ---> 2 is the number of selected eigenvectors
    # This is how we initialize the map, just by using the first two first eigen vals and eigenvectors
    # Further, we create a linear combination of them in the new map by giving values from -1 to 1 in each
    # Direction of SOM map
    # it shoud be noted that here, X is the covariance matrix of original data

    msize = getattr(self, "mapsize")
    #     rows = msize[0]
    cols = msize[1]
    nnodes = getattr(self, "nnodes")

    if np.min(msize) > 1:
        coord = np.zeros((nnodes, 2))
        for i in range(0, nnodes):
            coord[i, 0] = int(i / cols)  # x
            coord[i, 1] = int(i % cols)  # y
        mx = np.max(coord, axis=0)
        mn = np.min(coord, axis=0)
        coord = (coord - mn) / (mx - mn)
        coord = (coord - 0.5) * 2
        data = getattr(self, "data")
        me = np.mean(data, 0)
        data -= me
        codebook = np.tile(me, (nnodes, 1))
        pca = RandomizedPCA(n_components=2)  # Randomized PCA is scalable
        # pca = PCA(n_components=2)
        pca.fit(data)
        eigvec = pca.components_
        eigval = pca.explained_variance_
        norms = np.sqrt(np.einsum("ij,ij->i", eigvec, eigvec))
        eigvec = ((eigvec.T / norms) * eigval).T
        eigvec.shape

        for j in range(nnodes):
            for i in range(eigvec.shape[0]):
                codebook[j, :] = codebook[j, :] + coord[j, i] * eigvec[i, :]
        return np.around(codebook, decimals=6)
    elif np.min(msize) == 1:
        coord = np.zeros((nnodes, 1))
        for i in range(0, nnodes):
            # coord[i,0] = int(i/cols) #x
            coord[i, 0] = int(i % cols)  # y
        mx = np.max(coord, axis=0)
        mn = np.min(coord, axis=0)
        # LOGGER.debug(coord)

        coord = (coord - mn) / (mx - mn)
        coord = (coord - 0.5) * 2
        # LOGGER.debug(coord)
        data = getattr(self, "data")
        me = np.mean(data, 0)
        data -= me
        codebook = np.tile(me, (nnodes, 1))
        pca = RandomizedPCA(n_components=1)  # Randomized PCA is scalable
        # pca = PCA(n_components=2)
        pca.fit(data)
        eigvec = pca.components_
        eigval = pca.explained_variance_
        norms = np.sqrt(np.einsum("ij,ij->i", eigvec, eigvec))
        eigvec = ((eigvec.T / norms) * eigval).T
        eigvec.shape

        for j in range(nnodes):
            for i in range(eigvec.shape[0]):
                codebook[j, :] = codebook[j, :] + coord[j, i] * eigvec[i, :]
        return np.around(codebook, decimals=6)


def normalize(data, method="var"):
    """

    :param data: param method:  (Default value = 'var')
    :param method:  (Default value = 'var')

    """
    # methods  = ['var','range','log','logistic','histD','histC']
    # status = ['done', 'undone']
    me = np.mean(data, axis=0)
    st = np.std(data, axis=0)
    if method == "var":
        me = np.mean(data, axis=0)
        st = np.std(data, axis=0)
        n_data = (data - me) / st
        return n_data


def normalize_by(data_raw, data, method="var"):
    """

    :param data_raw: param data:
    :param method: Default value = 'var')
    :param data:

    """
    # methods  = ['var','range','log','logistic','histD','histC']
    # status = ['done', 'undone']
    # to have the mean and std of the original data, by which SOM is trained
    me = np.mean(data_raw, axis=0)
    st = np.std(data_raw, axis=0)
    if method == "var":
        n_data = (data - me) / st
        return n_data


def denormalize_by(data_by, n_vect, n_method="var"):
    """

    :param data_by: param n_vect:
    :param n_method: Default value = 'var')
    :param n_vect:

    """
    # based on the normalization
    if n_method == "var":
        me = np.mean(data_by, axis=0)
        st = np.std(data_by, axis=0)
        vect = n_vect * st + me
        return vect
    else:
        LOGGER.debug("data is not normalized before")
        return n_vect


def l(a, b):
    """

    :param a: param b:
    :param b:

    """
    c = np.zeros(b.shape)
    c[a - b >= 0] = 1
    return c


# Function to show hits
# som_labels = sm.project_data(Tr_Data)
# S = pd.DataFrame(data=som_labels,columns= ['label'])
# a = S['label'].value_counts()
# a = a.sort_index()
# a = pd.DataFrame(data=a.values, index=a.index,columns=['label'])
# d = pd.DataFrame(data= range(msz0*msz1),columns=['node_ID'])
# c  = d.join(a,how='outer')
# c.fillna(value=0,inplace=True)
# hits = c.values[:,1]
# hits = hits
# nodeID = np.arange(msz0*msz1)
# c_bmu = nodeID%msz1
# r_bmu = msz0 - nodeID/msz1
# fig, ax = plt.subplots()
# plt.axis([0, msz0, 0, msz1])
# ax.scatter(r_bmu, c_bmu, s=hits/2)
