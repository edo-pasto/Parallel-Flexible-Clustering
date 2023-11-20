# Copyright (c) 2017-2018 Symantec Corporation. All Rights Reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions
# are met:
#
# 1. Redistributions of source code must retain the above copyright
# notice, this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright
# notice, this list of conditions and the following disclaimer in the
# documentation and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS
# FOR A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE
# COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
# INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING,
# BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT
# LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN
# ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

import heapq
import hdbscan
import numpy as np
import scipy.sparse
from hdbscan import hdbscan_
from . import hnsw
from .unionfind import UnionFind
import time
import sys

MISSING = sys.maxsize
MISSING_WEIGHT = sys.float_info.max


def hnsw_hdbscan(
    data,
    d,
    m=5,
    ef=50,
    m0=None,
    level_mult=None,
    heuristic=True,
    balanced_add=True,
    **kwargs
):
    """Simple implementation for when you don't need incremental updates."""

    n = len(data)
    distance_matrix = scipy.sparse.lil_matrix((n, n))

    def decorated_d(i, j):
        res = d(data[i], data[j])
        distance_matrix[i, j] = distance_matrix[j, i] = res
        return res

    the_hnsw = hnsw.HNSW(decorated_d, m, ef, m0, level_mult, heuristic)
    add = the_hnsw.balanced_add if balanced_add else the_hnsw.add
    for i in range(len(data)):
        add(i)

    return hdbscan.hdbscan(distance_matrix, metric="precomputed", **kwargs)


class FISHDBC:
    """Class that represents the Flexible Incremental Scalable Hierarchical Density-Based Clustering.

    Attributes
    ----------
    d : func
        the dissimilarity function
    min_samples : int, optional
        controls the minimum number of samples in a neighborhood for a point to be considered a core point, default 5
    m : int, optional
        the number of each element's neighbros at the level > 0, default 5
    ef : int, optional
        number of closest neighbors of the inserted element in the layer
    m0 : int, optional
        the max number of each element's neighbors at the level 0, default None
    level_mult : bool, optional
        specify the level multiplier to normalize the probability to assign an element to a layer, default False
    heuristic : bool, optional
        used to enable the select_heuristic funtion instead of the select_naive function, default False
    balanced_add : bool, optional
        used to enable the balance_add function instead of the classical add function, default False
    vectorized: bool, optional
        used to vectorize the computation of the distance function, default False

    Methods
    -------
    add(elem)
        Prints the animals name and what sound it makes
    update(elems, mst_update_rate=1000000)
        Start the add procedure and the MST computation
    update_mst()
        Compute and update the MST
    cluster(
        mst=None,
        min_cluster_size=None,
        cluster_selection_method="eom",
        allow_single_cluster=False,
        match_reference_implementation=False,
        parallel=False
    )
        Performs the HDBSCAN clustering
    """

    def __init__(
        self,
        d,
        min_samples=5,
        m=5,
        ef=32,
        m0=None,
        level_mult=None,
        heuristic=True,
        balanced_add=True,
        vectorized=False,
    ):
        """Setup the algorithm. The only mandatory parameter is d, the
        dissimilarity function. min_samples is passed to hdbscan, and
        the other parameters are all passed to HNSW.

        The decorated_d internal function is used to compute the distance between element 
        but also to save distance values in cache

        Parameters
        ----------
        d : func
            the dissimilarity function
        min_samples : int, optional
            controls the minimum number of samples in a neighborhood for a point to be considered a core point, default 5
        m : int, optional
            the number of each element's neighbros at the level > 0, default 5
        ef : int, optional
            number of closest neighbors of the inserted element in the layer
        m0 : int, optional
            the max number of each element's neighbors at the level 0, default None
        level_mult : bool, optional
          specify the level multiplier to normalize the probability to assign an element to a layer, default False
        heuristic : bool, optional
            used to enable the select_heuristic funtion instead of the select_naive function, default False
        balanced_add : bool, optional
            used to enable the balance_add function instead of the classical add function, default False
        vectorized: bool, optional
            used to vectorize the computation of the distance function, default False
        """
        self.distance = d
        self.min_samples = min_samples

        self.data = data = []  # the data we're clustering

        self._mst_edges = []  # minimum spanning tree.
        # format: a list of (rd, i, j, dist) edges where nodes are
        # data[i] and data[j], dist is the dissimilarity between them, and rd
        # is the reachability distance.

        # (i, j) -> dist: the new candidates for the spanning tree
        # reachability distance will be computed afterwards
        self._new_edges = {}

        # for each data[i], _neighbor_heaps[i] contains a heap of
        # (mdist, j) where the data[j] are the min_sample closest distances
        # to i and mdist = -d(data[i], data[j]). Since heapq doesn't
        # currently support max-heaps, we use a min-heap with the
        # negative values of distances.
        self._neighbor_heaps = []

        # caches the distances computed to the last data item inserted
        self._distance_cache = distance_cache = {}

        self._tot_time = 0
        self._tot_MST_time = 0
        self.cache_hits = self.cache_misses = 0
        # decorated_d will cache the computed distances in distance_cache.
        if not vectorized:  # d is defined to work on scalars

            def decorated_d(i, j):
                if j in distance_cache:
                    self.cache_hits += 1
                    return distance_cache[j]
                self.cache_misses += 1
                distance_cache[j] = dist = d(data[i], data[j])
                return dist

            # used for testing the quality of search results
            def calc_dist(i, j):
                 return np.linalg.norm(i - data[j])

        else:  # d is defined to work on a scalar and a list

            def decorated_d(i, js):
                res = [None] * len(js)
                unknown_j, unknown_pos = [], []
                for pos, j in enumerate(js):
                    if j in distance_cache:
                        res[pos] = distance_cache[j]
                    else:
                        unknown_j.append(j)
                        unknown_pos.append(pos)
                if len(unknown_j) > 0:
                    for pos, j, dist in zip(
                        unknown_pos, unknown_j, d(data[i], unknown_j)
                    ):
                        distance_cache[j] = res[pos] = dist
                misses = len(unknown_j)
                self.cache_misses += misses
                self.cache_hits += len(js) - misses
                return res

        # We create the HNSW
        self.the_hnsw = hnsw.HNSW(
            decorated_d, calc_dist, m, ef, m0, level_mult, heuristic, vectorized
        )
        self._hnsw_add = (
            self.the_hnsw.balanced_add if balanced_add else self.the_hnsw.add
        )

    def __len__(self):
        return len(self.data)

    def add(self, elem):
        """Function to add elem to the data structure.

        Parameters
        ----------
        elem: int
            the element to add both to the hnsw and the data we are clustering
        """

        data = self.data
        distance_cache = self._distance_cache
        min_samples = self.min_samples
        nh = self._neighbor_heaps
        new_edges = self._new_edges

        minus_infty = -np.infty

        assert distance_cache == {}

        idx = len(data)
        data.append(elem)
        # let's start with min_samples values of infinity rather than
        # having to deal with heaps of less than min_samples values
        nh.append([(minus_infty, minus_infty)] * min_samples)

        start = time.time()
        self._hnsw_add(idx)
        end = time.time()
        self._tot_time = self._tot_time + (end - start)

        start = time.time()
        for j, dist in distance_cache.items():
            mdist = -dist
            heapq.heappushpop(nh[idx], (mdist, j))
            new_edges[j, idx] = dist

            # also update j's reachability distances
            nh_j = nh[j]
            old_mrd = heapq.heappushpop(nh_j, (mdist, idx))[0]
            new_mrd = nh_j[0][0]
            if old_mrd != new_mrd:
                # i is a new close neighbor for j and j's reachability
                # distance changed
                for md, k in nh_j:
                    if k == idx or k == minus_infty:
                        continue
                    if nh[k][0][0] > old_mrd:
                        # reachability distance between j and k decreased
                        key = (j, k) if j < k else (k, j)
                        # print(key, -min(md, new_mrd))
                        new_edges[key] = -min(md, new_mrd)
        end = time.time()
        self._tot_MST_time = self._tot_MST_time + (end - start)
        # print(distance_cache, idx, "\n" )
        distance_cache.clear()
        # print("hnsw graphs", self.the_hnsw._graphs)

    def update(self, elems, mst_update_rate=100000):
        """Function to add elements from elems and update the MST.
        To avoid clogging memory, the MST is updated every
        mst_update_rate elements are added.

        Parameters
        ----------
        elems:
             the list of elements to add to the hnsw and to the data we are clustering
        mst_update_rate: int
             update  the mst every mst_update_rate
        """

        for i, elem in enumerate(elems):
            self.add(elem)
            if i % mst_update_rate == 0:
                start = time.time()
                self.update_mst()
                end = time.time()
                self._tot_MST_time = self._tot_MST_time + (end - start)
        start = time.time()
        self.update_mst()
        end = time.time()
        self._tot_MST_time = self._tot_MST_time + (end - start)

    def update_mst(self):
        """Function to update the minimum spanning tree.
        It computes the MST using the candidate edges 
        and performing the kruskal algorithm
        """
        new_edges = self._new_edges

        if len(new_edges) == 0:
            return

        candidate_edges = self._mst_edges
        nh = self._neighbor_heaps

        candidate_edges.extend(
            (max(dist, -nh[i][0][0], -nh[j][0][0]), i, j, dist)
            for (i, j), dist in new_edges.items()
        )

        heapq.heapify(candidate_edges)
        # Kruskal's algorithm
        self._mst_edges = mst_edges = []
        n = len(self.data)
        needed_edges = n - 1
        uf = UnionFind(n)
        while needed_edges:
            _, i, j, _ = edge = heapq.heappop(candidate_edges)
            if uf.union(i, j):
                mst_edges.append(edge)
                needed_edges -= 1

        new_edges.clear()

    def cluster(
        self,
        mst=None,
        min_cluster_size=None,
        cluster_selection_method="eom",
        allow_single_cluster=False,
        match_reference_implementation=False,
        parallel=False,
    ):
        """
        Parameters
        ----------
        mst: np array
            the mst to be used as input for clustering, default None
        min_cluster_size: int
             controls the minimum number of samples in a neighborhood for a point to be considered a core point, default None
        cluster_selection_method: str
            used to extract a flat clustering from the hierarchical clustering, default eom
        allow_single_cluster: bool
           use it if you are getting lots of small clusters, but believe there should be some larger scale structure, default False
        match_reference_implementation: bool
            default False
        parallel: bool
            use it to specify if you are executing your fishdbc algorithm in parallel or in single process, default False

        Returns
        -------
        lps (labels, probs, stabilities): tuple of 3 arrays
            labels indicate which cluster a data point belongs to; 
            probs provide probability estimates for each cluster assignment;
            stabilities indicate how robust the cluster assignments are for each data point;
        condensed_tree:
            a numpy array representing the cluster resulting dendogram
        slt: unknown
        mst: numpy array
            the mst used for clustering
        """

        if min_cluster_size is None:
            min_cluster_size = self.min_samples

        if not parallel:
            self.update_mst()
            mst = np.array(self._mst_edges).astype(np.double)
        if parallel:
            mst = np.array(mst).astype(np.double)

        mst = np.concatenate((mst[:, 1:3], mst[:, 0].reshape(-1, 1)), axis=1)
        slt = hdbscan_.label(mst)
        condensed_tree = hdbscan_.condense_tree(slt, min_cluster_size)
        stability_dict = hdbscan_.compute_stability(condensed_tree)
        lps = hdbscan_.get_clusters(
            condensed_tree,
            stability_dict,
            cluster_selection_method,
            allow_single_cluster,
            match_reference_implementation,
        )
        return lps + (condensed_tree, slt, mst)


