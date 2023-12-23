from __future__ import division
from __future__ import print_function

from heapq import heapify, heappop, heappush, heapreplace, nlargest, nsmallest
# from operator import itemgetter
import numpy as np
import sys
import time
import heapq
from .unionfind import UnionFind
from multiprocessing import shared_memory, current_process
try:
    from math import log2
except ImportError:  # Python 2.x or <= 3.2
    from math import log

    def log2(x):
        return log(x, 2)


inf = float("inf")
MISSING = sys.maxsize
MISSING_WEIGHT = sys.float_info.max


class PARALLEL_FISHDBC:
    """Parallel Hierarchical Navigable Small World (HNSW) data structure.
    Based on the work by Yury Malkov and Dmitry Yashunin, available at
    http://arxiv.org/pdf/1603.09320v2.pdf
    HNSWs allow performing approximate nearest neighbor search with
    arbitrary data and non-metric dissimilarity functions, in a parallel way.

    Attributes
    ----------
    distance : func
        the dissimilarity function
    data: list
        the input data
    members: list[list]
        structure for describing how each level of the HNSW is composed
    levels: list[tuple]
        structure for describing at what level each element is originally assigned
    positions: list[dict]
        structure for describing at which position in each level an element can be found
    shm_adj: shared memory
        shared memory associated to the shared array of nodes of the HNSW structure
    shm_weights: shared memory
        shared memory associated to the shared array of weights of the HNSW structure
    shm_hnsw_data: shared memory
        shared memory to take track of the added element to the HNSW structure
    shm_enter_point: shared memory
        shared memory associated to the enter point to start the search for the hnsw add procedure
    shm_count: shared memory
        shared memory associated to the count for counting how much call to the distance function we make
    lock: Lock
        the lock used to synchronize the access and modification of the shared enter point
    m : int, optional
        the number of each element's neighbors at the level > 0, default 5
    ef : int, optional
        number of candidates closest neighbors of the inserted element in the layer, default 32
    m0 : int, optional
        the max number of each element's neighbors at the level 0, default None

    Methods
    -------
    decorated_d(distance_cache, i, j)
        computes the distance value bewteen two points and save it into the cache
    add_and_compute_local_mst(points)
        starts the add procedure of a range of points for each process
    hnsw_add(elem, ef=None)
        adds an element to the HNSW data structure
    calc_position(to_find, level_to_search)
        finds the position of a specific element inside a specific level in the HNSW structure
    calc_level(elem)
        finds the belonging level of an element
    search(graphs, q, k=None, ef=None, test=False)
        searches a query element in the HNSW graph
    _search_graph_ef1(count_dist, level_to_search, q, entry, dist, arr_adj, shm_adj, distance_cache)
        searches the candidates to be linked to the new inserted element when ef = 1 
    _search_graph(count_dist, level_to_search, q, ep, arr_adj, shm_adj, distance_cache, m, ef,)
        searches the candidates to be linked to the new inserted element 
    _search_graph_ef1_test(q, entry, dist, g)
        as the _search_graph_ef1, but used when accuracy test are performed
    _search_graph_test(q, ep, g, ef)
        as the _search_graph, but used when accuracy test are performed
    _select_heuristic(level_to_search, position, elem, to_insert, m, arr_adj, arr_weights, shm_adj, shm_weights, heap=False)
        links the new element to the existing items inside the HNSW, thanks to an heuristic
    local_mst(distances, points)
        computes the MST, locally to each process 
    global_mst(candidate_edges, n)
        computes the global MST with all the previous local MSTs
    """

    def __init__(
        self,
        distance,
        data,
        members,
        levels,
        positions,
        shm_adj,
        shm_weights,
        shm_hnsw_data,
        shm_enter_point,
        shm_count,
        lock,
        m=5,
        ef=32,
        m0=None,
    ):
        
        """
        Parameters
        ----------
        distance : func
            the dissimilarity function
        data: list
            the input data
        members: list[list]
            structure for describing how each level of the HNSW is composed
        levels: list[tuple]
            structure for describing at what level each element is originally assigned
        positions: list[dict]
            structure for describing at which position in each level an element can be found
        shm_adj: shared memory
            shared memory associated to the shared array of nodes of the HNSW structure
        shm_weights: shared memory
            shared memory associated to the shared array of weights of the HNSW structure
        shm_hnsw_data: shared memory
            shared memory to tack track of the added element to the HNSW structure
        shm_enter_point: shared memory
            shared memory associated to the enter point to start the search for the hnsw add procedure
        shm_count: shared memory
            shared memory associated to the count for counting how much call to the distance function we make
        lock: Lock
            the lock used to synchronize the access and modification of the shared enter point
        m : int, optional
            the number of each element's neighbors at the level > 0, default 5
        ef : int, optional
            number of closest neighbors of the inserted element in the layer, default 32
        m0 : int, optional
            the max number of each element's neighbors at the level 0, default None
        """
        self.data = data
        self.dim = len(data)
        self.distance = distance
        self.min_samples = 5
        self.distance_cache = {}
        self.shm_enter_point = shm_enter_point
        self.shm_count = shm_count
        self.sh_point = np.ndarray(shape=(1), dtype=int, buffer=shm_enter_point.buf)
        self.hnsw_data = np.ndarray(
            shape=(self.dim), dtype=int, buffer=shm_hnsw_data.buf
        )
        self.sh_count = np.ndarray(shape=(1), dtype=int, buffer=shm_count.buf)

        self.members = members
        self.levels = levels
        self.positions = positions

        self._m = m
        self._ef = ef
        self._m0 = 2 * m if m0 is None else m0
        self._enter_point = None

        # the hnsw graph now is composed by two numpy array for each level
        # that contain one the adjacency list of each elem,
        # and the other the weights list of each edges
        self.shm_adj = shm_adj
        self.shm_weights = shm_weights

        self.shared_weights = []
        self.shared_adjs = []
        for i in range(len(self.members)):
            self.shared_adjs.append(
                np.ndarray(
                    shape=(len(self.members[i]), self._m0 if i == 0 else self._m),
                    dtype=int,
                    buffer=shm_adj[i].buf,
                )
            )
            self.shared_weights.append(
                np.ndarray(
                    shape=(len(self.members[i]), self._m0 if i == 0 else self._m),
                    dtype=float,
                    buffer=shm_weights[i].buf,
                )
            )
        self.lock = lock

    def decorated_d(self, distance_cache, i, j):
        """Compute the distance bewteen i and j and save its value in the i's cache
        
        Parameters
        ----------
        distance_cache : dict
            cache of the distances computed between the element i and the others elements
        i : int
            the first element to be part of the distance computation
        j : int
            the second element to be part of the distance computation

        Returns
        -------
        dist : int
            the computed distance value
        """
        if j in distance_cache:
            return distance_cache[j]
        distance_cache[j] = dist = self.distance(self.data[i], self.data[j])
        return dist

    def add_and_compute_local_mst(self, points):
        """for each process, add a range of input elements in the HNSW structure and computes the local MST
        
        Parameters
        ----------
        points : list[int]
            cache of the distances computed between the element i and the others elements

        Returns
        -------
        local_mst : list[tuple]
            the computed distance value
        time_MST : float
            the time spent by the local mst computation
        time_HNSW : float
            the time spent by the partial HNSW computation
        """
        distances = []
        time_HNSW = 0
        start = time.time()
        for point in points:
            distances.append(self.hnsw_add(point))
        end = time.time()
        time_HNSW = end - start
        time_localMST = 0
        start = time.time()
        local_mst = self.local_mst(distances, points)
        end = time.time()
        time_localMST = end - start
        return local_mst, time_localMST, time_HNSW

    def hnsw_add(self, elem):
        """add the elem to the HNSW data structure
        
        Parameters
        ----------
        elem : int
            the element to be inserted in the HNSW structure

        Returns
        -------
        distance_cache : dict
            the distance cache of the element
        """
        distance_cache = {}
        level = self.calc_level(elem)
        """Add elem to the data structure"""
        sh_point = (
            np.ndarray(shape=(1), dtype=int, buffer=self.shm_enter_point.buf) + MISSING
            if elem == 0
            else np.ndarray(shape=(1), dtype=int, buffer=self.shm_enter_point.buf)
        )
        sh_count = np.ndarray(shape=(1), dtype=int, buffer=self.shm_count.buf)

        enter_point = sh_point[0]
        hnsw_data = self.hnsw_data
        hnsw_data[elem] = elem

        idx = elem
        ef = self._ef
        d = self.distance
        m = self._m

        shared_weights = []
        shared_adjs = []

        if enter_point != MISSING:  # the HNSW is not empty, we have an entry point
            dist = self.decorated_d(distance_cache, elem, enter_point)
            sh_count[0] = sh_count[0] + 1
            level_sh_point = self.calc_level(enter_point)

            # for all levels in which we dont have to insert elem,
            # we search for the closest neighbor
            if level_sh_point > level:
                level_to_search_pos = level_sh_point - 1
                for i in range(len(self.members)):
                    shared_adjs.append(
                        np.ndarray(
                            shape=(len(self.members[i]), self._m0 if i == 0 else m),
                            dtype=int,
                            buffer=self.shm_adj[i].buf,
                        )
                    )
                    shared_weights.append(
                        np.ndarray(
                            shape=(len(self.members[i]), self._m0 if i == 0 else m),
                            dtype=float,
                            buffer=self.shm_weights[i].buf,
                        )
                    )
                for g1, g2, sh1, sh2 in zip(
                    reversed(shared_adjs[level:level_sh_point]),
                    reversed(shared_weights[level:level_sh_point]),
                    reversed(self.shm_adj[level:level_sh_point]),
                    reversed(self.shm_weights[level:level_sh_point]),
                ):
                    level_m = self._m0 if level_to_search_pos == 0 else m
                    enter_point, dist = self._search_graph_ef1(
                        sh_count,
                        level_to_search_pos,
                        idx,
                        enter_point,
                        dist,
                        g1,
                        sh1,
                        distance_cache,
                    )
                    level_to_search_pos = level_to_search_pos - 1
            # at these levels we have to insert elem; ep is a heap of
            # entry points.
            ep = [(-dist, enter_point)]
            level_mod = level_sh_point if level_sh_point < level else level
            level_to_search_pos = level_mod - 1

            for i in range(len(self.members)):
                shared_adjs.append(
                    np.ndarray(
                        shape=(len(self.members[i]), self._m0 if i == 0 else m),
                        dtype=int,
                        buffer=self.shm_adj[i].buf,
                    )
                )
                shared_weights.append(
                    np.ndarray(
                        shape=(len(self.members[i]), self._m0 if i == 0 else m),
                        dtype=float,
                        buffer=self.shm_weights[i].buf,
                    )
                )

            for g1, g2, sh1, sh2 in zip(
                reversed(shared_adjs[:level_mod]),
                reversed(shared_weights[:level_mod]),
                reversed(self.shm_adj[:level_mod]),
                reversed(self.shm_weights[:level_mod]),
            ):
                level_m = self._m0 if level_to_search_pos == 0 else m
                ep = self._search_graph(
                    sh_count,
                    level_to_search_pos,
                    idx,
                    ep,
                    g1,
                    sh1,
                    distance_cache,
                    level_m,
                    ef,
                )

                pos = self.positions[level_to_search_pos].get(idx)
                self._select_heuristic(
                    level_to_search_pos,
                    pos,
                    ep,
                    level_m,
                    g1,
                    g2,
                    sh1,
                    sh2,
                    heap=True,
                )
                # insert backlinks to the new node
                for j, dist in zip(g1[pos], g2[pos]):
                    if j == MISSING or dist == MISSING_WEIGHT:
                        break
                    pos2 = self.positions[level_to_search_pos].get(j)
                    self._select_heuristic(
                        level_to_search_pos,
                        pos2,
                        (idx, dist),
                        level_m,
                        g1,
                        g2,
                        sh1,
                        sh2,
                    )

                level_to_search_pos = level_to_search_pos - 1

        if enter_point == MISSING or self.calc_level(enter_point) < level:
            self.lock.acquire()
            if enter_point == MISSING or self.calc_level(enter_point) < level:
                sh_point[0] = elem
            self.lock.release()

        return distance_cache

    def calc_position(self, to_find, level_to_search):
        """Function to find the position of an element in a specific level using the position structure

        Parameters
        ----------
        to_find : int
            element for which we want to find its position in the HNSW structure
        level_to_search : 
            level in which we want to search the element
        
        Returns
        -------
        the found position of the element at the specific level
        """
        return self.positions[level_to_search].get(to_find)

    def calc_level(self, elem):
        """Function to find the assigned level of an element

        Parameters
        ----------
        elem : int
            element for which we want to find its assigned level in the HNSW structure
        
        Returns
        -------
            the found belonging level of elem
        """
        for dic, i in zip(
            reversed(self.positions), reversed(range(len(self.positions)))
        ):
            if elem in dic:
                return i + 1

    def search(self, graphs, q, k=None, ef=None):
        """Find the k points closest to q.
        
        Parameters
        ----------
        graphs: list[dict]
            the graph representing the HNSW structure
        q : int
            the element for which find its k closest point
        k : int, optional
            the number of closest neighbors to find, default None
        ef : int, optional
            number of candidate closest neighbors of the inserted element in the layer, default None

        Returns
        -------
            the found K neighbors
        """

        d = self.distance
        graphs = graphs
        sh_point = np.ndarray(shape=(1), dtype=int, buffer=self.shm_enter_point.buf)
        point = sh_point[0]
        if ef is None:
            ef = self._ef

        if point is None:
            raise ValueError("Empty graph")

        dist = d(q, self.data[point])
        # look for the closest neighbor from the top to the 2nd level
        for g in reversed(graphs[1:]):
            point, dist = self._search_graph_ef1_test(q, point, dist, g)
        # look for ef neighbors in the bottom level
        ep = self._search_graph_test(q, [(-dist, point)], graphs[0], ef)

        if k is not None:
            ep = nlargest(k, ep)
        else:
            ep.sort(reverse=True)

        return [(idx, -md) for md, idx in ep]

    def _search_graph_ef1_test(self, q, entry, dist, g):
        """Equivalent to _search_graph_test when ef=1 used when we want to perform the accuracy test

        Parameters
        ----------
        q : int
            the element for which find its candidates to be linked to it
        entry : int
            the entry point from which starts the search fo the candidates
        dist : func
            the distance function
        g : dict
            the current level we are processing
        
        Returns
        -------
        best : int
            the best found neighbor of element
        best_dist : float
            the related distance of the best neighbor

        """

        d = self.distance
        data = self.data

        best = entry
        best_dist = dist
        candidates = [(dist, entry)]
        visited = set([entry])
        while candidates:
            dist, c = heappop(candidates)
            if dist > best_dist:
                break
            edges = [e for e in g[c] if e not in visited]
            if not edges:
                continue
            visited.update(edges)
            dists = [d(q, data[e]) for e in edges]
            for e, dist in zip(edges, dists):
                if dist < best_dist:
                    best = e
                    best_dist = dist
                    heappush(candidates, (dist, e))

        return best, best_dist

    def _search_graph_test(self, q, ep, g, ef):
        """Function used to find the candidates neighbors 
        of the element to be linked with the elem.
        when we want to perform the accuracy test

        Parameters
        ----------
        q : int
            the element for which find its candidates to be linked to it
        ep : heap
            the heap of entry points from which starts the search fo the candidates
        dist : func
            the distance function
        g : dict
            the current level we are processing
        ef : int
            number of closest neighbors of the inserted element in the layer, default None

        Returns
        -------
        ep : heap 
            the heap containing the found candidates of the element
        """
        d = self.distance
        data = self.data

        candidates = [(-mdist, p) for mdist, p in ep]
        heapify(candidates)
        visited = set(p for _, p in ep)

        while candidates:
            dist, c = heappop(candidates)
            mref = ep[0][0]
            if dist > -mref:
                break

            edges = [e for e in g[c] if e not in visited]
            if not edges:
                continue
            visited.update(edges)
            dists = [d(q, data[e]) for e in edges]
            for e, dist in zip(edges, dists):
                mdist = -dist
                if len(ep) < ef:
                    heappush(candidates, (dist, e))
                    heappush(ep, (mdist, e))
                    mref = ep[0][0]
                elif mdist > mref:
                    heappush(candidates, (dist, e))
                    heapreplace(ep, (mdist, e))
                    mref = ep[0][0]

        return ep

    def _search_graph_ef1(
        self,
        count_dist,
        level_to_search,
        q,
        entry,
        dist,
        arr_adj,
        shm_adj,
        distance_cache,
    ):
        """Equivalent to _search_graph when ef=1.

        Parameters
        ----------
        count_dist : shared memory
            counter for counting the number of call to the distance
        level_to_search : int
            level to search the neighbor of the elem
        q : int
            the element for which find its candidates to be linked to it
        entry : int
            the entry point from which starts the search fo the candidates
        dist : func
            the distance function
        arr_adj : shared numpy array
            the current shared array of the nodes in level we are processing
       shm_adj : shared memory
            the current shared memoryassociated to the shared array of the nodes
       distance_cache : dict
            cache of the distances computed between the element i and the others elements
        
        Returns
        -------
        best : int
            the best found neighbor of element
        best_dist : float
            the related distance of the best neighbor

        """
        g_adj = np.ndarray(shape=arr_adj.shape, dtype=int, buffer=shm_adj.buf)
        d = self.distance
        data = self.data

        best = entry
        best_dist = dist
        candidates = [(dist, entry)]
        visited = set([entry])

        while candidates:
            dist, c = heappop(candidates)
            if dist > best_dist:
                break
            pos = self.calc_position(c, level_to_search)
            edges = []
            for e in g_adj[pos]:
                if e == MISSING:
                    break
                if e not in visited:
                    edges.append(e)

            if not edges:
                continue
            visited.update(edges)
            count_dist[0] = count_dist[0] + len(edges)
            dists = [self.decorated_d(distance_cache, q, e) for e in edges]

            for e, dist in zip(edges, dists):
                if dist < best_dist:
                    best = e
                    best_dist = dist
                    heappush(candidates, (dist, e))
        return best, best_dist

    def _search_graph(
        self,
        count_dist,
        level_to_search,
        q,
        ep,
        arr_adj,
        shm_adj,
        distance_cache,
        m,
        ef,
    ):
        """Function used to find the candidates neighbors 
        of the element to be linked with the elem.

        Parameters
        ----------
        count_dist : shared memory
            counter for counting the number of call to the distance
        level_to_search : int
            level to search the neighbor of the elem
        q : int
            the element for which find its candidates to be linked to it
        ep : heap
            the heap of entry points from which starts the search of the candidates
        dist : func
            the distance function
        arr_adj : shared numpy array
            the current shared array of the nodes in the level we are processing
        shm_adj : shared memory
            the current shared memory associated to the shared array of the nodes
        distance_cache : dict
            cache of the distances computed between the element and the others elements
        m : int, optional
            the number of each element's neighbors at the level > 0, default 5
        ef : int, optional
            number of candidates closest neighbors of the inserted element in the layer
    
        Returns
        -------
        ep : heap 
            the heap containing the found candidates of the element
        """
        g_adj = np.ndarray(shape=(len(arr_adj), m), dtype=int, buffer=shm_adj.buf)

        d = self.distance
        data = self.data
        candidates = [(-mdist, p) for mdist, p in ep]
        heapify(candidates)
        visited = set(p for _, p in ep)
        while candidates:
            dist, c = heappop(candidates)
            mref = ep[0][0]
            if dist > -mref:  
                break
            pos = self.calc_position(c, level_to_search)
            edges = []
            for e in g_adj[pos]:
                if e == MISSING:
                    break
                if e not in visited:
                    edges.append(e)
            if not edges:
                continue

            visited.update(edges)
            count_dist[0] = count_dist[0] + len(edges)
            dists = [self.decorated_d(distance_cache, q, e) for e in edges]

            for e, dist in zip(edges, dists):
                mdist = -dist
                if len(ep) < ef:
                    heappush(candidates, (dist, e))
                    heappush(ep, (mdist, e))
                    mref = ep[0][0]
                elif mdist > mref:
                    heappush(candidates, (dist, e))
                    heapreplace(ep, (mdist, e))
                    mref = ep[0][0]

        return ep

    def _select_heuristic(
        self,
        level_to_search,
        position,
        to_insert,
        m,
        arr_adj,
        arr_weights,
        shm_adj,
        shm_weights,
        heap=False,
    ):
        """Function to select and link the right neighbors
        to the current element to be inserted, with the usage of an heuristic.

        Parameters
        ----------
        level_to_search : int
            level to search the neighbors of the elem
        position : int
            the position of the element in the considered level
        to_insert: heap
            heap of elements to be linked
        m : int
            the number of each element's neighbors 
        arr_adj : shared numpy array
            the current shared array of the nodes in the level we are processing
        arr_weights : shared numpy array
            the current shared array of the weights in the level we are processing
        shm_adj : shared memory
            the current shared memory associated to the shared array of the nodes
        shm_weights : shared memory
            the current shared memory associated to the shared array of the weights
        heap: bool, optional
            true if you have more than one element to insert, false otherwise. it's a shortcut when we've got only one thing to insert
        """
        g_adj = np.ndarray(shape=arr_adj.shape, dtype=int, buffer=shm_adj.buf)
        g_weights = np.ndarray(
            shape=arr_weights.shape, dtype=float, buffer=shm_weights.buf
        )

        def prioritize(idx, dist):
            b = False
            for ndw, nda in zip(nb_dicts_weights, nb_dicts_adj):
                p = np.where(nda == idx)[0]
                if len(p) == 0:
                    if inf < dist:
                        b = True
                        break
                elif ndw[p[0]] < dist:
                    b = True
                    break
            return b, dist, idx

        nb_dicts_adj = []
        nb_dicts_weights = []
        for idx, i in zip(g_adj[position], range(len(g_adj[position]))):
            if idx == MISSING:
                break
            pos = self.calc_position(idx, level_to_search)
            nb_dicts_adj.append(g_adj[pos])
            nb_dicts_weights.append(g_weights[pos])

        if not heap:
            idx, dist = to_insert
            to_insert = [prioritize(idx, dist)]
            if idx in g_adj[position]:
                to_insert = []
        else:
            tempList1 = []
            for mdist, idx in to_insert:
                if idx in g_adj[position]:
                    continue
                tempList1.append(prioritize(idx, -mdist))
            to_insert = nsmallest(m, tempList1)

        assert len(to_insert) > 0
        assert not any(
            idx in g_adj[position] for _, _, idx in to_insert
        ), "idx:{0}".format(
            idx
        )  # check if the assert make sense in concurrent version
        list_unchecked = list(filter(lambda i: i != MISSING, g_adj[position]))
        unchecked = m - len(list_unchecked)
        assert 0 <= unchecked <= m
        to_insert, checked_ins = to_insert[:unchecked], to_insert[unchecked:]
        to_check = len(checked_ins)

        if to_check > 0:
            tempList2 = []
            for idx, dist in zip(g_adj[position], g_weights[position]):
                if idx == MISSING:
                    break
                tempList2.append(prioritize(idx, dist))
            checked_del = nlargest(to_check, tempList2)

        else:
            checked_del = []
        # with self.elem_locks[lock_id]:
        for _, dist, idx in to_insert:
            for i, el in enumerate(g_weights[position]):
                if el == MISSING_WEIGHT:
                    g_weights[position][i] = abs(dist)
                    g_adj[position][i] = idx
                    break

        zipped = zip(checked_ins, checked_del)
        for (p_new, d_new, idx_new), (p_old, d_old, idx_old) in zipped:
            if (p_old, d_old) <= (p_new, d_new):
                break
            # with self.elem_locks[lock_id]:
            for i, el in enumerate(g_adj[position]):
                if el == idx_old:
                    g_adj[position][i] = idx_new
                    g_weights[position][i] = abs(d_new)
                    break

    def local_mst(self, distances, points):
        """Function that computes the local mst of each process, 
        based on the distance caches associated to its range of points

        Parameters
        ----------
        distances : list[dict]
            list of all the distance caches associated the points
        points : 
            the range of points for which we have to calculate the local mst
        
        Returns
        -------
        mst_edges : list[tuple]
            the mst edges, namely the local mst
        """
        shared_weights = []
        shared_adjs = []
        for i in range(len(self.members)):
            shared_adjs.append(
                np.ndarray(
                    shape=(len(self.members[i]), self._m0 if i == 0 else self._m),
                    dtype=int,
                    buffer=self.shm_adj[i].buf,
                )
            )
            shared_weights.append(
                np.ndarray(
                    shape=(len(self.members[i]), self._m0 if i == 0 else self._m),
                    dtype=float,
                    buffer=self.shm_weights[i].buf,
                )
            )
        data = self.data
        candidate_edges = []
        points = list(points)
        for d_cache, i in zip(distances, points):
            if i in d_cache:
                d_cache.pop(i)
            nhi = shared_weights[0][i]
            nhi = np.sort(nhi)
            for j, dist in d_cache.items():
                assert dist == self.distance(self.data[i], self.data[j])
                nhj = shared_weights[0][j]
                nhj = np.sort(nhj)
                candidate_edges.append(
                    (max(dist, nhi[self._m], nhj[self._m]), i, j, dist)
                )
        mst_edges = []
        n = len(data)
        uf = UnionFind(n)
        heapify(candidate_edges)
        while candidate_edges:
            mrd, i, j, dist = heappop(candidate_edges)
            if uf.union(i, j):
                mst_edges.append((mrd, i, j, dist))
        return mst_edges

    def global_mst(self, candidate_edges, n):
        """Function that computes the global MST, 
        based on the previous computed local MSTs, using Kruskal algorithm

        Parameters
        ----------
        candidates_edges : list[tuple]
           the candidate edges for the global mst, namely the partial local MSTs
        n : int
            the size used by the kruskal UnionFind 
        
        Returns
        -------
        final_mst : list[tuple]
            the final mst
        """
        uf = UnionFind(n)
        final_mst = []
        candidate_edges.sort()
        for mrd, i, j, dist in candidate_edges:
            if uf.union(i, j):
                final_mst.append((mrd, i, j, dist))
        return final_mst
