#!/usr/bin/env python3

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
# import timeit
import numpy as np
import pandas as pd
import argparse
from itertools import pairwise
import sys
import collections

# from functools import partial
from scipy.spatial import distance
from numba import njit
from Levenshtein import distance as lev
import sklearn.datasets
import matplotlib.pyplot as plt
from parallel_flexible_clustering import fishdbc
from parallel_flexible_clustering import hnsw_parallel
import create_text_dataset

# from line_profiler import LineProfiler
import time
import multiprocessing
from math import dist as mathDist
from random import random

try:
    from math import log2
except ImportError:  # Python 2.x or <= 3.2
    from math import log

    def log2(x):
        return log(x, 2)


MISSING = sys.maxsize
MISSING_WEIGHT = sys.float_info.max
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Show an example of running FISHDBC."
        "This will plot points that are naturally clustered and added incrementally,"
        "and then loop through all the hierarchical clusters recognized by the algorithm."
        "Original clusters are shown in different colors while each cluster found by"
        "FISHDBC is shown in red; press a key or click the mouse button to loop through clusters."
    )

    parser.add_argument(
        "--dataset",
        type=str,
        default="blob",
        help="dataset used by the algorithm (default: blob)." "try with: blob, string,",
    )
    parser.add_argument(
        "--distance",
        type=str,
        default="euclidean",
        help="distance metrix used by FISHDBC (default: hamming)."
        "try with: euclidean, squeclidean, cosine, dice, minkowsky, jaccard, hamming, jensenShannon, levensthein",
    )
    parser.add_argument(
        "--nitems", type=int, default=10000, help="Number of items (default 10000)."
    )
    parser.add_argument(
        "--niters",
        type=int,
        default=2,
        help="Clusters are shown in NITERS stage while being "
        "added incrementally (default 4).",
    )
    parser.add_argument(
        "--centers",
        type=int,
        default=5,
        help="Number of centers for the clusters generated " "(default 5).",
    )
    parser.add_argument(
        "--test",
        type=bool,
        default=False,
        help="Option to say to perform FISHDBC clustering accuracy test"
        "(default False).",
    )
    parser.add_argument(
        "--parallel",
        type=str,
        default="0",
        help="option to specify if we want to execute the parallel FISHDBC (specifying the number of processes from1 to 16) or single process FISHDBC (0 processes)",
    )
    args = parser.parse_args()

    def plot_cluster_result(size, ctree, x, y, labels):
        plt.figure(figsize=(9, 9))
        plt.gca().set_aspect("equal")
        nknown = (
            size  
        )
    
        clusters = collections.defaultdict(set)
        for parent, child, lambda_val, child_size in ctree[::-1]:
            if child_size == 1:
                clusters[parent].add(
                    child
                )  
            else:
                assert len(clusters[child]) == child_size
                clusters[parent].update(
                    clusters[child]
                )  
        clusters = sorted(
            clusters.items()
        ) 
        xknown, yknown, labels_known = x[:nknown], y[:nknown], labels[:nknown]
        color = ["rgbcmyk"[l % 7] for l in labels_known]
        plt.scatter(xknown, yknown, c=color, linewidth=0)
        plt.show(block=False)
        for _, cluster in clusters:
            plt.waitforbuttonpress()
            plt.gca().clear()
            color = ["kr"[i in cluster] for i in range(nknown)]
            plt.scatter(xknown, yknown, c=color, linewidth=0)
            plt.draw()

    def split(a, n):
        k, m = divmod(len(a), n)
        indices = [k * i + min(i, m) for i in range(n + 1)]
        return [a[l:r] for l, r in pairwise(indices)]

    dist = args.distance.lower()
    dataset = args.dataset
    parallel = int(args.parallel)

    if dataset == "blob":
        data, labels = sklearn.datasets.make_blobs(
            args.nitems, centers=args.centers, random_state=10
        )
        if dist == "euclidean":

            @njit
            def calc_dist(x, y):
                return np.linalg.norm(x - y)
                # return distance.euclidean(x, y)
                # return mathDist(x, y)

        elif dist == "sqeuclidean":

            def calc_dist(x, y):
                return distance.sqeuclidean(x, y)

        elif dist == "cosine":

            def calc_dist(x, y):
                return distance.cosine(x, y)

        elif dist == "minkowski":

            def calc_dist(x, y):
                return distance.minkowski(x, y, p=2)

        else:
            raise EnvironmentError(
                "At the moment the specified distance is not available for the blob dataset,"
                " try with: euclidean, sqeuclidean, cosine, minkowsky"
            )
    elif dataset == "text":
        realData = create_text_dataset.gen_dataset(args.centers, 20, args.nitems, 4)
        labels = create_text_dataset.gen_labels(args.centers, args.nitems)
        data = np.array(realData[0]).reshape(-1, 1)
        labels = np.asarray(labels).reshape(-1, 1)
        shuffled_indices = np.arange(len(data))
        np.random.shuffle(shuffled_indices)
        # Use the shuffled indices to rearrange both elements and labels
        data = data[shuffled_indices]
        labels = labels[shuffled_indices]
        labels = [item for sublist in labels for item in sublist]
        # if dist == 'hamming':
        #     def calc_dist(x,y):
        #         return distance.hamming(x,y)
        if dist == "levenshtein":

            def calc_dist(x, y):
                return lev(x, y)

        else:
            raise EnvironmentError(
                "At the moment the specified distance is not available for the string dataset,"
                " try with: levenshtein"
            )
    else:
        raise EnvironmentError(
            "The specified dataset doesn't exist at the moment," "try with: blob, text"
        )

    # x, y = data[:, 0], data[:, 1]
    if parallel > 0:
        print(
            "-------------------------- MULTI-PROCESS FISHDBC --------------------------"
        )
        start_tot = time.time()
        m = 5
        m0 = 2 * m
        # with fork method as starting process the child processes created starting from the main process
        # should inherit the calc_distance function and the orignal dataset
        multiprocessing.set_start_method("fork")

        # assign each element to a random level
        levels = [(int(-log2(random()) * (1 / log2(m))) + 1) for _ in range(len(data))]
        levels = sorted(enumerate(levels), key=lambda x: x[1])

        # insert the point in the list corresponding to the right level
        members = [[]]
        j = 1
        level_j = []
        for i in levels:
            elem, level = i
            if level > j:
                members.append(level_j)
                level_j = []
                j = j + 1
            level_j.append(elem)
            if j - 1 > 0:
                for i in range(j - 1, 0, -1):
                    members[i].append(elem)
        members.append(level_j)
        for i, l in zip(range(len(members)), members):
            sort = sorted(l)
            members[i] = sort
        del members[0]

        # create a list of dict to associate for each point in each levels its position
        positions = []
        for el, l in zip(members, range(len(members))):
            positions.append({})
            for i, x in enumerate(el):
                positions[l][x] = i
        # print("Levels: ",levels, "\n")
        # print("Members: ",members, "\n")
        # print("Positions: ",positions, "\n")

        # create the buffer of shared memory for each levels
        shm_hnsw_data = multiprocessing.shared_memory.SharedMemory(
            create=True, size=1000000000
        )
        shm_ent_point = multiprocessing.shared_memory.SharedMemory(create=True, size=10)
        shm_count = multiprocessing.shared_memory.SharedMemory(create=True, size=10)

        shm_adj = []
        shm_weights = []
        for i in range(len(members)):
            npArray = np.zeros(shape=(len(members[i]), m0 if i == 0 else m), dtype=int)
            shm1 = multiprocessing.shared_memory.SharedMemory(
                create=True, size=npArray.nbytes
            )
            np.ndarray(npArray.shape, dtype=int, buffer=shm1.buf)[:, :] = MISSING
            shm_adj.append(shm1)

            shm2 = multiprocessing.shared_memory.SharedMemory(
                create=True, size=npArray.nbytes
            )
            np.ndarray(npArray.shape, dtype=float, buffer=shm2.buf)[
                :, :
            ] = MISSING_WEIGHT
            shm_weights.append(shm2)

        num_processes = parallel
        manager = multiprocessing.Manager()
        lock = manager.Lock()

        hnswPar = hnsw_parallel.HNSW(
            calc_dist,
            data,
            members,
            levels,
            positions,
            shm_adj,
            shm_weights,
            shm_hnsw_data,
            shm_ent_point,
            shm_count,
            lock,
            m=m,
            m0=m0,
            ef=32,
        )

        # add the first element not in multiprocessing for correct initialization
        start_time = time.time()
        start_time_hnsw_par = time.time()

        partial_mst = []
        mst_times = []
        hnsw_times = []
        hnswPar.hnsw_add(0)
        pool = multiprocessing.Pool(num_processes)
        for local_mst, mst_time, hnsw_time in pool.map(
            hnswPar.add_and_compute_local_mst, split(range(1, len(data)), num_processes)
        ):
            mst_times.append(mst_time)
            hnsw_times.append(hnsw_time)
            partial_mst.extend(local_mst)
        pool.close()
        pool.join()

        end_time_hnsw_par = time.time()
        time_parHNSW = "{:.2f}".format(end_time_hnsw_par - start_time_hnsw_par)
        print(
            "The time of execution of Paralell HNSW and local MSTs is :", (time_parHNSW)
        )

        time_HNSW = np.mean(hnsw_times)
        print(
            "The time of execution of Paralell HNSW is :",
            "{:.3f}".format(time_HNSW),
        )
        time_localMST = np.mean(mst_times)
        print(
            "The time of execution of Paralell local MSTs is :",
            "{:.3f}".format(time_localMST),
        )

        tot_adjs = []
        tot_weights = []
        for shm1, shm2, memb, i in zip(
            shm_adj, shm_weights, members, range(len(members))
        ):
            adj = np.ndarray(
                shape=(len(memb), m0 if i == 0 else m), dtype=int, buffer=shm1.buf
            )
            tot_adjs.append(adj)
            weight = np.ndarray(
                shape=(len(memb), m0 if i == 0 else m), dtype=float, buffer=shm2.buf
            )
            tot_weights.append(weight)
        # print(tot_adjs, "\n", tot_weights, "\n")

        start = time.time()
        # perform the final fishdbc operation, the creation of the mst and the final clustering
        fishdbcPar = fishdbc.FISHDBC(
            calc_dist, m, m0, vectorized=False, balanced_add=False
        )
        final_mst = hnswPar.global_mst(shm_adj, shm_weights, partial_mst, len(data))
        end = time.time()
        time_globalMST = end - start
        print(
            "The time of execution of global MST is :", "{:.3f}".format(time_globalMST)
        )
        time_parallelMST = time_localMST + time_globalMST
        print(
            "The total time of execution of MST is :",
            "{:.3f}".format(time_parallelMST),
        )
        n = len(data)
        labels_cluster_par, _, _, ctree, _, _ = fishdbcPar.cluster(
            final_mst, parallel=True
        )
        end = time.time()
        time_parallelFISHDBC = "{:.3f}".format(end - start_time)
        print("The time of execution of Parallel FISHDBC is :", time_parallelFISHDBC)

        if args.test == True:
            from sklearn.metrics.cluster import (
                adjusted_mutual_info_score,
                adjusted_rand_score,
                rand_score,
                normalized_mutual_info_score,
                homogeneity_completeness_v_measure,
            )

            AMI = adjusted_mutual_info_score(labels, labels_cluster_par)
            NMI = normalized_mutual_info_score(labels, labels_cluster_par)
            ARI = adjusted_rand_score(labels, labels_cluster_par)
            RI = rand_score(labels, labels_cluster_par)
            homogeneity, completness, v_measure = homogeneity_completeness_v_measure(
                labels, labels_cluster_par
            )
            print(
                "Adjsuted Mutual Info Score: ",
                "{:.2f}".format(adjusted_mutual_info_score(labels, labels_cluster_par)),
            )
            print(
                "Normalized Mutual Info Score: ",
                "{:.2f}".format(
                    normalized_mutual_info_score(labels, labels_cluster_par)
                ),
            )
            print(
                "Adjusted Rand Score: ",
                "{:.2f}".format(adjusted_rand_score(labels, labels_cluster_par)),
            )
            print(
                "Rand Score: ", "{:.2f}".format(rand_score(labels, labels_cluster_par))
            )
            print(
                "Homogeneity, Completness, V-Measure: ",
                (homogeneity, completness, v_measure),
            )

        shm_hnsw_data.close()
        shm_hnsw_data.unlink()
        shm_ent_point.close()
        shm_ent_point.unlink()
        shm_count.unlink()
        shm_count.close()
        for i in range(len(members)):
            shm_adj[i].close()
            shm_adj[i].unlink()
            shm_weights[i].close()
            shm_weights[i].unlink()

        print(
            "___________________________________________________________________________________________\n"
        )
    else:
        print(
            "-------------------------- SINGLE PROCESS FISHDBC --------------------------"
        )
        start_single = time.time()
        fishdbcSingle = fishdbc.FISHDBC(calc_dist, vectorized=False, balanced_add=False)
        single_cand_edges = fishdbcSingle.update(data)
        graphs = fishdbcSingle.the_hnsw._graphs
        time_singleHNSW = "{:.2f}".format(fishdbcSingle._tot_time)
        print("The time of execution Single HNSW:", (time_singleHNSW))
        time_singleMST = "{:.2f}".format(fishdbcSingle._tot_MST_time)
        print("The time of execution Single MST:", (time_singleMST))
        labels_cluster, _, _, ctree, _, _ = fishdbcSingle.cluster(parallel=False)

        end_single = time.time()
        time_singleFISHDBC = end_single - start_single
        print(
            "The time of execution Single FISHDBC:",
            "{:.3f}".format(time_singleFISHDBC),
        )
        if args.test == True:
            from sklearn.metrics.cluster import (
                adjusted_mutual_info_score,
                adjusted_rand_score,
                rand_score,
                normalized_mutual_info_score,
                homogeneity_completeness_v_measure,
            )

            AMI = adjusted_mutual_info_score(labels, labels_cluster)
            NMI = normalized_mutual_info_score(labels, labels_cluster)
            ARI = adjusted_rand_score(labels, labels_cluster)
            RI = rand_score(labels, labels_cluster)
            homogeneity, completness, v_measure = homogeneity_completeness_v_measure(
                labels, labels_cluster
            )
            print(
                "Adjsuted Mutual Info Score: ",
                "{:.2f}".format(adjusted_mutual_info_score(labels, labels_cluster)),
            )
            print(
                "Normalized Mutual Info Score: ",
                "{:.2f}".format(normalized_mutual_info_score(labels, labels_cluster)),
            )
            print(
                "Adjusted Rand Score: ",
                "{:.2f}".format(adjusted_rand_score(labels, labels_cluster)),
            )
            print("Rand Score: ", "{:.2f}".format(rand_score(labels, labels_cluster)))
            print(
                "Homogeneity, Completness, V-Measure: ",
                (homogeneity, completness, v_measure),
            )
        print(
            "___________________________________________________________________________________________\n"
        )


