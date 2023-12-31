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

"""This script runs an example of the parallel FISHDBC example,
but you can also run the classical original single process FISHDBC.

Functions:

    * plot_cluster_result - plot the clustering result
    * split - split the dataset in a number of ranges as the number of processes
    * create_levels - assign for each point its level
    * create_members - create in the correct way each level of the HNSW
    * create_positions - assign for each element its position inside a level
    * compute_FISHDBC_accuracy - computes the FISHDBC clustering accuracy between the multi-process FISHDBC parallel version and the original labels of the orginal synthetic dataset 
    * make_texts - creates the synthetic text dataset
    * check_range_nitems - checks if the number of dataset's items is in the correct range
    * check_range_centers - checks if the number of centers is in the correct range
    * main - the main function of the scripts that, if you execute it, starts or the parallel FISHDBC or the single process FISHDBC
"""

import numpy as np
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
from parallel_flexible_clustering import fishdbc_parallel
import create_text_dataset

# from line_profiler import LineProfiler
import time
import multiprocessing
# from math import dist as mathDist
from random import random

try:
    from math import log2
except ImportError:  # Python 2.x or <= 3.2
    from math import log

    def log2(x):
        return log(x, 2)


MISSING = sys.maxsize
MISSING_WEIGHT = sys.float_info.max

def create_levels(data):
    """Function used to assign for each point its level

    Parameters
    ----------
    data : np array
           the input data set
    Returns
    -------
    levels : list of tuple
        the level of each point
    """
    levels = [(int(-log2(random()) * (1 / log2(m))) + 1) for _ in range(len(data))]
    levels = sorted(enumerate(levels), key=lambda x: x[1])
    return levels

def create_members(levels):  
    """Function used to create in the correct way each level of the HNSW

    Parameters
    ----------
    levels : list of tuple
           the level of each point
    Returns
    -------
    members : list of list
        the composition of each level of the HNSW
    """
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
    return members

def create_positions(members):
    """Function used to assign for each element its position inside a level

    Parameters
    ----------
    members : list of list
           the composition of each level of the HNSW
    Returns
    -------
    positions : list of dictionares
        the positions of each element in the various levels
    """
    positions = []
    for el, l in zip(members, range(len(members))):
        positions.append({})
        for i, x in enumerate(el):
            positions[l][x] = i
    return positions

def plot_cluster_result(size, ctree, x, y, labels):
    """Function to plot the resulting clusters
    Parameters
    ----------
    size: int
        the size of the dataset
    ctree : 
        the condensed tree of the hiererchical clustering
    labels : list
        the labels of the resulting hiererchical clustering

    """
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
    """Function used to split the input data in a number of range as the number of used processes

    Parameters
    ----------
    a : list
        the list of input data points
    n : 
        the number of processes
    Returns
    -------
        the splitted range of points
    """
    k, m = divmod(len(a), n)
    indices = [k * i + min(i, m) for i in range(n + 1)]
    return [a[l:r] for l, r in pairwise(indices)]

def compute_FISHDBC_accuracy(original_labels, resulting_labels):
    """Function to compute the FISHDBC (paralell and single process) clustering accuracy between the multi-process FISHDBC parallel version and the original labels of the orginal synthetic dataset

    Parameters
    ----------
    original_labels : list
            the labels assigned to the original data items
    resulting_labels : list
            the labels assigned to the clusterized data items, after the FISHDBC 
    """    
    from sklearn.metrics.cluster import (
                adjusted_mutual_info_score,
                adjusted_rand_score,
                rand_score,
                normalized_mutual_info_score,
                homogeneity_completeness_v_measure,
            )
    homogeneity, completness, v_measure = homogeneity_completeness_v_measure(
        original_labels, resulting_labels
    )
    print(
        "Adjsuted Mutual Info Score: ",
        "{:.2f}".format(adjusted_mutual_info_score(original_labels, resulting_labels)),
    )
    print(
        "Normalized Mutual Info Score: ",
        "{:.2f}".format(
            normalized_mutual_info_score(original_labels, resulting_labels)
        ),
    )
    print(
        "Adjusted Rand Score: ",
        "{:.2f}".format(adjusted_rand_score(original_labels, resulting_labels)),
    )
    print(
        "Rand Score: ", "{:.2f}".format(rand_score(original_labels, resulting_labels))
    )
    print(
        "Homogeneity, Completness, V-Measure: ",
        (homogeneity, completness, v_measure),
    )

def make_texts(centers, nitems):
    """Function to create the text dataset using the specific module

    Parameters
    ----------
    centers : int
        number of centroids
    nitems : int
        number of items 
   
    Returns
    -------
    data : list
        the created text dataset
    labels : list
        the associated labels of the text dataset
    """
    realData = create_text_dataset.gen_dataset(centers, 20, nitems, 4)
    labels = create_text_dataset.gen_labels(centers, nitems)

    data = np.array(realData[0]).reshape(-1, 1)
    labels = np.asarray(labels).reshape(-1, 1)

    shuffled_indices = np.arange(len(data))
    np.random.shuffle(shuffled_indices)

    # Use the shuffled indices to rearrange both elements and labels
    data = data[shuffled_indices]
    labels = labels[shuffled_indices]
    labels = [item for sublist in labels for item in sublist]

    return data, labels

def check_range_nitems(value):
    """Function to check if the number of dataset's items is in the correct range

    Parameters
    ----------
    value : int
        value to check if it is between the max and min values allowed
   
    Returns
    -------
    ivalue : int
        the same value if it passes the check
    """
    ivalue = int(value)
    if ivalue < 10 or ivalue > 1000000:
        raise argparse.ArgumentTypeError(f"{value} is not in the range 10-1000000")
    return ivalue

def check_range_centers(value):
    """Function to check if the number of centers is in the correct range

    Parameters
    ----------
    value : int
        value to check if it is between the max and min values allowed
   
    Returns
    -------
    ivalue : int
        the same value if it passes the check
    """
    ivalue = int(value)
    if ivalue < 1 or ivalue > 50:
        raise argparse.ArgumentTypeError(f"{value} is not in the range 1-50")
    return ivalue

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
        choices={"blob","text"},
        help="dataset used by the algorithm (default: blob)." "try with: blob, string,",
    )
    parser.add_argument(
        "--distance",
        type=str,
        default="euclidean",
        choices={"euclidean", "sqeuclidean", "cosine", "minkowsky", "levenshtein"},
        help="distance metrix used by FISHDBC (default: euclidean)."
        "try with: euclidean, squeclidean, cosine, minkowsky, levenshtein",
    )
    parser.add_argument(
        "--nitems", type=check_range_nitems, default=10000, help="Number of items (default 10000)."
    )
    parser.add_argument(
        "--niters",
        type=int,
        default=2,
        choices=range(1, 11),
        help="Clusters are shown in NITERS stage while being "
        "added incrementally (default 2).",
    )
    parser.add_argument(
        "--centers",
        type=check_range_centers,
        default=5,
        help="Number of centers for the clusters generated " "(default 5).",
    )
    parser.add_argument(
        "--parallel",
        type=int,
        default=16,
        choices=range(1,17),
        help="option to specify if we want to execute the parallel HNSW (specifying the number of processes from 1 to 16)",
    )
    parser.add_argument(
        "--test",
        type=str,
        default="False",
        choices={"True", "False"},
        help="Option to say to perform HNSW accuracy test, works only with blob dataset and euclidean distance"
        "(default False).",
    )
    args = parser.parse_args()
    dist = args.distance.lower()
    dataset = args.dataset
    parallel = int(args.parallel)
    test = True if args.test == "True" else False

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
        data, labels = make_texts(args.centers, args.nitems)

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
    if parallel > 1:
        print(
            "-------------------------- MULTI-PROCESS PARALLEL FISHDBC --------------------------"
        )
        start_tot = time.time()
        m = 5
        m0 = 2 * m
        # with fork method as starting process the child processes created starting from the main process
        # should inherit the calc_distance function and the orignal dataset
        multiprocessing.set_start_method("fork")

        levels = create_levels(data)
        members = create_members(levels)
        positions = create_positions(members)

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

        fishdbcPar = fishdbc_parallel.PARALLEL_FISHDBC(
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
        fishdbcPar.hnsw_add(0)
        pool = multiprocessing.Pool(num_processes)
        for local_mst, mst_time, hnsw_time in pool.map(
            fishdbcPar.add_and_compute_local_mst, split(range(1, len(data)), num_processes)
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
        final_fishdbc = fishdbc.FISHDBC(
            calc_dist, m, m0, vectorized=False, balanced_add=False
        )
        final_mst = fishdbcPar.global_mst(partial_mst, len(data))
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
        labels_cluster_par, _, _, ctree, _, _ = final_fishdbc.cluster(
            final_mst, parallel=True
        )
        end = time.time()
        time_parallelFISHDBC = "{:.3f}".format(end - start_time)
        print("The time of execution of Parallel FISHDBC is :", time_parallelFISHDBC)

        if test == True:
            compute_FISHDBC_accuracy(labels, labels_cluster_par)

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
            compute_FISHDBC_accuracy(labels, labels_cluster)
            
        print(
            "___________________________________________________________________________________________\n"
        )


