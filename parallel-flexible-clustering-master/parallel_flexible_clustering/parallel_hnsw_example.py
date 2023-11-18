import numpy as np
import sys
import argparse
from numba import njit
from scipy.spatial import distance, KDTree
from Levenshtein import distance as lev
from parallel_flexible_clustering import hnsw_parallel
import create_text_dataset
import sklearn.datasets
from sklearn.neighbors import KDTree
import time
from itertools import pairwise
from random import random
import multiprocessing

try:
    from math import log2
except ImportError:  # Python 2.x or <= 3.2
    from math import log

    def log2(x):
        return log(x, 2)


# importlib.reload(fishdbc)
MISSING = sys.maxsize
MISSING_WEIGHT = sys.float_info.max


def split(a, n):
    k, m = divmod(len(a), n)
    indices = [k * i + min(i, m) for i in range(n + 1)]
    return [a[l:r] for l, r in pairwise(indices)]


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
        "--parallel",
        type=str,
        default="16",
        help="option to specify if we want to execute the parallel FISHDBC (specifying the number of processes from1 to 16) or single process FISHDBC (0 processes)",
    )
    parser.add_argument(
        "--test",
        type=bool,
        default=False,
        help="Option to say to perform HNSW accuracy test, works only with blob dataset and euclidean distance "
        "(default False).",
    )
    args = parser.parse_args()
    dist = args.distance.lower()
    dataset = args.dataset
    parallel = int(args.parallel)

    if dataset == "blob":
        # create the input dataset, data element for creating the hnsw, Y element for testing the search over it
        data, labels = sklearn.datasets.make_blobs(
            args.nitems, centers=args.centers, random_state=10
        )
        if args.test == True and dist == "euclidean":
            np.random.shuffle(data)
            ten_percent_index = int(0.1 * len(data))
            print(ten_percent_index)
            Y = data[-ten_percent_index:]
            data = data[:-ten_percent_index]

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
        if dist == "levensthein":

            def calc_dist(x, y):
                return lev(x, y)

        else:
            raise EnvironmentError(
                "At the moment the specified distance is not available for the string dataset,"
                " try with: levensthein"
            )
    m = 5
    m0 = 2 * m

    print("-------------------------- MULTI-PROCESS HNSW --------------------------")
    multiprocessing.set_start_method("fork")

    start = time.time()
    levels = [(int(-log2(random()) * (1 / log2(m))) + 1) for _ in range(len(data))]
    levels = sorted(enumerate(levels), key=lambda x: x[1])
    members = [[]]
    # insert the point in the list corresponding to the right level
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
    # print("Members: ", members, "\n")

    # create a list of dict to associate for each point in each levels its position
    positions = []
    for el, l in zip(members, range(len(members))):
        positions.append({})
        for i, x in enumerate(el):
            positions[l][x] = i
    # print("Positions: ", positions, "\n")
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
        np.ndarray(npArray.shape, dtype=float, buffer=shm2.buf)[:, :] = MISSING_WEIGHT
        shm_weights.append(shm2)
    manager = multiprocessing.Manager()
    lock = manager.Lock()
    # create the hnsw parallel class object and execute with pool the add function in multiprocessing
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

    start_time = time.time()
    # add the first element not in multiprocessing for correct initialization
    start_time_hnsw_par = time.time()

    num_processes = parallel
    distances_cache = []
    distances_cache.append(hnswPar.hnsw_add(0))
    pool = multiprocessing.Pool(num_processes)
    for dist_cache in pool.map(hnswPar.hnsw_add, range(1, len(hnswPar.data))):
        distances_cache.append(dist_cache)
    pool.close()
    pool.join()
    end_time_hnsw_par = time.time()
    time_parHNSW = "{:.2f}".format(end_time_hnsw_par - start_time_hnsw_par)
    print("The time of execution of Paralell HNSW is :", (time_parHNSW))
    # take the shared numpy array from the shared memory buffer and print them
    start = time.time()
    tot_adjs = []
    tot_weights = []
    for shm1, shm2, memb, i in zip(shm_adj, shm_weights, members, range(len(members))):
        adj = np.ndarray(
            shape=(len(memb), m0 if i == 0 else m), dtype=int, buffer=shm1.buf
        )
        tot_adjs.append(adj)
        weight = np.ndarray(
            shape=(len(memb), m0 if i == 0 else m), dtype=float, buffer=shm2.buf
        )
        tot_weights.append(weight)
    # print(tot_adjs, "\n", tot_weights, "\n")

    if args.test == True and dataset == "blob" and dist == "euclidean":
        print(
            "-------------------------- HNSW ACCURACY RESULTS --------------------------"
        )

        graphs_par = []
        for adjs, weights, i in zip(tot_adjs, tot_weights, range(len(tot_adjs))):
            dic = {}
            for adj, weight, pos in zip(adjs, weights, range(len(adjs))):
                dic2 = {}
                for a, w in zip(adj, weight):
                    if a == MISSING:
                        continue
                    dic2[a] = w
                idx = list(positions[i].keys())[list(positions[i].values()).index(pos)]
                dic[idx] = dic2
            graphs_par.append(dic)

        X = data
        kdt = KDTree(X, leaf_size=30, metric="euclidean")
        knn_result = kdt.query(Y, k=5, return_distance=True)
        search_res_par = [hnswPar.search(graphs_par, i, 5) for i in Y]
        # # compute the quality of the search results over the two hnsw with respect to a knn on a kdTree
        diff_el_par = 0
        diff_dist_par = 0
        for i, j, el_par in zip(
            knn_result[1],
            knn_result[0],
            search_res_par,
        ):
            for n1, d1, t_par in zip(i, j, el_par):
                n_par, d_par = t_par
                if n1 != n_par:
                    diff_el_par += 1
                if d1 != d_par:
                    diff_dist_par += 1
        print(
            "Number of error during search between parallel version and state-of-the-art,",
            diff_el_par,
            "over",
            (ten_percent_index * m),
        )

    # close and unlink the shared memory objects
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


