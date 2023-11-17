# **Parallel Flexible Clustering**

## **Master Thesis in Computer Science - December 2023** 

### - Thesis Student: Edoardo Pastorino, 5169595

### - Supervisors: Matteo Dell'Amico, Daniele D'Agostino
### - Reviewer: Vito Paolo Pastore


## Introduction:
My master thesis in computer science is related to the implementation of a parallel clustering methodology, the Parallel Flexible Clustering algorithm. This concurrent version is based on the FISHDBC algorithm, originally implemented by the Professor Dell'Amico. As the FISHDBC, also the Parallel Flexible Clustering strictly depends on some main data structures and methods: the HNSW, a graph-based data structure used in approximate nearest neighbor search; the MST, a tree that spans all the vertices in the graph while minimizing the total weight of the edges; the HDBSCAN clustering, designed to perform robust clustering of data points based on their density. The main task of the thesis is to try to parallelize such FISHDBC algorithm to reduce the final execution time, after a careful analysis of the original single process algorithm and after a meticulous study of the concepts upon which it is based. My contribution allows to obtain a lock-free working parallel solution, without locks because we are talking about an approximated algorithm, thus it is possible to avoid the usage of a synchronization system, saving time and resources. But with my parallel implementation we can also obtain a correct version of only the HNSW data structure in a fast way. The procedure used to speed-up the algorithm is the multi-process approach, splitting the execution between many processes that works simultaneously, exploiting the cores of the reference machine. Everything was done inside the Python environment, hence the reason for the multi-processes instead of the multi-threading (since in python the GIL blocks the effectiveness of the multi-threads in many cases). Additionally, the Parallel Flexible Clustering algorithm is completely wrote in Python, without dependencies to other languages and it is very easy to use and to customize (with arbitrary Python user defined distance metrics used to compute similarity among data, but also it can works with both numerical and textual data). At the end of the work we have obtained the results we wanted. For all the two algorithm's main bottlenecks, discovered after profiling the code, the HNSW and MST creation, we were able to significantly reduce their running time, as for the overall FISHDBC procedure. With our thesis we have discovered more: we can assert that, in general, could be a good idea to implement concurrent approximated algorithm without synchronization, saving usually a lot of time and resources, while the quality of the outcomes could remain excellent. We believe that, in a situation in which it is need to perform a fast clustering operation in Python, our implementation could be useful, but also when there is the necessity to obtain a HNSW structure in a simple and quick way. 


## Project Structure:

```

.

├── parallel-flexible-clustering-master

│   ├── parallel_flexible_clustering

│       ├── __init__.py

│       ├── create_text_dataset.py

│       ├── fishdbc_example.py
    
│       ├── fishdbc.py

│       ├── hnsw_parallel.py

│       ├── hnsw.py

│       ├── test_parallel_HNSW.py

│       ├── unionfind.c


