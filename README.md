# **Parallel Flexible Clustering**

## **Master Thesis in Computer Science - December 2023** 

### - Thesis Student: Edoardo Pastorino, 5169595

### - Supervisors: Matteo Dell'Amico, Daniele D'Agostino
### - Reviewer: Vito Paolo Pastore


## Introduction:
My master thesis in computer science is related to the implementation of a parallel clustering methodology, the Parallel Flexible Clustering algorithm. This concurrent version is based on the FISHDBC algorithm, originally implemented by the Professor Dell'Amico. As the FISHDBC, also the Parallel Flexible Clustering strictly depends on some main data structures and methods: the HNSW, a graph-based data structure used in approximate nearest neighbor search; the MST, a tree that spans all the vertices in the graph while minimizing the total weight of the edges; the HDBSCAN clustering, designed to perform robust clustering of data points based on their density. The main task of the thesis is to try to parallelize such FISHDBC algorithm to reduce the final execution time, after a careful analysis of the original single process algorithm and after a meticulous study of the concepts upon which it is based. My contribution allows to obtain a lock-free working parallel solution, without locks because we are talking about an approximated algorithm, thus it is possible to avoid the usage of a synchronization system, saving time and resources. But with my parallel implementation we can also obtain a correct version of only the HNSW data structure in a fast way. 


## Project Structure:

```

.

├── parallel-flexible-clustering-master

│   ├── parallel_flexible_clustering

│       ├── __init__.py

│       ├── create_text_dataset.py

│       ├── parallel_fishdbc_example.py
    
│       ├── fishdbc.py

│       ├── hnsw_parallel.py

│       ├── hnsw.py

│       ├── parallel_hnsw_example.py

│       ├── unionfind.c

```

- **`parallel-flexible-clustering-master`**,  , 

- **`parallel_flexible_clustering`**,  (*`__init__.py`*, *`create_text_dataset.py`*, *`parallel_fishdbc_example.py`*, *`fishdbc.py`*, *`hnsw_parallel.py`*, *`hnsw.py`*, *`test_parallel_HNSW`*, *`unionfind.c`*  and *`unionfind.pyx`*).

## Insallation & Execution:

In this appendix section we will see some basic instructions to install everything you need to execute the FISHDBC algorithm on a Linux machine.
First of all you have to download or clone the GitHub repository: 
```
clone https://github.com/edo-pasto/Parallel-FISHDBC.git
```
Secondly, I really suggest to create a virtual environment, or with conda or with pip as package manager (in the following instructions i will use conda, but with pip the process is very similar and you can check \cite{pipInstr}):
```
conda create --name myenv python=3.10.12
```
After the creation of the environment (the suggestion is to create the environment with python version 3.10.12 or 3.11.3) we can activate it with:
```
conda activate myenv
```
Now that the preliminary operations are done we can start to install all the dependencies needed by the algorithm to work. These first packages can be installed directly with one single conda install command:
```
conda install numpy scipy pandas scikit-learn numba matplotlib   
```
The package allowing us to use the Leveshtein distance should be installed from a different conda source:
```
conda install -c conda-forge levenshtein
```
Also Cython is installed with a different command:
```
conda install -c anaconda cython
```
The last package to be installed is the one associated to the HDBSCAN, this time installed with pip (is not possible with conda):
```
pip install hdbscan
```
The last installation step is to execute the setup.py file in the following way:
```
python3 setup.py install
```
Now we will see how to execute experiments of the algorithm launching the fishdbc\_example.py file.
If you want to execute a classical run of the parallel FISHDBC you should type:
```
python3 fishdbc_example.py --dataset blob --nitems 10000 --centers 5 --distance euclidean --parallel 16
```
where you can specify the data set that you want to use as input (blob or text), the number of items of the input data set (--nitems), the numbers of the data set's centroids (--centers), depending on the type of data the distance to be used (the available distances are: euclidean, sqeuclidean, minkowski, cosine and levenshtein for text data) and the number of processes to use (--parallel 0 means original single process FISHDBC, --parallel > 0 means multi process, you can pass from 1 to 16 processes).    
If you want to execute the parallel FISHDBC with text data as input you can write:
```
python3 fishdbc_example.py --dataset text --distance levenshtein --nitems 1000 --centers 10 --parallel 16
```
If, instead, you desire to execute the original single process FISHDBC you can also write:
```
python3 fishdbc_example.py --dataset blob --nitems 1000 --centers 5 --distance euclidean 
```
It is possible to take a look at the accuracy of the final parallel (but also the single process) FISHDBC clustering enabling the --test option:
```
python3 fishdbc_example.py --dataset blob --nitems 10000 --centers 5 --distance euclidean --parallel 16 --test True
```
There is also the opportunity to execute an example of only the parallel HNSW creation (again with the possibility to perform some accuracy tests thank to the --test True option):
```
python3 test_parallel_HNSW.py --dataset blob --distance euclidean --parallel 16 --nitems 10000 --centers 5
```
Of course you can execute the parallel HNSW also for textual data (but in this case you cannot perform any kind of test)
```
python3 test_parallel_HNSW.py --dataset text --distance levenshtein --nitems 1000 --centers 10 --parallel 16
```

