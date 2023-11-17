Flexible clustering
===================

A project for scalable hierachical clustering, thanks to a Flexible,
Incremental, Scalable, Hierarchical Density-Based Clustering
algorithms (FISHDBC, for the friends).

This package lets you use an arbitrary dissimilarity function you write (or reuse from somebody else's work!) to cluster
your data.

Please see the paper at https://arxiv.org/abs/1910.07283

Dependencies
------------

* Python 3
* Cython
* hdbscan: https://github.com/scikit-learn-contrib/hdbscan
* scipy: https://www.scipy.org/


Installation
------------

    python3 setup.py install

A projects allowing scalable hierarchical clustering, thanks to an
approximated version of OPTICS, on arbitrary data and distance measures.

Quickstart
----------

Look at the HDBSCAN documentation for the meaning of the return values
of the `cluster` method.  There are plenty of configuration options,
inherited by HNSWs and HDBSCAN, but the only compulsory argument is a
dissimilarity function between arbitrary data elements::

    import parallel_flexible_clustering
    
    clusterer = parallel_flexible_clustering.FISHDBC(my_dissimilarity)
    for elem in my_data:
        clusterer.add(elem)
    labels, probs, stabilities, condensed_tree, slt, mst = clusterer.cluster()

    for elem in some_new_data: # support cheap incremental clustering
        clusterer.add(elem)
    # new clustering according to the newly available data
    labels, probs, stabilities, condensed_tree, slt, mst = clusterer.cluster()

Make sure to run everything from *outside* the source directory, to
avoid confusing Python path.

Demo/Example
------------

Look at the fishdbc_example.py file for something more (it requires
matplotlib to be run).

Want More Info?
---------------

Send me an email at `della@linux.it`. I'll improve the
docs as and if people use this.
    
Author
------

Matteo Dell'Amico

Copyright
---------

BSD 3-clause; see the LICENSE file.


