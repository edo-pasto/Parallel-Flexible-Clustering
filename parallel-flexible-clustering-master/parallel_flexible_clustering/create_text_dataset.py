"""This script allows to randomly create a synth text data set,
starting with a string as seed (the variable letters).
This file can also be imported as a module and contains the following
functions:

    * gen_value - returns the centroid
    * edit - return the edited original centroid string, modified by a certain number of edits 
    * gen_dataset - return the text generated dataset
    * gen_labels - return the labels of the associated text generated dataset
    * main - the main function of the scripts that, if you execute it, starts the generation of the text dataset and its labels
"""

from random import choice, randrange

letters = "abcdefghijklmnopqrstuvwxyz"


def gen_value(size: int) -> "list[str]":
    """Function to generate a centroid for the text dataset generation
    Parameters
    ----------
    size: int
        the size of the string that will be a centroid

    Returns
    -------
        a string that represents the centroid
    """
    return [choice(letters) for _ in range(size)]


def edit(s: "list[str]", n_edits: int) -> "list[str]":
    """Function to edit the original string
    Parameters
    ----------
    s: list[str]
        the original string
    n_edits:
        the number of allowed edits to modify the original string

    Returns
    -------
    s: list[str]
        the edited original string
    """
    for _ in range(n_edits):
        if not s:
            s = [choice(letters)]
            continue
        action = randrange(3)
        pos = randrange(len(s))
        if action == 0:  # delete a character
            del s[pos]
        elif action == 1:  # substitute a character
            s[pos] = choice(letters)
        else:  # add a character
            s[pos:pos] = [choice(letters)]
    return s


def gen_dataset(
    n_centroids: int, centroid_size: int, samples_per_cluster: int, n_edits: int
):
    """Function to generate the clusters of the text dataset
    Parameters
    ----------
    n_centroids: int
        the number of used centroids for the text dataset generation
    centroid_size: int
        the length of the centroid string
    samples_per_cluster: int
        the number of elements that belongs to each cluster
    n_edits: int
        the number of allowed edits to modify the original string

    Returns
    -------
    res: list[str]
        the cluster of strings
    """
    res = []
    labels = []
    for i in range(n_centroids):
        centroid = gen_value(centroid_size)
        res.append(
            [
                "".join(edit(centroid.copy(), n_edits))
                for _ in range(samples_per_cluster)
            ]
        )
        for _ in range(samples_per_cluster):
            labels.append(i)
    return res


def gen_labels(n_centroids: int, samples_per_cluster: int):
    """Function to generate the labels of the text dataset
    Parameters
    ----------
    n_centroids: int
        the number of used centroids for the text dataset generation
    samples_per_cluster: int
        the number of element that belongs to each cluster

    Returns
    -------
    res: list[int]
        the arrays of the leabels of the text dataset
    """
    labels = []
    for i in range(n_centroids):
        for _ in range(samples_per_cluster):
            labels.append(i)
    return labels


# if __name__ == "__main__":
#     tot_labels = []
#     for cluster in gen_dataset(5, 20, 20000, 4):
#         for string in cluster:
#             try:
#                 with open("namefile_data.csv", "a") as text_file:
#                     text_file.write(string + "\n")
#             except FileNotFoundError as e:
#                 print(e)

#     for labels in gen_labels(5, 20000):
#         tot_labels.append(labels)
#         try:
#             with open("namefile_labels.csv", "a") as text_file:
#                 text_file.write(str(labels) + "\n")
#         except FileNotFoundError as e:
#             print(e)