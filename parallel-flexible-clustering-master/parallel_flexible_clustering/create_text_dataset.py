from random import choice, randrange
import numpy as np
import pandas as pd

letters = "abcdefghijklmnopqrstuvwxyz"


def gen_value(size: int) -> list[str]:
    return [choice(letters) for _ in range(size)]


def edit(s: list[str], n_edits: int) -> list[str]:
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
    res = []
    labels = []
    for i in range(n_centroids):
        for _ in range(samples_per_cluster):
            labels.append(i)
    return labels


if __name__ == "__main__":
    tot_labels = []
    for cluster in gen_dataset(5, 20, 20000, 4):
        for string in cluster:
            with open("../data/textDataset100.csv", "a") as text_file:
                text_file.write(string + "\n")

    for labels in gen_labels(5, 20000):
        tot_labels.append(labels)
        with open("../data/textDatasetLabels100.csv", "a") as text_file:
            text_file.write(str(labels) + "\n")

    print(tot_labels)


