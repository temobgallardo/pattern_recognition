import enum
import math
import numpy as np
from collections import defaultdict


"""
Linde-Buzo-Gray / Generalized Lloyd algorithm implementation in Python *3*.
Heuristic process that can be used to generate cluster points from a big amount of multidimensional vectors.
"""


def lbg(signal, distance_function, number_regions=1, epsilon=0.001):
    """
    Generate codebook of size <number_regions> with convergence value <epsilon>.
    :param signal: input data with N k-dimensional vectors
    :param distance_function: a function that computes the distance
    :param number_regions: the number of regions or centroid to compute
    :param epsilon: convergence value
    :return list of: signal with their regions assigned
    :return list of: of code-values
    """

    old_distortion = 0
    codebook = [[None, None]]
    regions = np.full(
        (len(signal), len(signal[0])), compute_first_centroid(signal))
    # Generate the first centroid
    regions[0] = compute_first_centroid(signal)
    codebook[0] = regions[0]
    MAX_NUMBER_OF_RUNS = 20

    # Generate new centroids and cluster the vectors
    while len(codebook) < number_regions:
        regions, codebook, initial_distortion = generate_new_centroids(
            signal, regions, distance_function, epsilon, old_distortion, codebook)

        number_of_runs = 0
        mse = initial_distortion/len(signal)
        new_distortion = initial_distortion
        while number_of_runs < MAX_NUMBER_OF_RUNS:
            old_mse = mse
            old_distortion = new_distortion
            regions, codebook = cluster_vectors_to_nearest_centroid(
                signal, regions, codebook)
            regions, new_distortion = recompute_regions_to_centralize_centroid(
                signal, regions, distance_function, new_distortion, codebook)

            old_distortion = old_distortion if old_distortion > 0 else initial_distortion
            mse = math.sqrt((new_distortion - old_distortion)**2)/len(signal)

            # if not np.isclose(old_mse, mse):
            #     assert mse <= old_mse, 'uh-oh, MSE increased'

            if old_mse == 0.0:  # avoid divide by zero below
                break
            if (np.abs(old_mse - mse) / old_mse) < epsilon:
                break

            number_of_runs += 1

    return regions


def generate_new_centroids(signal, regions, distance_fuction, epsilon, global_distance, codebook):
    regions = np.copy(regions)
    codebook_new = []

    # generating two new centroids based on the ones created before, at the end we will has 2 times more codes
    for cb in codebook:
        c1 = cb + epsilon
        c2 = cb - epsilon

        # increasing the number of vectors in power of 2
        codebook_new.append(c1)
        codebook_new.append(c2)
        for s_i, vector in enumerate(signal):
            # the vectors that does not belong to region cb are not used for those new centroids
            if not (regions[s_i] == cb).all():
                continue

            d1 = distance_fuction(vector, c1)
            d2 = distance_fuction(vector, c2)

            if d1 < d2:
                regions[s_i] = c1
                global_distance = global_distance + d1
            else:
                regions[s_i] = c2
                global_distance = global_distance + d2

    # print(codebook)
    # print(codebook_new)
    return regions, codebook_new, global_distance


def recompute_regions_to_centralize_centroid(signal, regions, distance_fuction, global_distance, codebook):
    # only one region computed
    if len(regions) == 1:
        return [signal[0], 1], 0

    regions = np.copy(regions)

    for i, vector in enumerate(signal):
        min_dist = None
        min_dist_centroid = None

        for cb in codebook:
            d = distance_fuction(vector, cb)

            if min_dist is None or min_dist > d:
                min_dist = d
                min_dist_centroid = cb

        global_distance += min_dist
        regions[i] = min_dist_centroid

    return regions, global_distance


def compute_first_centroid(signal):
    # print(signal)
    dimension = len(signal[0])
    vectors = len(signal)
    centroids = []*dimension

    for i in range(dimension):
        given_dimension_sum = sum(signal[:, i])
        centroids.append(given_dimension_sum / vectors)

    return centroids


def cluster_vectors_to_nearest_centroid(signal, regions, codebook):
    """
    This will re-compute centroids based on <regions> set to each vector so they are at the center of their vector subset of <signal> data

    signal[0] -> regions[0] 
    signal[1] -> regions[1]*

    *take into consideration that each row/data in signal is k-dimensional.

    which means vector at <signal>[0] has a <region>[0]

    now <centroids_dics> is a dictionary holding the vectors that belongs to it on its first columns and in the second it holds a counter of this number 
    """
    signal = np.copy(signal)
    regions = np.copy(regions)
    # Stores all vectors given a centroid (which indicates a region)
    centroids_dic = defaultdict(list)
    dimension = len(signal[0])

    # Stores a key which is old centroid and new computed centroids given vectors in the region
    new_centroid_dic = defaultdict(list)
    for i, v in enumerate(codebook):
        code_key = tuple(v)
        centroids_dic[code_key]
        new_centroid_dic[code_key]

    for i, v in enumerate(signal):
        code_key = tuple(regions[i])
        centroids_dic[code_key].append(v)

    # Moving the centroids to the center of a region
    for key, vectors_near_centroid in centroids_dic.items():
        if not vectors_near_centroid:
            print("List is empty for centroid " + str(key))
            continue

        # Have an array of vectors: [[x1, y], [x2, y2], [x3, y3], ..., [xm, ym]]
        # transposing so we have [[x1, x2, x3, ..., xm], [y1, y2, y3, ..., ym]]
        current_vectors_in_centroid = np.array(vectors_near_centroid).T
        num_vectors_near_crnt_centrod = len(current_vectors_in_centroid[0])
        sum_current_vectors_in_centroids = [
            sum(current_vectors_in_centroid[:][i_d]) for i_d in range(dimension)]
        centroid = np.divide(
            sum_current_vectors_in_centroids, num_vectors_near_crnt_centrod)

        new_centroid_dic[key] = centroid

    for i, centroid in enumerate(regions):
        code_key = tuple(centroid)
        regions[i] = new_centroid_dic[code_key]

    codebook = [centroid if len(centroid) != 0 else np.array(key)
                for key,
                centroid in new_centroid_dic.items()]

    return regions, codebook


def euclid_squared(a, b):
    return math.sqrt(sum((x_a - x_b) ** 2 for x_a, x_b in zip(a, b)))


def itakura_saito(a, b):
    division = np.divide(a, b)
    log_division = math.log(division)
    return division - log_division - 1
