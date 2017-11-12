__author__ = "Georgios Rizos (georgerizos@iti.gr)"

try:
    import cPickle
except ImportError:
    import pickle as cPickle

import numpy as np
import scipy.sparse as spsp
from sklearn.cluster import KMeans

from thread2vec.common import get_package_path


def read_weights(dataset):
    if dataset == "youtube":
        base_model_filepath = get_package_path() + "/data_folder/models/youtube_model.pkl"
    elif dataset == "reddit":
        base_model_filepath = get_package_path() + "/data_folder/models/reddit_model.pkl"
    else:
        raise ValueError("Invalid dataset.")

    file_path_list = list()
    for i in range(3):
        file_path_list.append(base_model_filepath + "." + repr(i))

    fin = open(file_path_list[0], "rb")
    params = cPickle.load(fin)
    fin.close()

    user_embeddings = params[0]

    return user_embeddings


def make_features_vlad(dataset,
                       number_of_vlad_clusters,
                       filtered_item_to_user_matrix,
                       user_id_set,
                       do_power_norm,
                       do_l2_norm):
    user_embeddings = read_weights(dataset)

    number_of_items = filtered_item_to_user_matrix.shape[0]
    number_of_users = user_embeddings.shape[0]
    embedding_size = user_embeddings.shape[1]

    item_to_user_array = get_item_to_user_array(filtered_item_to_user_matrix, user_id_set)

    # K-means on user embeddings.
    community_dictionary = KMeans(n_clusters=number_of_vlad_clusters,
                                  init='k-means++',
                                  tol=0.0001,
                                  random_state=0).fit(user_embeddings)

    # Aggregation for all the items.
    centers = community_dictionary.cluster_centers_

    X = np.zeros([number_of_items, number_of_vlad_clusters * embedding_size], dtype=np.float32)

    for item_id in range(number_of_items):

        user_ids = item_to_user_array[item_id]

        item_user_embeddings = user_embeddings[user_ids, :]

        predictedLabels = community_dictionary.predict(item_user_embeddings)

        for centroid in range(number_of_vlad_clusters):
            # if there is at least one descriptor in that cluster
            if np.sum(predictedLabels == centroid) > 0:
                # add the diferences
                X[item_id, centroid * embedding_size:(centroid + 1) * embedding_size] = np.sum(
                    item_user_embeddings[predictedLabels == centroid, :] - centers[centroid], axis=0)

        # power normalization, also called square-rooting normalization
        if do_power_norm:
            X[item_id, :] = np.sign(X[item_id, :]) * np.sqrt(np.abs(X[item_id, :]))

        # L2 normalization
        if do_l2_norm:
            X[item_id, :] = X[item_id, :] / np.sqrt(np.dot(X[item_id, :], X[item_id, :]))

    return X


def make_features_mean(dataset,
                       filtered_item_to_user_matrix,
                       user_id_set,
                       do_power_norm,
                       do_l2_norm):
    user_embeddings = read_weights(dataset)

    number_of_items = filtered_item_to_user_matrix.shape[0]
    number_of_users = user_embeddings.shape[0]
    embedding_size = user_embeddings.shape[1]

    item_to_user_array = get_item_to_user_array(filtered_item_to_user_matrix, user_id_set)

    X = np.zeros([number_of_items, embedding_size], dtype=np.float32)

    for item_id in range(number_of_items):

        user_ids = item_to_user_array[item_id]

        item_user_embeddings = user_embeddings[user_ids, :]

        X[item_id, :] = np.mean(item_user_embeddings, axis=0)

        # power normalization, also called square-rooting normalization
        if do_power_norm:
            X[item_id, :] = np.sign(X[item_id, :]) * np.sqrt(np.abs(X[item_id, :]))

        # L2 normalization
        if do_l2_norm:
            X[item_id, :] = X[item_id, :] / np.sqrt(np.dot(X[item_id, :], X[item_id, :]))

    return X


def get_item_to_user_array(item_to_user, user_id_set):
    item_to_user = spsp.csr_matrix(item_to_user)

    item_to_user_array = np.ndarray(item_to_user.shape[0], dtype=np.ndarray)

    for i in range(item_to_user.shape[0]):
        array_row_indices = item_to_user.getrow(i).indices
        if array_row_indices.size > 0:

            array_row_indices = np.array([user for user in array_row_indices if user in user_id_set],
                                         dtype=np.int32)

            item_to_user_array[i] = array_row_indices
        else:
            raise ValueError

    return item_to_user_array


# TODO: Store the features.

