__author__ = "Georgios Rizos (georgerizos@iti.gr)"

import copy
import json

import numpy as np
import scipy.sparse as spsp

from thread2vec.common import get_package_path


def get_data(dataset):
    if dataset == "youtube":
        thread_data_filepath = get_package_path() + "/data_folder/uniform_data/youtube/new_thread_data.txt"
        item_to_userset_filepath = get_package_path() + "/data_folder/uniform_data/youtube/item_to_userset.txt"
        anonymize_user_filepath = get_package_path() + "/data_folder/uniform_data/youtube/anonymize_user.txt"
        popularity_filepath = get_package_path() + "/data_folder/uniform_data/youtube/youtube_popularity.txt"
        anonymous_coward_name = "REVEAL_FP7_anonymous_youtube_user"
        top_users = 200001
        total_number_of_items = 516995
    elif dataset == "reddit":
        thread_data_filepath = get_package_path() + "/data_folder/uniform_data/reddit/thread_data.txt"
        item_to_userset_filepath = get_package_path() + "/data_folder/uniform_data/reddit/item_to_userset_day.txt"
        anonymize_user_filepath = get_package_path() + "/data_folder/uniform_data/reddit/anonymize_user.txt"
        popularity_filepath = get_package_path() + "/data_folder/uniform_data/reddit/reddit_popularity.txt"
        anonymous_coward_name = "[deleted]"
        top_users = 20000
        total_number_of_items = 35844
    else:
        raise ValueError("Invalid dataset.")

    # Read popularity values.
    bad_popularity_items = list()
    popularity_matrix = np.empty((total_number_of_items, 4), dtype=np.float32)
    with open(popularity_filepath, "r") as fp:
        file_row = next(fp)

        item_counter = 0
        for file_row in fp:
            clean_row = file_row.strip().split("\t")

            if clean_row[0] == "None":
                popularity_matrix[item_counter, 0] = np.nan
                popularity_matrix[item_counter, 1] = np.nan
                popularity_matrix[item_counter, 2] = np.nan
                popularity_matrix[item_counter, 3] = np.nan
                bad_popularity_items.append(item_counter)
            else:
                popularity_matrix[item_counter, 0] = float(clean_row[0])
                popularity_matrix[item_counter, 1] = float(clean_row[1])
                popularity_matrix[item_counter, 2] = float(clean_row[2])
                popularity_matrix[item_counter, 3] = float(clean_row[3])

            item_counter += 1
    bad_popularity_items = np.array(bad_popularity_items, dtype=np.int32)

    # Read user anonymizer.
    anonymize_user = dict()
    with open(anonymize_user_filepath, "r") as fp:
        for file_row in fp:
            clean_row = file_row.strip().split("\t")

            anonymize_user[clean_row[0]] = int(clean_row[1])
    total_number_of_users = len(anonymize_user)

    true_anonymize_user = copy.copy(anonymize_user)

    user_list = list()
    for i in range(total_number_of_users):
        user_list.append(None)

    for k, v in anonymize_user.items():
        user_list[v] = k

    anonymous_coward_within_discussion = anonymize_user[anonymous_coward_name]

    # Read item to userset.
    item_to_user_row = list()
    item_to_user_col = list()

    item_to_user_matrix = spsp.coo_matrix((np.array(list(), dtype=np.int32),
                                           (np.array(list(), dtype=np.int32),
                                            np.array(list(), dtype=np.int32))),
                                          shape=(total_number_of_items,
                                                 total_number_of_users))

    item_to_user_matrix = spsp.csc_matrix(item_to_user_matrix)

    with open(item_to_userset_filepath, "r") as fp:
        counter = 0
        for file_row in fp:
            clean_row = file_row.strip().split("\t")

            for user in clean_row[1:]:
                item_to_user_row.append(int(clean_row[0]))
                item_to_user_col.append(int(user))

            counter += 1
            if counter % 10000 == 0:
                item_to_user_row = np.array(item_to_user_row, dtype=np.int32)
                item_to_user_col = np.array(item_to_user_col, dtype=np.int32)
                item_to_user_data = np.ones_like(item_to_user_row, dtype=np.int32)

                item_to_user_matrix_to_add = spsp.coo_matrix((item_to_user_data,
                                                              (item_to_user_row,
                                                               item_to_user_col)),
                                                             shape=(total_number_of_items,
                                                                    total_number_of_users))

                item_to_user_matrix_to_add = spsp.csc_matrix(item_to_user_matrix_to_add)
                item_to_user_matrix = item_to_user_matrix + item_to_user_matrix_to_add

                item_to_user_row = list()
                item_to_user_col = list()

    item_to_user_row = np.array(item_to_user_row, dtype=np.int32)
    item_to_user_col = np.array(item_to_user_col, dtype=np.int32)
    item_to_user_data = np.ones_like(item_to_user_row, dtype=np.int32)

    item_to_user_matrix_to_add = spsp.coo_matrix((item_to_user_data,
                                                  (item_to_user_row,
                                                   item_to_user_col)),
                                                 shape=(total_number_of_items,
                                                        total_number_of_users))

    item_to_user_matrix_to_add = spsp.csc_matrix(item_to_user_matrix_to_add)
    item_to_user_matrix = item_to_user_matrix + item_to_user_matrix_to_add

    if top_users is not None:
        user_to_item_distribution = item_to_user_matrix.sum(axis=0)

        user_indices_sorted = np.empty(top_users, dtype=np.int32)
        user_indices_sorted_to_add = np.argsort(user_to_item_distribution)[0, -top_users:]
        user_indices_sorted[:] = user_indices_sorted_to_add

        user_indices_sorted = user_indices_sorted[user_indices_sorted != anonymous_coward_within_discussion]

        user_indices_sorted_set = set(list(user_indices_sorted))

        filtered_item_to_user_matrix = item_to_user_matrix[:, user_indices_sorted]

        new_user_list = list()
        new_anonymize_user = dict()
        counter = 0
        for user in user_list:
            if anonymize_user[user] in user_indices_sorted_set:
                new_user_list.append(user)
                new_anonymize_user[user] = counter
                counter += 1
        user_list = new_user_list
        anonymize_user = new_anonymize_user

    else:
        top_users = total_number_of_users
        user_to_item_distribution = np.empty(top_users, dtype=np.int32)
        user_to_item_distribution[:] = item_to_user_matrix.sum(axis=0)[0, :]

        user_indices_sorted = np.arange(user_to_item_distribution.size, dtype=np.int32)
        user_indices_sorted = user_indices_sorted[user_to_item_distribution > 1]

        user_indices_sorted = user_indices_sorted[user_indices_sorted != anonymous_coward_within_discussion]

        user_indices_sorted_set = set(list(user_indices_sorted))

        filtered_item_to_user_matrix = item_to_user_matrix[:, user_indices_sorted]

        new_user_list = list()
        new_anonymize_user = dict()
        counter = 0
        for user in user_list:
            if anonymize_user[user] in user_indices_sorted_set:
                new_user_list.append(user)
                new_anonymize_user[user] = counter
                counter += 1
        user_list = new_user_list
        anonymize_user = new_anonymize_user

    # item_to_user_distribution = filtered_item_to_user_matrix.sum(axis=1)
    # item_to_user_distribution = item_to_user_distribution[item_to_user_distribution > 1]

    item_to_user_distribution = np.empty(total_number_of_items, dtype=np.int32)
    item_to_user_distribution[:] = filtered_item_to_user_matrix.sum(axis=1)[:, 0].transpose()

    item_indices_sorted = np.arange(total_number_of_items, dtype=np.int32)
    item_indices_sorted = item_indices_sorted[item_to_user_distribution > 0]

    item_indices_sorted = np.setdiff1d(item_indices_sorted, bad_popularity_items)

    filtered_item_to_user_matrix = spsp.csr_matrix(filtered_item_to_user_matrix)
    filtered_item_to_user_matrix = filtered_item_to_user_matrix[item_indices_sorted, :]

    popularity_matrix = popularity_matrix[item_indices_sorted, :]

    user_to_item_distribution = np.empty(len(anonymize_user), dtype=np.int32)
    user_to_item_distribution[:] = filtered_item_to_user_matrix.sum(axis=0)[0, :]

    user_indices_sorted = np.arange(user_to_item_distribution.size, dtype=np.int32)
    user_indices_sorted = user_indices_sorted[user_to_item_distribution > 0]

    user_indices_sorted = user_indices_sorted[user_indices_sorted != anonymous_coward_within_discussion]

    user_indices_sorted_set = set(list(user_indices_sorted))

    filtered_item_to_user_matrix = filtered_item_to_user_matrix[:, user_indices_sorted]

    new_user_list = list()
    new_anonymize_user = dict()
    counter = 0
    for user in user_list:
        if anonymize_user[user] in user_indices_sorted_set:
            new_user_list.append(user)
            new_anonymize_user[user] = counter
            counter += 1
    user_list = new_user_list
    anonymize_user = new_anonymize_user

    true_user_id_to_user_id = dict()
    for user in user_list:
        k = true_anonymize_user[user]
        v = anonymize_user[user]
        true_user_id_to_user_id[k] = v

    index_1 = int(np.ceil(filtered_item_to_user_matrix.shape[0] * 0.5))
    index_2 = int(np.ceil(filtered_item_to_user_matrix.shape[0] * 0.75))

    index_permutation = np.random.permutation(np.arange(filtered_item_to_user_matrix.shape[0], dtype=np.int32))

    train = index_permutation[:index_1]
    val = index_permutation[index_1:index_2]
    test = index_permutation[index_2:]

    data_splits = (train, val, test)

    data = dict()
    data["filtered_item_to_user_matrix"] = filtered_item_to_user_matrix
    data["popularity_matrix"] = popularity_matrix
    data["item_indices_sorted"] = item_indices_sorted
    data["anonymize_user"] = anonymize_user
    data["true_user_id_to_user_id"] = true_user_id_to_user_id
    data["user_list"] = user_list
    data["number_of_items"] = filtered_item_to_user_matrix.shape[0]
    data["number_of_users"] = filtered_item_to_user_matrix.shape[1]
    data["data_splits"] = data_splits

    return data


def read_indices(dataset):
    if dataset == "youtube":
        indices_filepath = get_package_path() + "/data_folder/uniform_data/youtube/data_splits.txt"
    elif dataset == "reddit":
        indices_filepath = get_package_path() + "/data_folder/uniform_data/reddit/data_splits.txt"
    else:
        raise ValueError

    with open(indices_filepath, "r") as fp:
        file_row = next(fp)

        clean_row = file_row.strip().split("\t")

        train_size = int(clean_row[0])
        val_size = int(clean_row[1])
        test_size = int(clean_row[2])

        indices = np.empty(train_size + val_size + test_size, dtype=np.int32)

        i = 0
        for file_row in fp:
            clean_row = file_row.strip()
            indices[i] = int(clean_row)

            i += 1

        train = indices[:train_size]
        val = indices[train_size:train_size + val_size]
        test = indices[train_size + val_size:]

    return train, val, test


def clean_thread_data(dataset):
    if dataset == "youtube":
        thread_data_filepath = get_package_path() + "/data_folder/uniform_data/youtube/thread_data.txt"
        new_thread_data_filepath = get_package_path() + "/data_folder/uniform_data/youtube/new_thread_data.txt"
        # item_to_userset_filepath = get_package_path() + "/data_folder/uniform_data/youtube/item_to_userset.txt"
        # anonymize_user_filepath = get_package_path() + "/data_folder/uniform_data/youtube/anonymize_user.txt"
        # popularity_filepath = get_package_path() + "/data_folder/uniform_data/youtube/youtube_popularity.txt"
        # anonymous_coward_name = "REVEAL_FP7_anonymous_youtube_user"
        # top_users = 200001
        # total_number_of_items = 516995
    elif dataset == "reddit":
        thread_data_filepath = get_package_path() + "/data_folder/uniform_data/reddit/thread_data.txt"
        new_thread_data_filepath = get_package_path() + "/data_folder/uniform_data/reddit/new_thread_data.txt"
        # item_to_userset_filepath = get_package_path() + "/data_folder/uniform_data/reddit/item_to_userset_day.txt"
        # anonymize_user_filepath = get_package_path() + "/data_folder/uniform_data/reddit/anonymize_user.txt"
        # popularity_filepath = get_package_path() + "/data_folder/uniform_data/reddit/reddit_popularity.txt"
        # anonymous_coward_name = "[deleted]"
        # top_users = 20000
        # total_number_of_items = 35844
    else:
        raise ValueError("Invalid dataset.")

    data = get_data(dataset=dataset)
    valid_users = set(list(data["true_user_id_to_user_id"].keys()))

    item_id_set = set(list(data["item_indices_sorted"]))

    with open(thread_data_filepath, "r") as i_fp:
        with open(new_thread_data_filepath, "w") as o_fp:
            for file_row in i_fp:
                item = json.loads(file_row.strip())

                item_id = int(item["item_id"])

                if item_id in item_id_set:

                    valid_comments = list()

                    comment_to_user = list()
                    comment_to_commentlist = list()
                    comment_to_timestamp = list()

                    i_to_i = dict()
                    ii = 0
                    for i in range(len(item["comment_to_user"])):
                        if int(item["comment_to_user"][i]) in valid_users:
                            valid_comments.append(i)
                            i_to_i[i] = ii
                            ii += 0
                    valid_comments = set(valid_comments)
                    for i in range(len(item["comment_to_user"])):
                        if i in valid_comments:
                            comment_to_user.append(item["comment_to_user"][i])
                            comment_to_commentlist.append(item["comment_to_commentlist"][i])
                            for j, reply in enumerate(comment_to_commentlist[-1]):
                                comment_to_commentlist[-1][j] = i_to_i[j]

                            comment_to_timestamp.append(item["comment_to_timestamp"][i])

                    item["comment_to_user"] = comment_to_user
                    item["comment_to_commentlist"] = comment_to_commentlist
                    item["comment_to_timestamp"] = comment_to_timestamp

                    json.dump(item, o_fp)
                    o_fp.write("\n")
