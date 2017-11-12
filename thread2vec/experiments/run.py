__author__ = "Georgios Rizos (georgerizos@iti.gr)"

import numpy as np
from sklearn.linear_model import LinearRegression

from thread2vec.representation.aggregation import make_features_vlad, make_features_mean
from thread2vec.representation.utility import get_data, read_indices
from thread2vec.common import get_package_path


def mean_versus_vlad_aggregation(dataset):
    method_names = list()
    vlad_parameters = [2, 3, 4, 5, 6, 7, 8, 9, 10, 20, 30]
    results_list = list()

    train, val, test = read_indices(dataset)

    train = np.append(train, val)

    data = get_data(dataset)

    y = data["popularity_matrix"]
    y_train = y[train, 2]
    y_test = y[test, 2]

    ####################################################################################################################
    # Mean
    ####################################################################################################################
    method_name = "mean"

    method_names.append(method_name + "")
    X = make_features_mean(dataset,
                           filtered_item_to_user_matrix=data["filtered_item_to_user_matrix"],
                           user_id_set=set(list(data["true_user_id_to_user_id"].values())),
                           do_power_norm=False,
                           do_l2_norm=False)

    X_train = X[train, :]
    X_test = X[test, :]

    model = LinearRegression().fit(X_train, y_train)

    y_pred = model.predict(X_test)

    loss = np.mean(np.power(y_pred - y_test, 2))

    print(loss)
    results_list.append(loss)

    method_names.append(method_name + "_pnorm")
    X = make_features_mean(dataset,
                           filtered_item_to_user_matrix=data["filtered_item_to_user_matrix"],
                           user_id_set=set(list(data["true_user_id_to_user_id"].values())),
                           do_power_norm=False,
                           do_l2_norm=True)

    X_train = X[train, :]
    X_test = X[test, :]

    model = LinearRegression().fit(X_train, y_train)

    y_pred = model.predict(X_test)

    loss = np.mean(np.power(y_pred - y_test, 2))

    print(loss)
    results_list.append(loss)

    method_names.append(method_name + "_l2norm")
    X = make_features_mean(dataset,
                           filtered_item_to_user_matrix=data["filtered_item_to_user_matrix"],
                           user_id_set=set(list(data["true_user_id_to_user_id"].values())),
                           do_power_norm=True,
                           do_l2_norm=False)

    X_train = X[train, :]
    X_test = X[test, :]

    model = LinearRegression().fit(X_train, y_train)

    y_pred = model.predict(X_test)

    loss = np.mean(np.power(y_pred - y_test, 2))

    print(loss)
    results_list.append(loss)

    method_names.append(method_name + "_allnorm")
    X = make_features_mean(dataset,
                           filtered_item_to_user_matrix=data["filtered_item_to_user_matrix"],
                           user_id_set=set(list(data["true_user_id_to_user_id"].values())),
                           do_power_norm=True,
                           do_l2_norm=True)

    X_train = X[train, :]
    X_test = X[test, :]

    model = LinearRegression().fit(X_train, y_train)

    y_pred = model.predict(X_test)

    loss = np.mean(np.power(y_pred - y_test, 2))

    print(loss)
    results_list.append(loss)

    ####################################################################################################################
    # VLAD
    ####################################################################################################################
    method_name = "vlad"

    for vlad_clusters in vlad_parameters:

        method_names.append(method_name + repr(vlad_clusters) + "")
        X = make_features_vlad(dataset,
                               number_of_vlad_clusters=vlad_clusters,
                               filtered_item_to_user_matrix=data["filtered_item_to_user_matrix"],
                               user_id_set=set(list(data["true_user_id_to_user_id"].values())),
                               do_power_norm=False,
                               do_l2_norm=False)

        X_train = X[train, :]
        X_test = X[test, :]

        model = LinearRegression().fit(X_train, y_train)

        y_pred = model.predict(X_test)

        loss = np.mean(np.power(y_pred - y_test, 2))

        print(loss)
        results_list.append(loss)

        method_names.append(method_name + repr(vlad_clusters) + "_pnorm")
        X = make_features_vlad(dataset,
                               number_of_vlad_clusters=vlad_clusters,
                               filtered_item_to_user_matrix=data["filtered_item_to_user_matrix"],
                               user_id_set=set(list(data["true_user_id_to_user_id"].values())),
                               do_power_norm=True,
                               do_l2_norm=False)

        X_train = X[train, :]
        X_test = X[test, :]

        model = LinearRegression().fit(X_train, y_train)

        y_pred = model.predict(X_test)

        loss = np.mean(np.power(y_pred - y_test, 2))

        print(loss)
        results_list.append(loss)

        method_names.append(method_name + repr(vlad_clusters) + "_l2norm")
        X = make_features_vlad(dataset,
                               number_of_vlad_clusters=vlad_clusters,
                               filtered_item_to_user_matrix=data["filtered_item_to_user_matrix"],
                               user_id_set=set(list(data["true_user_id_to_user_id"].values())),
                               do_power_norm=False,
                               do_l2_norm=True)

        X_train = X[train, :]
        X_test = X[test, :]

        model = LinearRegression().fit(X_train, y_train)

        y_pred = model.predict(X_test)

        loss = np.mean(np.power(y_pred - y_test, 2))

        print(loss)
        results_list.append(loss)

        method_names.append(method_name + repr(vlad_clusters) + "_allnorm")
        X = make_features_vlad(dataset,
                               number_of_vlad_clusters=vlad_clusters,
                               filtered_item_to_user_matrix=data["filtered_item_to_user_matrix"],
                               user_id_set=set(list(data["true_user_id_to_user_id"].values())),
                               do_power_norm=True,
                               do_l2_norm=True)

        X_train = X[train, :]
        X_test = X[test, :]

        model = LinearRegression().fit(X_train, y_train)

        y_pred = model.predict(X_test)

        loss = np.mean(np.power(y_pred - y_test, 2))

        print(loss)
        results_list.append(loss)

    with open(get_package_path() + "/data_folder/uniform_data/" + dataset + "/aggregation_benchmark.txt", "w") as fp:
        for name, loss in zip(method_names, results_list):
            fp.write(name + "\t" + repr(loss) + "\n")

    method_names = np.array(method_names)
    results_list = np.array(results_list)

    indices_sorted = np.argsort(results_list)
    print(method_names[indices_sorted])
    print(results_list[indices_sorted])


def handcrafted_features_versus_aggregation_comparison(dataset, vlad_clusters):
    method_names = list()
    results_list = list()

    train, val, test = read_indices(dataset)

    train = np.append(train, val)

    data = get_data(dataset)

    y = data["popularity_matrix"]
    y_train = y[train, 2]
    y_test = y[test, 2]

    handcrafted_parameters = ["hour", "day", "week", "final"]

    X = make_features_vlad(dataset,
                           number_of_vlad_clusters=vlad_clusters,
                           filtered_item_to_user_matrix=data["filtered_item_to_user_matrix"],
                           user_id_set=set(list(data["true_user_id_to_user_id"].values())),
                           do_power_norm=True,
                           do_l2_norm=False)

    for star in handcrafted_parameters:
        handcrafted_features = np.load(get_package_path() + "/data_folder/uniform_data/" + dataset + "/features_" + star + ".npy")

        method_names.append(star + "")
        X_train = handcrafted_features[train, :]
        X_test = handcrafted_features[test, :]

        model = LinearRegression().fit(X_train, y_train)

        y_pred = model.predict(X_test)

        loss = np.mean(np.power(y_pred - y_test, 2))

        print(loss)
        results_list.append(loss)

        method_names.append(star + "vlad" + repr(vlad_clusters))
        X = np.hstack([X, handcrafted_features])

        X_train = X[train, :]
        X_test = X[test, :]

        model = LinearRegression().fit(X_train, y_train)

        y_pred = model.predict(X_test)

        loss = np.mean(np.power(y_pred - y_test, 2))

        print(loss)
        results_list.append(loss)

    with open(get_package_path() + "/data_folder/uniform_data/" + dataset + "/handcrafted_benchmark.txt", "w") as fp:
        for name, loss in zip(method_names, results_list):
            fp.write(name + "\t" + repr(loss) + "\n")

    method_names = np.array(method_names)
    results_list = np.array(results_list)

    indices_sorted = np.argsort(results_list)
    print(method_names[indices_sorted])
    print(results_list[indices_sorted])


# def best_combo():
#     ####################################################################################################################
#     # YouTube
#     ####################################################################################################################
#
#     ####################################################################################################################
#     # Reddit
#     ####################################################################################################################


if __name__ == "__main__":
    # mean_versus_vlad_aggregation("reddit")
    #
    # mean_versus_vlad_aggregation("youtube")

    handcrafted_features_versus_aggregation_comparison("reddit", 7)

    # handcrafted_features_versus_aggregation_comparison("youtube", 7)
