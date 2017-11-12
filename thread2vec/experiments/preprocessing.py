from thread2vec.common import get_package_path
from thread2vec.representation.utility import get_data, clean_thread_data

if __name__ == "__main__":

    data = get_data("reddit")
    data_splits = data["data_splits"]
    with open(get_package_path() + "/data_folder/uniform_data/reddit/data_splits.txt", "w") as fp:
        fp.write(repr(len(data_splits[0])) + "\t" + repr(len(data_splits[1])) + "\t" + repr(len(data_splits[2])) + "\n")

        index = list(data_splits[0]) + list(data_splits[1]) + list(data_splits[2])

        for i in index:
            fp.write(repr(i) + "\n")

    # print(data["filtered_item_to_user_matrix"].shape)
    # print(data["popularity_matrix"].shape)
    # print(data["item_indices_sorted"].size)
    # print(len(data["anonymize_user"]))
    # print(len(data["true_user_id_to_user_id"]))
    # print(len(data["user_list"]))
    # print(data["item_indices_sorted"])
    # print(data["true_user_id_to_user_id"])

    # (349925, 199999)
    # (349925, 4)
    # 349925
    # 199999
    # 199999
    clean_thread_data("youtube")
    # (351768, 199999)
    # (351768, 4)
    # 351768
    # 199999
    # 199999

    data = get_data("youtube")
    data_splits = data["data_splits"]
    with open(get_package_path() + "/data_folder/uniform_data/youtube/data_splits.txt", "w") as fp:
        fp.write(repr(len(data_splits[0])) + "\t" + repr(len(data_splits[1])) + "\t" + repr(len(data_splits[2])) + "\n")

        index = list(data_splits[0]) + list(data_splits[1]) + list(data_splits[2])

        for i in index:
            fp.write(repr(i) + "\n")

    # print(data["filtered_item_to_user_matrix"].shape)
    # print(data["popularity_matrix"].shape)
    # print(data["item_indices_sorted"].size)
    # print(len(data["anonymize_user"]))
    # print(len(data["true_user_id_to_user_id"]))
    # print(len(data["user_list"]))
    # print(data["item_indices_sorted"].max())
    # print(max(data["true_user_id_to_user_id"].values))