__author__ = "Georgios Rizos (georgerizos@iti.gr)"

from thread2vec.preprocessing.anonymize_datasets import anonymize_reddit_dataset, anonymize_youtube_dataset
from thread2vec.preprocessing.anonymize_datasets import form_item_to_user, form_item_to_popularity
from thread2vec.representation.utility import get_data
from thread2vec.preprocessing.handcrafted import calculate_reddit_features, calculate_youtube_features
from thread2vec.common import get_package_path


if "__main__" == __name__:
    # Anonymize raw data
    anonymize_reddit_dataset(get_package_path() + "/data_folder/raw_data/reddit",
                             get_package_path() + "/data_folder/anonymized_data/reddit")
    anonymize_youtube_dataset(get_package_path() + "/data_folder/raw_data/youtube",
                              get_package_path() + "/data_folder/anonymized_data/youtube")

    # Form item to responding users arrays for different time scales.
    for scale in ["post", "min", "hour", "day", "week", "inf"]:
        form_item_to_user("youtube", scale)
        form_item_to_user("reddit", scale)

    # Extract label values from raw data.
    form_item_to_popularity("youtube")
    form_item_to_popularity("reddit")

    # Calculate engineered features.
    calculate_reddit_features()
    calculate_youtube_features()

    # Store data splits.
    data = get_data("reddit", "inf")
    data_splits = data["data_splits"]
    with open(get_package_path() + "/data_folder/anonymized_data/reddit/data_splits.txt", "w") as fp:
        fp.write(repr(len(data_splits[0])) + "\t" + repr(len(data_splits[1])) + "\t" + repr(len(data_splits[2])) + "\n")

        index = list(data_splits[0]) + list(data_splits[1]) + list(data_splits[2])

        for i in index:
            fp.write(repr(i) + "\n")

    data = get_data("youtube", "inf")
    data_splits = data["data_splits"]
    with open(get_package_path() + "/data_folder/anonymized_data/youtube/data_splits.txt", "w") as fp:
        fp.write(repr(len(data_splits[0])) + "\t" + repr(len(data_splits[1])) + "\t" + repr(len(data_splits[2])) + "\n")

        index = list(data_splits[0]) + list(data_splits[1]) + list(data_splits[2])

        for i in index:
            fp.write(repr(i) + "\n")
