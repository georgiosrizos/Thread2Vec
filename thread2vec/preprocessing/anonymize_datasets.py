__author__ = "Georgios Rizos (georgerizos@iti.gr)"

import os
import sys
import json

from thread2vec.preprocessing.social_media import youtube as youtube_extract
from thread2vec.preprocessing.social_media import reddit as reddit_extract
from thread2vec.preprocessing.social_media import anonymized as anonymized_extract
from thread2vec.preprocessing.common import safe_comment_generator
from thread2vec.common import get_package_path
# from thread2vec.preprocessing.social_media import anonymized as anonymized_extract


def anonymize_youtube_dataset(input_folder, output_folder):
    ####################################################################################################################
    # YouTube extraction functions.
    ####################################################################################################################
    youtube_extraction_functions = dict()
    youtube_extraction_functions["document_generator"] = youtube_extract.document_generator
    youtube_extraction_functions["comment_generator"] = youtube_extract.comment_generator
    youtube_extraction_functions["extract_comment_name"] = youtube_extract.extract_comment_name
    youtube_extraction_functions["extract_parent_comment_name"] = youtube_extract.extract_parent_comment_name
    youtube_extraction_functions["extract_timestamp"] = youtube_extract.extract_timestamp
    youtube_extraction_functions["extract_user_name"] = youtube_extract.extract_user_name
    youtube_extraction_functions["calculate_targets"] = youtube_extract.calculate_targets
    youtube_extraction_functions["extract_title"] = youtube_extract.extract_title
    youtube_extraction_functions["extract_text"] = youtube_extract.extract_text
    youtube_extraction_functions["anonymous_coward_name"] = "REVEAL_FP7_anonymous_youtube_user"

    ####################################################################################################################
    # Anonymizer dictionaries.
    ####################################################################################################################
    anonymize = dict()
    anonymize["user_name"] = dict()
    anonymize["user_name"]["REVEAL_FP7_anonymous_youtube_user"] = "0"

    ####################################################################################################################
    # Iterate over all videos.
    ####################################################################################################################
    # file_path_list = os.listdir(input_folder)
    # file_path_list = [name[:-4] for name in file_path_list]
    file_path_list = ["december_0_500",
                      "december_500_50000",
                      "december_50000_170000",
                      "november_0_170000",
                      "october_0_170000",
                      "september_0_170000"]

    input_file_path_list = [input_folder + "/" + name + ".txt.tar.gz" for name in file_path_list]
    output_file_path = output_folder + "/" + "anonymized_data" + ".txt"

    # We will store the post ids of the successfully preprocessed data in order to avoid duplicates.
    post_name_to_id = dict()

    with open(output_file_path, "w") as o_fp:
        for file_path, input_file_path in zip(file_path_list, input_file_path_list):
            document_gen = youtube_extraction_functions["document_generator"](input_file_path)
            for document in document_gen:
                if document["post_id"] in post_name_to_id.keys():
                    continue
                else:
                    document_counter = len(post_name_to_id)

                within_discussion_comment_anonymize = dict()
                safe_document_gen = safe_comment_generator(document,
                                                           youtube_extraction_functions,
                                                           within_discussion_comment_anonymize)
                comment_list = [comment for comment in safe_document_gen]

                initial_post = comment_list[0]
                initial_timestamp = youtube_extraction_functions["extract_timestamp"](initial_post)

                new_document = dict()
                new_document["fetch_timestamp"] = document["fetch_timestamp"]
                new_document["document_id"] = repr(document_counter)
                new_document["initial_post"] = anonymize_comment(youtube_extraction_functions,
                                                                 anonymize,
                                                                 within_discussion_comment_anonymize,
                                                                 document_counter,
                                                                 initial_post,
                                                                 initial_timestamp)
                new_document["comments"] = list()

                comment_name_set = list()
                user_name_set = list()
                comment_name_set.append(new_document["initial_post"]["comment_name"])
                user_name_set.append(new_document["initial_post"]["user_name"])
                for comment in comment_list[1:]:
                    new_comment = anonymize_comment(youtube_extraction_functions,
                                                    anonymize,
                                                    within_discussion_comment_anonymize,
                                                    document_counter,
                                                    comment,
                                                    initial_timestamp)
                    new_document["comments"].append(new_comment)

                    comment_name_set.append(new_comment["comment_name"])
                    user_name_set.append(new_comment["user_name"])

                comment_name_set = set(comment_name_set)
                user_name_set = set(user_name_set)

                try:
                    targets = youtube_extraction_functions["calculate_targets"](document,
                                                                                comment_name_set,
                                                                                user_name_set)
                except KeyError:
                    continue

                post_name_to_id[document["post_id"]] = document_counter
                print(document_counter)

                new_document["targets"] = targets

                json.dump(new_document, o_fp)
                o_fp.write("\n\n")


# def split_youtube_dataset(folder):
#     """
#     We do this in order to fit it in github.
#     """
#     input_file = folder + "/youtube_sep_dec_2014_data.txt"
#
#     file_counter = 0
#     document_counter = 0
#     fp = open(folder + "/youtube_sep_dec_2014_data_" + repr(file_counter) + ".txt", "w")
#
#     for document in anonymized_extract.document_generator(input_file):
#


def anonymize_reddit_dataset(input_folder, output_folder):
    ####################################################################################################################
    # Reddit extraction functions.
    ####################################################################################################################
    reddit_extraction_functions = dict()
    reddit_extraction_functions["document_generator"] = reddit_extract.document_generator
    reddit_extraction_functions["comment_generator"] = reddit_extract.comment_generator
    reddit_extraction_functions["extract_comment_name"] = reddit_extract.extract_comment_name
    reddit_extraction_functions["extract_parent_comment_name"] = reddit_extract.extract_parent_comment_name
    reddit_extraction_functions["extract_timestamp"] = reddit_extract.extract_timestamp
    reddit_extraction_functions["extract_user_name"] = reddit_extract.extract_user_name
    reddit_extraction_functions["calculate_targets"] = reddit_extract.calculate_targets
    reddit_extraction_functions["extract_title"] = reddit_extract.extract_title
    reddit_extraction_functions["extract_text"] = reddit_extract.extract_text
    reddit_extraction_functions["anonymous_coward_name"] = "[deleted]"

    ####################################################################################################################
    # Anonymizer dictionaries.
    ####################################################################################################################
    anonymize = dict()
    anonymize["user_name"] = dict()
    anonymize["user_name"]["[deleted]"] = "0"

    ####################################################################################################################
    # Iterate over all videos.
    ####################################################################################################################
    file_path_list = ["reddit_news_dataset"]

    input_file_path_list = [input_folder + "/" + name + ".txt" for name in file_path_list]

    document_counter = 0
    for file_path, input_file_path in zip(file_path_list, input_file_path_list):
        with open(output_folder + "/anonymized_data" + ".txt", "w") as o_fp:
            document_gen = reddit_extraction_functions["document_generator"](input_file_path)
            for document in document_gen:
                within_discussion_comment_anonymize = dict()
                safe_document_gen = safe_comment_generator(document,
                                                           reddit_extraction_functions,
                                                           within_discussion_comment_anonymize)
                comment_list = [comment for comment in safe_document_gen]

                initial_post = comment_list[0]
                initial_timestamp = reddit_extraction_functions["extract_timestamp"](initial_post)

                new_document = dict()
                new_document["fetch_timestamp"] = document["fetch_timestamp"]
                new_document["document_id"] = repr(document_counter)
                new_document["initial_post"] = anonymize_comment(reddit_extraction_functions,
                                                                 anonymize,
                                                                 within_discussion_comment_anonymize,
                                                                 document_counter,
                                                                 initial_post,
                                                                 initial_timestamp)
                new_document["comments"] = list()

                comment_name_set = list()
                user_name_set = list()
                comment_name_set.append(new_document["initial_post"]["comment_name"])
                user_name_set.append(new_document["initial_post"]["user_name"])
                for comment in comment_list[1:]:
                    new_comment = anonymize_comment(reddit_extraction_functions,
                                                    anonymize,
                                                    within_discussion_comment_anonymize,
                                                    document_counter,
                                                    comment,
                                                    initial_timestamp)
                    new_document["comments"].append(new_comment)

                    comment_name_set.append(new_comment["comment_name"])
                    user_name_set.append(new_comment["user_name"])

                comment_name_set = set(comment_name_set)
                user_name_set = set(user_name_set)

                try:
                    targets = reddit_extraction_functions["calculate_targets"](document,
                                                                               comment_name_set,
                                                                               user_name_set)
                except KeyError:
                    continue
                new_document["targets"] = targets

                document_counter += 1
                json.dump(new_document, o_fp)
                o_fp.write("\n\n")


def anonymize_comment(extraction_functions,
                      anonymize,
                      within_discussion_comment_anonymize,
                      document_counter,
                      comment,
                      initial_timestamp):
    new_comment = dict()
    timestamp = extraction_functions["extract_timestamp"](comment)
    comment_name = extraction_functions["extract_comment_name"](comment)
    parent_comment_name = extraction_functions["extract_parent_comment_name"](comment)
    user_name = extraction_functions["extract_user_name"](comment)

    new_comment_name = within_discussion_comment_anonymize[comment_name]
    if new_comment_name > 0:
        new_comment_name -= 1
    new_parent_comment_name = within_discussion_comment_anonymize[parent_comment_name]
    new_user_name = anonymize["user_name"].get(user_name,
                                               repr(len(anonymize["user_name"])))
    anonymize["user_name"][user_name] = new_user_name

    new_comment["lifetime"] = timestamp - initial_timestamp
    new_comment["comment_name"] = repr(document_counter) + "_" + repr(new_comment_name)
    if new_comment_name != 0:
        new_comment["parent_comment_name"] = repr(document_counter) + "_" + repr(new_parent_comment_name)
    new_comment["user_name"] = new_user_name

    return new_comment


def form_item_to_user(platform, time_scale):
    folder = get_package_path() + "/data_folder/anonymized_data/" + platform
    output_file_path = get_package_path() + "/data_folder/anonymized_data/" + platform + "/item_to_userset_" + time_scale + ".txt"
    anonymize_user_file_path = get_package_path() + "/data_folder/anonymized_data/" + platform + "/anonymize_user_" + time_scale + ".txt"

    time_scale_in_seconds = dict()
    time_scale_in_seconds["post"] = 0.0
    time_scale_in_seconds["min"] = 60.0
    time_scale_in_seconds["hour"] = 3600.0
    time_scale_in_seconds["day"] = 86400.0
    time_scale_in_seconds["week"] = 604810.0
    time_scale_in_seconds["inf"] = sys.maxsize

    ####################################################################################################################
    # Extraction functions.
    ####################################################################################################################
    extraction_functions = dict()
    extraction_functions["comment_generator"] = anonymized_extract.comment_generator
    extraction_functions["extract_comment_name"] = anonymized_extract.extract_comment_name
    extraction_functions["extract_parent_comment_name"] = anonymized_extract.extract_parent_comment_name
    extraction_functions["extract_lifetime"] = anonymized_extract.extract_lifetime
    extraction_functions["extract_user_name"] = anonymized_extract.extract_user_name
    extraction_functions["calculate_targets"] = anonymized_extract.calculate_targets
    extraction_functions["anonymous_coward_name"] = "0"

    ####################################################################################################################
    # Iterate over all videos.
    ####################################################################################################################
    input_file_path = folder + "/anonymized_data"+ ".txt"

    anonymize_user = dict()

    counter = 0

    fp = open(output_file_path, "w")

    document_gen = anonymized_extract.document_generator(input_file_path)
    for document in document_gen:
        if counter % 50 == 0:
            print(input_file_path, counter)

        user_set = list()

        ################################################################################################################
        # Within discussion anonymization.
        ################################################################################################################
        comment_gen = extraction_functions["comment_generator"](document)
        comment_list = [comment for comment in comment_gen]

        initial_post = comment_list[0]
        initial_timestamp = extraction_functions["extract_lifetime"](initial_post)

        for comment in comment_list:
            lifetime = extraction_functions["extract_lifetime"](comment) - initial_timestamp
            if lifetime > time_scale_in_seconds[time_scale]:
                continue

            user_name = extraction_functions["extract_user_name"](comment)
            user_id = anonymize_user.get(user_name, len(anonymize_user))
            anonymize_user[user_name] = user_id

            user_set.append(user_id)

        user_set = set(user_set)
        user_set = [repr(u) for u in user_set]
        fp.write(repr(counter) + "\t" + "\t".join(sorted(user_set)) + "\n")
        counter += 1

    fp.close()
    with open(anonymize_user_file_path, "w") as fp:
        for k, v in anonymize_user.items():
            fp.write(k + "\t" + repr(v) + "\n")


def form_item_to_popularity(platform):
    folder = get_package_path() + "/data_folder/anonymized_data/" + platform
    output_file_path = get_package_path() + "/data_folder/anonymized_data/" + platform + "/item_to_popularity" + ".txt"

    ####################################################################################################################
    # Extraction functions.
    ####################################################################################################################
    extraction_functions = dict()
    extraction_functions["comment_generator"] = anonymized_extract.comment_generator
    extraction_functions["extract_comment_name"] = anonymized_extract.extract_comment_name
    extraction_functions["extract_parent_comment_name"] = anonymized_extract.extract_parent_comment_name
    extraction_functions["extract_lifetime"] = anonymized_extract.extract_lifetime
    extraction_functions["extract_user_name"] = anonymized_extract.extract_user_name
    extraction_functions["calculate_targets"] = anonymized_extract.calculate_targets
    extraction_functions["anonymous_coward_name"] = "0"

    ####################################################################################################################
    # Iterate over all videos.
    ####################################################################################################################
    input_file_path = folder + "/anonymized_data" + ".txt"

    anonymize_user = dict()

    counter = 0

    fp = open(output_file_path, "w")

    document_gen = anonymized_extract.document_generator(input_file_path)
    for document in document_gen:
        if counter % 50 == 0:
            print(input_file_path, counter)

        targets = anonymized_extract.calculate_targets(document)

        fp.write(repr(targets["comments"]) + "\t" +
                 repr(targets["users"]) + "\t" +
                 repr(targets["score_wilson"]) + "\t" +
                 repr(targets["controversiality_wilson"]) + "\n")
        counter += 1
