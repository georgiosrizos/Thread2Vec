__author__ = "Georgios Rizos (georgerizos@iti.gr)"

import os
import heapq
import json
import collections

from thread2vec.preprocessing.social_media import youtube as youtube_extract
from thread2vec.preprocessing.social_media import reddit as reddit_extract
from thread2vec.common import get_package_path


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
    file_path_list = os.listdir(input_folder)
    file_path_list = [name[:-4] for name in file_path_list]

    input_file_path_list = [input_folder + "/" + name + ".txt" for name in file_path_list]

    document_counter = 0
    for file_path, input_file_path in zip(file_path_list, input_file_path_list):
        with open(output_folder + "/" + file_path + ".txt", "w") as o_fp:
            document_gen = youtube_extraction_functions["document_generator"](input_file_path)
            for document in document_gen:
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
                new_document["targets"] = targets

                document_counter += 1
                json.dump(new_document, o_fp)
                o_fp.write("\n\n")


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
        with open(output_folder + "/" + file_path + ".txt", "w") as o_fp:
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


def safe_comment_generator(document,
                           extraction_functions,
                           within_discussion_comment_anonymize):
    """
    We do this in order to correct for nonsensical or missing timestamps.
    """
    comment_generator = extraction_functions["comment_generator"]
    extract_comment_name = extraction_functions["extract_comment_name"]
    extract_parent_comment_name = extraction_functions["extract_parent_comment_name"]
    extract_timestamp = extraction_functions["extract_timestamp"]
    # anonymous_coward_name = extraction_functions["anonymous_coward_name"]

    comment_id_to_comment = dict()

    comment_gen = comment_generator(document)

    initial_post = next(comment_gen)
    yield initial_post

    within_discussion_comment_anonymize[extract_comment_name(initial_post)] = 0
    initial_post_id = within_discussion_comment_anonymize[extract_comment_name(initial_post)]
    within_discussion_comment_anonymize[None] = None

    comment_id_to_comment[initial_post_id] = initial_post

    comment_list = list(comment_gen)
    children_dict = collections.defaultdict(list)
    for comment in comment_list:
        # Anonymize comment.
        comment_name = extract_comment_name(comment)
        comment_id = within_discussion_comment_anonymize.get(comment_name, len(within_discussion_comment_anonymize))
        within_discussion_comment_anonymize[comment_name] = comment_id

        parent_comment_name = extract_parent_comment_name(comment)
        if parent_comment_name is None:
            parent_comment_id = None
        else:
            parent_comment_id = within_discussion_comment_anonymize.get(parent_comment_name,
                                                                        len(within_discussion_comment_anonymize))
        within_discussion_comment_anonymize[parent_comment_name] = parent_comment_id

        comment_id_to_comment[comment_id] = comment

        # Update discussion tree.
        children_dict[parent_comment_id].append(comment_id)

    # Starting from the root/initial post, we get the children and we put them in a priority queue.
    priority_queue = list()

    children = set(children_dict[initial_post_id])
    for child in children:
        comment = comment_id_to_comment[child]
        timestamp = extract_timestamp(comment)
        heapq.heappush(priority_queue, (timestamp, (child, comment)))

    # We iteratively yield the topmost priority comment and add to the priority list the children of that comment.
    while True:
        # If priority list empty, we stop.
        if len(priority_queue) == 0:
            break

        t = heapq.heappop(priority_queue)
        comment = t[1][1]
        yield comment

        children = set(children_dict[int(t[1][0])])
        for child in children:
            comment = comment_id_to_comment[child]
            timestamp = extract_timestamp(comment)
            heapq.heappush(priority_queue, (timestamp, (child, comment)))


if __name__ == "__main__":
    anonymize_reddit_dataset(get_package_path() + "/data_folder/raw_data/reddit",
                             get_package_path() + "/data_folder/anonymized_data/reddit")
    anonymize_youtube_dataset(get_package_path() + "/data_folder/raw_data/youtube",
                              get_package_path() + "/data_folder/anonymized_data/youtube")
