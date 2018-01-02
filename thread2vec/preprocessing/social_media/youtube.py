__author__ = 'Georgios Rizos (georgerizos@iti.gr)'

import tarfile
import json
import datetime
from dateutil import parser as duparser
from urllib.parse import urlparse

import numpy as np

from thread2vec.preprocessing.targets import ci_lower_bound


def document_generator(file_path):
    with tarfile.open(file_path, "r") as tar:
        for member in tar.getmembers():
            fp = tar.extractfile(member)
            while True:
                try:
                    file_row = next(fp).decode("utf-8")
                except StopIteration:
                    break
                except OSError:
                    continue
                clean_row = file_row.strip().split("\t")
                if len(clean_row) == 2:
                    document = dict()
                elif len(clean_row) == 0:
                    continue
                try:
                    youtube_document = json.loads(file_row.strip())
                    document["fetch_timestamp"] = youtube_document["fetch_timestamp"]
                    document["post_id"] = youtube_document["initial_post"]["id"]
                    document["initial_post"] = youtube_document["initial_post"]
                    document["comments"] = youtube_document["comments"]
                    yield document
                except ValueError:
                    continue


def extract_author_metadata(document):
    author_metadata = document["author_metadata"]

    return author_metadata


def comment_generator(document):
    initial_post = document["initial_post"]
    comments = document["comments"]
    additional_replies = list()
    for c in comments:
        if "replies" in c.keys():
            for reply in c["replies"]["comments"]:
                additional_replies.append(reply)
    comments = comments + additional_replies

    yield initial_post

    if len(comments) == 0:
        raise StopIteration

    for comment in comments:
        yield comment


def extract_comment_name(comment):
    comment_name = comment["id"]

    return comment_name


def extract_parent_comment_name(comment):
    if comment["kind"] == "youtube#comment":
        parent_comment_name = comment["snippet"]["parentId"]
    elif comment["kind"] == "youtube#commentThread":
        parent_comment_name = comment["snippet"]["videoId"]
    elif comment["kind"] == "youtube#video":
        parent_comment_name = None
    else:
        raise RuntimeError

    return parent_comment_name


def extract_user_name(comment):
    if comment["kind"] == "youtube#comment":
        if "authorChannelId" in comment["snippet"].keys():
            user_name = comment["snippet"]["authorChannelId"]["value"]
        elif "authorGoogleplusProfileUrl" in comment["snippet"].keys():
            parsed_url = urlparse(comment["snippet"]["authorGoogleplusProfileUrl"])
            user_name = "gp" + parsed_url.path
        else:
            user_name = "REVEAL_FP7_anonymous_youtube_user"
    elif comment["kind"] == "youtube#commentThread":
        if "authorChannelId" in comment["snippet"]["topLevelComment"]["snippet"].keys():
            user_name = comment["snippet"]["topLevelComment"]["snippet"]["authorChannelId"]["value"]
        elif "authorGoogleplusProfileUrl" in comment["snippet"]["topLevelComment"]["snippet"].keys():
            parsed_url = urlparse(comment["snippet"]["topLevelComment"]["snippet"]["authorGoogleplusProfileUrl"])
            user_name = "gp" + parsed_url.path
        else:
            user_name = "REVEAL_FP7_anonymous_youtube_user"
    elif comment["kind"] == "youtube#video":
        user_name = comment["snippet"]["channelId"]
    else:
        raise RuntimeError

    return user_name


def extract_title(initial_post):
    title = initial_post["snippet"]["title"]
    return title


def calculate_targets(document,
                      comment_name_set,
                      user_name_set):
    target_dict = dict()

    target_dict["comments"] = len(comment_name_set) - 1

    if "0" in user_name_set:
        target_dict["users"] = len(user_name_set) - 1  # -1 for Anonymous Coward.
    else:
        target_dict["users"] = len(user_name_set)

    target_dict["number_of_upvotes"] = int(document["initial_post"]["statistics"]["likeCount"])
    target_dict["number_of_downvotes"] = int(document["initial_post"]["statistics"]["dislikeCount"])
    target_dict["total_number_of_votes"] = target_dict["number_of_upvotes"] + target_dict["number_of_downvotes"]

    target_dict["score"] = target_dict["number_of_upvotes"] - target_dict["number_of_downvotes"]

    if target_dict["total_number_of_votes"] == 0:
        target_dict["score_div"] = 0

        target_dict["score_wilson"] = 0

        target_dict["controversiality_wilson"] = 0
    else:
        target_dict["score_div"] = target_dict["number_of_upvotes"] / target_dict["total_number_of_votes"]

        target_dict["score_wilson"] = ci_lower_bound(target_dict["number_of_upvotes"],
                                                     target_dict["total_number_of_votes"],
                                                     0.95)

        if np.floor(target_dict["total_number_of_votes"]/2) == 0.0:
            target_dict["controversiality_wilson"] = 0
        else:
            target_dict["controversiality_wilson"] = ci_lower_bound(min(np.floor(target_dict["number_of_upvotes"]),
                                                                        np.floor(target_dict["number_of_downvotes"])),
                                                                    np.floor(target_dict["total_number_of_votes"]/2),
                                                                    0.95)

    target_dict["controversiality"] = target_dict["total_number_of_votes"] /\
                                      max(abs(target_dict["number_of_upvotes"] -
                                              target_dict["number_of_downvotes"]), 1)

    return target_dict


def extract_timestamp(comment):
    if comment["kind"] == "youtube#comment":
        timestamp = comment["snippet"]["publishedAt"]
    elif comment["kind"] == "youtube#commentThread":
        timestamp = comment["snippet"]["topLevelComment"]["snippet"]["publishedAt"]
    elif comment["kind"] == "youtube#video":
        timestamp = comment["snippet"]["publishedAt"]
    else:
        raise RuntimeError

    timestamp = duparser.parse(timestamp)
    timestamp = (timestamp - datetime.datetime(1970, 1, 1, tzinfo=datetime.timezone.utc)) / datetime.timedelta(seconds=1)

    return timestamp


def extract_text(comment):
    if comment["kind"] == "youtube#comment":
        text = [comment["snippet"]["textDisplay"], ]
    elif comment["kind"] == "youtube#commentThread":
        text = [comment["snippet"]["topLevelComment"]["snippet"]["textDisplay"], ]
    elif comment["kind"] == "youtube#video":
        text = [comment["snippet"]["title"], comment["snippet"]["description"]]
    else:
        raise RuntimeError
    return text
