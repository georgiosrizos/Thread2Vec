__author__ = "Georgios Rizos (georgerizos@iti.gr)"

import json


def document_generator(file_path):
    with open(file_path, "r") as fp:
        while True:
            try:
                file_row = next(fp)
            except StopIteration:
                break
            except OSError:
                continue
            clean_row = file_row.strip().split("\t")
            if len(clean_row) == 0:
                continue
            try:
                document = json.loads(file_row.strip())
                yield document
            except ValueError:
                continue


def comment_generator(document):
    initial_post = document["initial_post"]
    comments = document["comments"]

    yield initial_post

    for comment in comments:
        yield comment


def extract_comment_name(comment):
    comment_name = comment["comment_name"]

    return comment_name


def extract_parent_comment_name(comment):
    parent_comment_name = comment["parent_comment_name"]

    return parent_comment_name


def extract_user_name(comment):
    user_name = comment["user_name"]

    return user_name


def extract_lifetime(comment):
    timestamp = comment["lifetime"]

    return timestamp


def calculate_targets(document):
    targets = document["targets"]

    return targets
