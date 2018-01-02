__author__ = "Georgios Rizos (georgerizos@iti.gr)"

import numpy as np
import scipy.sparse as spsp

from thread2vec.preprocessing.social_media import anonymized as anonymized_extract
from thread2vec.preprocessing import wrappers
from thread2vec.preprocessing.common import safe_comment_generator
from thread2vec.common import get_package_path


def calculate_reddit_features():
    input_file_path = get_package_path() + "/data_folder/anonymized_data/reddit/anonymized_data.txt"

    ####################################################################################################################
    # Iterate over all videos.
    ####################################################################################################################
    graph_generator = form_graphs([input_file_path, ], item_id_set=set(range(35844)))

    features_generator = extract_features(graph_generator, "reddit")

    reddit_feature_name_list = sorted(get_handcrafted_feature_names("Reddit"))
    number_of_reddit_features = len(reddit_feature_name_list)

    number_of_items = 35844  # TODO: Make this readable.

    features_post = np.empty((number_of_items, number_of_reddit_features), dtype=np.float32)
    features_minute = np.empty((number_of_items, number_of_reddit_features), dtype=np.float32)
    features_hour = np.empty((number_of_items, number_of_reddit_features), dtype=np.float32)
    features_day = np.empty((number_of_items, number_of_reddit_features), dtype=np.float32)
    features_week = np.empty((number_of_items, number_of_reddit_features), dtype=np.float32)
    features_inf = np.empty((number_of_items, number_of_reddit_features), dtype=np.float32)

    features_dict = dict()
    features_dict[0] = features_post
    features_dict[1] = features_minute
    features_dict[2] = features_hour
    features_dict[3] = features_day
    features_dict[4] = features_week
    features_dict[5] = features_inf

    counter = 0
    for features in features_generator:
        for s, snapshot in enumerate(features["snapshots"]):
            snapshot_features = snapshot["features"]

            for f, feature_name in enumerate(reddit_feature_name_list):
                features_dict[s][counter, f] = np.float32(snapshot_features[feature_name])

        if s < 5:
            for s_extra in range(s+1, 6):
                for f, feature_name in enumerate(reddit_feature_name_list):
                    features_dict[s_extra][counter, f] = np.float32(snapshot_features[feature_name])

        counter += 1

    np.save(get_package_path() + "/data_folder/anonymized_data/reddit/features_post", features_post)
    np.save(get_package_path() + "/data_folder/anonymized_data/reddit/features_minute", features_minute)
    np.save(get_package_path() + "/data_folder/anonymized_data/reddit/features_hour", features_hour)
    np.save(get_package_path() + "/data_folder/anonymized_data/reddit/features_day", features_day)
    np.save(get_package_path() + "/data_folder/anonymized_data/reddit/features_week", features_week)
    np.save(get_package_path() + "/data_folder/anonymized_data/reddit/features_inf", features_inf)


def calculate_youtube_features():
    input_file_path = get_package_path() + "/data_folder/anonymized_data/youtube/anonymized_data.txt"

    ####################################################################################################################
    # Iterate over all videos.
    ####################################################################################################################
    graph_generator = form_graphs([input_file_path, ], item_id_set=set(range(411288)))

    features_generator = extract_features(graph_generator, "youtube")

    youtube_feature_name_list = sorted(get_handcrafted_feature_names("YouTube"))
    number_of_youtube_features = len(youtube_feature_name_list)

    number_of_items = 411288  # TODO: Make this readable.

    features_post = np.empty((number_of_items, number_of_youtube_features), dtype=np.float32)
    features_minute = np.empty((number_of_items, number_of_youtube_features), dtype=np.float32)
    features_hour = np.empty((number_of_items, number_of_youtube_features), dtype=np.float32)
    features_day = np.empty((number_of_items, number_of_youtube_features), dtype=np.float32)
    features_week = np.empty((number_of_items, number_of_youtube_features), dtype=np.float32)
    features_inf = np.empty((number_of_items, number_of_youtube_features), dtype=np.float32)

    features_dict = dict()
    features_dict[0] = features_post
    features_dict[1] = features_minute
    features_dict[2] = features_hour
    features_dict[3] = features_day
    features_dict[4] = features_week
    features_dict[5] = features_inf

    counter = 0
    for features in features_generator:
        for s, snapshot in enumerate(features["snapshots"]):
            snapshot_features = snapshot["features"]

            for f, feature_name in enumerate(youtube_feature_name_list):
                features_dict[s][counter, f] = np.float32(snapshot_features[feature_name])

        if s < 5:
            for s_extra in range(s + 1, 6):
                for f, feature_name in enumerate(youtube_feature_name_list):
                    features_dict[s_extra][counter, f] = np.float32(snapshot_features[feature_name])

        counter += 1

    np.save(get_package_path() + "/data_folder/anonymized_data/youtube/features_post", features_post)
    np.save(get_package_path() + "/data_folder/anonymized_data/youtube/features_minute", features_minute)
    np.save(get_package_path() + "/data_folder/anonymized_data/youtube/features_hour", features_hour)
    np.save(get_package_path() + "/data_folder/anonymized_data/youtube/features_day", features_day)
    np.save(get_package_path() + "/data_folder/anonymized_data/youtube/features_week", features_week)
    np.save(get_package_path() + "/data_folder/anonymized_data/youtube/features_inf", features_inf)

########################################################################################################################


def form_graphs(input_file_path_list, item_id_set):
    counter = 0

    for input_file_path in input_file_path_list:
    # for i in range(1):
        document_gen = anonymized_extract.document_generator(input_file_path)

        for social_context_dict in document_gen:
            # print(item_id_set)

            print(counter)
            if counter in item_id_set:
                counter += 1
            else:
                counter += 1
                continue

            snapshots,\
            targets = get_snapshot_graphs(social_context_dict)

            if snapshots is None:
                raise ValueError
                continue

            graph_dict = social_context_dict
            graph_dict["snapshots"] = snapshots
            graph_dict["targets"] = targets
            yield graph_dict


def extract_features(graph_generator, platform):
    for graph_snapshot_dict in graph_generator:
        snapshots = graph_snapshot_dict["snapshots"]

        initial_post = graph_snapshot_dict["initial_post"]

        snapshots_with_features = list()
        # tweet_timestamp = graph_snapshot_dict["tweet_timestamp"]
        for snapshot_dict in snapshots:
            comment_tree = snapshot_dict["comment_tree"]
            user_graph = snapshot_dict["user_graph"]
            timestamp_list = snapshot_dict["timestamp_list"]

            features = extract_snapshot_features(comment_tree,
                                                 user_graph,
                                                 timestamp_list,
                                                 # tweet_timestamp,
                                                 initial_post,
                                                 platform)
            snapshot_dict["features"] = features

            snapshots_with_features.append(snapshot_dict)

        features_dict = graph_snapshot_dict
        features_dict["snapshots"] = snapshots_with_features

        yield features_dict


def get_snapshot_graphs(social_context):
    comment_generator = anonymized_extract.comment_generator
    extract_comment_name = anonymized_extract.extract_comment_name
    extract_parent_comment_name = anonymized_extract.extract_parent_comment_name
    extract_lifetime = anonymized_extract.extract_lifetime
    extract_user_name = anonymized_extract.extract_user_name
    calculate_targets = anonymized_extract.calculate_targets

    extraction_functions = dict()
    extraction_functions["comment_generator"] = comment_generator
    extraction_functions["extract_comment_name"] = extract_comment_name
    extraction_functions["extract_parent_comment_name"] = extract_parent_comment_name
    extraction_functions["extract_timestamp"] = extract_lifetime
    extraction_functions["extract_user_name"] = extract_user_name
    extraction_functions["calculate_targets"] = calculate_targets

    anonymous_coward_name = 0

    comment_gen = comment_generator(social_context)
    initial_post = next(comment_gen)
    initial_post_timestamp = extract_lifetime(initial_post)

    # post_lifetime_to_assessment = upper_timestamp - initial_post_timestamp

    # if post_lifetime_to_assessment < 0.0:
    #     print("Post timestamp smaller than assessment timestamp. Bad data. Continuing.")
    # elif post_lifetime_to_assessment > 604800:
    #     # Post is older than a week.
    #     return None, None, None
    # else:
    #     pass

    comment_gen = comment_generator(social_context)

    comment_name_set,\
    user_name_set,\
    within_discussion_comment_anonymize,\
    within_discussion_user_anonymize,\
    within_discussion_anonymous_coward = within_discussion_comment_and_user_anonymization(comment_gen,
                                                                                          extract_comment_name,
                                                                                          extract_user_name,
                                                                                          anonymous_coward_name)

    # safe_comment_gen = safe_comment_generator(social_context,
    #                                           comment_generator=comment_generator,
    #                                           within_discussion_comment_anonymize=within_discussion_comment_anonymize,
    #                                           extract_comment_name=extract_comment_name,
    #                                           extract_parent_comment_name=extract_parent_comment_name,
    #                                           extract_timestamp=extract_lifetime,
    #                                           safe=True)
    safe_comment_gen = safe_comment_generator(social_context,
                                              extraction_functions,
                                              within_discussion_comment_anonymize)

    snapshot_graphs = form_snapshot_graphs(safe_comment_gen,
                                           comment_name_set,
                                           user_name_set,
                                           extract_lifetime,
                                           extract_comment_name,
                                           extract_parent_comment_name,
                                           extract_user_name,
                                           within_discussion_comment_anonymize,
                                           within_discussion_user_anonymize,
                                           within_discussion_anonymous_coward)
    if snapshot_graphs is None:
        return None, None

    try:
        all_targets = calculate_targets(social_context)
    except KeyError:
        return None, None

    targets = dict()
    targets["comment_count"] = all_targets["comments"]
    targets["user_count"] = all_targets["users"]
    targets["upvote_count"] = all_targets["number_of_upvotes"]
    targets["downvote_count"] = all_targets["number_of_downvotes"]
    targets["score"] = all_targets["score_wilson"]
    targets["controversiality"] = all_targets["controversiality_wilson"]

    return snapshot_graphs, targets


def form_snapshot_graphs(safe_comment_gen,
                         comment_name_set,
                         user_name_set,
                         extract_timestamp,
                         extract_comment_name,
                         extract_parent_comment_name,
                         extract_user_name,
                         within_discussion_comment_anonymize,
                         within_discussion_user_anonymize,
                         within_discussion_anonymous_coward):
    # Keep only the social context until the tweet timestamp.
    comment_list = list()
    timestamp_list = list()
    try:
        initial_post = next(safe_comment_gen)
    except StopIteration:
        return None
    initial_timestamp = extract_timestamp(initial_post)
    comment_list.append(initial_post)
    timestamp_list.append(initial_timestamp)
    for comment in safe_comment_gen:
        comment_timestamp = extract_timestamp(comment)

        # Sanitize comment timestamps.
        if comment_timestamp < timestamp_list[-1]:
            comment_timestamp = timestamp_list[-1]

        comment_list.append(comment)
        timestamp_list.append(comment_timestamp)

    # Decide the snapshot timestamps.
    snapshot_timestamps = decide_snapshot_timestamps(timestamp_list,
                                                     max_number_of_snapshots_including_zero=10)
    # print(snapshot_timestamps)

    snapshot_gen = snapshot_generator(comment_list,
                                      timestamp_list,
                                      snapshot_timestamps,
                                      comment_name_set,
                                      user_name_set,
                                      extract_comment_name,
                                      extract_parent_comment_name,
                                      extract_user_name,
                                      within_discussion_comment_anonymize,
                                      within_discussion_user_anonymize,
                                      within_discussion_anonymous_coward)
    snapshot_graphs = [snapshot_graph_dict for snapshot_graph_dict in snapshot_gen]

    return snapshot_graphs


def decide_snapshot_timestamps(timestamp_list,
                               max_number_of_snapshots_including_zero):

    initial_timestamp = min(timestamp_list)
    final_timestamp = max(timestamp_list)

    snapshot_timestamps = [initial_timestamp,
                           initial_timestamp + 60,
                           initial_timestamp + 3600,
                           initial_timestamp + 86400,
                           initial_timestamp + 604800,
                           final_timestamp]

    # discrete_timestamp_list = sorted(set(timestamp_list))
    # discrete_timestamp_count = len(discrete_timestamp_list)
    # if discrete_timestamp_count < max_number_of_snapshots_including_zero:
    #     max_number_of_snapshots_including_zero = discrete_timestamp_count
    #
    # snapshot_timestamps = np.linspace(0,
    #                                   len(discrete_timestamp_list)-1,
    #                                   num=max_number_of_snapshots_including_zero,
    #                                   endpoint=True)
    #
    # snapshot_timestamps = np.rint(snapshot_timestamps)
    # snapshot_timestamps = list(snapshot_timestamps)
    # snapshot_timestamps = [discrete_timestamp_list[int(t)] for t in snapshot_timestamps]

    return snapshot_timestamps


def snapshot_generator(comment_list,
                       timestamp_list,
                       snapshot_timestamps,
                       comment_name_set,
                       user_name_set,
                       extract_comment_name,
                       extract_parent_comment_name,
                       extract_user_name,
                       within_discussion_comment_anonymize,
                       within_discussion_user_anonymize,
                       within_discussion_anonymous_coward):
    # Initialization.
    comment_tree = spsp.dok_matrix((len(comment_name_set),
                                    len(comment_name_set)),
                                   dtype=np.float64)

    user_graph = spsp.dok_matrix((len(user_name_set),
                                  len(user_name_set)),
                                 dtype=np.float64)
    comment_id_to_user_id = dict()
    comment_id_to_user_id[0] = 0

    user_name_list = list()

    initial_post = comment_list[0]
    initial_post_timestamp = timestamp_list[0]

    user_name = extract_user_name(initial_post)
    if user_name is not None:
        user_name_list.append(user_name)

    # snapshot_graph_dict = dict()
    # snapshot_graph_dict["comment_tree"] = spsp.coo_matrix(comment_tree)
    # snapshot_graph_dict["user_graph"] = spsp.coo_matrix(user_graph)
    # snapshot_graph_dict["timestamp_list"] = [initial_post_timestamp]
    # snapshot_graph_dict["user_set"] = set(user_name_list)
    # yield snapshot_graph_dict

    snapshot_counter = 0
    for counter in range(len(comment_list)):
        comment = comment_list[counter]
        comment_timestamp = timestamp_list[counter]

        user_name = extract_user_name(comment)
        if user_name is not None:
            user_name_list.append(user_name)

        if comment_timestamp > snapshot_timestamps[snapshot_counter]:
            snapshot_graph_dict = dict()
            snapshot_graph_dict["comment_tree"] = spsp.coo_matrix(comment_tree)
            snapshot_graph_dict["user_graph"] = spsp.coo_matrix(user_graph)
            snapshot_graph_dict["timestamp_list"] = timestamp_list[:counter+1]
            snapshot_graph_dict["user_set"] = set(user_name_list)
            yield snapshot_graph_dict

            snapshot_counter += 1
            if snapshot_counter >= len(snapshot_timestamps):
                raise StopIteration

        comment_tree,\
        user_graph,\
        comment_id,\
        parent_comment_id,\
        commenter_id,\
        parent_commenter_id,\
        comment_id_to_user_id = update_discussion_and_user_graphs(comment,
                                                                  extract_comment_name,
                                                                  extract_parent_comment_name,
                                                                  extract_user_name,
                                                                  comment_tree,
                                                                  user_graph,
                                                                  within_discussion_comment_anonymize,
                                                                  within_discussion_user_anonymize,
                                                                  within_discussion_anonymous_coward,
                                                                  comment_id_to_user_id)

    snapshot_graph_dict = dict()
    snapshot_graph_dict["comment_tree"] = spsp.coo_matrix(comment_tree)
    snapshot_graph_dict["user_graph"] = spsp.coo_matrix(user_graph)
    snapshot_graph_dict["timestamp_list"] = timestamp_list
    snapshot_graph_dict["user_set"] = set(user_name_list)
    yield snapshot_graph_dict


def update_discussion_and_user_graphs(comment,
                                      extract_comment_name,
                                      extract_parent_comment_name,
                                      extract_user_name,
                                      discussion_tree,
                                      user_graph,
                                      within_discussion_comment_anonymize,
                                      within_discussion_user_anonymize,
                                      within_discussion_anonymous_coward,
                                      comment_id_to_user_id):
    """
    Update the discussion tree and the user graph for a discussion.

    Does not handle the initial post.
    """
    # Extract comment.
    comment_name = extract_comment_name(comment)
    comment_id = within_discussion_comment_anonymize[comment_name]

    # Extract commenter.
    commenter_name = extract_user_name(comment)
    commenter_id = within_discussion_user_anonymize[commenter_name]

    # Update the comment to user map.
    comment_id_to_user_id[comment_id] = commenter_id

    # Check if this is a comment to the original post or to another comment.
    try:
        parent_comment_name = extract_parent_comment_name(comment)
    except KeyError:
        parent_comment_name = None
    if parent_comment_name is None:
        # The parent is the original post.
        parent_comment_id = 0
        parent_commenter_id = 0
    else:
        # The parent is another comment.
        try:
            parent_comment_id = within_discussion_comment_anonymize[parent_comment_name]
        except KeyError:
            print("Parent comment does not exist. Comment name: ", comment_name)
            raise RuntimeError

        # Extract parent comment in order to update user graph.
        try:
            parent_commenter_id = comment_id_to_user_id[parent_comment_id]
        except KeyError:
            print("Parent user does not exist. Comment name: ", comment_name)
            raise RuntimeError

    try:
        if within_discussion_anonymous_coward is None:
            if user_graph[commenter_id, parent_commenter_id] > 0.0:
                user_graph[commenter_id, parent_commenter_id] += 1.0
            elif user_graph[parent_commenter_id, commenter_id] > 0.0:
                user_graph[parent_commenter_id, commenter_id] += 1.0
            else:
                user_graph[commenter_id, parent_commenter_id] = 1.0
        else:
            if within_discussion_anonymous_coward not in (parent_commenter_id,
                                                          commenter_id):
                if user_graph[commenter_id, parent_commenter_id] > 0.0:
                    user_graph[commenter_id, parent_commenter_id] += 1.0
                elif user_graph[parent_commenter_id, commenter_id] > 0.0:
                    user_graph[parent_commenter_id, commenter_id] += 1.0
                else:
                    user_graph[commenter_id, parent_commenter_id] = 1.0
    except IndexError:
        print("Index error: ", user_graph.shape, commenter_id, parent_commenter_id)
        raise RuntimeError

    # Update discussion radial tree.
    discussion_tree[comment_id, parent_comment_id] = 1

    return discussion_tree,\
           user_graph,\
           comment_id,\
           parent_comment_id,\
           commenter_id,\
           parent_commenter_id,\
           comment_id_to_user_id


# def safe_comment_generator(document,
#                            comment_generator,
#                            within_discussion_comment_anonymize,
#                            extract_comment_name,
#                            extract_parent_comment_name,
#                            extract_timestamp,
#                            safe):
#     """
#     We do this in order to correct for nonsensical or missing timestamps.
#     """
#     if not safe:
#         comment_gen = comment_generator(document)
#
#         initial_post = next(comment_gen)
#         yield initial_post
#
#         comment_list = sorted(comment_gen, key=extract_timestamp)
#         for comment in comment_list:
#             yield comment
#     else:
#         comment_id_to_comment = dict()
#
#         comment_gen = comment_generator(document)
#
#         initial_post = next(comment_gen)
#         yield initial_post
#
#         initial_post_id = within_discussion_comment_anonymize[extract_comment_name(initial_post)]
#
#         comment_id_to_comment[initial_post_id] = initial_post
#
#         if initial_post_id != 0:
#             print("This cannot be.")
#             raise RuntimeError
#
#         comment_list = list(comment_gen)
#         children_dict = collections.defaultdict(list)
#         for comment in comment_list:
#             # Anonymize comment.
#             comment_name = extract_comment_name(comment)
#             comment_id = within_discussion_comment_anonymize[comment_name]
#
#             parent_comment_name = extract_parent_comment_name(comment)
#             if parent_comment_name is None:
#                 parent_comment_id = 0
#             else:
#                 parent_comment_id = within_discussion_comment_anonymize[parent_comment_name]
#
#             comment_id_to_comment[comment_id] = comment
#
#             # Update discussion tree.
#             children_dict[parent_comment_id].append(comment_id)
#
#         # Starting from the root/initial post, we get the children and we put them in a priority queue.
#         priority_queue = list()
#
#         children = set(children_dict[initial_post_id])
#         for child in children:
#             comment = comment_id_to_comment[child]
#             timestamp = extract_timestamp(comment)
#             heapq.heappush(priority_queue, (timestamp, (child, comment)))
#
#         # We iteratively yield the topmost priority comment and add to the priority list the children of that comment.
#         while True:
#             # If priority list empty, we stop.
#             if len(priority_queue) == 0:
#                 break
#
#             t = heapq.heappop(priority_queue)
#             comment = t[1][1]
#             yield comment
#
#             children = set(children_dict[int(t[1][0])])
#             for child in children:
#                 comment = comment_id_to_comment[child]
#                 timestamp = extract_timestamp(comment)
#                 heapq.heappush(priority_queue, (timestamp, (child, comment)))


def within_discussion_comment_and_user_anonymization(comment_gen,
                                                     extract_comment_name,
                                                     extract_user_name,
                                                     anonymous_coward_name):
    """
    Reads all distinct users and comments in a single document and anonymizes them. Roots are 0.
    """
    comment_name_set = list()
    user_name_set = list()

    append_comment_name = comment_name_set.append
    append_user_name = user_name_set.append

    ####################################################################################################################
    # Extract comment and user name from the initial post.
    ####################################################################################################################
    initial_post = next(comment_gen)

    initial_post_name = extract_comment_name(initial_post)
    op_name = extract_user_name(initial_post)

    append_comment_name(initial_post_name)
    append_user_name(op_name)

    ####################################################################################################################
    # Iterate over all comments.
    ####################################################################################################################
    for comment in comment_gen:
        comment_name = extract_comment_name(comment)
        commenter_name = extract_user_name(comment)

        append_comment_name(comment_name)
        append_user_name(commenter_name)

    ####################################################################################################################
    # Perform anonymization.
    ####################################################################################################################
    # Remove duplicates and then remove initial post name because we want to give it id 0.
    comment_name_set = set(comment_name_set)
    comment_name_set.remove(initial_post_name)

    # Remove duplicates and then remove OP because we want to give them id 0.
    user_name_set = set(user_name_set)
    user_name_set.remove(op_name)

    # Anonymize.
    within_discussion_comment_anonymize = dict(zip(comment_name_set, range(1, len(comment_name_set) + 1)))
    within_discussion_comment_anonymize[initial_post_name] = 0  # Initial Post gets id 0.

    within_discussion_user_anonymize = dict(zip(user_name_set, range(1, len(user_name_set) + 1)))
    within_discussion_user_anonymize[op_name] = 0            # Original Poster gets id 0.

    comment_name_set.add(initial_post_name)
    user_name_set.add(op_name)

    if anonymous_coward_name is not None:
        # if op_name == anonymous_coward_name:
            # print("The Original Poster is Anonymous.")
        try:
            within_discussion_anonymous_coward = within_discussion_user_anonymize[anonymous_coward_name]
        except KeyError:
            within_discussion_anonymous_coward = None
    else:
        within_discussion_anonymous_coward = None

    return comment_name_set,\
           user_name_set,\
           within_discussion_comment_anonymize,\
           within_discussion_user_anonymize,\
           within_discussion_anonymous_coward


def extract_snapshot_features(comment_tree,
                              user_graph,
                              timestamp_list,
                              initial_post,
                              platform):
    graph_snapshot_input = dict()
    graph_snapshot_input["comment_tree"] = comment_tree
    graph_snapshot_input["user_graph"] = user_graph
    graph_snapshot_input["timestamp_list"] = timestamp_list
    # graph_snapshot_input["tweet_timestamp"] = tweet_timestamp
    graph_snapshot_input["initial_post"] = initial_post
    # graph_snapshot_input["author"] = author

    feature_names = sorted(get_handcrafted_feature_names(platform))

    handcrafted_function_list = [getattr(wrappers, "wrapper_" + feature_name) for feature_name in feature_names]

    features = calculate_handcrafted_features(graph_snapshot_input,
                                              feature_names,
                                              handcrafted_function_list)

    return features


def calculate_handcrafted_features(graph_snapshot_input,
                                   feature_names,
                                   handcrafted_function_list):
    features = dict()
    for feature_name, calculation_function in zip(feature_names, handcrafted_function_list):
        feature_value = calculation_function(graph_snapshot_input)
        features[feature_name] = feature_value

    return features


def get_handcrafted_feature_names(platform):
    """
    Returns a set of feature names to be calculated.

    Output: - names: A set of strings, corresponding to the features to be calculated.
    """
    names = set()

    ####################################################################################################################
    # Add basic discussion tree features.
    ####################################################################################################################
    names.update(["comment_count",
                  "max_depth",
                  "avg_depth",
                  "max_width",
                  "avg_width",
                  "max_depth_over_max_width",
                  "avg_depth_over_width"])

    ####################################################################################################################
    # Add branching discussion tree features.
    ####################################################################################################################
    names.update(["comment_tree_hirsch",
                  "comment_tree_wiener",
                  "comment_tree_randic"])

    ####################################################################################################################
    # Add user graph features.
    ####################################################################################################################
    names.update(["user_count",
                  "user_graph_hirsch",
                  "user_graph_randic",
                  "outdegree_entropy",
                  "norm_outdegree_entropy",
                  "indegree_entropy",
                  "norm_indegree_entropy"])

    ####################################################################################################################
    # Add temporal features.
    ####################################################################################################################
    names.update(["avg_time_differences_1st_half",
                  "avg_time_differences_2nd_half",
                  "time_differences_std"])

    ####################################################################################################################
    # Add YouTube channel features.
    ####################################################################################################################
    # if platform == "YouTube":
    #     names.update(["author_privacy_status_youtube",
    #                   "author_is_linked_youtube",
    #                   "author_long_uploads_status_youtube",
    #                   "author_comment_count_youtube",
    #                   "author_comment_rate_youtube",
    #                   "author_view_count_youtube",
    #                   "author_view_rate_youtube",
    #                   "author_video_upload_count_youtube",
    #                   "author_video_upload_rate_youtube",
    #                   "author_subscriber_count_youtube",
    #                   "author_subscriber_rate_youtube",
    #                   "author_hidden_subscriber_count_youtube",
    #                   "author_channel_lifetime_youtube"])

    ####################################################################################################################
    # Add Reddit author features.
    ####################################################################################################################
    # elif platform == "Reddit":
    #     names.update(["author_has_verified_mail_reddit",
    #                   "author_account_lifetime_reddit",
    #                   "author_hide_from_robots_reddit",
    #                   "author_is_mod_reddit",
    #                   "author_link_karma_reddit",
    #                   "author_link_karma_rate_reddit",
    #                   "author_comment_karma_reddit",
    #                   "author_comment_karma_rate_reddit",
    #                   "author_is_gold_reddit"])
    # else:
    #     print("Invalid platform name.")
    #     raise RuntimeError

    return names


# print(sorted(get_handcrafted_feature_names("YouTube")))
# print(sorted(get_handcrafted_feature_names("Reddit")))


def make_features_vector(features_dict, platform):
    feature_names = sorted(get_handcrafted_feature_names(platform))

    features_vector_list = list()
    for feature_name in feature_names:
        feature_value = features_dict[feature_name]

        features_vector_list.append(feature_value)
    features_vector = np.empty((1, len(feature_names)), dtype=np.float64)
    for i, v in enumerate(features_vector_list):
        features_vector[0, i] = v

    return features_vector
