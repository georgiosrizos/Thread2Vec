__author__ = 'Georgios Rizos (georgerizos@iti.gr)'

import heapq
import collections

import numpy as np
import scipy.sparse as spsp


def get_binary_graph(graph):
    graph = spsp.coo_matrix(graph)
    binary_graph = spsp.coo_matrix((np.ones_like(graph.data,
                                                 dtype=np.float64),
                                    (graph.row, graph.col)),
                                   shape=graph.shape)
    return binary_graph


def get_degree_undirected(graph):
    graph = spsp.coo_matrix(graph)

    total_degree_vector = np.zeros(graph.shape[0], dtype=np.float64)
    total_weighted_degree_vector = np.zeros(graph.shape[0], dtype=np.float64)

    for i, j, d in zip(graph.row, graph.col, graph.data):
        total_degree_vector[i] += 1.0
        total_degree_vector[j] += 1.0

        total_weighted_degree_vector[i] += d
        total_weighted_degree_vector[j] += d

    return total_degree_vector,\
           total_weighted_degree_vector


def get_degree_directed(graph):
    graph = spsp.coo_matrix(graph)

    out_degree_vector = np.zeros(graph.shape[0], dtype=np.float64)
    out_weighted_degree_vector = np.zeros(graph.shape[0], dtype=np.float64)

    in_degree_vector = np.zeros(graph.shape[0], dtype=np.float64)
    in_weighted_degree_vector = np.zeros(graph.shape[0], dtype=np.float64)

    for i, j, d in zip(graph.row, graph.col, graph.data):
        out_degree_vector[i] += 1.0
        in_degree_vector[j] += 1.0

        out_weighted_degree_vector[i] += d
        in_weighted_degree_vector[j] += d

    total_degree_vector = out_degree_vector + in_degree_vector
    total_weighted_degree_vector = out_weighted_degree_vector + in_weighted_degree_vector

    return out_degree_vector,\
           out_weighted_degree_vector, \
           in_degree_vector, \
           in_weighted_degree_vector, \
           total_degree_vector,\
           total_weighted_degree_vector


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
