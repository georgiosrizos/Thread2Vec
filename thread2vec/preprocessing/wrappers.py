__author__ = 'Georgios Rizos (georgerizos@iti.gr)'

from thread2vec.preprocessing import comment_tree
from thread2vec.preprocessing import user_graph
from thread2vec.preprocessing import temporal


def wrapper_comment_count(graph_snapshot_input):
    comment_count = comment_tree.calculate_comment_count(graph_snapshot_input["comment_tree"])
    return comment_count


def wrapper_max_depth(graph_snapshot_input):
    basic_max_depth = comment_tree.calculate_max_depth(graph_snapshot_input["comment_tree"])
    return basic_max_depth


def wrapper_avg_depth(graph_snapshot_input):
    avg_depth = comment_tree.calculate_avg_depth(graph_snapshot_input["comment_tree"])
    return avg_depth


def wrapper_max_width(graph_snapshot_input):
    max_width = comment_tree.calculate_max_width(graph_snapshot_input["comment_tree"])
    return max_width


def wrapper_avg_width(graph_snapshot_input):
    avg_width = comment_tree.calculate_avg_width(graph_snapshot_input["comment_tree"])
    return avg_width


def wrapper_max_depth_over_max_width(graph_snapshot_input):
    max_depth_over_max_width = comment_tree.calculate_max_depth_over_max_width(graph_snapshot_input["comment_tree"])
    return max_depth_over_max_width


def wrapper_avg_depth_over_width(graph_snapshot_input):
    avg_depth_over_width = comment_tree.calculate_avg_depth_over_width(graph_snapshot_input["comment_tree"])
    return avg_depth_over_width


def wrapper_comment_tree_hirsch(graph_snapshot_input):
    comment_tree_hirsch = comment_tree.calculate_comment_tree_hirsch(graph_snapshot_input["comment_tree"])
    return comment_tree_hirsch


def wrapper_comment_tree_wiener(graph_snapshot_input):
    comment_tree_wiener = comment_tree.calculate_comment_tree_wiener(graph_snapshot_input["comment_tree"])
    return comment_tree_wiener


def wrapper_comment_tree_randic(graph_snapshot_input):
    comment_tree_randic = comment_tree.calculate_comment_tree_randic(graph_snapshot_input["comment_tree"])
    return comment_tree_randic


def wrapper_user_count(graph_snapshot_input):
    user_count = user_graph.calculate_user_count(graph_snapshot_input["user_graph"])
    return user_count


def wrapper_user_graph_hirsch(graph_snapshot_input):
    user_graph_hirsch = user_graph.calculate_user_graph_hirsch(graph_snapshot_input["user_graph"])
    return user_graph_hirsch


def wrapper_user_graph_randic(graph_snapshot_input):
    user_graph_randic = user_graph.calculate_user_graph_randic(graph_snapshot_input["user_graph"])
    return user_graph_randic


def wrapper_norm_outdegree_entropy(graph_snapshot_input):
    norm_outdegree_entropy = user_graph.calculate_norm_outdegree_entropy(graph_snapshot_input["user_graph"])
    return norm_outdegree_entropy


def wrapper_outdegree_entropy(graph_snapshot_input):
    outdegree_entropy = user_graph.calculate_outdegree_entropy(graph_snapshot_input["user_graph"])
    return outdegree_entropy


def wrapper_indegree_entropy(graph_snapshot_input):
    indegree_entropy = user_graph.calculate_indegree_entropy(graph_snapshot_input["user_graph"])
    return indegree_entropy


def wrapper_norm_indegree_entropy(graph_snapshot_input):
    norm_indegree_entropy = user_graph.calculate_norm_indegree_entropy(graph_snapshot_input["user_graph"])
    return norm_indegree_entropy


def wrapper_avg_time_differences_1st_half(graph_snapshot_input):
    avg_time_differences_1st_half = temporal.calculate_avg_time_differences_1st_half(graph_snapshot_input["timestamp_list"])
    return avg_time_differences_1st_half


def wrapper_avg_time_differences_2nd_half(graph_snapshot_input):
    avg_time_differences_2nd_half = temporal.calculate_avg_time_differences_2nd_half(graph_snapshot_input["timestamp_list"])
    return avg_time_differences_2nd_half


def wrapper_time_differences_std(graph_snapshot_input):
    time_differences_std = temporal.calculate_time_differences_std(graph_snapshot_input["timestamp_list"])
    return time_differences_std


def wrapper_last_comment_lifetime(graph_snapshot_input):
    last_comment_lifetime = temporal.calculate_last_comment_lifetime(graph_snapshot_input["timestamp_list"],
                                                                     graph_snapshot_input["tweet_timestamp"])
    return last_comment_lifetime
