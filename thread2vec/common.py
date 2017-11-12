import os
import inspect
try:
    import cPickle as pickle
except ImportError:
    import pickle

import thread2vec


def get_package_path():
    """
    Returns the folder path that the package lies in.
    :return: folder_path: The package folder path.
    """
    return os.path.dirname(inspect.getfile(thread2vec))
