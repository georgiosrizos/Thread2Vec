__author__ = "Georgios Rizos (georgerizos@iti.gr)"

from thread2vec.common import get_package_path
from thread2vec.representation.neural_embedding import Thread2Vec
from thread2vec.representation.utility import get_data


if __name__ == "__main__":
    data_folder = get_package_path() + "/data_folder"

    ####################################################################################################################
    # Run Reddit experiment.
    ####################################################################################################################
    batch_size = 64
    negative_samples = .4
    embedding_size = 64
    window_size = 500
    learning_rate = 1e-3
    dropout = 0.2

    dataset = "reddit"

    data = get_data(dataset, "week")
    print("Read data.")

    async_batch_size = 1000

    shuffle = True

    user_user_iterations_number = None

    number_of_vlad_clusters = 50

    number_of_epochs = 700
    patience = 50
    model_file_path = data_folder + "/models/reddit_model.pkl"

    thread2vec = Thread2Vec(batch_size=batch_size,
                            negative_samples=negative_samples,
                            embedding_size=embedding_size,
                            window_size=window_size,
                            learning_rate=learning_rate,
                            dropout=dropout,
                            data=data,
                            dataset=dataset,
                            async_batch_size=async_batch_size,
                            shuffle=shuffle,
                            user_user_iterations_number=user_user_iterations_number,
                            number_of_vlad_clusters=number_of_vlad_clusters)

    # thread2vec.load_params(data_folder + "/models/model.pkl")
    print("Loaded model parameters.")

    thread2vec.train(number_of_epochs,
                     patience,
                     model_file_path)

    ####################################################################################################################
    # Run YouTube experiment.
    ####################################################################################################################
    batch_size = 64
    negative_samples = .4
    embedding_size = 64
    window_size = 500
    learning_rate = 1e-3
    dropout = 0.2

    dataset = "youtube"

    data = get_data(dataset, "week")
    print("Read data.")

    async_batch_size = 10000

    shuffle = True

    user_user_iterations_number = None

    number_of_vlad_clusters = 50

    number_of_epochs = 700
    patience = 50
    model_file_path = data_folder + "/models/youtube_model.pkl"

    thread2vec = Thread2Vec(batch_size=batch_size,
                            negative_samples=negative_samples,
                            embedding_size=embedding_size,
                            window_size=window_size,
                            learning_rate=learning_rate,
                            dropout=dropout,
                            data=data,
                            dataset=dataset,
                            async_batch_size=async_batch_size,
                            shuffle=shuffle,
                            user_user_iterations_number=user_user_iterations_number,
                            number_of_vlad_clusters=number_of_vlad_clusters)

    # thread2vec.load_params(data_folder + "/models/model.pkl")
    print("Loaded model parameters.")

    thread2vec.train(number_of_epochs,
                     patience,
                     model_file_path)
