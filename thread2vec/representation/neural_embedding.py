__author__ = "Georgios Rizos (georgerizos@iti.gr)"

try:
    import cPickle
except ImportError:
    import pickle as cPickle

import lasagne
import numpy as np
import scipy.sparse as spsp
import theano
from lasagne.layers import EmbeddingLayer, InputLayer
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from theano import tensor as T


class Thread2Vec():
    def __init__(self,
                 batch_size,
                 negative_samples,
                 embedding_size,
                 window_size,
                 learning_rate,
                 dropout,
                 data,
                 dataset,
                 async_batch_size,
                 shuffle,
                 user_user_iterations_number,
                 number_of_vlad_clusters):

        self.batch_size = batch_size
        self.negative_samples = negative_samples
        self.embedding_size = embedding_size
        self.window_size = window_size
        self.learning_rate = learning_rate
        self.dropout = dropout

        self.data = data
        self.filtered_item_to_user_matrix = data["filtered_item_to_user_matrix"]
        self.popularity_matrix = data["popularity_matrix"]
        self.anonymize_user = data["anonymize_user"]
        self.true_user_id_to_user_id = data["true_user_id_to_user_id"]
        self.true_user_id_set = set(self.true_user_id_to_user_id.keys())
        self.user_id_set = set(self.true_user_id_to_user_id.values())
        self.user_list = data["user_list"]
        self.shuffle = shuffle

        self.item_to_user_array = self.item_to_user_array(data["filtered_item_to_user_matrix"])
        self.item_to_user_array_of_arrays_csr,\
        self.item_to_user_array_of_arrays_csc,\
        self.item_to_user_effective_doc = self.sparse_matrix_to_array_of_arrays(data["filtered_item_to_user_matrix"])

        self.number_of_items = self.data["number_of_items"]
        self.number_of_users = self.data["number_of_users"]

        self.data_splits = data["data_splits"]
        self.train_index, self.val_index, self.test_index = self.data_splits

        self.dataset = dataset
        self.async_batch_size = async_batch_size
        self.item_ids_to_true_item_ids = data["item_indices_sorted"]

        self.user_user_iterations_number = user_user_iterations_number
        self.number_of_vlad_clusters = number_of_vlad_clusters
        if self.number_of_vlad_clusters > 0:
            self.vector_aggregation = "vlad"
        elif self.number_of_vlad_clusters == 0:
            self.vector_aggregation = "mean"
        else:
            raise ValueError("Invalid number of VLAD clusters selected.")
        self.shuffle = shuffle

        print("Set class fields.")

        self.l = list()

        self.user_user_batch_train_function, \
        self.l1_user = self.build_model()

        print("Built model.")

    def build_model(self):
        # Define tensor variables.
        x_user = T.ivector("x_user")
        x_user_context = T.ivector("x_user_context")

        y_labels = T.vector("y_emb")

        ################################################################################################################
        # Unsupervised embedding learning.
        ################################################################################################################
        l_in_user = InputLayer(shape=(None,), input_var=x_user)
        l_in_user_context = InputLayer(shape=(None,), input_var=x_user_context)

        l1_user = EmbeddingLayer(l_in_user,
                                 input_size=self.number_of_users,
                                 output_size=self.embedding_size,
                                 W=lasagne.init.GlorotUniform(gain=1.0))
        l1_user_context = EmbeddingLayer(l_in_user_context,
                                         input_size=self.number_of_users,
                                         output_size=self.embedding_size,
                                         W=lasagne.init.GlorotUniform(gain=1.0))

        l_user_user_merge = lasagne.layers.ElemwiseMergeLayer([l1_user, l1_user_context],
                                                              T.mul)

        self.l.append(l_user_user_merge)

        user_user_embedding_merge = lasagne.layers.get_output(l_user_user_merge)

        user_user_loss = - T.log(T.nnet.sigmoid(T.sum(user_user_embedding_merge, axis=1) * y_labels)).sum()

        l_user_user_merge_params = lasagne.layers.get_all_params(l_user_user_merge, trainable=True)

        user_user_updates = lasagne.updates.adam(user_user_loss,
                                                 l_user_user_merge_params,
                                                 learning_rate=self.learning_rate)

        user_user_batch_train_function = theano.function([x_user,
                                                          x_user_context,
                                                          y_labels],
                                                         user_user_loss,
                                                         updates=user_user_updates,
                                                         on_unused_input="ignore")

        return user_user_batch_train_function, \
               l1_user

    @staticmethod
    def item_to_user_array(item_to_user):
        item_to_user = spsp.csr_matrix(item_to_user)

        item_to_user_array = np.ndarray(item_to_user.shape[0], dtype=np.ndarray)

        for i in range(item_to_user.shape[0]):
            array_row_indices = item_to_user.getrow(i).indices
            if array_row_indices.size > 0:
                item_to_user_array[i] = array_row_indices
            else:
                raise ValueError

        return item_to_user_array

    @staticmethod
    def sparse_matrix_to_array_of_arrays(sparse_array):
        sparse_array = spsp.csr_matrix(sparse_array)
        array_of_arrays_csr = np.ndarray(sparse_array.shape[0], dtype=np.ndarray)
        for i in range(sparse_array.shape[0]):
            array_of_arrays_csr[i] = sparse_array.getrow(i).indices

        effective_doc = list()

        sparse_array = spsp.csc_matrix(sparse_array)
        array_of_arrays_csc = np.ndarray(sparse_array.shape[1], dtype=np.ndarray)
        for j in range(sparse_array.shape[1]):
            array_of_arrays_csc[j] = sparse_array.getcol(j).indices
            if array_of_arrays_csc[j].size > 1:
                effective_doc.append(j)

        effective_doc = np.array(effective_doc, dtype=np.int32)

        return array_of_arrays_csr, array_of_arrays_csc, effective_doc

    def gen_batches_doc_based(self, array_of_arrays_csr, array_of_arrays_csc, effective_doc, number_of_samples):
        if number_of_samples is None:
            doc_basis = effective_doc
        else:
            doc_basis = np.random.choice(effective_doc, size=number_of_samples, replace=True)

        # doc_list = np.empty((self.batch_size,), dtype=np.int32)
        target_list = np.empty((self.batch_size,), dtype=np.int32)
        context_list = np.empty((self.batch_size,), dtype=np.int32)
        label_list = np.empty((self.batch_size,), dtype=np.float32)
        counter = 0

        positive_samples_per_batch = int(self.batch_size * (1.0 - self.negative_samples))
        negative_samples_per_batch = self.batch_size - positive_samples_per_batch

        for doc in doc_basis:
            # context_size = min(self.window_size, array_of_arrays_csc[doc].size)
            double_context_size = min(2*self.window_size, array_of_arrays_csr[doc].size)
            context_size = double_context_size // 2
            if context_size == 0:
                continue

            # target_list_to_add = np.random.choice(array_of_arrays_csc[doc],
            #                                       size=context_size,
            #                                       replace=False)
            # context_list_to_add = np.random.choice(array_of_arrays_csc[doc],
            #                                        size=context_size,
            #                                        replace=True)

            word_list_to_add = np.random.choice(array_of_arrays_csr[doc],
                                                size=2*context_size,
                                                replace=False)

            # for c_index in range(context_size):
            #     retry = 5
            #     while target_list_to_add[c_index] == context_list_to_add[c_index]:
            #         if retry == 0:
            #             break
            #         context_list_to_add[c_index] = np.random.choice(array_of_arrays_csc[doc])
            #         retry -= 1

            for target, context in zip(word_list_to_add[:context_size], word_list_to_add[context_size:]):
                # doc_list[counter] = doc
                target_list[counter] = target
                context_list[counter] = context

                label_list[counter] = 1.0
                counter += 1

                if counter == positive_samples_per_batch:
                    if self.negative_samples > 0:
                        appeared_index = np.random.choice(counter,
                                                          negative_samples_per_batch)
                        # doc_list[counter:] = doc_list[:counter][appeared_index]
                        target_list[counter:] = target_list[:counter][appeared_index]
                        context_list[counter:] = np.random.randint(low=0,
                                                                   high=array_of_arrays_csc.size,
                                                                   size=(negative_samples_per_batch,))
                        label_list[counter:] = [-1.0]

                    if self.shuffle:
                        perm_index = np.random.permutation(np.arange(len(label_list)))
                        # doc_list = doc_list[perm_index]
                        target_list = target_list[perm_index]
                        context_list = context_list[perm_index]
                        label_list = label_list[perm_index]

                    yield target_list, \
                          context_list, \
                          label_list

                    # doc_list = np.empty((self.batch_size,), dtype=np.int32)
                    target_list = np.empty((self.batch_size,), dtype=np.int32)
                    context_list = np.empty((self.batch_size,), dtype=np.int32)
                    label_list = np.empty((self.batch_size,), dtype=np.float32)
                    counter = 0

        if counter == 0:
            raise StopIteration
        else:
            if self.negative_samples > 0:
                num_negative_samples = self.batch_size - counter

                appeared_index = np.random.choice(counter,
                                                  num_negative_samples)
                # doc_list[counter:] = doc_list[:counter][appeared_index]
                target_list[counter:] = target_list[:counter][appeared_index]
                context_list[counter:] = np.random.randint(low=0,
                                                           high=array_of_arrays_csc.size,
                                                           size=(num_negative_samples,))
                label_list[counter:] = [-1.0]

            if self.shuffle:
                perm_index = np.random.permutation(np.arange(len(label_list)))
                # doc_list = doc_list[perm_index]
                target_list = target_list[perm_index]
                context_list = context_list[perm_index]
                label_list = label_list[perm_index]

            yield target_list, \
                  context_list, \
                  label_list

    def gen_user_user(self):
        for target_list, context_list, label_list in self.gen_batches_doc_based(array_of_arrays_csr=self.item_to_user_array_of_arrays_csr,
                                                                                array_of_arrays_csc=self.item_to_user_array_of_arrays_csc,
                                                                                effective_doc=self.item_to_user_effective_doc,
                                                                                number_of_samples=self.user_user_iterations_number):
            yield target_list, context_list, label_list

    def aggregate_vectors_function_mean(self):
        # Get user embeddings.
        params = self.l1_user.get_params()

        user_embeddings = params[0].get_value()

        # Aggregation for all the items.
        X = np.zeros([self.number_of_items, self.embedding_size], dtype=np.float32)

        for item_id in range(self.number_of_items):

            user_ids = self.item_to_user_array[item_id]

            if user_ids.size > 0:
                item_user_embeddings = user_embeddings[user_ids, :]

                X[item_id, :] = item_user_embeddings.mean(axis=0)

        return X

    def aggregate_vectors_function_vlad(self):
        # K-means on user embeddings.
        params = self.l1_user.get_params()

        user_embeddings = params[0].get_value()

        community_dictionary = KMeans(n_clusters=self.number_of_vlad_clusters,
                                      init='k-means++',
                                      tol=0.0001,
                                      random_state=0).fit(user_embeddings)

        # Aggregation for all the items.
        centers = community_dictionary.cluster_centers_

        X = np.zeros([self.number_of_items, self.number_of_vlad_clusters * self.embedding_size], dtype=np.float32)

        for item_id in range(self.number_of_items):

            user_ids = self.item_to_user_array[item_id]

            item_user_embeddings = user_embeddings[user_ids, :]

            predictedLabels = community_dictionary.predict(item_user_embeddings)

            for centroid in range(self.number_of_vlad_clusters):
                # if there is at least one descriptor in that cluster
                if np.sum(predictedLabels == centroid) > 0:
                    # add the diferences
                    X[item_id, centroid * self.embedding_size:(centroid + 1) * self.embedding_size] = np.sum(item_user_embeddings[predictedLabels == centroid, :] - centers[centroid], axis=0)

            # power normalization, also called square-rooting normalization
            X[item_id, :] = np.sign(X[item_id, :]) * np.sqrt(np.abs(X[item_id, :]))

            # L2 normalization
            X[item_id, :] = X[item_id, :] / np.sqrt(np.dot(X[item_id, :], X[item_id, :]))

        return X

    def evaluate_supervised_function(self):
        if self.vector_aggregation == "mean":
            X = self.aggregate_vectors_function_mean()
        elif self.vector_aggregation == "vlad":
            X = self.aggregate_vectors_function_vlad()
        else:
            raise ValueError("Invalid vector aggragation method.")

        # Linear Regression.
        model = LinearRegression()
        model.fit(X[self.train_index, :], self.popularity_matrix[self.train_index, 2])

        y_pred = model.predict(X[self.val_index, :])

        loss_val = np.mean(np.power(y_pred - self.popularity_matrix[self.val_index, 2], 2))

        return loss_val

    def gen_instance_supervised(self, indices, shuffle):
        if shuffle:
            indices_effective = np.array(np.random.permutation(indices), dtype=np.int32)
        else:
            indices_effective = indices.astype(dtype=np.int32)
        i = 0
        while i < indices_effective.size:
            j = i + self.batch_size
            if j > indices_effective.size:
                j = indices_effective.size

                x_item = np.empty(j - i, dtype=np.int32)
                x_item[:] = indices_effective[i: j]
                y_supervision_labels = self.popularity_labels[indices_effective[i: j]]

                yield x_item, \
                      y_supervision_labels
                break
            else:
                x_item = np.empty(self.batch_size, dtype=np.int32)
                x_item[:] = indices_effective[i: j]
                y_supervision_labels = self.popularity_labels[indices_effective[i: j]]

                yield x_item, \
                      y_supervision_labels
                i = j

    def train(self, number_of_epochs, patience, model_file_path):
        # previous_best_loss = self.evaluate_supervised_function()
        # 
        # print(previous_best_loss)
        # previous_best_loss = 500.0
        # 
        # for x_user_target, x_user_context, y_label in self.gen_user_user():
        #     _ = self.user_user_batch_train_function(x_user_target, x_user_context, y_label)
        # 
        # for x_user_target, x_user_context, y_label in self.gen_user_user():
        #     _ = self.user_user_batch_train_function(x_user_target, x_user_context, y_label)
        # 
        # for x_user_target, x_user_context, y_label in self.gen_user_user():
        #     _ = self.user_user_batch_train_function(x_user_target, x_user_context, y_label)

        for x_user_target, x_user_context, y_label in self.gen_user_user():
            _ = self.user_user_batch_train_function(x_user_target, x_user_context, y_label)

        previous_best_loss = self.evaluate_supervised_function()
        print(previous_best_loss)

        no_improvement_counter = 0
        epoch = 0

        while (no_improvement_counter < patience) and (epoch < number_of_epochs):
            loss = 0.
            user_user_loss = 0.

            for x_user_target, x_user_context, y_label in self.gen_user_user():
                loss_to_add = self.user_user_batch_train_function(x_user_target, x_user_context, y_label)
                loss += loss_to_add
                user_user_loss += loss_to_add

            loss_val = self.evaluate_supervised_function()

            if loss_val < previous_best_loss:
                self.store_params(model_file_path)
                previous_best_loss = loss_val
                no_improvement_counter = 0
            else:
                no_improvement_counter += 1

            epoch += 1

            print(epoch,
                  loss,
                  user_user_loss,
                  loss_val)

    def store_params(self, model_file):
        for i, l in enumerate(self.l):
            fout = open("{}.{}".format(model_file, i), "wb")
            params = lasagne.layers.get_all_param_values(l)
            cPickle.dump(params, fout, cPickle.HIGHEST_PROTOCOL)
            fout.close()

    def load_params(self, model_file):
        for i, l in enumerate(self.l):
            fin = open("{}.{}".format(model_file, i), "rb")
            params = cPickle.load(fin)
            lasagne.layers.set_all_param_values(l, params)
            fin.close()
