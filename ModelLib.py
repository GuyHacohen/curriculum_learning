
import numpy as np
import transfer_learning


class ModelLib():

    def build_classifier_model(self, dataset):
        raise NotImplementedError


    def corriculum_svm_based_training_data(self, dataset, anti_corriculum=False, random=False):
        (transfer_values_train, transfer_values_test) = transfer_learning.get_transfer_values_inception(dataset)
        train_scores, test_scores = transfer_learning.get_svm_scores(transfer_values_train, dataset.y_train,
                                                                     transfer_values_test, dataset.y_test, dataset)
        order = transfer_learning.rank_data_according_to_score(train_scores, dataset.y_train, reverse=anti_corriculum,
                                                               random=random)
        size_data = dataset.x_train.shape[0]
        epochs_each_data = 10
        jumps = 0.1
        data_sizes = list(int(size_data * frac) for frac in (np.arange(0, 1, jumps) + jumps))
        epochs = [epochs_each_data] * len(data_sizes)
        total_batchs = sum(epoch * data_size for epoch, data_size in zip(epochs, data_sizes))
        total_batchs_original = 100 * size_data
        epochs[-1] += (total_batchs_original - total_batchs) // size_data

        def data_function(x, y, cur_phase, num_phases):
            data_limit = data_sizes[cur_phase]
            new_data = order[:data_limit]
            return x[new_data, :, :, :], y[new_data, :]

        return epochs, data_function
