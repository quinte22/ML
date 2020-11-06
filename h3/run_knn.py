import numpy as np
from l2_distance import l2_distance
from matplotlib import pyplot
def run_knn(k, train_data, train_labels, valid_data):
    """Uses the supplied training inputs and labels to make
    predictions for validation data using the K-nearest neighbours
    algorithm.

    Note: N_TRAIN is the number of training examples,
          N_VALID is the number of validation examples,
          and M is the number of features per example.

    Inputs:
        k:            The number of neighbours to use for classification
                      of a validation example.
        train_data:   The N_TRAIN x M array of training
                      data.
        train_labels: The N_TRAIN x 1 vector of training labels
                      corresponding to the examples in train_data
                      (must be binary).
        valid_data:   The N_VALID x M array of data to
                      predict classes for.

    Outputs:
        valid_labels: The N_VALID x 1 vector of predicted labels
                      for the validation data.
    """

    dist = l2_distance(valid_data.T, train_data.T)
    nearest = np.argsort(dist, axis=1)[:,:k]

    train_labels = train_labels.reshape(-1)
    valid_labels = train_labels[nearest]

    # note this only works for binary labels
    valid_labels = (np.mean(valid_labels, axis=1) >= 0.5).astype(np.int)
    valid_labels = valid_labels.reshape(-1,1)

    return valid_labels


if "__main__" == __name__:
    # set it up
    training_set = np.load("mnist_train.npz")
    train_inputs = training_set["train_inputs"]
    train_targets = training_set["train_targets"]
    valid_set = np.load("mnist_valid.npz")
    valid_inputs = valid_set['valid_inputs']
    valid_target = valid_set['valid_targets']
    test_set = np.load("mnist_test.npz")
    test_inputs = test_set["test_inputs"]
    test_target = test_set["test_targets"]
    correct_list = [0.0] * 5
    k_vals = [1,3,5,7,9]
    for k in range(len(k_vals)):
        valid_predicted = run_knn(k_vals[k],train_inputs, train_targets, valid_inputs)
        correct = 0
        for i in range(len(valid_predicted)):
            if valid_target[i] == valid_predicted[i]:
                correct+=1
        correct_list[k] = correct / len(valid_predicted)
    print(correct_list)
    pyplot.plot(k_vals, correct_list)
    pyplot.xlabel("k_vals")
    pyplot.ylabel("classification rate")
    pyplot.show()
    k_test_vals = [3,5,7]
    correct_test = [0.0] * 3
    for k in range(len(k_test_vals)):
        test_predict = run_knn(k_test_vals[k], train_inputs, train_targets, test_inputs)
        correct = 0
        for i in range(len(test_predict)):
            if test_predict[i] == test_target[i]:
                correct += 1
        correct_test[k] = correct/len(test_predict)
    pyplot.plot(k_test_vals, correct_test)
    pyplot.xlabel("k_vals")
    pyplot.ylabel("classification rate")
    pyplot.show()











