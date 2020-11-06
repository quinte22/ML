""" Methods for doing logistic regression."""

import numpy as np
from math import log2
from check_grad import check_grad

from utils import sigmoid

def logistic_predict(weights, data):
    """
    Compute the probabilities predicted by the logistic classifier.

    Note: N is the number of examples and
          M is the number of features per example.

    Inputs:
        weights:    (M+1) x 1 vector of weights, where the last element
                    corresponds to the bias (intercepts).
        data:       N x M data matrix where each row corresponds
                    to one data point.
    Outputs:
        y:          :N x 1 vector of probabilities. This is the output of the classifier.
    """

    # TODO: Finish this function
    # compute b
    #
    # b = weights[weights.shape[0]-1, :]
    # weight_final = np.delete(weights, weights.shape[0]-1, 0)
    # mat mul vs dot
    ar = [[1] for i in range(data.shape[0])]
    array_1 = np.array(ar)
    new_data = np.c_[data,array_1 ]
    z = new_data.dot(weights)
    y = sigmoid(z)

    return y

def evaluate(targets, y):
    """
    Compute evaluation metrics.
    Inputs:
        targets : N x 1 vector of targets.
        y       : N x 1 vector of probabilities.
    Outputs:
        ce           : (scalar) Cross entropy. CE(p, q) = E_p[-log q]. Here we want to compute CE(targets, y)
        frac_correct : (scalar) Fraction of inputs classified correctly.
    """
    class_1_cost = targets * np.log(y)
    class_2_cost = (1-targets) * np.log(1-y)
    ce = - (np.sum((class_1_cost + class_2_cost)))/targets.shape[0]
    correct = 0
    for i in range(len(targets)):
        pred = 0 if y[i] < 0.5 else 1
        if targets[i] == pred:
            correct += 1
    frac_correct = correct/targets.shape[0]

    return ce, frac_correct

def logistic(weights, data, targets, hyperparameters):
    """
    Calculate negative log likelihood and its derivatives with respect to weights.
    Also return the predictions.

    Note: N is the number of examples and
          M is the number of features per example.

    Inputs:
        weights:    (M+1) x 1 vector of weights, where the last element
                    corresponds to bias (intercepts).
        data:       N x M data matrix where each row corresponds
                    to one data point.
        targets:    N x 1 vector of targets class probabilities.
        hyperparameters: The hyperparameters dictionary.

    Outputs:
        f:       The sum of the loss over all data points. This is the objective that we want to minimize.
        df:      (M+1) x 1 vector of derivative of f w.r.t. weights.
        y:       N x 1 vector of probabilities.
    """

    # TODO: Finish this function
    y = logistic_predict(weights, data)

    f, frac_co = evaluate(targets, y)
# df to be determine
    df = data.transpose().dot((y - targets))
    df = np.r_[df,np.array([[0]])]


    return f, df, y



def logistic_pen(weights, data, targets, hyperparameters):
    """
    Calculate negative log likelihood and its derivatives with respect to weights.
    Also return the predictions.

    Note: N is the number of examples and
          M is the number of features per example.

    Inputs:
        weights:    (M+1) x 1 vector of weights, where the last element
                    corresponds to bias (intercepts).
        data:       N x M data matrix where each row corresponds
                    to one data point.
        targets:    N x 1 vector of targets class probabilities.
        hyperparameters: The hyperparameters dictionary.

    Outputs:
        f:             The sum of the loss over all data points. This is the objective that we want to minimize.
        df:            (M+1) x 1 vector of derivative of f w.r.t. weights.
    """

    y = logistic_predict(weights, data)

    f, frac_co = evaluate(targets, y)
    # df to be determine
    # find norm of the weights?
    df = data.transpose().dot((y - targets))

    df = np.r_[df, np.array([[0]])] + (hyperparameters['weight_regularization'] * weights)


    return f, df, y



# if "__main__" == __name__:
#     # training_set = np.load("mnist_train_small.npz")
#     # print(training_set.files)
#     # training_inputs = training_set["train_inputs_small"]
#     # print("training", training_inputs.shape)
#     # weights_1= np.random.rand(785,1)
#     # print("weights", weights_1.shape)
#     # y_final = logistic_predict(weights_1,training_inputs)
#     # print(y_final)


