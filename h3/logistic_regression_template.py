import numpy as np
from check_grad import check_grad
from utils import *
from logistic import *
from matplotlib import pyplot



def run_logistic_regression(hype, w):
    # train_inputs, train_targets = load_train()
    train_inputs, train_targets = load_train_small()
    valid_inputs, valid_targets = load_valid()
    test_inputs , test_targets = load_test()

    N, M = train_inputs.shape

    # TODO: Set hyperparameters
    hyperparameters = hype

    # Logistic regression weights
    # TODO:Initialize to random weights here.
    np.random.seed(311)
    weights = np.random.rand(M + 1, 1) / w

    # Verify that your logistic function produces the right gradient.
    # diff should be very close to 0.
    run_check_grad(hyperparameters)


    # Begin learning with gradient descent
    entropy_val = [None] * hyperparameters['num_iterations']
    entropy_train = [None] * hyperparameters['num_iterations']
    ce_val = [None] * hyperparameters['num_iterations']
    ce_train = [None] * hyperparameters['num_iterations']
    for t in range(hyperparameters['num_iterations']):

        # TODO: you may need to modify this loop to create plots, etc.

        # Find the negative log likelihood and its derivatives w.r.t. the weights.
        f, df, predictions = logistic_pen(weights, train_inputs, train_targets, hyperparameters)

        # Evaluate the prediction.
        cross_entropy_train, frac_correct_train = evaluate(train_targets, predictions)

        if np.isnan(f) or np.isinf(f):
            raise ValueError("nan/inf error")

        # update parameters

        weights = weights - hyperparameters['learning_rate'] * df / N

        # Make a prediction on the valid_inputs.
        predictions_valid = logistic_predict(weights, valid_inputs)

        # Evaluate the prediction.
        cross_entropy_valid, frac_correct_valid = evaluate(valid_targets, predictions_valid)
        entropy_train[t] = cross_entropy_train
        ce_train[t] = frac_correct_train
        ce_val[t] = frac_correct_valid
        entropy_val[t] = cross_entropy_valid



        # print some stats
        print ("ITERATION:{:4d}  TRAIN NLOGL:{:4.2f}  TRAIN CE:{:.6f} TRAIN FRAC:{:2.2f}  VALID CE:{:.6f}  VALID FRAC:{:2.2f}".format(t+1, f / N, cross_entropy_train, frac_correct_train*100,
                   cross_entropy_valid, frac_correct_valid*100))
        # make pred on the test inputs
    # prediction_test = logistic_predict(weights, test_inputs)
    # cross_entropy_test, frac_correct_test = evaluate(test_targets, prediction_test)
    # print("test FRAC:{:2.2f}  test CE:{:.6f}".format(frac_correct_test, cross_entropy_test))
    # print("entro train")
    # print(entropy_train)
    # print("entro val")
    # print(entropy_val)
    # hype_list = [i for i in range(hyperparameters["num_iterations"])]
    # pyplot.plot(hype_list, entropy_train, hype_list, entropy_val)
    # pyplot.gca().legend(("training", "valid"))
    # pyplot.xlabel("num_iterations")
    # pyplot.ylabel("cross entropy")
    # pyplot.show()
    return entropy_val[-1], entropy_train[-1], ce_val[-1], ce_train[-1]


def run_check_grad(hyperparameters):
    """Performs gradient check on logistic function.
    """

    # This creates small random data with 20 examples and
    # 10 dimensions and checks the gradient on that data.
    num_examples = 20
    num_dimensions = 10

    weights = np.random.randn(num_dimensions+1, 1)
    data    = np.random.randn(num_examples, num_dimensions)
    targets = np.random.rand(num_examples, 1)

    diff = check_grad(logistic,      # function to check
                      weights,
                      0.001,         # perturbation
                      data,
                      targets,
                      hyperparameters)

    print ("diff =", diff)

if __name__ == '__main__':
    # hyper = {}
    # run_check_grad({})
    # learning_rate = [1, 0.2, 0.3, 0.333, 0.333333333, 0.5]
    # num_iterations = [ 2, 3, 5, 10]
    weight = [0, 0.001, 0.01, 0.1, 1]
    hyper = {}
    # min_cross_val = -9999999999
    # min_cross_train = -99999999
    # min_frac_cros = -9999999999
    # min_frac_val = -99999999999
    # saved = []
    # final_cross = []
    # saved2 = []
    # final_cross2 = []
    # final_cross3 = []
    # final_cross4 = []
    # saved3 = []
    # saved4 = []
    ce_val_list = [None] * len(weight)
    ce_t_list = [None] * len(weight)
    frac_val_list = [None] * len(weight)
    frac_train_list = [None] * len(weight)

    for w2 in range(len(weight)):
        ce_avg_v = 0
        frac_val_v = 0
        ce_avg_t = 0
        frac_val_t = 0
        for val in range(10):
                hyper['learning_rate'] = 0.333,
                hyper['weight_regularization'] = weight[w2]
                hyper['num_iterations'] = 3
                e_v, e_t, c_v, c_t = run_logistic_regression(hyper, 7)
                ce_avg_v += e_v
                ce_avg_t += e_t
                frac_val_t += c_t
                frac_val_v += c_v
        ce_val_list[w2] = ce_avg_v /10
        frac_val_list[w2] = frac_val_v / 10
        ce_t_list[w2] = ce_avg_t /10
        frac_train_list[w2] = frac_val_t /10

    pyplot.plot(weight, ce_val_list, weight, ce_t_list)
    pyplot.gca().legend(("training", "valid"))
    pyplot.xlabel("weights")
    pyplot.ylabel("cross entropy")
    pyplot.show()
    pyplot.plot(weight, frac_val_list, weight, frac_train_list)
    pyplot.gca().legend(("training", "valid"))
    pyplot.xlabel("weights")
    pyplot.ylabel("classification error")
    pyplot.show()




    print("the right valid params are {}".format(saved))
    print(min_cross_val)

    #
    # print("the right  params are {}".format(saved2))
    # print(min_cross_train)
    # # best params for 2.2
    # hyperparams = {'learning_rate': 0.3333333333333333, 'weight_regularization': 1, 'num_iterations':10}
    # run_logistic_regression(hyperparams, 7 )



