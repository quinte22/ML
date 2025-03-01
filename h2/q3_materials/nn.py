"""
Instruction:

In this section, you are asked to train a NN with different hyperparameters.
To start with training, you need to fill in the incomplete code. There are 3
places that you need to complete:
a) Backward pass equations for an affine layer (linear transformation + bias).
b) Backward pass equations for ReLU activation function.
c) Weight update equations with momentum.

After correctly fill in the code, modify the hyperparameters in "main()".
You can then run this file with the command: "python nn.py" in your terminal.
The program will automatically check your gradient implementation before start.
The program will print out the training progress, and it will display the
training curve by the end. You can optionally save the model by uncommenting
the lines in "main()".
"""

from __future__ import division
from __future__ import print_function
import matplotlib.pyplot as plt

from util import LoadData, Load, Save, DisplayPlot
import sys
import numpy as np


def InitNN(num_inputs, num_hiddens, num_outputs):
    """Initializes NN parameters.

    Args:
        num_inputs:    Number of input units.
        num_hiddens:   List of two elements, hidden size for each layer.
        num_outputs:   Number of output units.

    Returns:
        model:         Randomly initialized network weights.
    """
    np.random.seed(311)
    W1 = 0.1 * np.random.randn(num_inputs, num_hiddens[0])
    W2 = 0.1 * np.random.randn(num_hiddens[0], num_hiddens[1])
    W3 = 0.01 * np.random.randn(num_hiddens[1], num_outputs)
    b1 = np.zeros((num_hiddens[0]))
    b2 = np.zeros((num_hiddens[1]))
    b3 = np.zeros((num_outputs))
    model = {
        'W1': W1,
        'W2': W2,
        'W3': W3,
        'b1': b1,
        'b2': b2,
        'b3': b3
    }
    return model


def Affine(x, w, b):
    """Computes the affine transformation.

    Args:
        x: Inputs (or hidden layers)
        w: Weights
        b: Bias

    Returns:
        y: Outputs
    """
    # y = np.dot(w.T, x) + b

    y = x.dot(w) + b
    return y


def AffineBackward(grad_y, h, w):
    """Computes gradients of affine transformation.
    hint: you may need the matrix transpose np.dot(A,B).T = np.dot(B,A) and (A.T).T = A

    Args:
        grad_y: gradient from last layer
        h: inputs from the hidden layer
        w: weights

    Returns:
        grad_h: Gradients wrt. the inputs/hidden layer.
        grad_w: Gradients wrt. the weights.
        grad_b: Gradients wrt. the biases.
    """
    ###########################
    # Insert your code here.


    grad_h = grad_y.dot(w.transpose())
    grad_w = (h.transpose().dot(grad_y))
    grad_b = np.sum(grad_y, axis=0)
    return grad_h, grad_w, grad_b
    ###########################


def ReLU(z):
    """Computes the ReLU activation function.

    Args:
        z: Inputs

    Returns:
        h: Activation of z
    """
    return np.maximum(z, 0.0)


def ReLUBackward(grad_h, z, h2):
    """Computes gradients of the ReLU activation function wrt. the unactivated inputs.

    Returns:
        grad_z: Gradients wrt. the hidden state prior to activation.
    """
    ###########################
    # Insert your code here.

    grad_z = np.where(z<=0, 0, 1)
    grad_z = grad_z * grad_h
    return grad_z
    ###########################


def Softmax(x):
    """Computes the softmax activation function.

    Args:
        x: Inputs

    Returns:
        y: Activation
    """
    return np.exp(x) / np.exp(x).sum(axis=1, keepdims=True)


def NNForward(model, x):
    """Runs the forward pass.

    Args:
        model: Dictionary of all the weights.
        x:     Input to the network.

    Returns:
        var:   Dictionary of all intermediate variables.
    """
    z1 = Affine(x, model['W1'], model['b1'])
    h1 = ReLU(z1)
    z2 = Affine(h1, model['W2'], model['b2'])
    h2 = ReLU(z2)
    y = Affine(h2, model['W3'], model['b3'])
    var = {
        'x': x,
        'z1': z1,
        'h1': h1,
        'z2': z2,
        'h2': h2,
        'y': y
    }
    return var


def NNBackward(model, err, var):
    """Runs the backward pass.

    Args:
        model:    Dictionary of all the weights.
        err:      Gradients to the output of the network.
        var:      Intermediate variables from the forward pass.
    """
    dE_dh2, dE_dW3, dE_db3 = AffineBackward(err, var['h2'], model['W3'])
    dE_dz2 = ReLUBackward(dE_dh2, var['z2'], var['h2'])
    dE_dh1, dE_dW2, dE_db2 = AffineBackward(dE_dz2, var['h1'], model['W2'])
    dE_dz1 = ReLUBackward(dE_dh1, var['z1'], var['h1'])
    _, dE_dW1, dE_db1 = AffineBackward(dE_dz1, var['x'], model['W1'])
    model['dE_dW1'] = dE_dW1
    model['dE_dW2'] = dE_dW2
    model['dE_dW3'] = dE_dW3
    model['dE_db1'] = dE_db1
    model['dE_db2'] = dE_db2
    model['dE_db3'] = dE_db3
    pass


def NNUpdate(model, eps, momentum):
    """Update NN weights.

    Args:
        model:    Dictionary of all the weights.
        eps:      Learning rate.
        momentum: Momentum.
    """
    ###########################
    # Insert your code here.
    # Update the weights.
    # model['dE_dW1'] = dE_dW1
    # model['dE_dW2'] = dE_dW2
    # model['dE_dW3'] = dE_dW3
    # model['dE_db1'] = dE_db1
    # model['dE_db2'] = dE_db2
    # model['dE_db3'] = dE_db3

    v = model['W1'] - eps * model['dE_dW1']
    v2 = model['W2'] - eps * model['dE_dW2']
    v3 = model['W3'] - eps * model['dE_dW3']
    v4 = model['b1'] - eps * model['dE_db1']
    v5 = model['b2'] - eps * model['dE_db2']
    v6 = model['b3'] - eps * model['dE_db3']



    model['W1'] = momentum*model['W1'] + (1-momentum)* v
    model['W2'] = momentum*model['W2'] + (1-momentum)* v2
    model['W3'] = momentum*model['W3'] + (1-momentum)* v3
    model['b1'] = momentum*model['b1'] + (1-momentum)* v4
    model['b2'] = momentum*model['b2'] + (1-momentum)* v5
    model['b3'] = momentum*model['b3'] + (1-momentum)* v6
    ###########################

def plot_uncertain_images(x, t, prediction, threshold=0.5):
    """
    """
    low_index = np.max(prediction, axis=1)< threshold
    class_names = ['anger', 'disgust', 'fear', 'happy', 'sad', 'surprised', 'neutral']
    if np.sum(low_index)>0:
        for i in np.where(low_index>0)[0]:

            plt.figure()
            img_w, img_h = int(np.sqrt(2304)), int(np.sqrt(2304)) #2304 is input size
            plt.imshow(x[i].reshape(img_h,img_w))
            plt.title('P_max: {}, Predicted: {}, Target: {}'.format(np.max(prediction[i]), class_names[np.argmax(prediction[i])], class_names[np.argmax(t[i])]))
            plt.show()
            input("press enter to continue")
    return

def Train(model, forward, backward, update, eps, momentum, num_epochs,
          batch_size):
    """Trains a simple MLP.

    Args:
        model:           Dictionary of model weights.
        forward:         Forward prop function.
        backward:        Backward prop function.
        update:          Update weights function.
        eps:             Learning rate.
        momentum:        Momentum.
        num_epochs:      Number of epochs to run training for.
        batch_size:      Mini-batch size, -1 for full batch.

    Returns:
        stats:           Dictionary of training statistics.
            - train_ce:       Training cross entropy.
            - valid_ce:       Validation cross entropy.
            - train_acc:      Training accuracy.
            - valid_acc:      Validation accuracy.
    """
    inputs_train, inputs_valid, inputs_test, target_train, target_valid, \
        target_test = LoadData('toronto_face.npz')
    rnd_idx = np.arange(inputs_train.shape[0])
    train_ce_list = []
    valid_ce_list = []
    train_acc_list = []
    valid_acc_list = []
    num_train_cases = inputs_train.shape[0]
    if batch_size == -1:
        batch_size = num_train_cases
    num_steps = int(np.ceil(num_train_cases / batch_size))
    for epoch in range(num_epochs):
        np.random.shuffle(rnd_idx)
        inputs_train = inputs_train[rnd_idx]
        target_train = target_train[rnd_idx]
        for step in range(num_steps):
            # Forward prop.
            start = step * batch_size
            end = min(num_train_cases, (step + 1) * batch_size)
            x = inputs_train[start: end]
            t = target_train[start: end]

            var = forward(model, x)
            prediction = Softmax(var['y'])

            train_ce = -np.sum(t * np.log(prediction)) / x.shape[0]
            train_acc = (np.argmax(prediction, axis=1) ==
                         np.argmax(t, axis=1)).astype('float').mean()
            # print(('Epoch {:3d} Step {:2d} Train CE {:.5f} '
            #        'Train Acc {:.5f}').format(
            #     epoch, step, train_ce, train_acc))

            # Compute error.
            error = (prediction - t) / x.shape[0]

            # Backward prop.
            backward(model, error, var)

            # Update weights.
            update(model, eps, momentum)

            valid_ce, valid_acc = Evaluate(
                inputs_valid, target_valid, model, forward, batch_size=batch_size)
            # print(('Epoch {:3d} '
            #        'Validation CE {:.5f} '
            #        'Validation Acc {:.5f}\n').format(
            #     epoch, valid_ce, valid_acc))
            train_ce_list.append((epoch, train_ce))
            train_acc_list.append((epoch, train_acc))
            valid_ce_list.append((epoch, valid_ce))
            valid_acc_list.append((epoch, valid_acc))
    DisplayPlot(train_ce_list, valid_ce_list, 'Cross Entropy', number=0)
    DisplayPlot(train_acc_list, valid_acc_list, 'Accuracy', number=1)

    print()
    train_ce, train_acc = Evaluate(
        inputs_train, target_train, model, forward, batch_size=batch_size)
    valid_ce, valid_acc = Evaluate(
        inputs_valid, target_valid, model, forward, batch_size=batch_size)
    test_ce, test_acc = Evaluate(
        inputs_test, target_test, model, forward, batch_size=batch_size)
    print('CE: Train %.5f Validation %.5f Test %.5f' %
          (train_ce, valid_ce, test_ce))
    print('Acc: Train {:.5f} Validation {:.5f} Test {:.5f}'.format(
        train_acc, valid_acc, test_acc))

    stats = {
        'train_ce': train_ce_list,
        'valid_ce': valid_ce_list,
        'train_acc': train_acc_list,
        'valid_acc': valid_acc_list
    }

    return model, stats


def Evaluate(inputs, target, model, forward, batch_size=-1):
    """Evaluates the model on inputs and target.

    Args:
        inputs: Inputs to the network.
        target: Target of the inputs.
        model:  Dictionary of network weights.
    """
    num_cases = inputs.shape[0]
    if batch_size == -1:
        batch_size = num_cases
    num_steps = int(np.ceil(num_cases / batch_size))
    ce = 0.0
    acc = 0.0
    last_x = None
    last_t = None
    last_pred = None
    for step in range(num_steps):
        start = step * batch_size
        end = min(num_cases, (step + 1) * batch_size)
        x = inputs[start: end]
        t = target[start: end]
        prediction = Softmax(forward(model, x)['y'])
        ce += -np.sum(t * np.log(prediction))
        acc += (np.argmax(prediction, axis=1) == np.argmax(
            t, axis=1)).astype('float').sum()
        if step ==(num_steps-1):
            last_t = t
            last_x = x
            last_pred = prediction

    # call here
    plot_uncertain_images(last_x, last_t, last_pred, threshold=0.5)
    ce /= num_cases
    acc /= num_cases
    return ce, acc


def CheckGrad(model, forward, backward, name, x):
    """Check the gradients

    Args:
        model: Dictionary of network weights.
        name: Weights name to check.
        x: Fake input.
    """
    np.random.seed(0)
    var = forward(model, x)
    loss = lambda y: 0.5 * (y ** 2).sum()
    grad_y = var['y']
    backward(model, grad_y, var)
    grad_w = model['dE_d' + name].ravel()
    w_ = model[name].ravel()
    eps = 1e-7
    grad_w_2 = np.zeros(w_.shape)
    check_elem = np.arange(w_.size)
    np.random.shuffle(check_elem)
    # Randomly check 20 elements.
    check_elem = check_elem[:20]
    for ii in check_elem:
        w_[ii] += eps
        err_plus = loss(forward(model, x)['y'])
        w_[ii] -= 2 * eps
        err_minus = loss(forward(model, x)['y'])
        w_[ii] += eps
        grad_w_2[ii] = (err_plus - err_minus) / 2 / eps
    np.testing.assert_almost_equal(grad_w[check_elem], grad_w_2[check_elem],
                                   decimal=3)


def main():
    """Trains a NN."""
    model_fname = 'nn_model.npz'
    stats_fname = 'nn_stats.npz'

    # Hyper-parameters. Modify them if needed.
    num_hiddens = [16, 32]
    eps = 0.01
    momentum = 0.9
    num_epochs = 1000
    batch_size = 200

    # Input-output dimensions.
    num_inputs = 2304
    num_outputs = 7

    # Initialize model.
    model = InitNN(num_inputs, num_hiddens, num_outputs)

    # Uncomment to reload trained model here.
    model = Load("default.npz)

    # Check gradient implementation.

    x = np.random.rand(10, 48 * 48) * 0.1
    CheckGrad(model, NNForward, NNBackward, 'W3', x)
    CheckGrad(model, NNForward, NNBackward, 'b3', x)
    CheckGrad(model, NNForward, NNBackward, 'W2', x)
    CheckGrad(model, NNForward, NNBackward, 'b2', x)
    CheckGrad(model, NNForward, NNBackward, 'W1', x)
    CheckGrad(model, NNForward, NNBackward, 'b1', x)
    # Train model.
    stats = Train(model, NNForward, NNBackward, NNUpdate, eps,
                  momentum, num_epochs, batch_size)
    # print(stats)

    # Uncomment if you wish to save the model.
    #Save("default", model)

    # Uncomment if you wish to save the training statistics.
    # Save(stats_fname, stats)

if __name__ == '__main__':
    main()

    # eps = 0.01
    # momentum = 0.0
    # batch_size = 100
    #
    # for eps in [0.002, 0.05, 0.3, 0.7, 1.0]:
    #     print("\nEps: " + str(eps))
    #     main(eps, momentum, batch_size)
    #
    # eps = 0.01
    # momentum = 0.0
    # batch_size = 100
    # for momentum in [0.25, 0.5, 0.95]:
    #     print("\nMomentum: " + str(momentum))
    #     main(eps, momentum, batch_size)
    #
    # eps = 0.01
    # momentum = 0.0
    # # batch_size = 100
    # for batch_size in [200, 250, 450, 750, 900]:
    #     print("\nBatch Size : " + str(batch_size))
    #     main(eps, momentum, batch_size)

    # a = [10, 15, 40]
    # b = [20, 50, 70]
    # for i in a:
    #     print("First layer: {}".format(i))
    #     for j in b:
    #         print("second layer: {}".format(j))
    #         main(i,j)



