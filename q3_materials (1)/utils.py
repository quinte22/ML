import numpy as np
from matplotlib import pyplot as plt


def l2_distance(a, b):
    """Computes the Euclidean distance matrix between a and b.
    """

    if a.shape[0] != b.shape[0]:
        raise ValueError("A and B should be of same dimensionality")

    aa = np.sum(a**2, axis=0)
    bb = np.sum(b**2, axis=0)
    ab = np.dot(a.T, b)

    return np.sqrt(aa[:, np.newaxis] + bb[np.newaxis, :] - 2*ab)

def run_knn(train_data, train_labels, valid_data):
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
    nearest = np.argsort(dist, axis=1)[:,:1]

    train_labels = train_labels.reshape(-1)
    valid_labels = train_labels[nearest]

    # note this only works for binary labels
    valid_labels = (np.mean(valid_labels, axis=1) >= 0.5).astype(np.int)
    valid_labels = valid_labels.reshape(-1,1)

    return valid_labels

def load_data(filename, load2=True, load3=True):
  """Loads data for 2's and 3's
  Inputs:
    filename: Name of the file.
    load2: If True, load data for 2's.
    load3: If True, load data for 3's.
  """
  assert (load2 or load3), "Atleast one dataset must be loaded."
  data = np.load(filename)
  if load2 and load3:
    inputs_train = np.hstack((data['train2'], data['train3']))
    inputs_valid = np.hstack((data['valid2'], data['valid3']))
    inputs_test = np.hstack((data['test2'], data['test3']))
    target_train = np.hstack((np.zeros((1, data['train2'].shape[1])), np.ones((1, data['train3'].shape[1]))))
    target_valid = np.hstack((np.zeros((1, data['valid2'].shape[1])), np.ones((1, data['valid3'].shape[1]))))
    target_test = np.hstack((np.zeros((1, data['test2'].shape[1])), np.ones((1, data['test3'].shape[1]))))
  else:
    if load2:
      inputs_train = data['train2']
      target_train = np.zeros((1, data['train2'].shape[1]))
      inputs_valid = data['valid2']
      target_valid = np.zeros((1, data['valid2'].shape[1]))
      inputs_test = data['test2']
      target_test = np.zeros((1, data['test2'].shape[1]))
    else:
      inputs_train = data['train3']
      target_train = np.zeros((1, data['train3'].shape[1]))
      inputs_valid = data['valid3']
      target_valid = np.zeros((1, data['valid3'].shape[1]))
      inputs_test = data['test3']
      target_test = np.zeros((1, data['test3'].shape[1]))

  return inputs_train.T, inputs_valid.T, inputs_test.T, target_train.T, target_valid.T, target_test.T

# end def load_data

if __name__ == "__main__":
    input_train, input_valid, input_test, targ_train, targ_valid, targ_test = load_data("digits.npz")

    # PCA ALGORITHM
    # Subtract the mean from each dimension (centering)
    m = np.mean(input_train, axis=0)
    valid_centered = input_train - np.tile(m, (input_train.shape[0], 1))

    # Calculate the covariance matrix of the data;
    C = np.cov(valid_centered.T)
    # PCA (or equivalently SVD or EVD) SVD and EVD are equaivalent since C is symmetric PSD
    U, S, V = np.linalg.svd(C)
    # S is eigen
    # Project the data onto the first principal component, then back into 2D space
    kns = [2, 5, 10, 20, 30]
    class_err = []
    for k in kns:
        X_recon = valid_centered.dot(U[:, :k])
        nn_1 = run_knn( X_recon, targ_train, np.matmul(input_valid, U[:, :k]))
        acc = 0
        for p in range(len(nn_1)):
            if nn_1[p] == targ_valid[p]:
                acc += 1
        acc/= len(nn_1)
        print("acc is {} for k : {}".format(acc, k))

        class_err.append(1-acc)



    #testing
    X_recon = valid_centered.dot(U[:, :30])
    nn_1 = run_knn(X_recon, targ_train, np.matmul(input_test, U[:, :30]))
    acc = 0
    for p in range(len(nn_1)):
        if nn_1[p] == targ_test[p]:
            acc += 1
    acc /= len(nn_1)
    print("acc is {} for k : {}".format(acc, 30))

    # # plot centered data and its reconstruction
    plt.plot(kns, class_err)
    plt.xlabel("K Eig Used")
    plt.ylabel("classification error")
    plt.show()



