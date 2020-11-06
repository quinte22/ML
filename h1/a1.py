"""class for decision trees"""
# from sklearn import DecisionTreeClassifier
from sklearn.feature_extraction.text import CountVectorizer
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import scipy.sparse
from sklearn.datasets import load_iris
from sklearn import tree
import matplotlib
from matplotlib import pyplot
from math import log2

# -----------------------Part 2 --------------------------------------------------------------------------------
def load_data(fake_news, real_news):
    """loading data, vectorizing it and spliting training test and
    validation """
    # readlines and extract the features
    with open(fake_news, 'r') as file:
        fake_news_headlines = file.readlines()
    with open(real_news, 'r') as file:
        real_news_headlines = file.readlines()
    # call feature extraction.text.CountVectorizer
    cv = CountVectorizer()
    fit_matrix = cv.fit(fake_news_headlines+real_news_headlines)
    fake_matrix = cv.transform(fake_news_headlines)
    real_matrix = cv.transform(real_news_headlines)
    feature_names = cv.get_feature_names()
    fake_array = scipy.sparse.lil_matrix(fake_matrix).toarray()
    real_array = scipy.sparse.lil_matrix(real_matrix).toarray()
    # split the data using train_test_split [0.3] then split the test in half to get test and validation
    y_fake = [0] * len(fake_news_headlines)
    y_real = [1] * len(real_news_headlines)
    X = np.concatenate([fake_array, real_array])
    # X = fake_array.concat(real_array)
    y = y_fake + y_real
    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size=0.30, random_state=42,)
    split = len(X_test)//2
    #shallow copy
    X_spliting = X_test[:]
    X_test = X_test[:split]
    X_valid = X_spliting[split:]
    y_split = y_test[:]
    y_test = y_test[:split]
    y_valid = y_split[split:]
    return (X_train, X_test, X_valid, y_train, y_test, y_valid, feature_names)

# end def load_data

def select_model(X_train, X_valid, y_train, y_valid):
    # choose depth list
    max_depth_list = [3, 9, 27, 54, 81, 162, 243, 1000]
    max_accurancy = 0
    valid_params = (None, None)
    for max_depth in max_depth_list:
        dtc = DecisionTreeClassifier(max_depth=max_depth)
        dtc.fit(X=X_train, y=y_train)
        Y_predicted = dtc.predict(X=X_valid)
        accurate = 0
        total = len(X_valid)
        for i in range(len(X_valid)):
            if Y_predicted[i] == y_valid[i]:
                accurate+=1
        if accurate > max_accurancy:
            max_accurancy = accurate
            valid_params = (max_depth, "gini")
        print("max depth {}, accurancy: {} and type: {}".format(max_depth, accurate/total, "gini") )
        dtc_entropy = DecisionTreeClassifier(criterion='entropy', max_depth=max_depth)
        dtc_entropy.fit(X=X_train, y=y_train)
        Y_predicted_entropy = dtc_entropy.predict(X=X_valid)
        accurate = 0
        total = len(X_valid)
        for i in range(len(X_valid)):
            if Y_predicted_entropy[i] == y_valid[i]:
                accurate += 1
        if accurate > max_accurancy:
            max_accurancy = accurate
            valid_params = (max_depth, "entropy")
        print("max depth {}, accurancy: {} and type: {}".format(max_depth, accurate/total, "entropy") )

    return valid_params, max_accurancy/total

# end def select_model

def illustrate_tree(depth, type, X_train, y_train, features):
    dtc= DecisionTreeClassifier(criterion=type, max_depth=depth)
    # fit with training
    dtc.fit(X=X_train, y=y_train)
    with open("tree_{0}_{1}".format(type, depth), "w") as file:
      tree.export_graphviz(dtc, file, max_depth=2, feature_names=features)

    return tree
    # end def illustrate_tree

def compute_information_gain(X_train, y_train, feature_sample):
    fin = {}
    x_shape = X_train.shape
    total_train = x_shape[0]
    ig_dic = {}
    for key in list(features_sample.keys()):
        row = X_train[:,features_sample[key]]
        sum_1 = sum(row)
        print("sum of y  : {}".format(sum(y_train)))
        real_left_child = sum([y_train[i] for i in range(len(y_train)) if row[i]== 0])
        print("real left {}".format(real_left_child))
        total_left_child = len([y_train[i] for i in range(len(y_train)) if row[i]== 0])
        fake_left_child = total_left_child - real_left_child
        real_right_child = sum([y_train[i] for i in range(len(y_train)) if row[i]== 1])
        total_right_child = len([y_train[i] for i in range(len(y_train)) if row[i]== 1])
        fake_right_child = total_right_child - real_right_child
        print("real right {}".format(real_right_child))
        print("total left {}".format(total_left_child))
        print("total right {}".format(total_right_child))
        print("fake left {}".format(fake_left_child))
        print("fake right {}".format(fake_right_child))
        print("total train {}".format(total_train))



        fin[key] = sum_1
        prob_fake_root = (total_train - sum(y_train))/total_train
        prob_real_root = sum(y_train)/total_train
        prob_fake_left = fake_left_child/total_left_child
        prob_fake_right =fake_right_child/total_right_child
        prob_real_left = real_left_child/total_left_child
        prob_real_right = real_right_child/total_right_child
        prob_left = total_left_child/total_train
        prob_right = total_right_child/total_train
        root_entro = ((-prob_real_root*log2(prob_real_root)) -(prob_fake_root*log2(prob_fake_root)))
        left_entro =((-prob_fake_left*log2(prob_fake_left)) -(prob_real_left*log2(prob_real_left)))
        right_entro = ((-prob_real_right*log2(prob_real_right)) -(prob_fake_right*log2(prob_fake_right)))
        ig_dic[key] = root_entro -((left_entro*prob_left) + (right_entro*prob_right))
    return ig_dic
# ----------------------------------------- Part 3 --------------------------------


def load_data_vector(target_name, feature_name):
    """Returns an array with vector and a matrix, target_vector and feature_matrix"""
    data = {'X': np.genfromtxt(feature_name, delimiter=','),
            't': np.genfromtxt(target_name, delimiter=',')}
    print(data['t'])
    # create ref x_i to t_1
    return data
# end of load_data_vector

def shuffle_data(data):
    #np.shape , np.random.permutation , np._c (concat),
    # append t to x
    col = data['X'].shape[1]

    ref = np.c_[data['X'], data['t']]
    shuffled = np.random.permutation(ref)
    # shuffle_split = np.split(shuffled, [col], axis=1)
    # print("shuffle is {}".format(shuffle_split))
    t = shuffled[:,col]
    X = np.delete(shuffled, col, 1)
    # add them back together.
    # t = [i[0] for i in shuffle_split[1]]
    new_data = {'X': X, 't': t}


    return new_data

# end def shuffle_data

def split_data(data, num_folds, fold):
    # partition as num_folds
    ref = np.c_[data['X'], data['t']]
    col = data['X'].shape[1]
    split_data1 = np.split(ref, num_folds)
    data_left = []
    for i in range(num_folds):
        if i == fold:
            fold_x = split_data1[i]
        else:
            data_left.append(split_data1[i])

    data_left_2 = data_left[0]
    for i in range(len(data_left) -1):
        data_left_2 = np.concatenate((data_left_2,data_left[i+1]), axis=0)
    new_fold = np.split(fold_x, [col], axis=1)
    new_data = np.split(data_left_2, [col], axis=1)

    t_fold = np.array([i[0] for i in new_fold[1]])
    t_rest = np.array([i[0] for i in new_data[1]])
    data_fold = {'X': new_fold[0], 't': t_fold}
    data_rest = {'X': new_data[0], 't': t_rest}
    return data_rest, data_fold
# end def split_data

def train_model(data, lambd):
    # X^t X + lm
    x_matrix = data['X']
    t_vector = data['t']
    w_1= (np.matmul(np.transpose(x_matrix), x_matrix)) + (np.eye(x_matrix.shape[1])*lambd)
    w_2 = np.linalg.inv(w_1)
    w = np.matmul( w_2,(np.transpose(x_matrix).dot(t_vector)))
#     w =np.multiply(np.invert(np.add(
# np.multiply(np.transpose(x_matrix), x_matrix), np.identity()*lambd)), np.multiply(np.transpose(x_matrix), t_vector))
    return w
# end def train_model

def predict(data, model):
    # predictions?
    prediction = np.matmul(data, model)
    return prediction
#end def predict

def loss(data, model):
    t = data['t']
    X = data['X']
    p = predict(X,model)
    final = t - p
    # look at linear algebra rules
    dist = (np.linalg.norm(final)**2) /t.shape[0]
    return dist
# end def loss

def cross_validation(data, num_folds, lambd_seq):
    cv_error = [None]*50
    data = shuffle_data(data)
    for i in range(len(lambd_seq)):
        lambd = lambd_seq[i]
        cv_loss_lmd = 0
        for fold in range(num_folds):
            train_cv , val_cv= split_data(data, num_folds, fold)
            model = train_model(train_cv, lambd)
            cv_loss_lmd += loss(val_cv, model)
            print(cv_loss_lmd)
        cv_error[i] = cv_loss_lmd / num_folds
    return cv_error

def train_test_error(data, data_2, lambd_seq):
    errors_trai = []
    errors_test = []
    for i in range(len(lambd_seq)):
        model = train_model(data, lambd_seq[i])
        error = loss(data, model)
        error2 = loss(data_2, model)
        errors_trai.append(error)
        errors_test.append(error2)

    return errors_trai, errors_test

def plot(training_error, test_error, lamda_seq, cross_val_error_5, cross_val_error_10):
    # plot
    pyplot.plot(lamda_seq, training_error, lamda_seq, test_error, lamda_seq, cross_val_error_5, lamda_seq, cross_val_error_10)
    pyplot.gca().legend(("Training", "Testing", "5F", "10F"))
    pyplot.xlim(0.02,1.5)
    pyplot.xlabel("Lambda")
    pyplot.ylabel("Error")
    pyplot.show()



if "__main__" == __name__:
    X_train, X_test, X_valid, y_train, y_test, y_valid, features= load_data("clean_fake.txt", "clean_real.txt")
    valid_params, acct =select_model(X_train, X_valid, y_train, y_valid)
    depth, type = valid_params
    tree = illustrate_tree(depth, type, X_train, y_train,features)
    features_sample = {'the':features.index('the'), 'trump':features.index('trump'), 'hillary': features.index('hillary'), 'donald':features.index('donald')}
    print(features_sample)
    ig_dict = compute_information_gain(X_train,y_train, features_sample)
    print(ig_dict)
    data_train = load_data_vector("data_train_y.csv", "data_train_X.csv")
    data_test = load_data_vector("data_test_y.csv", "data_test_X.csv")
    num_folds = 5
    lambd_seq = np.linspace(0.02, 1.5, num=50)
    error_cv_5 = cross_validation(data_train, num_folds, lambd_seq)
    num_folds = 10
    error_cv_10 = cross_validation(data_train, num_folds, lambd_seq)
    error_train, error_test = train_test_error(data_train, data_test, lambd_seq)

    print("error test", error_test)
    print("error train", error_train)
    print("error cv5", error_cv_5)
    print("error cv10", error_cv_10)
    #
    plot(error_train, error_test, lambd_seq, error_cv_5, error_cv_10)
    arr = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
    arr2 = np.array([13,14,15])
    data = {'X': arr, 't':arr2}
    print(shuffle_data(data))













