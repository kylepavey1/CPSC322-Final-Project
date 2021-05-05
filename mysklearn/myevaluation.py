"""
Programmers: Kyle Pavey and Adam Lee
Class: 322-01, Spring 2021
Final Project
4/21/21
"""
import mysklearn.myutils as myutils
import random
import math
from itertools import islice

def train_test_split(X, y, test_size=0.33, random_state=None, shuffle=True):
    """Split dataset into train and test sets (sublists) based on a test set size.

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X)
            The shape of y is n_samples
        test_size(float or int): float for proportion of dataset to be in test set (e.g. 0.33 for a 2:1 split)
            or int for absolute number of instances to be in test set (e.g. 5 for 5 instances in test set)
        random_state(int): integer used for seeding a random number generator for reproducible results
        shuffle(bool): whether or not to randomize the order of the instances before splitting

    Returns:
        X_train(list of list of obj): The list of training samples
        X_test(list of list of obj): The list of testing samples
        y_train(list of obj): The list of target y values for training (parallel to X_train)
        y_test(list of obj): The list of target y values for testing (parallel to X_test)
    
    Note:
        Loosely based on sklearn's train_test_split(): https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html
    """
    if random_state is not None:
        random.seed(random_state)
        # TODO: seed your random number generator
       # you can use the math module or use numpy for your generator
       # choose one and consistently use that generator throughout your code
        pass
    
    if shuffle:
        z = list(zip(X, y))
        random.shuffle(z)
        X, y = zip(*z)
        # TODO: shuffle the rows in X and y before splitting
        # be sure to maintain the parallel order of X and y!!
        # note: the unit test for train_test_split() does not test
        # your use of random_state or shuffle, but you should still 
        # implement this and check your work yourself
        pass
    n = len(X)
    if isinstance(test_size, float):
        test_size = math.ceil(n * test_size)
    split_index = n - test_size 
    return X[:split_index], X[split_index:], y[:split_index], y[split_index:]

def kfold_cross_validation(X, n_splits=3):
    """Split dataset into cross validation folds.

    Args:
        X(list of list of obj): The list of samples
            The shape of X is (n_samples, n_features)
        n_splits(int): Number of folds.

    Returns:
        X_train_folds(list of list of int): The list of training set indices for each fold
        X_test_folds(list of list of int): The list of testing set indices for each fold

    Notes: 
        The first n_samples % n_splits folds have size n_samples // n_splits + 1, 
            other folds have size n_samples // n_splits, where n_samples is the number of samples
            (e.g. 11 samples and 4 splits, the sizes of the 4 folds are 3, 3, 3, 2 samples)
        Loosely based on sklearn's KFold split(): https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.KFold.html
    """
    X_train_folds = []
    X_test_folds = []
    random.shuffle(X)
    if len(X) % n_splits != 0:
        remainder = len(X) % n_splits
        n_odd_splits = []
        for i in range(remainder):
            n_odd_splits.append(len(X) // n_splits + 1)
        for i in range(n_splits - remainder):
            n_odd_splits.append(len(X) // n_splits)
        indices = []
        for i in range(len(X)):
            indices.append(i)
        indices_iter = iter(indices)
        folds = [list(islice(indices_iter, elem)) for elem in n_odd_splits]
        for i, fold in enumerate(folds):
            X_test_folds.append(fold)
            fold_indices = []
            for val in folds:
                if val != fold:
                    for elem in val:
                        fold_indices.append(elem)
            X_train_folds.append(fold_indices)

    else:
        indices = []
        for i in range(len(X)):
            indices.append(i)
        n = len(X) // n_splits
        folds = [indices[i * n:(i + 1) * n] for i in range((len(indices) + n - 1) // n )]  
        for i, fold in enumerate(folds):
            X_test_folds.append(fold)
            fold_indices = []
            for val in folds:
                if val != fold:
                    for elem in val:
                        fold_indices.append(elem)
            X_train_folds.append(fold_indices)

    return X_train_folds, X_test_folds

def stratified_kfold_cross_validation(X, y, n_splits=4):
    """Split dataset into stratified cross validation folds.

    Args:
        X(list of list of obj): The list of instances (samples). 
            The shape of X is (n_samples, n_features)
        y(list of obj): The target y values (parallel to X). 
            The shape of y is n_samples
        n_splits(int): Number of folds.
 
    Returns:
        X_train_folds(list of list of int): The list of training set indices for each fold.
        X_test_folds(list of list of int): The list of testing set indices for each fold.

    Notes: 
        Loosely based on sklearn's StratifiedKFold split(): https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedKFold.html#sklearn.model_selection.StratifiedKFold
    """
    if len(X) % n_splits != 0:
        remainder = len(X) % n_splits
        n_odd_splits = []
        for i in range(remainder):
            n_odd_splits.append(len(X) // n_splits + 1)
        for i in range(n_splits - remainder):
            n_odd_splits.append(len(X) // n_splits)
        indices = []
        for i in range(len(X)):
            indices.append(i)
        indices_iter = iter(indices)
        folds = [list(islice(indices_iter, elem)) for elem in n_odd_splits]
    else:
        indices = []
        for i in range(len(X)):
            indices.append(i)
        n = len(X) // n_splits
        folds = [indices[i * n:(i + 1) * n] for i in range((len(indices) + n - 1) // n )]

    header = ["att1", "att2", "result"]
    for i, val in enumerate(X):
        val.append(y[i])
        val.append(i)
    _, group_subtables = myutils.group_by(X, header, "result")
    
    grouped_indices = []
    for group in group_subtables:
        for instance in group:
            grouped_indices.append(instance[3])

    fold_x = 0
    fold_y = 0
    for index in grouped_indices:
        folds[fold_x][fold_y] = index
        if fold_x / (n_splits - 1) == 1:
            fold_x = 0
            fold_y += 1
        else: 
            fold_x += 1

    X_train_folds = []
    X_test_folds = []
    if len(X) % n_splits != 0:
        for i, fold in enumerate(folds):
            X_test_folds.append(fold)
            fold_indices = []
            for val in folds:
                if val != fold:
                    for elem in val:
                        fold_indices.append(elem)
            X_train_folds.append(fold_indices)
    else:
        for i, fold in enumerate(folds):
            X_test_folds.append(fold)
            fold_indices = []
            for val in folds:
                if val != fold:
                    for elem in val:
                        fold_indices.append(elem)
            X_train_folds.append(fold_indices)
    return X_train_folds, X_test_folds

def confusion_matrix(y_true, y_pred, labels):
    """Compute confusion matrix to evaluate the accuracy of a classification.

    Args:
        y_true(list of obj): The ground_truth target y values
            The shape of y is n_samples
        y_pred(list of obj): The predicted target y values (parallel to y_true)
            The shape of y is n_samples
        labels(list of str): The list of all possible target y labels used to index the matrix

    Returns:
        matrix(list of list of int): Confusion matrix whose i-th row and j-th column entry 
            indicates the number of samples with true label being i-th class 
            and predicted label being j-th class

    Notes:
        Loosely based on sklearn's confusion_matrix(): https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
    """
    
    labels_copy = labels
    matrix = []
    for loop in range(len(labels)):
        row = []
        for val in labels:
            row.append(0)
        matrix.append(row)

    for row_index, row in enumerate(matrix):
        for col_index, col in enumerate(row):
            for i in range(len(y_true)):
                if y_true[i] == labels[row_index] and y_pred[i] == labels_copy[col_index]:
                    matrix[row_index][col_index] += 1

    return matrix

# def confusion_matrix(y_true, y_pred, labels):
#     """Compute confusion matrix to evaluate the accuracy of a classification.
#     Args:
#         y_true(list of obj): The ground_truth target y values
#             The shape of y is n_samples
#         y_pred(list of obj): The predicted target y values (parallel to y_true)
#             The shape of y is n_samples
#         labels(list of str): The list of all possible target y labels used to index the matrix
#     Returns:
#         matrix(list of list of int): Confusion matrix whose i-th row and j-th column entry 
#             indicates the number of samples with true label being i-th class 
#             and predicted label being j-th class
#     Notes:
#         Loosely based on sklearn's confusion_matrix(): https://scikit-learn.org/stable/modules/generated/sklearn.metrics.confusion_matrix.html
#     """
#     matrix = []
#     #start in actual
#     for actual in labels:
#         row = []
#         #compare with predicted
#         for predicted in labels:
#             #loop through true and pred
#             count = 0
#             for i in range(0, (len(y_true))):
#                 if (y_true[i] == actual) & (y_pred[i] == predicted):
#                     count = count + 1
#             row.append(count)
#         matrix.append(row)
                
#     return matrix