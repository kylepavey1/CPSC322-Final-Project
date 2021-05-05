"""
Programmers: Kyle Pavey and Adam Lee
Class: 322-01, Spring 2021
Final Project
4/21/21
"""

import mysklearn.myutils as myutils
import operator
import random
import numpy as np 
from itertools import groupby
from collections import Counter
import importlib

import mysklearn.myutils
importlib.reload(mysklearn.myutils)
import mysklearn.myutils as myutils
import mysklearn.myevaluation
importlib.reload(mysklearn.myevaluation)
import mysklearn.myevaluation as myevaluation

class MySimpleLinearRegressor:
    """Represents a simple linear regressor.

    Attributes:
        slope(float): m in the equation y = mx + b
        intercept(float): b in the equation y = mx + b

    Notes:
        Loosely based on sklearn's LinearRegression: https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    
    def __init__(self, slope=None, intercept=None):
        """Initializer for MySimpleLinearRegressor.

        Args:
            slope(float): m in the equation y = mx + b (None if to be computed with fit())
            intercept(float): b in the equation y = mx + b (None if to be computed with fit())
        """
        self.slope = slope 
        self.intercept = intercept

    def fit(self, X_train, y_train):
        """Fits a simple linear regression line to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training samples
                The shape of X_train is (n_train_samples, n_features)
                Note that n_features for simple regression is 1, so each sample is a list 
                    with one element e.g. [[0], [1], [2]]
            y_train(list of numeric vals): The target y values (parallel to X_train) 
                The shape of y_train is n_train_samples
        """
        self.slope, self.intercept = myutils.compute_slope_intercept(X_train, y_train)

    def predict(self, X_test):
        """Makes predictions for test samples in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)
                Note that n_features for simple regression is 1, so each sample is a list 
                    with one element e.g. [[0], [1], [2]]

        Returns:
            y_predicted(list of numeric vals): The predicted target y values (parallel to X_test)
        """
        y_predicted = []
        for val in X_test:
            y_predicted.append((self.slope * val) + self.intercept)
        return y_predicted # TODO: fix this


class MyKNeighborsClassifier:
    """Represents a simple k nearest neighbors classifier.

    Attributes:
        n_neighbors(int): number of k neighbors
        X_train(list of list of numeric vals): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train). 
            The shape of y_train is n_samples

    Notes:
        Loosely based on sklearn's KNeighborsClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
        Assumes data has been properly normalized before use.
    """
    def __init__(self, n_neighbors):
        """Initializer for MyKNeighborsClassifier.

        Args:
            n_neighbors(int): number of k neighbors
        """
        self.n_neighbors = n_neighbors
        self.X_train = None 
        self.y_train = None

    def fit(self, X_train, y_train):
        """Fits a kNN classifier to X_train and y_train.

        Args:
            X_train(list of list of numeric vals): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since kNN is a lazy learning algorithm, this method just stores X_train and y_train
        """
        for i, instance in enumerate(X_train):
            instance.append(y_train[i])
            instance.append(i)

        self.X_train = X_train 
        self.y_train = y_train 

    def kneighbors(self, X_test):
        """Determines the k closes neighbors of each test instance.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            distances(list of list of float): 2D list of k nearest neighbor distances 
                for each instance in X_test
            neighbor_indices(list of list of int): 2D list of k nearest neighbor
                indices in X_train (parallel to distances)
        """
        distances = []
        distances_2d_list = []
        neighbors = []
        neighbor_indices = []
        if isinstance(X_test[0], list):
            for test_instance in X_test:
                for i, instance in enumerate(self.X_train):
                    distance = myutils.compute_euclidean_distance(instance[:2], test_instance)
                    instance.append(distance)
                    distances.append(distance)
                distances_2d_list.append(distances)
                train_sorted = sorted(self.X_train, key=operator.itemgetter(-1))
                top_k = train_sorted[:self.n_neighbors]
                for instance in top_k:
                    neighbors.append(instance[3])
                neighbor_indices.append(neighbors)
                distances = []
                neighbors = []
                for instance in self.X_train:
                    del instance[-1:]
        else:
            for i, instance in enumerate(self.X_train):
                    distance = myutils.compute_euclidean_distance(instance[:2], X_test)
                    instance.append(distance)
                    distances.append(distance)
            train_sorted = sorted(self.X_train, key=operator.itemgetter(-1))
            top_k = train_sorted[:self.n_neighbors]
            for instance in top_k:
                neighbor_indices.append(instance[3])
        return distances_2d_list, neighbor_indices

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of numeric vals): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        y_predicted = []
        _, neighbor_indices = self.kneighbors(X_test)
        new_list = []
        if isinstance(X_test[0], list):
            for test_set in neighbor_indices:
                for index in test_set:
                    new_list.append(self.X_train[index])
                vals, freqs = myutils.get_frequencies_index(new_list, 2)
                max_index = freqs.index(max(freqs))
                y_predicted.append(vals[max_index])
                new_list = []
        else:
            for i, val in enumerate(neighbor_indices):
                if type(val) == str:
                    new_list.append(self.X_train[i])
                else:
                    new_list.append(self.X_train[int(val)])
            vals, freqs = myutils.get_frequencies_index(new_list, 2)
            max_index = freqs.index(max(freqs))
            y_predicted.append(vals[max_index])
            new_list = []
        return y_predicted 

class MyNaiveBayesClassifier:
    """Represents a Naive Bayes classifier.

    Attributes:
        X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train). 
            The shape of y_train is n_samples
        priors(list of list of obj): The prior probabilities computed for each
            label in the training set.
        posteriors(list of list of obj): The posterior probabilities computed for each
            attribute value/label pair in the training set.

    Notes:
        Loosely based on sklearn's Naive Bayes classifiers: https://scikit-learn.org/stable/modules/naive_bayes.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyNaiveBayesClassifier.

        """
        self.X_train = None 
        self.y_train = None
        self.priors = None 
        self.posteriors = None

    def fit(self, X_train, y_train):
        """Fits a Naive Bayes classifier to X_train and y_train.

        Args:
            X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since Naive Bayes is an eager learning algorithm, this method computes the prior probabilities
                and the posterior probabilities for the training data.
            You are free to choose the most appropriate data structures for storing the priors
                and posteriors.
        """
        # print("1, ", X_train)
        unique_list = []
        for x in y_train:
            if [x] not in unique_list:
                unique_list.append([x])
        for val in unique_list:
            val.append(0)
        for y in y_train:
            for val in unique_list:
                if y == val[0]:
                    val[1] += 1
        train_size = len(y_train)
        for val in unique_list:
            val[1] = val[1] / train_size
        self.priors = unique_list
        prior_vals = [item[0] for item in self.priors]
        posterior = []

        for prior in prior_vals:
            posterior.append([prior])

        X_train_copy = list(X_train)
        for i, val in enumerate(y_train):
            X_train_copy[i].append(val)

        names, grouped = myutils.group_by_index(X_train_copy, len(X_train_copy[0]) - 1)
        for i, group in enumerate(grouped):
            posterior_group = []
            for attr in range(len(group[0]) - 1):
                values, counts = myutils.get_frequencies_index(group, attr)
                result_sum = sum(counts)
                testing = []
                for y, attr_val in enumerate(values):
                    attr_posterior = counts[y] / result_sum
                    testing.append((attr_val, attr_posterior))
                for val in testing:
                    posterior_group.append(val)
            posterior[i].append(posterior_group)
        self.posteriors = posterior
        for row in X_train:
            del row[-1]

    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        greater_p = 0
        y_index = 0
        for i, c in enumerate(self.posteriors):
            value_list = []
            for n in range(len(X_test)):
                try:
                    pos_index = [y[0] for y in c[1]].index(X_test[n])
                except ValueError:
                    continue
                value_list.append(c[1][pos_index])
            result = 1
            for x in value_list:
                result = result * x[1]
            p = result * self.priors[i][1]
            if p >= greater_p:
                greater_p = p
                y_index = i
        y_predicted = self.priors[y_index][0]
        return y_predicted

class MyZeroRClassifier:
    def __init__(self):
        """Initializer for MyZeroRClassifier.
            Chooses the most common class
        """
        self.y_train = None

    def fit(self, y_train):
        """Fits a Zero R classifier to X_train and y_train.

        Args:
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples
        """
        self.y_train = y_train

    def predict(self):
        """Predicts a Zero R value for y_train

        Returns:
            obj: The most common class label in y_train
        """
        c = Counter(self.y_train)
        return c.most_common()[0][0]

class MyRandomClassifier:
    def __init__(self):
        """Initializer for MyRandomClassifier.
            Chooses a random value in the y list
        """
        self.X_train = None
        self.y_train = None

    def fit(self, X_train, y_train):
        self.X_train = X_train
        self.y_train = y_train

    def predict(self):
        """Predicts a random y_train value.

        Returns:
            obj: A random obj in y_train
        """
        values, counts = myutils.get_frequencies_index(self.y_train, None)
        return myutils.weighted_choice(values, counts)

class MyDecisionTreeClassifier:
    """Represents a decision tree classifier.

    Attributes:
        X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train). 
            The shape of y_train is n_samples
        tree(nested list): The extracted tree model.

    Notes:
        Loosely based on sklearn's DecisionTreeClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    # 1. all_same_class()
    # 2. append subtree to values_subtree and to tree appropriately
    # 3. work on CASE 1, then CASE 2, then CASE 3 (write helper functions!!)
    # e.g. compute_partition_stats()
    # 4. finish the TODOs in fit_starter_code()
    # 5. replace random w/entropy (compare tree w/interview_tree)
    # 6. move over starter code to PA6 OOP w/unit test fit()
    # 7. move on to predict()...
    def __init__(self):
        """Initializer for MyDecisionTreeClassifier.

        """
        self.X_train = None 
        self.y_train = None
        self.tree = None

    def fit(self, X_train, y_train):
        """Fits a decision tree classifier to X_train and y_train using the TDIDT (top down induction of decision tree) algorithm.

        Args:
            X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since TDIDT is an eager learning algorithm, this method builds a decision tree model
                from the training data.
            Build a decision tree using the nested list representation described in class.
            Store the tree in the tree attribute.
            Use attribute indexes to construct default attribute names (e.g. "att0", "att1", ...).
        """
        self.X_train = X_train
        self.y_train = y_train
        # calculate a header (e.g. ["att0", "att1", ...])
        header = []
        for i in range(len(X_train[0])):
            att = 'att' + str(i)
            header.append(att)
        #print("header:", header)

        # calculate the attribute domains dictionary
        attributeDomain = {}
        for attribute in header:
            attributeDomain[attribute] = []
            attributeIndex = header.index(attribute)
            for row in X_train:
                if row[attributeIndex] not in attributeDomain[attribute]:
                    attributeDomain[attribute].append(row[attributeIndex])
        #print("attributeDomain: ", attributeDomain)

        # stitch together X_train and y_train
        train = [X_train[i] + [y_train[i]] for i in range(len(X_train))]
        available_attributes = header.copy() # recall: Python is pass by object reference

        # initial tdidt() call
        self.tree = myutils.tdidt(train, available_attributes, header, attributeDomain)
        
        #print("tree:", self.tree)
        
        
    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        #["Junior", "Java", "yes", "no"],
        predictions = []
        
        # calculate a header (e.g. ["att0", "att1", ...])
        header = []
        for i in range(len(self.X_train[0])):
            att = 'att' + str(i)
            header.append(att)
        #print("header:", header)

        # test for all instances in X_test
        for test in X_test:
            try:
                predictions.append(myutils.tdidt_predict(header, self.tree, test))
            except:
                return None
        
        return predictions

    def print_decision_rules(self, attribute_names=None, class_name="class"):
        """Prints the decision rules from the tree in the format "IF att == val AND ... THEN class = label", one rule on each line.

        Args:
            attribute_names(list of str or None): A list of attribute names to use in the decision rules
                (None if a list is not provided and the default attribute names based on indexes (e.g. "att0", "att1", ...) should be used).
            class_name(str): A string to use for the class name in the decision rules
                ("class" if a string is not provided and the default name "class" should be used).
        """

        # calculate a header (e.g. ["att0", "att1", ...])   
        header = []
        for i in range(len(self.X_train[0])):
            att = 'att' + str(i)
            header.append(att)


        

        pass # TODO: fix this





# class MyDecisionTreeClassifier:
#     """Represents a decision tree classifier.

#     Attributes:
#         X_train(list of list of obj): The list of training instances (samples). 
#                 The shape of X_train is (n_train_samples, n_features)
#         y_train(list of obj): The target y values (parallel to X_train). 
#             The shape of y_train is n_samples
#         tree(nested list): The extracted tree model.

#     Notes:
#         Loosely based on sklearn's DecisionTreeClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
#         Terminology: instance = sample = row and attribute = feature = column
#     """
#     def __init__(self):
#         """Initializer for MyDecisionTreeClassifier.

#         """
#         self.X_train = None 
#         self.y_train = None
#         self.tree = None

#     def fit(self, X_train, y_train):
#         """Fits a decision tree classifier to X_train and y_train using the TDIDT (top down induction of decision tree) algorithm.

#         Args:
#             X_train(list of list of obj): The list of training instances (samples). 
#                 The shape of X_train is (n_train_samples, n_features)
#             y_train(list of obj): The target y values (parallel to X_train)
#                 The shape of y_train is n_train_samples

#         Notes:
#             Since TDIDT is an eager learning algorithm, this method builds a decision tree model
#                 from the training data.
#             Build a decision tree using the nested list representation described in class.
#             Store the tree in the tree attribute.
#             Use attribute indexes to construct default attribute names (e.g. "att0", "att1", ...).
#         """
#         train = [X_train[i] + [y_train[i]] for i in range(len(X_train))]
#         available_attributes = ["attr" + str(i) for i in range(len(train[0]) - 1)]
#         attribute_domains = myutils.get_attribute_domains(train)
#         tree = myutils.tdidt(train, available_attributes, attribute_domains) 
#         self.tree = tree 
        
#     def predict(self, X_test):
#         """Makes predictions for test instances in X_test.

#         Args:
#             X_test(list of list of obj): The list of testing samples
#                 The shape of X_test is (n_test_samples, n_features)

#         Returns:
#             y_predicted(list of obj): The predicted target y values (parallel to X_test)
#         """
#         isClass = False
#         class_label = None
#         y_predicted = []
#         current_list = self.tree.copy()
#         for x_val in X_test:
#             while not isClass:
#                 attr_position = int(current_list[1].split("attr", 1)[1])
#                 value = x_val[attr_position]
#                 next_size = len(current_list) - 2
#                 for x in range(2, 2 + next_size):
#                     if current_list[x][1] == value:
#                         current_list = current_list[x][2]
#                         break
#                 if (current_list[0] == "Leaf"):
#                     class_label = current_list[1]
#                     isClass = True
#             y_predicted.append(class_label)
#             isClass = False
#             class_label = None
#             current_list = self.tree.copy()
#         return y_predicted


#     def print_decision_rules(self, attribute_names=None, class_name="class"):
#         """Prints the decision rules from the tree in the format "IF att == val AND ... THEN class = label", one rule on each line.

#         Args:
#             attribute_names(list of str or None): A list of attribute names to use in the decision rules
#                 (None if a list is not provided and the default attribute names based on indexes (e.g. "att0", "att1", ...) should be used).
#             class_name(str): A string to use for the class name in the decision rules
#                 ("class" if a string is not provided and the default name "class" should be used).
#         """
#         print("\n\n===========================================")
#         print("Decision Rules")
#         print("===========================================")
#         all_rules = []
#         counted_leaves = []
#         leaf_Found = False
#         next_branch_list = {}
        
#         rule = ""
#         current_list = self.tree.copy()
#         for _ in range(len(self.tree)):
#             branch = len(current_list) - 2
#             attr_position = current_list[1]
#             if attr_position in next_branch_list:
#                 value = next_branch_list[attr_position]
#             else:
#                 value = current_list[branch + 1][1]
#             if current_list[0] == "Leaf":
#                 rule += " THEN {} == {}".format(class_name, current_list[1])
#                 all_rules.append(rule)
#                 print(all_rules)
#                 rule = ""
#                 current_list = self.tree.copy()
#                 continue
#             else: 
#                 if rule == "":
#                     rule = "IF {} == {}".format(attr_position, value)   
#                 else: 
#                     rule += " AND IF {} == {}".format(attr_position, value)   
            
#             if attr_position.__contains__("attr"):
#                 if attr_position in next_branch_list:
#                     next_branch_list[attr_position] = next_branch_list[attr_position] - 1
#                 else: 
#                     next_branch_list[attr_position] = branch - 1
#             current_list = current_list[(next_branch_list[attr_position]+2)][2]   
#         pass

class MyRandomForestClassifier:
    """Represents a random forest classifier.

    Attributes:
        X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
        y_train(list of obj): The target y values (parallel to X_train). 
            The shape of y_train is n_samples
        tree(nested list): The extracted tree model.

    Notes:
        Loosely based on sklearn's DecisionTreeClassifier: https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeClassifier.html
        Terminology: instance = sample = row and attribute = feature = column
    """
    def __init__(self):
        """Initializer for MyDecisionTreeClassifier.

        """
        self.best_M_trees = []
        self.accuracy = None
    def fit(self, X_train, y_train, header, matrix_labels):
        """Fits a decision tree classifier to X_train and y_train using the TDIDT (top down induction of decision tree) algorithm.

        Args:
            X_train(list of list of obj): The list of training instances (samples). 
                The shape of X_train is (n_train_samples, n_features)
            y_train(list of obj): The target y values (parallel to X_train)
                The shape of y_train is n_train_samples

        Notes:
            Since TDIDT is an eager learning algorithm, this method builds a decision tree model
                from the training data.
            Build a decision tree using the nested list representation described in class.
            Store the tree in the tree attribute.
            Use attribute indexes to construct default attribute names (e.g. "att0", "att1", ...).
        """
        # Divide instances into a test and remainder set

        # take 1/3 for test set, 2/3 for remainder set
        remainder_set, test_set, remainder_y, test_y = myevaluation.train_test_split(X_train, y_train)
        print("first split: ", len(remainder_set), len(test_set), len(remainder_y), len(test_y))
        # Create N bootstrap samples from remainder set
        N = 4 #number of trees
        forest = []
        tree_accuracy = []
        for i in range(N):
            #get bootstrapped example
            boot_set, boot_y = myutils.compute_bootstrapped_sample(remainder_set, remainder_y)
            
            #split into training set 63% and validation set 36% set
            training_set, validation_set, training_y, validation_y = myevaluation.train_test_split(boot_set, boot_y, 0.36)
            # print("trainingset: ", training_set)
            
            # build random attribute subsets of size F
            F = 4
            randomAtts = myutils.random_attribute_subset(header, F)
    
            attIndicies = []    
            for att in randomAtts:
                attIndicies.append(header.index(att))
            # delete unwanted attributes
            F_training_subset = []
            F_validation_subset = []
            for instance in training_set:
                row = []
                for i in range(len(instance)):
                    if i in attIndicies:
                        row.append(instance[i])
                F_training_subset.append(row)
            for instance in validation_set:
                row = []
                for i in range(len(instance)):
                    if i in attIndicies:
                        row.append(instance[i])
                F_validation_subset.append(row)
            
            # print("validationset: ", validation_set)

            tree = MyDecisionTreeClassifier()
            #print("tree: ", tree)
            tree.fit(F_training_subset, training_y)

            forest.append(tree)
        

            #run validation set through each tree to get accuracy
            y_pred = tree.predict(F_validation_subset)
            matrix = myevaluation.confusion_matrix(validation_y, y_pred, matrix_labels)
            accuracy = myutils.compute_accuracy_2(matrix)
            tree_accuracy.append(accuracy)
        #choose best M trees 
        M = 3

        tree_accuracy.sort()
        for i in range(M):
            # print("index: ", -i-1)
            self.best_M_trees.append(forest[-i-1])

        # Using test set, determine performace of the ensemble of M via simple majority voting
        predictions = []
        for tree in self.best_M_trees:
            y_pred = tree.predict(test_set)
            predictions.append(y_pred)
        majority_vote = myutils.get_majority_vote(predictions)
        matrix = myevaluation.confusion_matrix(test_y, majority_vote, matrix_labels)
        self.accuracy = myutils.compute_accuracy_2(matrix)
        pass
        
    def predict(self, X_test):
        """Makes predictions for test instances in X_test.

        Args:
            X_test(list of list of obj): The list of testing samples
                The shape of X_test is (n_test_samples, n_features)

        Returns:
            y_predicted(list of obj): The predicted target y values (parallel to X_test)
        """
        
        # Using majority voting, make predictions from the M learners for each instance in the test set
        predictions = []
        for tree in self.best_M_trees:
            y_pred = tree.predict(X_test)
            print(y_pred)
            predictions.append(y_pred)
        majority_vote = myutils.get_majority_vote(predictions)

        return predictions