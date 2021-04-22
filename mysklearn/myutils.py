"""
Programmers: Kyle Pavey and Adam Lee
Class: 322-01, Spring 2021
Final Project
4/21/21
"""
import math
import numpy as np
import random
# from numpy import random

def getRandomSearch(characters): 
    randomCharacter = random.choice(characters)
    randomSearch = ''
    
    rand = random.randint(0, 1)
    if rand == 0:
        randomSearch = randomCharacter + '%'
    elif rand == 1:
        randomSearch = '%' + randomCharacter + '%'
    offset = random.randint(0, 1000)
    return randomSearch, offset

def tdidt(train, available_attributes, attribute_domains):
    """Creates a tree from a table of data based on its attributes.

        Args: 
            train: the table of values
            available_attributes: the remaining available attributes to parse over
            attribute_domains: a dictionary of possible values for each attribute
        Returns:
            list of list: the decision tree
    """
    split_attribute = select_attribute(train, available_attributes)
    available_attributes.remove(split_attribute)
    tree = ["Attribute", split_attribute]
    partitions = partition_instances(train, split_attribute, attribute_domains)
    value_tracker = []
    # for each partition, repeat unless one of the following occurs (base case)
    previous_partition = list(partitions.values())[0]
    
    for attribute_value, partition in partitions.items():
        value_tracker.append(attribute_value)
        value_subtree = ["Value", attribute_value]
        # compute_partition_stats(partition, split_attribute)

        # TODO: appending leaf nodes and subtrees appropriately to value_subtree
        #    CASE 1: all class labels of the partition are the same => make a leaf node
        if len(partition) > 0 and all_same_class(partition):
            num, total = get_leaf_occurences(value_tracker, train, split_attribute)
            value_subtree.append(["Leaf", partition[0][-1]])
            tree.append(value_subtree)
        #    CASE 2: no more attributes to select (clash) => handle clash w/majority vote leaf node
        elif len(partition) > 0 and len(available_attributes) == 0:
            # classes, counts = compute_partition_stats(partition, attribute_value)
            # tree.append(value_subtree)
            num, total = get_leaf_occurences(value_tracker, train, split_attribute)
            counts = count_item_occurences(partition)
            class_label = get_class_label(counts)
            value_subtree.append(["Leaf", class_label])
            tree.append(value_subtree)
        #    CASE 3: no more instances to partition (empty partition) => backtrack and replace attribute node with majority vote leaf node
        elif len(partition) == 0:
            # if "Attribute" in tree[-1]:
            #     if tree[-1].index("Attribute") == len(tree[-1] - 1):
            #         tree[-1] = ["Leaf", "TEST_TEST"]
            # print("---------case3 ", split_attribute)
            # print(tree[-1])
            # print("")
            # value_subtree.append(["Leaf", "TEST_TEST"])
            # tree.append(value_subtree)
            num, total = get_leaf_occurences(value_tracker, train, split_attribute)
            counts = count_item_occurences(previous_partition)
            class_label = get_class_label(counts)
            tree = []
            # use current instances
            tree = ["Leaf", class_label]

            # print(tree, "\n")
            # counts = count_class_occurences(partition)
            # class_label = get_class_label(counts)
            # tree[-1] = ["Leaf", class_label]
        else: # all base cases are false... recurse!!
            subtree = tdidt(partition, available_attributes.copy(), attribute_domains)
            value_subtree.append(subtree)
            tree.append(value_subtree)
        previous_partition = partition
    #print("tree: \n", tree)
    return tree

def calculate_entropy(instances, available_attributes):
    """Calculates the entropy of each attribute group.

        Args: 
            instances: the data values of the column
            available_attributes: the remaining available attributes to parse over
        Returns:
            int: the index of the minimum entropy value
    """
    entropy_list = []
    instances_total = len(instances)
    for attr in available_attributes:
        values = count_attr_occurences(instances, attr)
        enew = 0
        for value in values:
            values_list, total = count_value_occurences(instances, attr, value)
            val1 = values_list[0]
            val2 = values_list[1]
            if val1 == 0 or val2 == 0:
                e = 0
            else:
                e = (-val1/total) * math.log(val1/total, 2) - (val2/total) * math.log(val2/total, 2)  
            enew += ((total/instances_total) * e)
        entropy_list.append(enew)
    min_e = min(entropy_list)
    min_e_index = entropy_list.index(min_e)
    return min_e_index

def select_attribute(instances, available_attributes):
    """Selects an attribute to split on.

        Args: 
            instances: the data values of the column
            available_attributes: the remaining available attributes to parse over
        Returns:
            obj: the attribute
    """
    min_e_index = calculate_entropy(instances, available_attributes)
    return available_attributes[min_e_index]

def partition_instances(instances, split_attribute, attribute_domains):
    """Creates partitions of the instances.

        Args: 
            instances: the data values of the column
            split_attribute: the attribute that the instances will be split based on
            attribute_domains: a dictionary of possible values for each attribute
        Returns:
            list of list: the partitions
    """
    attribute_domain = attribute_domains[split_attribute] 
    index = split_attribute.split("attr",1)[1] 
    attribute_index = int(index) # 0
    partitions = {} 
    for attribute_value in attribute_domain:
        partitions[attribute_value] = []
        for instance in instances:
            if instance[attribute_index] == attribute_value:
                partitions[attribute_value].append(instance)
    return partitions

def get_attribute_domains(train):
    """Creates a dictionary of the possible values for each attribute

        Args: 
            train: the data table
        Returns:
            dict: the attributes and their possible values
    """
    domain_dict = {}
    for col_index in range(len(train[0])):
        col = get_column(train, col_index)
        list_set = set(col)
        unique_col_list = (list(list_set))
        key_name = "attr" + str(col_index)
        domain_dict[key_name] = unique_col_list
    return domain_dict

def all_same_class(instances):
    """Checks if all of the instances in a list have the same class.

        Args: 
            instances: the data values of the column
        Returns:
            boolean
    """
    # assumption: instances is not empty and class label is at index -1
    first_label = instances[0][-1]
    for instance in instances:
        if instance[-1] != first_label:
            return False 
    return True # if we get here, all instance labels matched the first label

def get_leaf_occurences(val_list, train, split_attribute):
    """Counts the number of leaf occurences for a given attribute and value

        Args: 
            val_list: a list of the values for the split attribute
            train: the data table
            split_attribute: the attribute that the instances will be split based on
        Returns:
            num: the number of leaf occurences
            total:  the total possible leaf nodes
    """
    num = 0
    total = 0
    if len(val_list) == 1:
        for row in train:
            if all(elem in row for elem in val_list):
                num += 1 
        total = len(train)
    else:
        for row in train:
            if all(elem in row for elem in val_list):
                num += 1
            if val_list[0] in row:
                total += 1
    return num, total

def count_item_occurences(instances):
    """Counts the number of item occurences for a given instance

        Args: 
            instances: the data values of the column
        Returns:
            list of list: the value and the number of times it appears
    """
    occurrences_list = []
    for instance in instances:
        if not occurrences_list:
            occurrences_list.append([instance[-1], 1])
        elif instance[-1] not in [val for sublist in occurrences_list for val in sublist]:
            occurrences_list.append([instance[-1], 1])
        else:
            for val in occurrences_list:
                if val[0] == instance[-1]:
                    val[1] += 1
    return occurrences_list

def count_attr_occurences(instances, attr):
    """Counts the number of attribute occurences for a given instance

        Args: 
            instances: The data values of the column
            attr:  The attribute to count
        Returns:
            list of list: the value and the number of times it appears
    """
    occurrences_list = []
    index = int(attr.split("attr",1)[1]) 
    attr_col = get_column(instances, index)
    for instance in attr_col:
        if instance not in occurrences_list:
            occurrences_list.append(instance)
    return occurrences_list

def count_value_occurences(instances, attr, value):
    """Counts the number of value occurences for a given instance and attribute

        Args: 
            instances: The data values of the column
            attr:  The attribute to count
            value:  The value to count
        Returns:
            list of list: the value and the number of times it appears
    """
    occurrences_list = [0, 0]
    index = int(attr.split("attr",1)[1]) 
    attr_col = get_column(instances, index)
    class_col = get_column(instances, len(instances[0]) - 1)
    class1 = class_col[0]
    total = 0
    for i, instance in enumerate(attr_col):
        if instance == value:
            if class_col[i] == class1:
                occurrences_list[0] += 1 
            else:
                occurrences_list[1] += 1 
            total += 1
    return occurrences_list, total

def get_class_label(counts):
    """Gets the class label based on majority voting

        Args: 
            counts:  The list of values and number of occurences
        Returns:
            obj: The class label 
    """
    count_list = []
    choice = 0
    for label in counts:
        count_list.append(label[1])
    all_same = count_list.count(count_list[0]) == len(count_list)
    if all_same:
        choice = random.randint(0, 1)
    else:
        max_val = max(count_list)
        choice = count_list.index(max_val)
    return counts[choice][0]


def get_frequencies(table, index):
    """Computes the number of occurrences in a column.

        Args: 
            table: the table to be parsed
            index: the index of the table to be counted
        Returns:
            list: the values
            list: the number of occurrences
    """
    if index is None:
        col = table
    else:
        col = get_column(table, index)
    # if type(col) == int or type(col) == float:
        # col.sort() 
    values = []
    counts = []

    for value in col:
        if value not in values:
            values.append(value)
            counts.append(1)
        else:
            if type(col) == int or type(col) == float:
                counts[-1] += 1 
            else:
                for index, val in enumerate(values):
                    if value == val:
                        counts[index] += 1
    return values, counts

def get_column(table, index):
    """Retrieves a column from the table.

        Args: 
            table: the table to be parsed
            header: the header row of the table
            col_name: the column to be counted from
        Returns:
            list: the column values
    """
    col = []
    for row in table: 
        # if row[index] != "NA":
        val = row[index]
        col.append(val)
    return col


def compute_slope_intercept(x, y):
    """Computes the slope intercept.

        Args: 
            x: an individual value
            y: an individual value
        Returns:
            int: the slope
            int: the y intercept
    """
    mean_x = sum(x) / len(x)
    mean_y = sum(y) / len(y)
    m = sum([(x[i] - mean_x) * (y[i] - mean_y) for i in range(len(x))]) / sum([(x[i] - mean_x) ** 2 for i in range(len(x))])
    b = mean_y - m * mean_x
    return m, b

def compute_euclidean_distance(v1, v2):
    """Computes the slope intercept.

        Args: 
            v1: an individual value
            v2: an individual value
        Returns:
            float: the euclidean distance  between points
    """
    str_present = False
    for i in range(len(v1)):
        if v1[i] == str or v2[i] == str:
            str_present = True
    if len(v1) == len(v2) and str_present is False:
        dist = np.sqrt(sum([(v1[i] - v2[i]) ** 2 for i in range(len(v1))]))
    elif len(v1) == len(v2) and str_present is True:
        dist = 0
    else:
        dist = 1
    return dist

def group_by(table, header, group_by_col_name):
    """Computes the number of occurrences in a column.

        Args: 
            table: the table to be parsed
            header: the header row of the table
            group_by_col_name: the column to be grouped by
        Returns:
            list: the names of each group
            list of list: a new table based on group names
    """
    col_index = header.index(group_by_col_name)
    col = get_column(table, col_index)
    group_names = sorted(list(set(col))) 
    group_subtables = [[] for _ in group_names] 
    for i, row in enumerate(table):
        group_by_value = col[i]
        group_index = group_names.index(group_by_value)
        group_subtables[group_index].append(row.copy())
    return group_names, group_subtables

def compute_equal_width_cutoffs(values, num_bins):
    """Computes equal bin widths to divide values.

        Args: 
            values: a list of values
            bins: an int of the number of bins
        Returns:
            list: the cutoffs for each bin
    """
    values_range = max(values) - min(values)
    bin_width = values_range / num_bins 
    cutoffs = list(np.arange(min(values), max(values), bin_width)) 
    cutoffs.append(max(values))
    cutoffs = [round(cutoff, 2) for cutoff in cutoffs]
    return cutoffs 

def get_train_labels(X_train, index):
    """Computes the number of occurrences in a column.

        Args: 
            table: the table to be parsed
            header: the header row of the table
            group_by_col_name: the column to be grouped by
        Returns:
            list: the names of each group
            list of list: a new table based on group names
    """
    col = get_column(X_train, index)
    for row in X_train:
        del row[index]
    return col, X_train

def group_by_index(table, index):
    """Computes the number of occurrences in a column.

        Args: 
            table: the table to be parsed
            index: the index of the table
        Returns:
            list: the names of each group
            list of list: a new table based on group names
    """
    col = get_column(table, index)
    group_names = sorted(list(set(col))) 
    group_subtables = [[] for _ in group_names] 
    for i, row in enumerate(table):
        group_by_value = col[i]
        group_index = group_names.index(group_by_value)
        group_subtables[group_index].append(row.copy())
    return group_names, group_subtables

def weighted_choice(values, weights):
    """Returns a random value in a list given the number of occurances of each value

        Args: 
            values(list of obj):  a list of values
            weight(list of int):  the corresponding number of instances for each value
        Returns:
            obj: a random value
    """
    choice_list = []
    for i, val in enumerate(values):
        choice_list.append([val] * weights[i])
    flat_list = [item for sublist in choice_list for item in sublist]
    return np.random.choice(flat_list)

def compute_accuracy(matrix):
    """Computes the accuracy of a classifier

        Args: 
            matrix(list of list of obj): the number of occurances for True and False Positive and True and False negatives 
        Returns:
            float: accuracy as a percent
    """
    tp_tn = 0
    p_n = 0
    for i in range(len(matrix)):
        tp_tn += matrix[i][i + 1]
    for row in matrix:
        p_n += sum(row[1:])
    if tp_tn == 0 or p_n == 0:
        return 0
    else:
        return tp_tn / p_n

def compute_error_rate(matrix):
    """Computes the error rate of a classifier

        Args: 
            matrix(list of list of obj): the number of occurances for True and False Positive and True and False negatives 
        Returns:
            float: error rate as a percent
    """
    index = len(matrix[0]) - 1
    fp_fn = 0
    p_n = 0
    for i in range(len(matrix)):
        fp_fn += matrix[i][index]
        index -= 1
    for row in matrix:
        p_n += sum(row[1:])
    if fp_fn == 0 or p_n == 0:
        return 0
    else:
        return fp_fn / p_n