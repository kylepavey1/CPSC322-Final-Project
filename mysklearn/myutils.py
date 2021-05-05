"""
Programmers: Kyle Pavey and Adam Lee
Class: 322-01, Spring 2021
Final Project
5/5/21
"""
# import math
# import numpy as np
# import random
# # from numpy import random

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

def get_frequencies_index(table, index):
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

def group_by_table(table, header, group_by_col_name):
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

def compute_accuracy_2(matrix):
    """Computes the accuracy of a classifier

        Args: 
            matrix(list of list of obj): the number of occurances for True and False Positive and True and False negatives 
        Returns:
            float: accuracy as a percent
    """
    tp_tn = 0
    p_n = 0
    for i in range(len(matrix)-1):
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

import math

def get_column_by_index(array, index):
    col = []
    for row in array: 
        # ignore missing values ("NA")
        if row[index] != "NA":
            col.append(row[index])
    return col
import numpy as np
import random

def compute_slope_intercepts(x, y):
    mean_x = np.mean(x)
    mean_y = np.mean(y)
    
    m = sum([(x[i] - mean_x) * (y[i] - mean_y) for i in range(len(x))]) \
        / sum([(x[i] - mean_x) ** 2 for i in range(len(x))])

    b = mean_y - m * mean_x
    return m, b

def convert2DArray(arr):
    listOfLists = False
    for elements in arr:
        if elements.count == 1:
            listOfLists = True
    if listOfLists == True:
        oneD = []
        oneD = sum(arr, []) 
    return oneD


def compute_euclidean_distance(v1, v2):
    assert len(v1) == len(v2)

    dist = np.sqrt(sum([(v1[i] - v2[i]) ** 2 for i in range(len(v1))]))
    return dist

def get_column(array, index):
    col = []
    for row in array: 
        # ignore missing values ("NA")
        if row[index] != "NA":
            col.append(row[index])
    return col

def get_frequencies(col):
    col.sort()

    values = []
    counts = []

    for value in col:
        if value not in values:
            # first time we have seen this value
            values.append(value)
            counts.append(1)
        else:
            # we have seen this value before 
            counts[-1] += 1 # ok because the list is sorted

    return values, counts

def get_majority(values, counts):
    highestCount = 0
    highestIndex = 0
    for i, num in enumerate(counts):
        if num > highestCount:
            highestCount = num
            highestIndex = i
    highestVal = values[highestIndex]

    return highestVal

def get_random_num(range):
    return random.randint(0, range)

def group_by(y):
    groups = []
    groupNames = []
    #determine groups
    for instances in y:
        if instances not in groupNames:
            groupNames.append(instances)
    #print("groupNames: ", groupNames)
    
    #group instances by name
    for n, name in enumerate(groupNames):
        group = []
        #print("n: ", n)
        for i, instances in enumerate(y):
            #print("i: ", i)
            if instances == name:
                #print("instances: ", instances, "name: ", name)
                group.append(i)
        groups.append(group)
    
    #print("groups: ", groups)

    return groupNames, groups

def probability_of_array(array, target):
    count = 0
    for items in array:
        if items == target:
            count = count + 1

    probability = count / len(array)
    return probability

def probability_two_array(X, y, x_target, y_target):
    assert len(X) == len(y)
    count = 0
    for i in range(len(y)):
        if (X[i] == x_target) & (y[i] == y_target):
            count = count + 1
    probability = count / len(y)
    return probability 

def compute_ranking(mpg):
    if mpg <= 1999:
        return 1
    elif mpg < 2499:
        return 2
    elif mpg < 2999:
        return 3
    elif mpg < 3499:
        return 4
    else:
        return 5

def select_attribute(instances, available_attributes, header, attribute_domains):
    # build a tree replace random w/entropy
    #rand_index = random.randrange(0, len(available_attributes))
    #return available_attributes[rand_index]

    #entrophy
    attribute_entrophy = []
    for att in available_attributes:
        partitions = partition_instances(instances, att, header, attribute_domains)
        E_new = []
        for attribute_value, partition in partitions.items():
            col = get_column_by_index(partition, -1)
            vals, counts = get_frequencies(col)
            if (len(vals) > 1):
                p_yes = counts[0]/len(partition)
                p_no = counts[1]/len(partition)
                E = -(p_yes * math.log(p_yes, 2) + p_no * math.log(p_no, 2))
                E = E * (len(partition) / len(instances))
                E_new.append(E)
        attribute_entrophy.append(sum(E_new))
    #print("E's: ", attribute_entrophy)
    E_index = attribute_entrophy.index(min(attribute_entrophy))
    return available_attributes[E_index]
   

def partition_instances(instances, split_attribute, header, attribute_domains):
    # comments refer to split_attribute "level"
    attribute_domain = attribute_domains[split_attribute] # ["Senior", "Mid", "Junior"]
    attribute_index = header.index(split_attribute) # 0

    partitions = {} # key (attribute value): value (partition)
    # task: try to finish this
    for attribute_value in attribute_domain:
        partitions[attribute_value] = []
        for instance in instances:
            if instance[attribute_index] == attribute_value:
                partitions[attribute_value].append(instance)

    return partitions 

def all_same_class(instances):
    # assumption: instances is not empty and class label is at index -1
    assert len(instances) != 0
    

    first_label = instances[0][-1]
    for instance in instances:
        if instance[-1] != first_label:
            return False
    return True # if we get here, all instance labels matched the first label

def tdidt(current_instances, available_attributes, header, attribute_domains):
    # select an attribute to split on
    split_attribute = select_attribute(current_instances, available_attributes, header, attribute_domains)
    #print("splitting on:", split_attribute)
    
    # remove split attribute from available attributes
    # because, we can't split on the same attribute twice in a branch
    available_attributes.remove(split_attribute) # Python is pass by object reference!!
    tree = ["Attribute", split_attribute]

    # group data by attribute domains (creates pairwise disjoint partitions)
    partitions = partition_instances(current_instances, split_attribute, header, attribute_domains)
    #print("partitions:", partitions)
    
    # for each partition, repeat unless one of the following occurs (base case)
    for attribute_value, partition in partitions.items():
        values_subtree = ["Value", attribute_value]
        #print("Partition:", partition)
        #print("values_subtree", values_subtree)

        #    CASE 1: all class labels of the partition are the same => make a leaf node
        if len(partition) > 0 and all_same_class(partition):
            #print("CASE 1")
            leaf_node = ["Leaf", partition[0][-1], len(partition), len(current_instances)]
            #print("leaf_node", leaf_node)
            values_subtree.append(leaf_node)

        #    CASE 2: no more attributes to select (clash) => handle clash w/majority vote leaf node
        elif len(partition) > 0 and len(available_attributes) == 0:
            #print("CASE 2")
            # group possible answers
            answers = []
            for parts in partition:
                answers.append(parts[-1])
            # find majoity
            majority = max(set(answers), key = answers.count)
            leaf_node = ["Leaf", majority, len(partition), len(current_instances)]
            #print("leaf_node", leaf_node)
            values_subtree.append(leaf_node)
        #    CASE 3: no more instances to partition (empty partition) => backtrack and replace attribute node with majority vote leaf node
        elif len(partition) == 0:
            #print("CASE 3")
            answers = []
            for attribute_value, new_partition in partitions.items():
                if len(new_partition) > 0:
                    # group possible answers      
                    for parts in new_partition:         
                        answers.append(parts[-1])
            # find majoity
            majority = max(set(answers), key = answers.count)
            leaf_node = ["Leaf", majority, len(partition), len(current_instances)]
            return leaf_node
        else: # all base cases are false, recurse!!
            subtree = tdidt(partition, available_attributes.copy(), header, attribute_domains)
            values_subtree.append(subtree)

        tree.append(values_subtree)
        #print("tree_after_loop", tree)
    return tree

def tdidt_predict(header, tree, instance):
    info_type = tree[0]
    if info_type == "Attribute":
        attribute_index = header.index(tree[1])
        instance_value = instance[attribute_index]
        for i in range(2, len(tree)):
            value_list = tree[i]
            if value_list[1] == instance_value:
                return tdidt_predict(header, value_list[2], instance)
    else: # leaf
        return tree[1]

def tdidt_rules(header, tree):
    info_type = tree[0]
    if info_type == "Attribute":
        
        return tdidt_rules(header, tree)
    else: # leaf
        return tree[1]
def build_dot_graph(tree, higher_att):
    g = ""
    curr_val = ""
    curr_att = ""
    for i, value in enumerate(tree):
        g += "\t" #tab

        #if it is a list
        if isinstance(value, list):
            g += build_dot_graph(value, curr_att)
        elif value == "Attrubute":
            #g += tree[i+1], ";\n"
            g += (higher_att, " -- ", tree[i+1], "[label='", curr_val, "'")
        elif value == "value":
            #g.append(prev_attribute, " -- ", next_attribute, "[label='", tree[i+1], "'")
            curr_val = tree[i+1]

        g.append("\n")
    # level -- phd [label="Junior"];
    return g
    

def combine_genre(table):
    for instance in table:
        if "pop" in instance[3]:
            instance[3] = "pop"
        elif "hip hop" in instance[3]:
            instance[3] = "hip hop"
        elif "rap" in instance[3]:
            instance[3] = "rap"
        elif "rock" in instance[3]:
            instance[3] = "rock"
        elif "latin" in instance[3]:
            instance[3] = "latin"
        elif "r&b" in instance[3]:
            instance[3] = "r&b"
        elif "indie" in instance[3]:
            instance[3] = "indie"
        elif "country" in instance[3]:
            instance[3] = "country"
        elif "metal" in instance[3]:
            instance[3] = "metal"
        elif "classical" in instance[3]:
            instance[3] = "classical"
        else:
            instance[3] = ""
    i = len(table) - 1
    for row in reversed(table):
        if row[3] == "":
            del table[i]
        i -= 1
    pass

def get_majority_vote(table):
    votes = []

    for i in range(len(table[0])):
        vals, counts = get_frequencies_index(table, i)
        majority = get_majority(vals, counts)
        votes.append(majority)
    return votes



def random_attribute_subset(attributes, F):
    # shuffle and pick first F
    shuffled = attributes[:] # make a copy
    random.shuffle(shuffled)
    return shuffled[:F]

def compute_bootstrapped_sample(table, y):
    n = len(table)
    sample = []
    sample_y = []
    for _ in range(n):
        rand_index = random.randrange(0, n)
        sample.append(table[rand_index])
        sample_y.append(y[rand_index])
    return sample, y 

def discretize_popularity(music_data, index):
    for i, value in enumerate(music_data):
        if value[index] <= 20:
            value[index] = "Low"
        elif value[index] > 20 and value[index] <= 40:
            value[index] = "Below Average"
        elif value[index] > 40 and value[index] <= 60:
            value[index] = "Average"
        elif value[index] > 60 and value[index] <= 80:
            value[index] = "Above Average"
        elif value[index] > 80:
            value[index] = "High"
