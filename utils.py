"""
Programmer: Kyle Pavey
Class: 322-01, Spring 2021
Programming Assignment #3
2/25/21
"""
from collections import Counter
def get_column(table, header, col_name):
    """Retrieves a column from the table.

        Args: 
            table: the table to be parsed
            header: the header row of the table
            col_name: the column to be counted from
        Returns:
            list: the column values
    """
    col_index = header.index(col_name)
    col = []
    for row in table: 
        if row[col_index] != "NA":
            col.append(row[col_index])
    return col

def get_min_max(values):
    """Computes the number of occurrences in a column.

        Args: 
            values: a list of values
        Returns:
            int or float: the max and min values
    """
    return min(values), max(values)

def get_frequencies(table, header, col_name):
    """Computes the number of occurrences in a column.

        Args: 
            table: the table to be parsed
            header: the header row of the table
            col_name: the column to be counted from
        Returns:
            list: the values
            list: the number of occurrences
    """
    col = get_column(table, header, col_name)
    col.sort()
    values = Counter(col).keys() # equals to list(set(words))
    counts = Counter(col).values() #



    # col = get_column(table, header, col_name)
    # print(col)
    # if type(col) == int or type(col) == float:
    #     col.sort() 
    # values = []
    # counts = []

    # for value in col:
    #     if value not in values:
    #         values.append(value)
    #         counts.append(1)
    #     else:
    #         if type(col) == int or type(col) == float:
    #             counts[-1] += 1 
    #         else:
    #             for index, val in enumerate(values):
    #                 if value == val:
    #                     counts[index] += 1
    return values, counts

def convert_column_to_string(tablename, column_name):
    """Converts values in a column to a string.

        Args: 
            tablename: the table to be parsed
            column_name: the column to be converted
    """
    index = 0
    for i, col in enumerate(tablename.column_names):
        if column_name == col:
            index = i
    for rowindex, row in enumerate(tablename.data):
        try:  
            string_val = str(row[index])
            tablename.data[rowindex][index] = string_val
        except ValueError:
            pass

def get_percentages(table, col_list):
    """Gets the percentages of each item out of the whole column.

        Args: 
            table: the table to be parsed
            col_list: the column to be counted from
        Returns:
            list: the percentages based on index
    """
    sum_list = []
    percent_list = []
    for col in col_list:
        col_sum = get_column_sum(table.data, table.column_names, col)
        sum_list.append(col_sum)
    total = sum_list[-1]
    for sum_val in sum_list[:-1]:
        percent_list.append(sum_val / total)
    return percent_list


def get_column_sum(table, header, col_name):
    """Computes the sum of the items in a column.

        Args: 
            table: the table to be parsed
            header: the header row of the table
            col_name: the column to be counted from
        Returns:
            int: the sum
    """
    col = get_column(table, header, col_name)
    col_sum = 0
    for val in col:
        col_sum += val
    return col_sum

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

def remove_empty_values(column):
    """Removes empty values from a column.

        Args: 
            column: the column to remove values from
        Returns:
            list: a list of existing values
    """
    refined_list = [i for i in column if i] 
    return refined_list

def strip_value(column, value):
    """Removes a substring from a string.

        Args: 
            col_name: the column to be altered
    """
    for index, x in enumerate(column):
        column[index] = float(x.strip(value))
    
def remove_parallel_missing_values(col1, col2):
    """Removes values from two columns simultaneously.

        Args: 
            col1: the first column
            col2: the second column    
    """
    for i, val in reversed(list(enumerate(col1))):
        if not col1[i] or col1[i] == "NA" or not col2[i] or col2[i] == "NA":
            col1.pop(i)
            col2.pop(i)

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
    col = get_column(table, header, group_by_col_name)
    col = remove_empty_values(col)
    col_index = header.index(group_by_col_name)
    split_col = split_multiple_string_values(col)
    group_names = sorted(list(set(split_col))) 
    group_subtables = [[] for _ in group_names] 
    for i, row in enumerate(table):
        group_by_value = split_col[i]
        group_index = group_names.index(group_by_value)
        group_subtables[group_index].append(row.copy())
    return group_names, group_subtables

def split_multiple_string_values(col):
    """Splits strings with ,.

        Args: 
            col: the column to split values in
        Returns:
            list: the column with every substring 
    """
    new_col = []
    for item in col:
        word_list = item.split(',')
        for word in word_list:
            if word:
                new_col.append(word)
    return new_col
