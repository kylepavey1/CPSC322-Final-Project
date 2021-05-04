"""
Programmers: Kyle Pavey and Adam Lee
Class: 322-01, Spring 2021
Final Project
4/21/21
"""


import importlib

import mysklearn.myutils
importlib.reload(mysklearn.myutils)
import mysklearn.myutils as myutils
import copy
import csv 
import math
import statistics
from tabulate import tabulate # uncomment if you want to use the pretty_print() method
# install tabulate with: pip install tabulate

# required functions/methods are noted with TODOs
# provided unit tests are in test_mypytable.py
# do not modify this class name, required function/method headers, or the unit tests
class MyPyTable:
    """Represents a 2D table of data with column names.

    Attributes:
        column_names(list of str): M column names
        data(list of list of obj): 2D data structure storing mixed type data. There are N rows by M columns.
    """

    def __init__(self, column_names=None, data=None):
        """Initializer for MyPyTable.

        Args:
            column_names(list of str): initial M column names (None if empty)
            data(list of list of obj): initial table data in shape NxM (None if empty)
        """
        if column_names is None:
            column_names = []
        self.column_names = copy.deepcopy(column_names)
        if data is None:
            data = []
        self.data = copy.deepcopy(data)

    # def pretty_print(self):
    #     """Prints the table in a nicely formatted grid structure.
    #     """
    #     print(tabulate(self.data, headers=self.column_names))

    def get_shape(self):
        """Computes the dimension of the table (N x M).

        Returns:
            int: number of rows in the table (N)
            int: number of cols in the table (M)
        """
        rows = len(self.data)
        columns = len(self.data[0])
        return rows, columns

    def get_column(self, col_identifier, include_missing_values=True):
        """Extracts a column from the table data as a list.

        Args:
            col_identifier(str or int): string for a column name or int
                for a column index
            include_missing_values(bool): True if missing values ("NA")
                should be included in the column, False otherwise.

        Returns:
            list of obj: 1D list of values in the column

        Notes:
            Raise ValueError on invalid col_identifier
        """
        col_list = []  
        int_col_identifier = 0      
        if isinstance(col_identifier, str):
            int_col_identifier = self.column_names.index(col_identifier)
            # for col in range(len(self.column_names)):
            #     if self.column_names[col] == col_identifier:
            #         int_col_identifier = col
        else:
            int_col_identifier = col_identifier
        if(include_missing_values):
            for row in self.data:
                col_list.append(row[int_col_identifier])
        else:
            for row in self.data:
                if(row[int_col_identifier] == "NA" or row[int_col_identifier] == "" or row[int_col_identifier] == " "):
                    return
                else:
                    col_list.append(row[int_col_identifier])
        return col_list

    def convert_to_numeric(self):
        """Try to convert each value in the table to a numeric type (float).

        Notes:
            Leave values as is that cannot be converted to numeric.
        """
        for row_index, row in enumerate(self.data):
            for col_index, col in enumerate(row):
                try:
                    numeric_value = 0
                    if type(col) == float:
                        numeric_value = float(col)
                    elif "." not in col:
                        numeric_value = int(col)
                    else:
                        numeric_value = float(col)
                    self.data[row_index][col_index] = numeric_value
                except ValueError:
                    pass

    def convert_column_to_string(self, column_name):
        index = 0
        for i, col in self.column_names:
            if column_name == col:
                index = i
        for rowindex, row in enumerate(self.data):
            try:
                string_val = str(row[i])
                self.data[rowindex][i] = string_val
            except ValueError:
                pass

    def drop_rows(self, rows_to_drop):
        """Remove rows from the table data.

        Args:
            rows_to_drop(list of list of obj): list of rows to remove from the table data.
        """
        for row_in_table in list(self.data):
            for row in rows_to_drop:
                if row == row_in_table:
                    try:
                        self.data.remove(row)
                    except ValueError:
                        pass

    def load_from_file(self, filename):
        """Load column names and data from a CSV file.

        Args:
            filename(str): relative path for the CSV file to open and load the contents of.

        Returns:
            MyPyTable: return self so the caller can write code like: table = MyPyTable().load_from_file(fname)
        
        Notes:
            Use the csv module.
            First row of CSV file is assumed to be the header.
            Calls convert_to_numeric() after load
        """
        data = []
        with open(filename, 'rt') as f:
            csv_reader = csv.reader(f, quotechar='"', delimiter=',',
                        quoting=csv.QUOTE_ALL, skipinitialspace=True)
            for line in csv_reader:
                data.append(line)
        self.data = data
        self.column_names = data.pop(0)
        self.convert_to_numeric()
        return self 

    def save_to_file(self, filename):
        """Save column names and data to a CSV file.

        Args:
            filename(str): relative path for the CSV file to save the contents to.

        Notes:
            Use the csv module.
        """
        # open(filename, "w").close()
        with open(filename, 'a+', newline='') as csvfile: 
            # creating a csv writer object 
            csvwriter = csv.writer(csvfile)

            # writing the fields 
            # csvwriter.writerow(self.column_names) 
                
            # writing the data rows 
            csvwriter.writerows(self.data)

        # with open(filename, "w") as f:
        #     csv_writer = csv.writer(f)
        #     csv_writer.writerow(self.column_names)
        # with open(filename, "a+", newline="") as f:
        #     csv_writer = csv.writer(f)
        #     csv_writer.writerows(self.data)

    def find_duplicates(self, key_column_names):
        """Returns a list of duplicates. Rows are identified uniquely baed on key_column_names.

        Args:
            key_column_names(list of str): column names to use as row keys.

        Returns: 
            list of list of obj: list of duplicate rows found
        
        Notes:
            Subsequent occurrence(s) of a row are considered the duplicate(s). The first instance is not
                considered a duplicate.
        """
        passed_vals = []
        duplicates = []
        column_lists = []
        duplicate_indexes = []
        for col_name in key_column_names:
            column = self.get_column(col_name)
            column_lists.append(column)
        size = len(column_lists[0])
        test = []
        for i in range(size):
            val = [item[i] for item in column_lists]
            test.append(val)
        for index, elem in enumerate(test):
            if elem not in passed_vals:
                passed_vals.append(elem)
            else:
                duplicate_indexes.append(index)
        for index, row in enumerate(self.data):
            if index in duplicate_indexes:
                duplicates.append(row)
        return duplicates

    def remove_rows_with_missing_values(self):
        """Remove rows from the table data that contain a missing value ("NA").
        """
        indexes = []
        for i, row in enumerate(self.data):
            for col in row:
                if col == "NA" or col == "N/A":
                    indexes.append(i)
        for i in reversed(indexes):
            self.data.pop(i)

    def remove_empty_rows(self, value):
        col_index = self.column_names.index(value)
        indexes = []
        for i, row in enumerate(self.data):
                if row[col_index] == '':
                    indexes.append(i)
        for i in reversed(indexes):
            self.data.pop(i)
    
    def replace_missing_values_with_column_average(self, col_name):
        """For columns with continuous data, fill missing values in a column by the column's original average.

        Args:
            col_name(str): name of column to fill with the original average (of the column).
        """
        col = self.get_column(col_name)
        col_index = 0
        for i, col in enumerate(self.column_names):
            if col == col_name:
                col_index = i
        
        for row in self.data:
            if row[col_index] == "NA":
                try:
                    avg = sum(col) / len(col)
                    self.data.row[col_index] = avg
                except:
                    pass
            # try:
            #     avg = sum(map(float, col))
            #     for row in self.data:
            #         if row[col_name] == "NA":
            #             row[col_name] = avg
            # except ValueError:
        
        

    def compute_summary_statistics(self, col_names):
        """Calculates summary stats for this MyPyTable and stores the stats in a new MyPyTable.

        Args:
            col_names(list of str): names of the continuous columns to compute summary stats for.

        Returns:
            MyPyTable: stores the summary stats computed. The column names and their order
                is as follows: ["attribute", "min", "max", "mid", "avg", "median"]
        """
        stats_column_names = ["attribute", "min", "max", "mid", "avg", "median"]
        stats_data = []
        if self.data:
            for name in col_names:
                col = self.get_column(name)
                min_val = min(col)
                max_val = max(col)
                mid = (float(max_val) + float(min_val)) / 2
                avg = sum(col) / len(col)
                median = statistics.median(col)
                stats_data.append([name, min_val, max_val, mid, avg, median])
        return MyPyTable(stats_column_names, stats_data)

    def perform_inner_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable inner joined with other_table based on key_column_names.

        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.

        Returns:
            MyPyTable: the inner joined table.
        """
        
        return MyPyTable() # TODO: fix this

    def perform_full_outer_join(self, other_table, key_column_names):
        """Return a new MyPyTable that is this MyPyTable fully outer joined with other_table based on key_column_names.

        Args:
            other_table(MyPyTable): the second table to join this table with.
            key_column_names(list of str): column names to use as row keys.

        Returns:
            MyPyTable: the fully outer joined table.

        Notes:
            Pad the attributes with missing values with "NA".
        """
        return MyPyTable() # TODO: fix this
