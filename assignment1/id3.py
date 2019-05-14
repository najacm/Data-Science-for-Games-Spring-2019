from math import log
import pandas as pd
import numpy as np
import operator

# rename and set types for chosen attributes
def rename_and_typesetter(D):
    D = D.rename(columns={'Total intl minutes': 'min', 'Total intl calls': 'calls', 'Total intl charge': 'charge', 'International plan': 'plan'})
    D[['min', 'calls', 'charge']] = D[['min', 'calls', 'charge']].astype(np.float)
    D[['plan']] = D[['plan']].astype(np.str)
    return D

# function to split values by the median if they are float
def split_continous_values(D):
    high_no = 5
    low_no = -1

    for column in D:     # for each column split by median
        if type(data_frame[column][1]) is np.float64:  # only split if it is float
            median = np.median(D[column])
            low_no = low_no * 10  # generate new number
            high_no = high_no * 10 # generate new number
            D[column] = np.where(D[column] >= median, high_no, D[column])  # if >= median, replace with high no
            D[column] = np.where(D[column] < median, low_no, D[column])    # if < median, replace with low no
    return D

# calculate majority for the leaf note
def majority_voting(classes):
    class_count = {}
    for occurence in classes:
        if occurence not in class_count.keys():
            class_count[occurence] = 0
        class_count[occurence] += 1
    sorted_class_count = sorted(class_count.items(), key=operator.itemgetter(1), reverse=True)  # sort to get majority first
    return sorted_class_count[0][0]


# calculate entropy - probability of each class in feature
def entropy(D):
    no_of_entries = len(D)
    labels = {}
    for feature in D:
        label = feature[0]
        if label not in labels.keys():
            labels[label] = 0
            labels[label] += 1
    e = 0.0
    for l in labels:
        probability = float(labels[l])/no_of_entries
        e -= probability * log(probability, 2)
    return e

# split data to subsets
def split_to_subsets(D, column, value_to_check):
    sub_dataset = []
    for feature in D:
        if feature[column] == value_to_check:     # if feature class matches the current check-value
            unique_feature = list(feature[:column])    # add feature to list with the unique feature
            unique_feature.extend(feature[column + 1:])
            sub_dataset.append(unique_feature)
    return sub_dataset

# pick winning feature - to add to tree
def find_winner_feature(D):
    winner_info_gain = 0.0
    winner_feature = -1
    features = len(D[0]) - 1
    for feature in range(features):
        feature_list = [d[feature] for d in D]   # create list of all values in a feature
        unique_values = set(feature_list) # get unique values from list of all values in a feature
        curr_entropy = 0.0
        for value in unique_values:
            new_data = split_to_subsets(D, feature, value)    # split to subset
            probability = len(new_data)/len(D)   # occurence of this subset
            curr_entropy += probability * entropy(new_data)  # calculate entropy
        info_gain = curr_entropy
        if info_gain > winner_info_gain:  # is info gain higher than current winner info gain
            print(winner_info_gain)
            winner_info_gain = info_gain
            winner_feature = feature
    return winner_feature

def id3(D, labels):
    # create a node N;
    N = 0
    C = [ex[-1] for ex in D]

    # if tubles in D are all of the same class, C, then that class is the leafnode
    if C.count(C[0]) == len(C):
        return C[0]

    # if this is the last attribute, get majority class in d as leafnode
    if len(D[0]) == 1:
        return majority_voting(C)

    # apply Attribute selection method to find the “best” splitting criterion and label node N with splitting criterion;
    N = find_winner_feature(D)
    winner_feature_label = labels[N]
    the_tree = {winner_feature_label: {}}   # adds label (e.g. 'calls') to tree

    del(labels[N])  # delete winning feature from labels, since it is already on the tree
    feature_values = [ex[N] for ex in D]
    unique_values = set(feature_values)

    # for each unique value, find subset and add to tree
    for value in unique_values:
        sub_labels = labels[:]
        the_tree[winner_feature_label][value] = id3(split_to_subsets(D, N, value), sub_labels)
    return the_tree

# load dataset
data_raw = pd.read_csv('telecom_churn.csv', delimiter=",")  # load dataset
chosen_attr = ['Total intl minutes','Total intl calls', 'Total intl charge','International plan']  # save relevant attr

data_frame = pd.DataFrame(data_raw, columns=chosen_attr)    # convert to dataframe with relevant attr
data_frame = rename_and_typesetter(data_frame)  # run preprocessing function

data_frame = split_continous_values(data_frame) # create binary option for continous values in dataset
print(data_frame.head(2))   # print for reference
feature_labels = list(data_frame.columns.values)
data_array = np.array(data_frame)

# create and print tree
great_tree = id3(data_array, feature_labels)
print("******** TREE:", great_tree, "********")

# EVALUATION

# calculate occurences for leaf node branches
def count_leaf_occurences(data_array):
    data_list = data_array.tolist()
    full_set_size = len(data_array)

    winner_list1 = list([50.0, -100.0, 5000.0, 'No'])
    winner_list2 = list([50.0, 500.0, 5000.0, 'No'])
    winner_list3 = list([-10.0, -100.0, -1000.0, 'No'])
    winner_list4 = list([-10.0, 500.0, -1000.0, 'No'])
    win = list([winner_list1, winner_list2, winner_list3, winner_list4])

    for winner_list in win:
        occurences = 0
        probability = 0
        for element in data_list:
            if element == winner_list:
                occurences = occurences + 1 # count occurence of leaf
        probability = (occurences/full_set_size)
        print(winner_list, 'occurence: ', occurences, 'percentage:',float(probability))



count_leaf_occurences(data_array) # run if evaluation is wanted


