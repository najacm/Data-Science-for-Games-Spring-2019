
# import libraries


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster
from matplotlib import style
from scipy.spatial import distance
import math
from math import *

style.use('seaborn-white')

from sklearn import preprocessing

# remove outliers higher than Q3 and lower than Q1
def remove_outliers(data, column):
    data = data.sort_values(by = column, ascending = True)

    q3 = data[column].quantile(0.90)
    data = data[data[column] < q3]

    q1 = data[column].quantile(0.10)
    data = data[data[column] > q1]

    filtered_data = data

    return filtered_data


# PREPROCESSING
def start_preprocessing(df, del_outliers):
    # set State to string
    df[['state']] = df[['state']].astype(str)

    # set 'total night calls' and 'total international calls' to numerical
    df[['nightCall','intCall']] = df[['nightCall','intCall']].astype(np.float64)


    # min max normalization/standardization for nightcall and intcall respectivily
    scaler = preprocessing.MinMaxScaler()
    preprocessing.MinMaxScaler(copy=True, feature_range=(0, 1))
    scaler.fit(df[['nightCall']])
    df[['nightCall']] = scaler.transform(df[['nightCall']])
    scaler.fit(df[['intCall']])
    df[['intCall']] = scaler.transform(df[['intCall']])

    # load dataset with states
    states_data_raw = pd.read_csv('state_table4.csv', delimiter=';')
    states_chosen_attr = ['abbreviation', 'census_region_name']
    states_df = pd.DataFrame(states_data_raw, columns=states_chosen_attr)
    states_df = states_df.rename(columns={'abbreviation':'state','census_region_name':'region'})
    states_df[['state','region']] = states_df[['state','region']].astype(str)

    # add regions to new region collumn in df
    df = pd.merge(states_df, df, on='state')

    # create new binary collumn based on region
    df['south'] = df['region'].str.contains('South', regex=False)
    df['west'] = df['region'].str.contains('West', regex=False)
    df['northeast'] = df['region'].str.contains('Northeast', regex=False)
    df['midwest'] = df['region'].str.contains('Midwest', regex=False)

    # remove outliers
    if del_outliers:
        df = remove_outliers(df, 'nightCall')
        df = remove_outliers(df, 'intCall')

    # if nan drop object
    df.replace(["nan","NaN","NaT",'nan','NaN','NaT'], np.nan, inplace = True)
    df.dropna()

    return df

# convert boolean columns to 0 and 1
def convert_boolean_columns(df):
    for column_name, column in df.transpose().iterrows():
        bool_columns = list({'west','south','northeast','midwest'}) # columns that are boolean
        if column_name in bool_columns:
            df[column_name] = df[column_name].map({False: 0, True: 1})
    return df

# Input: k: the number of clusters, Data: a data set containing n objects.
# Output: A set of k clusters.
class KMeans:
    def __init__(self, k=3, data=None):
        self.k = k # k = how many clusters I want
        self.max_iterations = 500 # maximum iterations before final clusters are found # TODO: no of iterations remove
        self.reset_centroids(data)

    def reset_centroids(self, D):
        self.centroids = {}

# 1) arbitrarily choose k objects from D as the initial cluster centers;
        for i in range(self.k):
            self.centroids[i] = D[i]

# create class per cluster
        for i in range(self.max_iterations):
            self.classes = {}
            for j in range(self.k):
                self.classes[j] = []

# 2) (re)assign each object to the cluster to which the object is the most similar,
# based on the mean value of the objects in the cluster
            for column in D:
                distances = [distance.euclidean(column, self.centroids[centroid]) for centroid in self.centroids]
                classification = distances.index(min(distances))    # data object in cluster with min distance
                self.classes[classification].append(column)     # append data object to the choosen cluster

            previous = dict(self.centroids)

# 3) update the cluster means, that is, calculate the mean value of the objects for each cluster
            for classification in self.classes:
                self.centroids[classification] = np.mean(self.classes[classification], axis=0)

            for centroid in self.centroids:
                curr_centroid = self.centroids[centroid]
                prev_centroid = previous[centroid]

                # calculate euclidian distance #
                dist_from_last = distance.euclidean(curr_centroid, prev_centroid)

                if dist_from_last == 0:
                    clusters_are_optimal = True
                else:
                    clusters_are_optimal = False

# 4) until no change
            if clusters_are_optimal:
                break

# get classification. used for visualization
    def pred(self, data):
        distances = [np.linalg.norm(data - self.centroids[centroid]) for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification


def start_kmeans(data):
    d = data  # convert to a numpy array
    final_clusters = KMeans(3, d)   # get clusters
    visualize_kmeans(final_clusters)    # visualize clusters

def visualize_kmeans(kmeans_algorithm):
    color_scheme = ["#D46A6A", "#A5C663", "#565695", "#D4AC6A", '#B5E9E6', '#E9B5E4']


# visualize centroids
    for centroid in kmeans_algorithm.centroids:
        color = color_scheme[centroid]
        plt.scatter(kmeans_algorithm.centroids[centroid][0], kmeans_algorithm.centroids[centroid][1], color=color, s=130, marker="x") # X centroids in plotter

# visialize datapoints
    for classification in kmeans_algorithm.classes:
        color = color_scheme[classification]
        for columns in kmeans_algorithm.classes[classification]:
            plt.scatter(columns[0], columns[1], color=color, s=30) # X and Y in plotter

# show plotter
    plt.show()


## IMPORT DATASET

# load dataset
data_raw = pd.read_csv('telecom_churn.csv')

# save relevant attr
chosen_attr = ['State', 'Total night calls','Total intl calls']

df = pd.DataFrame(data_raw, columns=chosen_attr)

# Rename columns if neccessary
df = df.rename(columns={'State':'state','Total night calls':'nightCall','Total intl calls':'intCall'})
df = start_preprocessing(df, del_outliers=False)

convert_boolean_columns(df)
print(df.head()) # for reference

data_to_cluster = df[['nightCall', 'intCall']]
data_to_cluster = data_to_cluster.values # returns a numpy array

start_kmeans(data_to_cluster)
