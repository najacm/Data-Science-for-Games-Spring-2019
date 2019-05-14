# Loading dependencies by running the code block below.
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.spatial import distance_matrix
import seaborn as sns
sns.set()
sns.set_style("dark")


class SOM:
    def __init__(self, m: int, n: int, dim: int, n_iterations: int = 100, radius: float = None, sigma: float = None):
        self._m = m
        self._n = n
        self._dim = dim
        self._n_iterations = n_iterations
        self._radius = radius or max(m, n) / 10.0
        self._sigma = sigma or 0.1

        # np doesn't allow argmin over multiple axis, so we squash the first two dimensions (will make find the BMU easy)
        self.map = np.random.rand(m * n, dim)

        self.locations = self._location_matrix()

        # Get the map distance between each neuron (i.e. not the weight distance).
        # distance[i][j] = distance between neuron i and neuron j
        #  [[0.        , 1.        , 1.        , 1.41421356],
        # [1.        , 0.        , 1.41421356, 1.        ],
        # [1.        , 1.41421356, 0.        , 1.        ],
        # [1.41421356, 1.        , 1.        , 0.        ]]
        self.distances = distance_matrix(self.locations, self.locations)

    def train(self, data):
        # TODO
        pass

    def eval(self, data):
        # TODO
        pass

    def get_umatrix(self):
        """
        Generates an M x N u-matrix of the SOM's weights.

        Used to visualize higher-dimensional data. Shows the average distance between a SOM unit and its neighbors.
        When displayed, areas of a darker color separated by lighter colors correspond to clusters of units which
        encode similar information.
        """
        umatrix = np.zeros((self._m * self._n, 1))

        for i in range(self._m * self._n):
            # Get the indices of the units which neighbor i
            neighbor_idxs = self.distances[i] <= 1  # Change this to `< 2` if you want to include diagonal neighbors
            # Get the weights of those units
            neighbor_weights = self.map[neighbor_idxs]
            # Get the average distance between unit i and all of its neighbors
            # Expand dims to broadcast to each of the neighbors
            umatrix[i] = distance_matrix(np.expand_dims(self.map[i], 0), neighbor_weights).mean()

        return umatrix.reshape((self._m, self._n))

    def get_weight_matrix(self):
        """
        Returns the map weights with the proper M x N
        """
        return self.map.reshape((self._m, self._n, self._dim))

    def _get_bmu(self, d):
        # TODO
        pass

    def _location_matrix(self):
        # Get the location of the neurons on the map to figure out their neighbors
        neuron_locs = list()
        for i in range(self._m):
            for j in range(self._n):
                neuron_locs.append(np.array([i, j]))

        return neuron_locs

    def _proximity_constraint(self, bmu):
        # TODO
        pass

    def _iteration_constraint(self, t):
        # TODO formula from the slides
        pass



colors_dataset = pd.read_csv('colors1.csv')

features = ['R', 'G', 'B']
labels = 'Name'
colors_dataset['Train'] = 1
colors_dataset.loc[15:, 'Train'] = 0

som = SOM(20, 30, len(features), 100)
som.train(colors_dataset[colors_dataset['Train'] == 1][features].values)
evaluated = som.eval(colors_dataset[colors_dataset['Train'] == 0][features].values)


def visualize_weights(w, evaluation=None, eval_labels=None, train_eval=None, train_labels=None, ax=None):
    if ax is None:
        ax = plt

    ax.imshow(w[:, :, :3])

    if evaluation is not None:
        for i, m in enumerate(evaluation):
            ax.text(m[1], m[0], eval_labels[i], ha='center', va='center',
                    bbox=dict(facecolor='red', alpha=0.5, lw=0))

    if train_eval is not None:
        for i, m in enumerate(train_eval):
            ax.text(m[1], m[0], train_labels[i], ha='center', va='center',
                    bbox=dict(facecolor='white', alpha=0.5, lw=0))


def visualize_map(umatrix, evaluation, lbls, ax=None, legend=None):
    if ax is None:
        ax = plt

    vis_eval = np.array(evaluation).transpose()
    ax.imshow(umatrix, cmap='gray_r')
    sns.scatterplot(x=vis_eval[1], y=vis_eval[0], hue=lbls, legend=legend, palette='tab10', ax=ax)
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)


def som_summary(som, evaluation, labels, classes):
    _, axs = plt.subplots(1, 2, figsize=(15, 30))

    visualize_weights(som.get_weight_matrix(), evaluation, classes, ax=axs[0])
    visualize_map(som.get_umatrix(), evaluation, labels, axs[1], legend='full')

visualize_map(som.get_umatrix(), evaluated, colors_dataset[colors_dataset['Train'] == 0][labels].squeeze().values)
# visualize_weights(som.get_weight_matrix(), evaluated, eval_labels=colors_dataset[colors_dataset['Train'] == 0][labels].squeeze().values)
