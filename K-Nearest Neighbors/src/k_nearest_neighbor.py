import numpy as np 
from scipy import stats
from .distances import euclidean_distances, manhattan_distances

class KNearestNeighbor():    
    def __init__(self, n_neighbors, distance_measure='euclidean', aggregator='mode'):
        """
        K-Nearest Neighbor is a straightforward algorithm that can be highly
        effective. Training time is...well...is there any training? At test time, labels for
        new points are predicted by comparing them to the nearest neighbors in the
        training data.

        ```distance_measure``` lets you switch between which distance measure you will
        use to compare data points. The behavior is as follows:

        If 'euclidean', use euclidean_distances, if 'manhattan', use manhattan_distances.

        ```aggregator``` lets you alter how a label is predicted for a data point based 
        on its neighbors. If it's set to `mean`, it is the mean of the labels of the
        neighbors. If it's set to `mode`, it is the mode of the labels of the neighbors.
        If it is set to `median`, it is the median of the labels of the neighbors. If the
        number of dimensions returned in the label is more than 1, the aggregator is
        applied to each dimension independently. For example, if the labels of 3 
        closest neighbors are:
            [
                [1, 2, 3], 
                [2, 3, 4], 
                [3, 4, 5]
            ] 
        And the aggregator is 'mean', applied along each dimension, this will return for 
        that point:
            [
                [2, 3, 4]
            ]

        Arguments:
            n_neighbors {int} -- Number of neighbors to use for prediction.
            distance_measure {str} -- Which distance measure to use. Can be one of
                'euclidean' or 'manhattan'. This is the distance measure
                that will be used to compare features to produce labels. 
            aggregator {str} -- How to aggregate a label across the `n_neighbors` nearest
                neighbors. Can be one of 'mode', 'mean', or 'median'.
        """
        # raise NotImplementedError()
        
        self.n_neighbors = n_neighbors
        self.distance_measure = distance_measure
        self.aggregator = aggregator

    def fit(self, features, targets):
        """Fit features, a numpy array of size (n_samples, n_features). For a KNN, this
        function should store the features and corresponding targets in class 
        variables that can be accessed in the `predict` function. Note that targets can
        be multidimensional! 
        
        Arguments:
            features {np.ndarray} -- Features of each data point, shape of (n_samples,
                n_features).
            targets {[type]} -- Target labels for each data point, shape of (n_samples, 
                n_dimensions).
        """
        # raise NotImplementedError()

        self.training_features = features
        self.training_targets = np.array(targets)

    def predict(self, features, ignore_first=False):
        """Predict from features, a numpy array of size (n_samples, n_features) Use the
        training data to predict labels on the test features. For each testing sample, compare it
        to the training samples. Look at the self.n_neighbors closest samples to the 
        test sample by comparing their feature vectors. The label for the test sample
        is the determined by aggregating the K nearest neighbors in the training data.

        Note that when using KNN for imputation, the predicted labels are the imputed testing data
        and the shape is (n_samples, n_features).

        Arguments:
            features {np.ndarray} -- Features of each data point, shape of (n_samples,
                n_features).
            ignore_first {bool} -- If this is True, then we ignore the closest point
                when doing the aggregation. This is used for collaborative
                filtering, where the closest point is itself and thus is not a neighbor. 
                In this case, we would use 1:(n_neighbors + 1).

        Returns:
            labels {np.ndarray} -- Labels for each data point, of shape (n_samples,
                n_dimensions). This n_dimensions should be the same as n_dimensions of targets in fit function.
        """
        # raise NotImplementedError()

        M = features.shape[0]                   # Number of rows to predict
        N = self.training_features.shape[0]     # Number of rows in training_features
        K = features.shape[1]                   # Number of features
        dim = self.training_targets.shape[1]    # Number of dimensions of target

        dist = np.zeros((M, N))

        if self.distance_measure == 'euclidean':
            dist = euclidean_distances(features, self.training_features)
        else: # manhattan
            dist = manhattan_distances(features, self.training_features)

        labels = np.zeros((M, dim))

        for m in range(M):
            current_instance = dist[m]
            
            if ignore_first == False:
                idx = np.argsort(current_instance)[:self.n_neighbors]    # idx = array of indices where their dist values are the smallest n_neighbors
            else:
                idx = np.argsort(current_instance)[1:self.n_neighbors+1]    # idx = array of indices where their dist values are the smallest n_neighbors

            nn_labels = np.array(self.training_targets[idx])

            label = np.zeros(dim)

            if self.aggregator == 'mean':
                label = np.mean(nn_labels, axis=0)
            elif self.aggregator == 'mode':
                label = stats.mode(nn_labels)[0]
            else: # median
                label = np.median(nn_labels, axis=0)

            labels[m] = label

        return labels



        

        

