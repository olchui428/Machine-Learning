import numpy as np

class KMeans():
    def __init__(self, n_clusters):
        """
        This class implements the traditional KMeans algorithm with hard assignments:

        https://en.wikipedia.org/wiki/K-means_clustering

        The KMeans algorithm has two steps:

        1. Update assignments
        2. Update the means

        While you only have to implement the fit and predict functions to pass the
        test cases, we recommend that you use an update_assignments function and an
        update_means function internally for the class.

        Use only numpy to implement this algorithm.

        Args:
            n_clusters (int): Number of clusters to cluster the given data into.

        """
        self.n_clusters = n_clusters
        self.means = None

    def fit(self, features):
        """
        Fit KMeans to the given data using `self.n_clusters` number of clusters.
        Features can have greater than 2 dimensions.

        Args:
            features (np.ndarray): array containing inputs of size
                (n_samples, n_features).
        Returns:
            None (saves model - means - internally)
        """
        # raise NotImplementedError()

        # Initiating means
        self.means = np.zeros((self.n_clusters, features.shape[1]))
        for i in range(self.n_clusters):
            for j in range(features.shape[1]):
                self.means[i][j] = np.random.choice(features[:,j], size=1)
        
        self.labels = np.zeros(features.shape[0], dtype=int)

        # While assignments dont update
        while np.array_equal(self.labels, self.update_assignments(features)) == False:
            # update assignments
            self.labels = self.update_assignments(features)
            # update means
            self.update_means(features)

    def update_assignments(self, features):
        dist = np.zeros((features.shape[0], self.n_clusters))
        labels = np.zeros(features.shape[0], dtype=int)

        for i in range(features.shape[0]):
            for j in range(self.n_clusters):
                dist[i][j] = np.linalg.norm(features[i] - self.means[j])
        
        for i in range(features.shape[0]):
            labels[i] = np.argmin(dist[i])

        return labels
    
    def update_means(self, features):
        cluster_freq = np.zeros((self.n_clusters), dtype=int)
        cluster_total = np.zeros(self.means.shape)

        for i in range(features.shape[0]):
            cluster_id = self.labels[i]
            cluster_freq[cluster_id] += 1
            cluster_total[cluster_id] = np.add(cluster_total[cluster_id],features[i])

        for i in range(self.means.shape[0]):
            for j in range(self.means.shape[1]):
                if cluster_freq[i] == 0:
                    self.means[i][j] = 1
                else:
                    self.means[i][j] = cluster_total[i][j] / cluster_freq[i]

    def predict(self, features):
        """
        Given features, an np.ndarray of size (n_samples, n_features), predict cluster
        membership labels.

        Args:
            features (np.ndarray): array containing inputs of size
                (n_samples, n_features).
        Returns:
            predictions (np.ndarray): predicted cluster membership for each features,
                of size (n_samples,). Each element of the array is the index of the
                cluster the sample belongs to.
        """
        # raise NotImplementedError()
        
        dist = np.zeros((features.shape[0], self.n_clusters))
        predictions = np.zeros(features.shape[0])

        for i in range(features.shape[0]):
            for j in range(self.n_clusters):
                dist[i][j] = np.linalg.norm(features[i] - self.means[j])
        
        for i in range(features.shape[0]):
            predictions[i] = np.argmin(dist[i])

        return predictions
