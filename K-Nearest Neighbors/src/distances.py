import numpy as np 

def euclidean_distances(X, Y):
    """Compute pairwise Euclidean distance between the rows of two matrices X (shape MxK) 
    and Y (shape NxK). The output of this function is a matrix of shape MxN containing
    the Euclidean distance between two rows.
    
    Arguments:
        X {np.ndarray} -- First matrix, containing M examples with K features each.
        Y {np.ndarray} -- Second matrix, containing N examples with K features each.

    Returns:
        D {np.ndarray}: MxN matrix with Euclidean distances between rows of X and rows of Y.
    """
    # raise NotImplementedError()

    M = X.shape[0]
    N = Y.shape[0]
    K = X.shape[1] # Number of features

    D = np.zeros((M, N))

    for m in range(M):
        for n in range(N):
            terms = np.zeros(K)
            for k in range(K):
                x = X[m][k]
                y = Y[n][k]
                term = (np.abs(x-y))**2
                terms[k] = term
            D[m][n] = (terms.sum())**(1/2)
    
    return D

def manhattan_distances(X, Y):
    """Compute pairwise Manhattan distance between the rows of two matrices X (shape MxK) 
    and Y (shape NxK). The output of this function is a matrix of shape MxN containing
    the Manhattan distance between two rows.
    
    Arguments:
        X {np.ndarray} -- First matrix, containing M examples with K features each.
        Y {np.ndarray} -- Second matrix, containing N examples with K features each.

    Returns:
        D {np.ndarray}: MxN matrix with Manhattan distances between rows of X and rows of Y.
    """
    # raise NotImplementedError()

    M = X.shape[0]
    N = Y.shape[0]
    K = X.shape[1] # Number of features

    D = np.zeros((M, N))

    for m in range(M):
        for n in range(N):
            terms = np.zeros(K)
            for k in range(K):
                x = X[m][k]
                y = Y[n][k]
                term = (np.abs(x-y))**1
                terms[k] = term
            D[m][n] = (terms.sum())**(1/1)
    
    return D
