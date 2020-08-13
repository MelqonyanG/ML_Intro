import numpy as np

class LDA(object):

    def __init__(self, K):
        self.K = K
        self.explained_variance_ = np.array([])
        self.explained_variance_ratio_ = np.array([])
        self.eigenvectors_ = np.array([])
    
    def compute_scatters(self, X, y):
        """
        param X: numpy array of shape (M,N)
        param y: numpy array of shape (M), shows to which class each row of X belongs
        return S_w, S_b: scatter within and scatter between matrices which are of shape (N,N)
        """
        raise NotImplementedError
    
    def fit(self, X, y):
        """
        param X: numpy array of shape (M,N)
        param y: numpy array of shape (M), shows to which class each row of X belongs
        """
        """TODO fit the model,(compute scatter matrices and compute 
        eigenvalues and eigenvectors of S_w^{-1}S_b) and update values of 
        self.explained_variance_, self.explained_variance_ratio_, self.eigenvectors"""
        raise NotImplementedError

    def transform(self, X):
        """
        param X: numpy array of shape (M,N)
        return X_proj: numpy array of shape (M,K)
        """
        """TODO use self.explained_variance_ratio_ and self.eigenvectors_
        to project X from dimension N to K"""
        raise NotImplementedError


