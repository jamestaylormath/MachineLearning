import numpy as np
import pandas as pd
import math

def get_mean_vector(X):
    return X.mean(axis=0)

def euclidean_distance(A,B):
    return math.sqrt(((A-B)**2).sum())

def initialize_rand_labels(k, X):
    labels = np.random.randint(k, size=X.shape[0])
    return np.c_[X,labels]

class KMC():
    def __init__(self, k=3, max_iterations=100):
        self.k = k
        self.max_iterations = max_iterations
        self.labels = None
    
    def fit(self,X):
        n,p = X.shape
        
        #Add a column to the end of X that stores each entry's current label
        X = initialize_rand_labels(self.k, X)
        
        for _ in range(self.max_iterations):
            new_labels = []
            means = []
            
            for i in range(self.k):
                #Get cluster with label i and compute its mean vector
                cluster = X[X[:,p] == i]
                mean = get_mean_vector(cluster[:,:-1])
                means.append(mean)
            
            for x in X[:,:-1]:
                #distances[j] will contain the distance between x and the center of cluster i
                distances = []
                for j in range(self.k):
                    distances.append(euclidean_distance(x, means[j]))
                    
                #Find cluster whose center is closest to x
                x_label = np.argsort(distances)[0]
                #Assign x to this new cluster for next iteration
                new_labels.append(x_label)
                
            #Update labels
            X[:,-1] = new_labels
            
        self.labels = X[:,-1]
