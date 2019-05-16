# -*- coding: utf-8 -*-
"""
Created on Wed May 15 19:19:13 2019

@author: Benutzer1
"""

import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import math

dataset = np.random.choice(100, 300)

class kmeans_:
    def __init__(self, n_clusters, random_state = 0):
        self.n_clusters = n_clusters
        self.random_state = random_state
        np.random.seed(random_state)
        
    def assign(self, new_centroids, min_index, value, counter):
        
        '''This method assign data points to nearest centroid,
        new centroids contain sum of assign data points and 
        counter contain how many data assign to new_centroids'''
        
        new_centroids[min_index] += value
        counter[min_index] += 1
    
    def cal_centroids(self, new_centroids, counter):
        
        ''' For calculating new centroids values by using new_centroids and counter'''
        
        
        for i in range(self.n_clusters):
            temp = int(round(new_centroids[i] / counter[i]))
            new_centroids[i] = temp
        
    def fit_(self, X):
        
        centroids = np.random.choice(X, self.n_clusters)
        while True:
            y = []
            new_centroids = np.zeros(self.n_clusters, dtype = int) #random assigning centroids from Dataset
            counter = np.zeros(self.n_clusters, dtype = int)
            for i in X:
                min_dist = []
                for j in centroids:
                    temp = abs(i - j) #Simple Mahattan Distance 
                    min_dist.append(temp)
                    
                ''' Taking index of centroid which is nearer to data point'''
                min_index = min_dist.index(min(min_dist)) 
                
                self.assign(new_centroids, min_index, i, counter)
                y.append(min_index)
                
            self.cal_centroids(new_centroids, counter)
            
            if Counter(centroids) != Counter(new_centroids):
                centroids = new_centroids
            else:
                break
        return y
                
    '''This one is for elbow method or within cluster sum of square
    
        def inertia(self, X):
        np.random.seed(seed = self.random_state)
        centroids = np.random.choice(X, self.n_clusters)
        print(centroids)
        dist = np.zeros(self.n_clusters, dtype = int)
        for i in range(self.n_clusters):
            for j in X:
                temp = abs(centroids[i] - j)
                temp = temp ** 2
                dist[i] = dist[i] + temp
                
        squared_dist = 0
        for i in range(self.n_clusters):
            squared_dist = dist[i] 
            
        return squared_dist'''
        
        
k = kmeans_(n_clusters = 2)
y = k.fit_(dataset)

'''print(y)'''

len_cluster = Counter(y)

plt.scatter([dataset[i] for i in range(len(dataset)) if y[i] == 0], [i for i in range(len_cluster[0])])
plt.scatter([dataset[i] for i in range(len(dataset)) if y[i] == 1], [i for i in range(len_cluster[1])])
plt.show()







       
        