# Libraries
from sklearn.datasets import load_iris
import math
import random
import pandas as pd
import numpy as np
from sklearn.metrics.cluster import *
import matplotlib.pyplot as plt
import seaborn as sns

MAX_ITER = 1000
THRESHOLD = 0.00001

def distance(data1, data2):
# Euclidean distance between two data
    res = 0
    for i in range(len(data1)):
        res += (data2[i] - data1[i]) ** 2
    return math.sqrt(res)

def assignCluster(centroid, instance):
# Assign data to closest centroid
    d = distance(centroid[0], instance), distance(centroid[1], instance), distance(centroid[2], instance)
    return d.index(min(d))

def means(data, cluster):
# Count mean in the clusters
    df = pd.DataFrame(np.column_stack([data, cluster]))
    avg = df.groupby(4).mean()
    avg = avg.values.tolist()
    return avg

def stopIter(error, epoch):
    totalErr = (error[0] + error[1] + error[2])/3

    return ((totalErr <= THRESHOLD) or (epoch >= MAX_ITER))

def kMeans(data):
    # Initialize centroid
    c1, c2, c3 = random.sample(range(0, len(data) - 1), 3)
    centroid = data[c1], data[c2], data[c3]

    cluster = [-1] * len(data)

    error = (99999, 99999, 99999)
    epoch = 0

    while not stopIter(error, epoch):
        for i in range(len(data)):
            cluster[i] = assignCluster(centroid, data[i])

        c = means(data, cluster)

        # Error (distance between old and new centroid)
        error = distance(centroid[0], c[0]), distance(centroid[1], c[1]), distance(centroid[2], c[2])

        centroid = c
        epoch += 1
    return cluster

def fowlkesMallows(target, prediction):
    return fowlkes_mallows_score(target, prediction)

def silhuoette(data, prediction):
    return silhouette_score(data, prediction, sample_size=150)

def main():
    # Read iris dataset
    iris = load_iris()
    data = iris.data
    pred = kMeans(data)
    fm = fowlkesMallows(iris.target, pred)
    sc = silhuoette(data, pred)
    print("Prediction:", pred)
    print("Folkes-Mallows Score:", fm)
    print("Silhouette Coefficient:", sc)

    sns.set(style="white", color_codes=True)
    dfIris = pd.DataFrame(data= np.c_[data, pred],
                     columns= iris['feature_names'] + ['Clusters'])
    g = sns.pairplot(dfIris, hue='Clusters')

    plt.show()

if __name__ == "__main__":
    main()