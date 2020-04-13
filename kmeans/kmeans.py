# Libraries
from sklearn.datasets import load_iris
import math
import random

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
    sum = [[0] * 4] * 3
    count = [0] * 3
    
    for i in range(len(cluster)):
        for k in range(3):
            if (cluster[i] == k):
                for x in range(4):
                    sum[k][x] += data[i][x]
                count[k] += 1
    avg = [[sum[0][0] / count[0], sum[0][1] / count[0], sum[0][2] / count[0], sum[0][3] / count[0]],
          [sum[1][0] / count[1], sum[1][1] / count[0], sum[1][2] / count[1], sum[1][3] / count[1]],
          [sum[2][0] / count[2], sum[2][1] / count[0], sum[2][2] / count[2], sum[2][3] / count[2]]]
    return avg

def kMeans(data):
    epoch = 0
    # Initialize centroid
    c1, c2, c3 = random.sample(range(0, len(data)), 3)
    centroid = data[c1], data[c2], data[c3]
    print(centroid)

    cluster = []

    # Start iteration
    
    
    for d in data:
        cluster.append(assignCluster(centroid, d))
    print(cluster)

    c = means(data, cluster)
    print(c)

    # Error (distance between old and new centroid)
    e = distance(centroid[0], c[0]), distance(centroid[1], c[1]), distance(centroid[2], c[2])
    print(e)
    
def main():
    # Read iris dataset
    iris = load_iris()
    data = iris.data
    kMeans(data)

if __name__ == "__main__":
    main()