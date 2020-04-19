from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import math
import random
import copy
import numpy as np

COMPLETE_LINKAGE = 0
SINGLE_LINKAGE = 1
AVERAGE_LINKAGE = 2
AVERAGE_GROUP_LINKAGE = 3
DATA = 0
INDEX = 1

def euclideanDistance(array1, array2):
    res = 0
    for i in range(len(array1)):
        res += (array2[i] - array1[i]) ** 2
    return math.sqrt(res)

def updateDendogram(dendogram, iteration, i_result, j_result):
    print(iteration)
    for i in range(0,len(dendogram)-iteration+1):
        if i < j_result:
            if i == i_result:
                dendogram[iteration][i_result] = copy.deepcopy(dendogram[iteration-1][i])
                dendogram[iteration][i_result].append(copy.deepcopy(dendogram[iteration-1][j_result]))
            else:
                dendogram[iteration][i]= copy.deepcopy(dendogram[iteration-1][i]) 
        elif i != j_result :
            dendogram[iteration][i-1] = copy.deepcopy(dendogram[iteration-1][i])
    pass

def stripCluster (dendogram, data, clusterResult, mode):
    if not(isinstance(dendogram, list)):
        if mode == DATA:
            clusterResult.append(data[dendogram])
        elif mode == INDEX: 
            clusterResult.append(dendogram)
    else:
        for items in dendogram:
            stripCluster(items, data,  clusterResult, mode)

def averageGroupReplace(distanceMetrics, i_result, j_result, i, j):
    clusterA = stripCluster(dendogram)
    if i == i_result:
        if j < j_result:
            return (distanceMetrics[j][i_result] + distanceMetrics[j][j_result]) / 2
        else:
            return (distanceMetrics[j+1][i_result] + distanceMetrics[j+1][j_result]) /2
    else:
        return (distanceMetrics[i][i_result] + distanceMetrics[i][j_result]) / 2
    return -1


def updateDistanceMetricsAverage(distanceMetrics, i_result, j_result, data, dendogram, iteration):
#Update distance matrix average linkage
    new_distanceMetrics = []
    jumlah_data = len(distanceMetrics)

    # make empty
    for i in range(0,jumlah_data-1):
        new_distanceMetrics.append([])
        for j in range(0, jumlah_data-1):
            new_distanceMetrics[i].append(0)
    
    for i in range(0,jumlah_data-1):
        for j in range(i+1, jumlah_data-1):
            if i == i_result or j == i_result:
                # average group linkage
                clusterA = []
                clusterB = []
                stripCluster(dendogram[iteration][i], data, clusterA, DATA)
                stripCluster(dendogram[iteration][j], data, clusterB, DATA)
                new_distanceMetrics[i][j] = euclideanDistance(np.mean(clusterA, axis= 0), np.mean(clusterB, axis = 0))
            else :
                # shift or copy element
                if j < j_result:
                    new_distanceMetrics[i][j] = distanceMetrics[i][j]
                else:
                    if i < j_result:
                        new_distanceMetrics[i][j] = distanceMetrics[i][j+1]
                    else:
                        new_distanceMetrics[i][j] = distanceMetrics[i+1][j+1] 
            new_distanceMetrics[j][i] = copy.deepcopy(new_distanceMetrics[i][j])
    return new_distanceMetrics

def averageReplace(distanceMetrics, i_result, j_result, i, j ):
    if i == i_result:
        if j < j_result:
            return (distanceMetrics[j][i_result] + distanceMetrics[j][j_result]) / 2
        else:
            return (distanceMetrics[j+1][i_result] + distanceMetrics[j+1][j_result]) /2
    else:
        return (distanceMetrics[i][i_result] + distanceMetrics[i][j_result]) / 2
    return -1

def singleCompleteReplace(single, distanceMetrics, i_result, j_result, i, j ):
    if i == i_result:
        if single:
            if j < j_result:
                if distanceMetrics[j][i_result] <= distanceMetrics[j][j_result] :
                    return copy.deepcopy(distanceMetrics[j][i_result])
                else:

                    return copy.deepcopy(distanceMetrics[j][j_result])
            else:
                if distanceMetrics[j+1][i_result] <= distanceMetrics[j+1][j_result] :
                    return copy.deepcopy(distanceMetrics[j+1][i_result])
                else:
                    return copy.deepcopy(distanceMetrics[j+1][j_result])
        else:
            if j < j_result:
                if distanceMetrics[j][i_result] >= distanceMetrics[j][j_result] :
                    return copy.deepcopy(distanceMetrics[j][i_result])
                else:
                    return copy.deepcopy(distanceMetrics[j][j_result])
            else:
                if distanceMetrics[j+1][i_result] >= distanceMetrics[j+1][j_result] :
                    return copy.deepcopy(distanceMetrics[j+1][i_result])
                else:
                    return copy.deepcopy(distanceMetrics[j+1][j_result])
    else:
        if single:
            if distanceMetrics[i][i_result] <= distanceMetrics[i][j_result] :
                return copy.deepcopy(distanceMetrics[i][i_result])
            else:
                return copy.deepcopy(distanceMetrics[i][j_result])
        else:
            if distanceMetrics[i][i_result] >= distanceMetrics[i][j_result] :
                return copy.deepcopy(distanceMetrics[i][i_result])
            else:
                return copy.deepcopy(distanceMetrics[i][j_result])
    return -1

def updateDistanceMetrics(distanceMetrics, i_result, j_result, mode ):
    
    # single linkage and complete linkage only
    new_distanceMetrics = []
    jumlah_data = len(distanceMetrics)
    
    # make empty
    for i in range(0,jumlah_data-1):
        new_distanceMetrics.append([])
        for j in range(0, jumlah_data-1):
            new_distanceMetrics[i].append(0)
    
    for i in range(0,jumlah_data-1):
        for j in range(i+1, jumlah_data-1):
            if i == i_result or j == i_result:
                # single or complete linkage
                if mode == SINGLE_LINKAGE:
                    new_distanceMetrics[i][j] = singleCompleteReplace(True,distanceMetrics,i_result,j_result,i,j)
                elif mode == COMPLETE_LINKAGE:
                    new_distanceMetrics[i][j] = singleCompleteReplace(False,distanceMetrics,i_result,j_result,i,j)
                elif mode == AVERAGE_LINKAGE:
                    new_distanceMetrics[i][j] = averageReplace(distanceMetrics,i_result,j_result,i,j)
            else :
                # shift or copy element
                if j < j_result:
                    new_distanceMetrics[i][j] = distanceMetrics[i][j]
                else:
                    if i < j_result:
                        new_distanceMetrics[i][j] = distanceMetrics[i][j+1]
                    else:
                        new_distanceMetrics[i][j] = distanceMetrics[i+1][j+1] 
            new_distanceMetrics[j][i] = copy.deepcopy(new_distanceMetrics[i][j])
    return new_distanceMetrics

def preprocessing(dendogram, distanceMetrics, data):
    jumlah_data = len(data)
    
    # make empty data
    for i in range(0,jumlah_data):
        dendogram.append(dict())
        distanceMetrics.append([])
        for j in range(0, jumlah_data):
            distanceMetrics[i].append(0)
    
    # initiate distanceMetrics
    for i in range(0,jumlah_data):
        for j in range(i+1, jumlah_data):
            distanceMetrics[i][j] = euclideanDistance(data[i], data[j])
            distanceMetrics[j][i] = copy.deepcopy(distanceMetrics[i][j])
    
    #initiate dendogram
    for i in range(0, jumlah_data):
        for j in range(0,jumlah_data-i):
            if (i == 0):
                dendogram[i][j] = [j]
            else:
                dendogram[i][j] = []
    pass

def searchMinDist(distanceMetrics):
    minimum = 9999999
    i_result = -1
    j_result = -1
    for i in range(0,len(distanceMetrics)):
        for j in range(i+1,len(distanceMetrics)):
            if distanceMetrics[i][j] < minimum:
                minimum = distanceMetrics[i][j]
                i_result = i
                j_result = j
    return i_result, j_result

def buildResult(dendogram, n_cluster, data):
    clusters = []
    for cluster in dendogram[len(dendogram)-n_cluster]:
        temp_cluster = []
        print(cluster)
        stripCluster(dendogram[len(dendogram)-n_cluster][cluster] , data, temp_cluster, INDEX)
        clusters.append(copy.deepcopy(temp_cluster))
    
    # make empty result
    label_result = []
    for i in range(0,len(data)):
        label_result.append(-1)
    cluster_id = 0
    for cluster in clusters:
        for index in cluster:
            label_result[index] = cluster_id
        cluster_id += 1 
    return label_result

def agglomerative(data, mode, n_cluster):
    dendogram = []
    distanceMetrics = []
    preprocessing(dendogram, distanceMetrics, data)
    
    # Main Agglomerative iteration n-1
    for iteration in range(1,len(data)):
        # print(distanceMetrics)
        # print("=======================")
        i, j = searchMinDist(distanceMetrics)
        if i==-1 or j == -1 :
            print("internal error searchMinDist")
            break
        # if iteration ==2 :
        #     break
        # print(i)
        # print(j)
        updateDendogram(dendogram, iteration, i, j )
        if mode == SINGLE_LINKAGE or mode == AVERAGE_LINKAGE or mode == COMPLETE_LINKAGE:
            distanceMetrics = updateDistanceMetrics(distanceMetrics, i, j, mode)
        elif mode == AVERAGE_GROUP_LINKAGE:
            distanceMetrics = updateDistanceMetricsAverage(distanceMetrics,i, j, data, dendogram, iteration)
    # for row in range(0,len(dendogram)):
    #     print("========================")
    #     print(dendogram[row])
    print("========================")
    print(dendogram[len(dendogram) - 3])
    return buildResult(dendogram, n_cluster, data)
    pass 

def main():
    iris = load_iris()
    data = iris.data
    print(data)
    prediction = agglomerative(data, AVERAGE_GROUP_LINKAGE, 3)
    print(prediction)
    
    clustering = AgglomerativeClustering(n_clusters=3, linkage="average").fit(data)
    print(clustering)

    # visualization
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    img = ax.scatter(iris.data[:, 0], iris.data[:, 1],iris.data[:, 2] , c=prediction, cmap=plt.hot())
    fig.colorbar(img)
    plt.show()
    print(clustering.labels_)

    pass

if __name__ == "__main__":
    main()