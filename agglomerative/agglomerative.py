from sklearn.datasets import load_iris
from sklearn.cluster import AgglomerativeClustering
import math
import random
import copy


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

def updateDistanceMetricsAverage(distanceMetrics, i_result, j_result):
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
                # average linkage
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

def averageReplace(distanceMetrics, i_result, j_result, i, j ):
    if i == i_result:
        if j < j_result:
            return (distanceMetrics[j][i_result] + distanceMetrics[j][j_result]) / 2
        else:
            return (distanceMetrics[j+1][i_result] + distanceMetrics[j+1][j_result]) /2

    else:
        return (distanceMetrics[i][i_result] + distanceMetrics[i][j_result]) / 2
    
    return -1

def updateDistanceMetrics(distanceMetrics, i_result, j_result ):
    
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
                new_distanceMetrics[i][j] = singleCompleteReplace(False,distanceMetrics,i_result,j_result,i,j)
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

def agglomerative(data):
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
        distanceMetrics = updateDistanceMetricsAverage(distanceMetrics,i, j)
    # for row in range(0,len(dendogram)):
    #     print("========================")
    #     print(dendogram[row])
    print("========================")
    print(dendogram[len(dendogram) - 3])    
    print(len(dendogram[150-3][0]))
    print(len(dendogram[150-3][1]))
    print(len(dendogram[150-3][2]))
    pass 

def main():
    iris = load_iris()
    print(iris.data)
    data = iris.data
    agglomerative(data)
    
    clustering = AgglomerativeClustering(n_clusters=3, linkage="average").fit(data)
    print(clustering)

    print(clustering.labels_)
    pass

if __name__ == "__main__":
    main()