from sklearn.datasets import load_iris
import math
import random


def preprocessing():
    makedendogram()
    initdistancemetrics()   
    pass

def agglomerative():
    preprocessing()
    # Main Agglomerative iteration n-1
    searchMinDist()
    updateDendogram()
    updateDistanceMetrics()
    pass 

def main():
    pass

if __name__ == "__main__":
    main()