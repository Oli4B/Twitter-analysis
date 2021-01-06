import numpy as np

def c3e(objects, similarity_matrix, k, n, a):
    n = len(objects)
    #k is dimensions
    #check if a can be a vector over k
    yl = yr = [np.array([1/k] * k)] * n
    while 1:
        for i in range(n):
            g = gamma(a, n, similarity_matrix, j)
            yr[i] = objects[i] + g
    y = []
    for i in range(n):
        y = np.sum([yr[i], yl[i]], axis=0) / 2
    return y

def gamma(a, n, s, j):
    return a * np.sum([s[i, j] for i in range(n)], axis=0)