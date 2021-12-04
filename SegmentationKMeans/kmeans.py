"""
Elif Cansu YILDIZ 11/2021
"""
import numpy as np

def Kmeans(data, k):
    """
    args:
        data: list of data points
        k: number of clusters
    returns:
        centers and list of indices that store the cluster index for each data point
    """
    centers = np.zeros((k, data.shape[1]))  #RGB: (4,3)  Intensity: (4,1)
    index = np.zeros(data.shape[0], dtype=int)

    # initialize centers using some random points from data
    x = np.random.randint(0, data.shape[0], size=k)
    centers = data[x].copy()

    convergence = False
    iterationNo = 0
    new_centers = np.zeros(centers.shape)
    MAX_ITERATION = 100

    while not convergence:

        diff = data.reshape((data.shape[0], 1, -1)) - centers.reshape((1, k, -1))  # RGB: (117200, 1, 3) - (1, 4, 3) Intensity: (117200, 1, 1) - (1, 4, 1)
        distance = np.sum(diff**2, axis=2)      #(117200, 4)
        clusters = np.argmin(distance, axis=1)  #(117200)

        for c in range(k):
            mask = clusters == c
            new_centers[c] = np.mean(data[mask], axis=0)
        if np.min(centers == new_centers) == 1 or iterationNo==MAX_ITERATION:
            convergence = True

        centers = new_centers.copy()

        #print("\ncenters: {} \nnew_centers: {}".format(centers, new_centers))
        iterationNo += 1
    print('Has been completed with {} iterations for {} centers'.format(iterationNo, k))

    return centers, clusters