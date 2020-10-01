import math
import numpy as np

def linear_motion (centroids2,centroids1):

    velx = centroids1[0] - centroids2[0]
    vely = centroids1[1] - centroids2[1]
    if velx > 0:
        velx = 0
    if vely < 0:
        vely = 0

    # p = p_ini + velocity * t + 1/2 acceleration * t^2
    # We cnder t = 1 (in frames)
    predictedx = centroids1[0] + velx
    predictedy = (centroids1[1] + vely)
    predicted = np.array([int(predictedx),int(predictedy)])

    return predicted



