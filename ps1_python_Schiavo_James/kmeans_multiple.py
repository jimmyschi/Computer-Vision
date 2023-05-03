import numpy as np
from matplotlib import pyplot as plt

from kmeans_single import kmeans_single

def kmeans_multiple(X,K,iters,R):
    min_ids,min_means,min_ssd = kmeans_single(X,K,iters)
    for r in range(1,R):
        ids,means,ssd = kmeans_single(X,K,iters)
        if ssd < min_ssd:
            min_ids = ids
            min_means = means
            min_ssd = ssd
    return min_ids,min_means,min_ssd

        