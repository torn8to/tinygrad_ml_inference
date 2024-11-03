from numpy import np
cimport numpy as cnp
cimport c


cpdef secnp.ndarray(ndim=2) depthToPointCloud(cnp.ndarr arr):
    cpdef int u v

