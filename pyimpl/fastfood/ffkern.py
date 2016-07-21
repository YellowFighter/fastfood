import numpy as np
from .fastfood_for_kernel import FastfoodForKernel


def ffkern(U, V, para, sgm):
    phi1 = FastfoodForKernel(U.T, para, sgm, False)
    phi2 = FastfoodForKernel(V.T, para, sgm, False)
    k_appro = np.dot(phi1.T, phi2)
    return k_appro
