import numpy as np
from numba import jit

@jit(nopython=True)
def construct_GC_2d(cut: float, l: int, ylocs: np.mat) -> np.mat:
    """ construct the GC localization matrix

    Args:
        cut (float): cutoff
        l (int): model size
        ylocs (np.mat): model grids

    Returns:
        np.mat: GC localization matrix
    """

    nobs = len(ylocs)
    V = np.mat(np.zeros((nobs, l)))

    for iobs in range(0, nobs):
        yloc = ylocs[iobs]
        for iCut in range(0, l):
            dist = min(abs(iCut+1 - yloc), abs(iCut+1 - l - yloc), abs(iCut+1 + l - yloc))
            r = dist / (0.5 * cut)

            if dist >= cut:
                V[iobs, iCut] = 0.0
            elif 0.5*cut <= dist < cut:
                V[iobs, iCut] = r**5 / 12.0 - r**4 / 2.0 + r**3 * 5.0 / 8.0 + r**2 * 5.0 / 3.0 - 5.0 * r + 4.0 - 2.0 / (3.0 * r)
            else:
                V[iobs, iCut] = r**5 * (-0.25) + r**4 / 2.0 + r**3 * 5.0/8.0 - r**2 * 5.0/3.0 + 1.0

    return V