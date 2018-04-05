import numpy as np
from scipy.special import j0, j1, jn_zeros

def spheromak(Bx, By, Bz, domain, center=(0,0,0), B0=1, R=1, L=1):
    """domain must be a dedalus domain
    Bx, By, Bz must be Dedalus fields

    """

    # parameters
    xx, yy, zz = domain.grids()

    j1_zero1 = jn_zeros(1,1)[0]
    kr = j1_zero1/R
    kz = np.pi/L
    
    lam = np.sqrt(kr**2 + kz**2)

    # construct cylindrical coordinates centered on center
    r = np.sqrt((xx- center[0])**2 + (yy- center[1])**2)
    theta = np.arctan2(yy,xx)
    z = zz - center[2]

    
    # calculate cylindrical fields
    Br = B0 * kz/kr * j1(kr*r) * np.cos(kz*z)
    Bt = B0 * lam/kr * j1(kr*r) * np.sin(kz*z)

    # convert back to cartesian, place on grid.
    Bx['g'] = Br*np.cos(theta) - Bt*np.sin(theta)
    By['g'] = Br*np.sin(theta) + Bt*np.cos(theta)
    Bz['g'] = B0 * j0(kr*r) * np.sin(kz*z)
