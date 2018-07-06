import dedalus.public as de
import numpy as np
from scipy.special import j0, j1, jn_zeros

def spheromak_A(domain, center=(0,0,0), B0=1, R=1, L=1):
    """Solve 

    Laplacian(A) = - J0

    J0 = S(r) l_sph [ -pi J1(a r) cos(pi z) rhat + l_sph*J1(a r)*sin(pi z)

    """

    problem = de.LBVP(domain, variables=['Ax', 'Ay', 'Az'])
    problem.meta['Ax']['y', 'z']['parity'] =  -1
    problem.meta['Ax']['x']['parity'] = 1
    problem.meta['Ay']['x', 'z']['parity'] = -1
    problem.meta['Ay']['y']['parity'] = 1
    problem.meta['Az']['x', 'y']['parity'] = -1
    problem.meta['Az']['z']['parity'] = 1

    J0_x = domain.new_field()
    J0_y = domain.new_field()
    J0_z = domain.new_field()
    problem.parameters['J0_x'] = J0_x
    problem.parameters['J0_y'] = J0_y
    problem.parameters['J0_z'] = J0_z


    problem.add_equation("dx(dx(Ax)) + dy(dy(Ax)) + dz(dz(Ax)) = J0_x", condition="(nx != 0) or (ny != 0) or (nz != 0)")
    problem.add_equation("Ax = 0", condition="(nx == 0) and (ny == 0) and (nz == 0)")

    problem.add_equation("dx(dx(Ay)) + dy(dy(Ay)) + dz(dz(Ay)) = J0_y", condition="(nx != 0) or (ny != 0) or (nz != 0)")
    problem.add_equation("Ay = 0", condition="(nx == 0) and (ny == 0) and (nz == 0)")
    
    problem.add_equation("dx(dx(Az)) + dy(dy(Az)) + dz(dz(Az)) = J0_z", condition="(nx != 0) or (ny != 0) or (nz != 0)")
    problem.add_equation("Az = 0", condition="(nx == 0) and (ny == 0) and (nz == 0)")

    # Build solver
    solver = problem.build_solver()
    solver.solve()

    return solver.state['Ax']['g'], solver.state['Ay']['g'], solver.state['Az']['g']
    

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
