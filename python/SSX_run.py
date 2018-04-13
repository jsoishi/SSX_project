import dedalus.public as de
from spheromak import spheromak
import matplotlib.pyplot as plt

nx = 100
ny = 100
nz = 100

r = 0.08
length = 2

x = de.Fourier('x', nx, interval=(-r, r))
y = de.Fourier('y', ny, interval=(-r, r))
z = de.Chebyshev('z', nz, interval=(0,length))

domain = de.Domain([x,y,z],grid_dtype='float')

SSX = de.IVP(domain, variables=['lnrho','T', 'vx', 'vy', 'vz', 'Ax', 'Ay', 'Az'])

SSX.parameters['mu'] = mu
SSX.parameters['chi'] kappa/rho0

SSX.substituions['divv'] = "dx(vx) + dy(vy) + dz(vz)"
SSX.substitutions['udotgrad(A)'] = "vx*dx(A) + vy*dy(A) + vz*dz(A)"
SSX.substitutions['Bdotgrad(A)'] = "Bx*dx(A) + By*dy(A) + Bz*dz(A)"
SSX.substitutions['Bx'] = "dy(Az) - dz(Ay)"
SSX.substitutions['By'] = "dz(Ax) - dx(Az)"
SSX.substitutions['Bz'] = "dx(Ay) - dy(Ax)"
SSX.substitutions['jx'] = "dy(bz) - dz(by)"
SSX.substitutions['jy'] = "dz(bx) - dx(bz)"
SSX.substitutions['jz'] = "dx(by) - dy(bx)"
SSX.substituions['J2'] = "jx**2 + jy**2 + jz**2"
SSX.substituions['rho'] = "exp(lnrho)"

# Continuity
SSX.add_equation("dt(lnrho) + divv = - udotgrad(lnrho)")

# Momentum 
SSX.add_equation("Dt(vx) + dx(T) = T*dx(lnrho) - udotgrad(vx) + (jy*Bz - jz*By)/rho")
SSX.add_equation("Dt(vy) + dy(T) = T*dy(lnrho) - udotgrad(vy) + (jz*Bx - jx*Bz)/rho")
SSX.add_equation("Dt(vz) + dz(T) = T*dz(lnrho) - udotgrad(vz) + (jx*By - jy*Bx)/rho")

# MHD equations: A
SSX.add_equation("dt(Ax) - eta*Jx = vy*Bz - vz*By")
SSX.add_equation("dt(Ay) - eta*Jy = vz*Bx - vx*Bz")
SSX.add_equation("dt(Az) - eta*Jz = vx*By - vy*Bx")

solver = SSX.build_solver(de.timesteppers.RK443)

Bx = solver.state['Bx']
By = solver.state['By']
Bz = solver.state['Bz']

L = 0.1*length
R = L

spheromak(Bx, By, Bz, domain, center=(0,0, length/2), R=R, L=L)


plt.imshow(Bx['g'][:,:,10])
plt.colorbar(label='Bx')
plt.savefig('Bx_spheromak.png', dpi=100)

plt.clf()
plt.imshow(By['g'][:,:,10])
plt.colorbar(label='By')
plt.savefig('By_spheromak.png', dpi=100)

plt.clf()
plt.imshow(Bz['g'][:,:,10])
plt.colorbar(label='Bz')
plt.savefig('Bz_spheromak.png', dpi=100)

plt.clf()
plt.imshow((Bx['g'][:,:,10]**2 + By['g'][:,:,10]**2))
plt.colorbar(label=r'$B_x^2 + B_y^2$')
plt.savefig('Br_spheromak.png', dpi=100)

