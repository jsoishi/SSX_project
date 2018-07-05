"""SSX_model_A.py

This is the *simplest* model we will consider for modelling spheromaks evolving in the SSX wind tunnel. 

Major simplificiations fall in two categories

Geometry
--------
We consider a square duct using parity bases (sin/cos) in all directions.

Equations
---------
The equations themselves are those from Schaffner et al (2014), with the following simplifications

* hall term off
* constant eta instead of Spitzer
* no wall recycling term
* no mass diffusion

For this first model, rather than kinematic viscosity nu and thermal
diffusivitiy chi varying with density rho as they should, we are here
holding them *constant*. This dramatically simplifies the form of the
equations in Dedalus.

We use the vector potential, and enforce the Coulomb Gauge, div(A) = 0.

"""
import dedalus.public as de
from spheromak import spheromak_A
import matplotlib.pyplot as plt

nx = 20#0
ny = 20#0
nz = 20#0
r = 0.08
length = 2

kappa = 0.1
mu = 0.1
eta = 0.1
rho0 = 0.01
gamma = 5./3.

x = de.SinCos('x', nx, interval=(-r, r))
y = de.SinCos('y', ny, interval=(-r, r))
z = de.SinCos('z', nz, interval=(0,length))

domain = de.Domain([x,y,z],grid_dtype='float')

SSX = de.IVP(domain, variables=['lnrho','T', 'vx', 'vy', 'vz', 'Ax', 'Ay', 'Az', 'phi'])


SSX.meta['T','lnrho']['x', 'y', 'z']['parity'] = 1
SSX.meta['phi']['x', 'y', 'z']['parity'] = -1

SSX.meta['vx']['y', 'z']['parity'] =  1
SSX.meta['vx']['x']['parity'] = -1
SSX.meta['vy']['x', 'z']['parity'] = 1
SSX.meta['vy']['y']['parity'] = -1
SSX.meta['vz']['x', 'y']['parity'] = 1
SSX.meta['vz']['z']['parity'] = -1

SSX.meta['Ax']['y', 'z']['parity'] =  -1
SSX.meta['Ax']['x']['parity'] = 1
SSX.meta['Ay']['x', 'z']['parity'] = -1
SSX.meta['Ay']['y']['parity'] = 1
SSX.meta['Az']['x', 'y']['parity'] = -1
SSX.meta['Az']['z']['parity'] = 1

SSX.parameters['mu'] = mu
SSX.parameters['chi'] = kappa/rho0
SSX.parameters['nu'] = mu/rho0
SSX.parameters['eta'] = eta
SSX.parameters['gamma'] = gamma

SSX.substitutions['divv'] = "dx(vx) + dy(vy) + dz(vz)"
SSX.substitutions['vdotgrad(A)'] = "vx*dx(A) + vy*dy(A) + vz*dz(A)"
SSX.substitutions['Bdotgrad(A)'] = "Bx*dx(A) + By*dy(A) + Bz*dz(A)"
SSX.substitutions['Lap(A)'] = "dx(dx(A)) + dy(dy(A)) + dz(dz(A))"
SSX.substitutions['Bx'] = "dy(Az) - dz(Ay)"
SSX.substitutions['By'] = "dz(Ax) - dx(Az)"
SSX.substitutions['Bz'] = "dx(Ay) - dy(Ax)"

# Coulomb Gauge implies J = -Laplacian(A)
SSX.substitutions['jx'] = "-Lap(Ax)" 
SSX.substitutions['jy'] = "-Lap(Ay)"
SSX.substitutions['jz'] = "-Lap(Az)" 
SSX.substitutions['J2'] = "jx**2 + jy**2 + jz**2"
SSX.substitutions['rho'] = "exp(lnrho)"

# Continuity
SSX.add_equation("dt(lnrho) + divv = - vdotgrad(lnrho)")

# Momentum 
SSX.add_equation("dt(vx) + dx(T) - nu*Lap(vx) = T*dx(lnrho) - vdotgrad(vx) + (jy*Bz - jz*By)/rho")
SSX.add_equation("dt(vy) + dy(T) - nu*Lap(vy) = T*dy(lnrho) - vdotgrad(vy) + (jz*Bx - jx*Bz)/rho")
SSX.add_equation("dt(vz) + dz(T) - nu*Lap(vz) = T*dz(lnrho) - vdotgrad(vz) + (jx*By - jy*Bx)/rho")

# MHD equations: A
SSX.add_equation("dt(Ax) - eta*jx + dx(phi) = vy*Bz - vz*By")
SSX.add_equation("dt(Ay) - eta*jy + dy(phi) = vz*Bx - vx*Bz")
SSX.add_equation("dt(Az) - eta*jz + dz(phi) = vx*By - vy*Bx")
SSX.add_equation("dx(Ax) + dy(Ay) + dz(Az) = 0")

# Energy
SSX.add_equation("dt(T) - chi*Lap(T) = - (gamma - 1) * T * divv - vx*dx(T) - vdotgrad(T) + (gamma - 1)*eta*J2")

solver = SSX.build_solver(de.timesteppers.RK443)

Ax = solver.state['Ax']
Ay = solver.state['Ay']
Az = solver.state['Az']

L = 0.1*length
R = L

aa_x, aa_y, aa_z = spheromak_A(domain, center=(0,0, length/2), R=R, L=L)

Ax['g'] = aa_x
Ay['g'] = aa_y
Az['g'] = aa_z


