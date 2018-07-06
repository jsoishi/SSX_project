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

import os
import sys
import time
import numpy as np

import dedalus.public as de
from dedalus.extras import flow_tools

from spheromak import spheromak_A

import logging
logger = logging.getLogger(__name__)

nx = 64
ny = 64
nz = 64
r = 0.08
length = 2

# for 3D runs, you can divide the work up over two dimensions (x and y).
# The product of the two elements of mesh *must* equal the number
# of cores used.
# mesh = None
mesh = [8,8]

kappa = 0.1
mu = 0.1
eta = 0.1
rho0 = 0.01
gamma = 5./3.

x = de.SinCos('x', nx, interval=(-r, r))
y = de.SinCos('y', ny, interval=(-r, r))
z = de.SinCos('z', nz, interval=(0,length))

domain = de.Domain([x,y,z],grid_dtype='float', mesh=mesh)

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
SSX.add_equation("dt(Ax) + eta*jx + dx(phi) = vy*Bz - vz*By")
SSX.add_equation("dt(Ay) + eta*jy + dy(phi) = vz*Bx - vx*Bz")
SSX.add_equation("dt(Az) + eta*jz + dz(phi) = vx*By - vy*Bx")
SSX.add_equation("dx(Ax) + dy(Ay) + dz(Az) = 0", condition="(nx != 0) or (ny != 0) or (nz != 0)")
SSX.add_equation("phi = 0", condition="(nx == 0) and (ny == 0) and (nz == 0)")


# Energy
SSX.add_equation("dt(T) - (gamma - 1) * chi*Lap(T) = - (gamma - 1) * T * divv  - vdotgrad(T) + (gamma - 1)*eta*J2")

solver = SSX.build_solver(de.timesteppers.RK443)

# Initial timestep
dt = 1e-3

# Integration parameters
solver.stop_sim_time = 100
solver.stop_wall_time = np.inf
solver.stop_iteration = 100 # np.inf


# Initial conditions
Ax = solver.state['Ax']
Ay = solver.state['Ay']
Az = solver.state['Az']
lnrho = solver.state['lnrho']
T = solver.state['T']

x = domain.grid(0)
y = domain.grid(1)
z = domain.grid(2)

# Initial condition parameters
L = 0.1*length
R = L
lambda_rho = 0.4 # half-width of transition region for initial conditions
rho_min = 0.011
T0 = 0.1

## Not implemented yet
# aa_x, aa_y, aa_z = spheromak_A(domain, center=(0,0, length/2), R=R, L=L)
# Ax['g'] = aa_x
# Ay['g'] = aa_y
# Az['g'] = aa_z

transition_mask = ((1-lambda_rho) <= z) & ((1 + lambda_rho) >= z)
outside_mask = (1 + lambda_rho) < z

rho0 = domain.new_field()
rho0['g'] = 1.
rho0['g'][:,:,transition_mask[0,0,:]] = (1 + rho_min)/2 + (1 - rho_min)/2*np.sin((1-z[transition_mask]) * np.pi/(2*lambda_rho))
rho0['g'][:,:,outside_mask[0,0,:]] = rho_min

lnrho['g'] = np.log(rho0['g'])

T['g'] = T0 * rho0['g']**(gamma - 1)


# analysis output
data_dir = './'+sys.argv[0].split('.py')[0]
wall_dt_checkpoints = 60*55
output_cadence = 100. # FIX THIS

checkpoint = solver.evaluator.add_file_handler(os.path.join(data_dir, 'checkpoints'), max_writes=1, wall_dt=wall_dt_checkpoints, mode='overwrite')
checkpoint.add_system(solver.state, layout='c')

field_writes = solver.evaluator.add_file_handler(os.path.join(data_dir, 'fields'), max_writes=50, sim_dt = 10*output_cadence, mode='overwrite')
field_writes.add_task('vx')
field_writes.add_task('vy')
field_writes.add_task('vz')
field_writes.add_task('Bx')
field_writes.add_task('By')
field_writes.add_task('Bz')
field_writes.add_task('lnrho')
field_writes.add_task('T')

# Flow properties
flow = flow_tools.GlobalFlowProperty(solver, cadence=1)
flow.add_property("sqrt(vx*vx + vy*vy + vz*vz) / nu", name='Re')
flow.add_property("sqrt(vx*vx + vy*vy + vz*vz) / sqrt(T)", name='Ma')


char_time = 1. # this should be set to a characteristic time in the problem (the alfven crossing time of the tube, for example)
CFL_safety = 0.3
CFL = flow_tools.CFL(solver, initial_dt=dt, cadence=1, safety=CFL_safety,
                     max_change=1.5, min_change=0.5, max_dt=output_cadence, threshold=0.05)
CFL.add_velocities(('vx', 'vy', 'vz'))


good_solution = True
# Main loop
try:
    logger.info('Starting loop')
    start_time = time.time()
    while solver.ok and good_solution:
        #dt = CFL.compute_dt()
        solver.step(dt)

        if (solver.iteration-1) % 1 == 0:
            logger_string = 'iter: {:d}, t/tb: {:.2e}, dt/tb: {:.2e}'.format(solver.iteration, solver.sim_time/char_time, dt/char_time)
            Re_avg = flow.grid_average('Re')
            logger_string += ' Max Re = {:.2g}, Avg Re = {:.2g}, Max Ma = {:.1g}'.format(flow.max('Re'), Re_avg, flow.max('Ma'))
            logger.info(logger_string)
            if not np.isfinite(Re_avg):
                good_solution = False
                logger.info("Terminating run.  Trapped on Reynolds = {}".format(Re_avg))
except:
    logger.error('Exception raised, triggering end of main loop.')
    raise
finally:
    end_time = time.time()
    logger.info('Iterations: %i' %solver.iteration)
    logger.info('Sim end time: %f' %solver.sim_time)
    logger.info('Run time: %.2f sec' %(end_time-start_time))
    logger.info('Run time: %f cpu-hr' %((end_time-start_time)/60/60*domain.dist.comm_cart.size))
    logger.info('Iter/sec: {:g}'.format(solver.iteration/(end_time-start_time)))
