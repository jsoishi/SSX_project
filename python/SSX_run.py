import dedalus.public as de
from spheromak import spheromak

nx = 100
ny = 100
nz = 100

r = 0.08
length = 2

x = de.Fourier('x', nx, interval=(-r, r))
y = de.Fourier('y', ny, interval=(-r, r))
z = de.Chebyshev('z', nz, interval=(0,length))

domain = de.Domain([x,y,z],grid_dtype='float')

Bx = domain.new_field()
By = domain.new_field()
Bz = domain.new_field()

L = 0.1*length
R = L

spheromak(Bx, By, Bz, domain, center=(0,0, length/2), R=R, L=L)

