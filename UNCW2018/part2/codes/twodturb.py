import numpy as np
import matplotlib.pyplot as plt
import time

from numpy import pi


# Parameters
nu = 3e-4
Lx = 2.0*pi
nx = 64
dt = 2e-2
nsteps = 60000
nthreads = 4


# Construct grid.
dx = Lx/nx
nk, nl = nx//2+1, nx
x = np.arange(0.0, Lx, dx)

k = 2.0*pi/Lx * np.arange(0.0, nk)
l = 2.0*pi/Lx * np.append(
    np.arange(0.0, nx/2.0), np.arange(-nx/2.0, 0.0))

X, Y = np.meshgrid(x, x)
K, L = np.meshgrid(k, l)
Ksq = K**2.0 + L**2.0

divsafeKsq = Ksq.copy()
divsafeKsq[0, 0] = float('Inf')
invKsq = 1.0/divsafeKsq


# Define fft
fft2 = np.fft.rfft2
ifft2 = np.fft.irfft2


# Initial condition
t = 0.0
zeta = np.random.rand(nx, nx)
zetah = fft2(zeta)


# Step forward
fig = plt.figure(1)
t1 = time.time()
for step in range(nsteps):

    if step % 1000 is 0:
        zeta = ifft2(zetah)
        plt.clf()
        plt.imshow(zeta)
        plt.pause(0.01)

        print("step = {:4d}, t = {:06.1f} s, wall time = {:.3f}".format(
            step, t, time.time()-t1))
        t1=time.time()


    # Calculate right hand side
    psih = -zetah * invKsq

    zetax = ifft2(1j*K*zetah)
    zetay = ifft2(1j*L*zetah)
    u = ifft2(1j*L*psih)
    v = -ifft2(1j*K*psih)

    rhs = fft2(u*zetax + v*zetay) - nu*Ksq*zetah

    zetah += dt*rhs
    t += dt
