import numpy as np
import pyfftw
import time
import matplotlib.pyplot as plt

from numpy import pi


# Parameters
nu = 2e-5
Lx = 2*pi
nx = 256
dt = 1e-1
nsteps = 10000
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
usefftw = True
if usefftw:
    effort = 'FFTW_MEASURE'
    fft2 = (lambda x: pyfftw.interfaces.numpy_fft.rfft2(x,
        threads=nthreads, planner_effort=effort))
    ifft2 = (lambda x: pyfftw.interfaces.numpy_fft.irfft2(x,
        threads=nthreads, planner_effort=effort))
else:
    fft2 = np.fft.rfft2
    ifft2 = np.fft.irfft2


# Initial condition
t = 0.0
q = np.random.rand(nx, nx)
qh = fft2(q)


# Step forward
fig = plt.figure(1)
t1 = time.time()
for step in range(nsteps):

    if step % 100 is 0:
        plt.clf(); plt.imshow(q); plt.axis('square')
        plt.xlabel('$x$'); plt.xlabel('$y$'); plt.pause(0.01)

        print("step = {:4d}, t = {:06.1f} s, wall time = {:.3f}".format(
            step, t, time.time()-t1))
        t1=time.time()


    # Calculate right hand side
    psih = -qh * invKsq

    q = ifft2(qh)
    u = ifft2(1j*L*psih)
    v = -ifft2(1j*K*psih)

    rhs = -1j*K*fft2(u*q) - 1j*L*fft2(v*q) - nu*Ksq*qh

    qh += dt*rhs
    t += dt
