#= twodturb.jl

Solves the 2D vorticity equation using a Fourier pseudospectral method
in a doubly-periodic box. Time-stepping is Forward Euler. Does not 
dealias.
=#

using PyPlot


# Parameters
nx = 128            # Resolution 
Lx = 2.0*pi         # Physical box size
nu = 8e-5           # Laplacian viscosity
dt = 1e-2           # Time step
nsteps = 5000       # Number of time steps


# Construct the grid
ny, Ly = nx, Lx
dx, dy = Lx/nx, Ly/ny
nk, nl = Int(nx/2+1), ny

x = 0.0:dx:Lx-dx
y = 0.0:dy:Ly-dy 
k = 2.0*pi/Lx*(0:nk)
l = 2.0*pi/Ly*cat(1, 0:nl/2, -nl/2+1:-1)

# 2D wavenumber grids with shapes (nk, nl)
K = [ k[i] for i in 1:nk, j in 1:nl ]
L = [ l[j] for i in 1:nk, j in 1:nl ]

# Need K^2 = k^2 + l^2 and 1/K^2.
Ksq = K.^2.0 + L.^2.0
invKsq = 1.0./Ksq
invKsq[1, 1] = 0.0  # Will eliminate 0th mode during inversion


# Random initial condition
t = 0.0
q = rand(nx, ny)
qh = rfft(q)

u = v = zeros(nx, ny)
psih = rhs = zeros(nk, nl)


# Step forward
fig = figure(); tic()
for step = 1:nsteps

  if step % 100 == 0 && step > 1
    clf(); imshow(q); pause(0.01)

    cfl = maximum(u)*dx/dt
    @printf("step: %04d, t: %6.1f, cfl: %.2f, ", step, t, cfl)
    toc(); tic()
  end


  # Calculate right hand side of vorticity equation.
  # (Note that this algorithm is not as memory efficient as it 
  # could be because it creates many temporary arrays.)
  psih = -invKsq .* qh

  q = irfft(qh, nx)
  u = -irfft(im.*L.*psih, nx)
  v = irfft(im.*K.*psih, nx)

  rhs = (-im).*K.*rfft(u.*q) .- im*L.*rfft(v.*q) .- nu*Ksq.*qh

  # Step forward
  qh .+= dt.*rhs
  t += dt
end


# Final plot
q = irfft(qh, nx)
clf()
pcolormesh(x, y, q)

colorbar()
xlabel(L"x")
ylabel(L"y")
title("\$q(x, y, t = $t)\$")

show()
