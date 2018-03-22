#= twodturb.jl

Solves the 2D vorticity equation using a Fourier pseudospectral method
in a doubly-periodic box. Time-stepping is Forward Euler. Does not
dealias.
=#

using PyPlot


# Parameters
nx = 64            # Resolution
Lx = 2.0*pi        # Physical box size
nu = 3e-4          # Kinematic viscosity
dt = 2e-2          # Time step
nsteps = 60000      # Number of time steps


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
zeta = rand(nx, ny)
zetah = rfft(q)

u = v = zeros(nx, ny)
psih = rhs = zeros(nk, nl)


# Step forward
fig = figure(); tic()
for step = 1:nsteps

  if step % 1000 == 0 && step > 1
    zeta = irfft(zetah, nx)
    clf(); pcolormesh(x, y, zeta);
    xlabel(L"x")
    ylabel(L"y")
    title("\$\\zeta(x, y, t = $t)\$")
    pause(0.01)

    cfl = maximum(u)*dx/dt
    @printf("step: %04d, t: %6.1f, cfl: %.2f, ", step, t, cfl)
    toc(); tic()
  end

  # Calculate right hand side of vorticity equation.
  # (Note that this algorithm is not as memory efficient as it
  # could be because it creates many temporary arrays.)
  psih = -invKsq .* zetah

  zetax = irfft(im*K.*zetah, nx)
  zetay = irfft(im*L.*zetah, nx)
  u = -irfft(im.*L.*psih, nx)
  v =  irfft(im.*K.*psih, nx)

  rhs = -rfft( u.*zetax + v.*zetay) .- nu*Ksq.*zetah

  # Step forward
  zetah .+= dt.*rhs
  t += dt
end


# Final plot
zeta = irfft(zetah, nx)
clf()
pcolormesh(x, y, zeta)

colorbar()
xlabel(L"x")
ylabel(L"y")
title("\$\\zeta(x, y, t = $t)\$")

show()
