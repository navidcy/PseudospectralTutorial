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

FFTW.set_num_threads(Sys.CPU_CORES)


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

# Preallocate
u  = zeros(Float64, nx, ny)
v  = zeros(Float64, nx, ny)
uq = zeros(Float64, nx, ny)
vq = zeros(Float64, nx, ny)

qsh  = zeros(Complex{Float64}, nk, nl)
psih = zeros(Complex{Float64}, nk, nl)
rhs  = zeros(Complex{Float64}, nk, nl)
uh   = zeros(Complex{Float64}, nk, nl)
vh   = zeros(Complex{Float64}, nk, nl)
uqh  = zeros(Complex{Float64}, nk, nl)
vqh  = zeros(Complex{Float64}, nk, nl)

# FFT plans
effort = FFTW.MEASURE
rfftplan  = plan_rfft(Array{Float64,2}(nx, ny); flags=effort)
irfftplan = plan_irfft(
  Array{Complex{Float64},2}(nk, nl), nx; flags=effort)


# Step forward
fig = figure(); tic()
for step = 1:nsteps

  if step % 1000 == 0 && step > 1
    clf(); imshow(q); pause(0.01)

    cfl = maximum(u)*dx/dt
    @printf("step: %04d, t: %6.1f, cfl: %.2f, ", step, t, cfl)
    toc(); tic()
  end


  # Calculate right hand side of vorticity equation.
  qsh .= qh  # Necessary because irfft destroys its input.
  A_mul_B!(q, irfftplan, qsh)

  uh .=    im .* L .* invKsq .* qh
  vh .= (-im) .* K .* invKsq .* qh

  A_mul_B!(u, irfftplan, uh)
  A_mul_B!(v, irfftplan, vh)

  uq .= u.*q
  vq .= v.*q

  A_mul_B!(uqh, rfftplan, uq)
  A_mul_B!(vqh, rfftplan, vq)

  rhs .= (-im).*K.*uqh .- im.*L.*vqh .- nu.*Ksq.*qh

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
