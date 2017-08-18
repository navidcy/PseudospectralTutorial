#= betaplaneturb.jl

Solves the 2D vorticity equation on a beta plane using a Fourier 
pseudospectral method in a doubly-periodic box. Uses Forward Euler
time-stepping and does not dealias.

=#

using PyPlot
include("./turbtools.jl")


# Parameters
nx = 128                  # Resolution 
Lx = 2.0*pi               # Physical box size
nu = 8e-5                 # Laplacian viscosity
dt = 5e-4                 # Time step
nsteps = 50000            # Number of time steps

kinit = 32 * 2*pi/Lx      # Initial annulus wavenumber
beta = 4*kinit            # Planetary vorticity gradient


# Construct the grid
ny, Ly = nx, Lx
dx, dy, nk, nl, x, y, k, l, K, L, Ksq, invKsq = makegrid(nx, Lx, ny, Ly)
FFTW.set_num_threads(Sys.CPU_CORES)
rfftplan, irfftplan = makefftplans()


# Annular initial condition in wavenumber space
t = 0.0
dk = 4*pi/Lx          # Width of annulus

qh = expannulus(Ksq, kinit, dk)
q = irfft(qh, nx)
q *= 1.0 / maximum(abs.(q))
qh = rfft(q)

u, v, uq, vq, qsh, psih, rhs, uh, vh, uqh, vqh = allocateturbfields(nx, ny, nk, nl)


# Step forward
fig, axs = subplots(ncols=2, nrows=1, figsize=(12, 6))
tic()
for step = 1:nsteps

  if step % 1000 == 0 && step > 1
    axs[1][:cla]; axs[2][:cla]
    axs[1][:imshow](q) 

    axs[2][:imshow](fftshift( 
      abs.( cat(1, qh, flipdim(qh[2:end-1, :], 1)) ), [1, 2]))
    pause(0.01)

    cfl = maximum(u)*dt/dx
    @printf("step: %04d, t: %6.1f, cfl: %.2f, ", step, t, cfl)
    toc(); tic()
  end


  # Calculate right hand side of vorticity equation.
  qsh .= qh  # Necessary because irfft destroys its input.
  A_mul_B!(q, irfftplan, qsh)

  psih .= (-invKsq) .* qh

  uh .=    im .* L .* invKsq .* qh
  vh .= (-im) .* K .* invKsq .* qh

  A_mul_B!(u, irfftplan, uh)
  A_mul_B!(v, irfftplan, vh)

  uq .= u.*q
  vq .= v.*q

  A_mul_B!(uqh, rfftplan, uq)
  A_mul_B!(vqh, rfftplan, vq)

  rhs .= (-im).*K.*uqh .- im.*L.*vqh .- nu.*Ksq.*qh .- beta.*im.*K.*psih

  # Step forward
  qh .+= dt.*rhs
  t += dt
end


# Final plot
q = irfft(qh, nx)



clf()
axs[1][:cla]; axs[2][:cla]

axs[1][:pcolormesh](x, y, q)
axs[2][:pcolormesh](
  fftshift(cat(1, K, flipdim(K[2:end-1, :], 1)), [1, 2]),
  fftshift(cat(1, L, flipdim(L[2:end-1, :], 1)), [1, 2]),
  fftshift(abs.( cat(1, qh, flipdim(qh[2:end-1, :], 1)) ), [1, 2])
)

#colorbar()
axs[1][:set_xlabel](L"x")
axs[1][:set_ylabel](L"y")
axs[1][:set_title]("\$q(x, y, t = $t)\$")

axs[2][:set_xlabel](L"x")
axs[2][:set_title]("\$abs(\\hat{q}(k, l, t = $t)\$")

show()
