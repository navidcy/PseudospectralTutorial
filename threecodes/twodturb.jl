using PyPlot

nu = 2e-5
Lx = 2.0*pi
nx = 128
dt = 1e-1
nsteps = 10000


# Construct the grid
ny, Ly = nx, Lx

nk, nl = Int(nx/2+1), ny
dx, dy = Lx/nx, Ly/ny
  x, y = 0.0:dx:Lx-dx, 0.0:dy:Ly-dy 

k = 2.0*pi/Lx * (0:nk)
l = 2.0*pi/Ly * cat(1, 0:nl/2, -nl/2+1:-1)

X, Y = zeros(nx, ny), zeros(nx, ny)
K, L = zeros(nk, nl), zeros(nk, nl)

for i = 1:nx, j = 1:ny
  X[i, j], Y[i, j] = x[i], y[j]
end

for i = 1:nk, j = 1:nl
  K[i, j], L[i, j] = k[i], l[j]
end

Ksq = K.^2.0 + L.^2.0
invKsq = 1.0 ./ Ksq
invKsq[1, 1] = 0.0


# Initial condition
t = 0.0
q = rand(nx, ny)
qh = rfft(q)


# Step forward
fig = figure()
tic()
for step = 1:nsteps

  if (step-1) % 100 == 0
    clf(); imshow(q); pause(0.01)
    @printf("step: %04d, t: %6.1f, ", step, t)
    toc(); tic()
  end

  psih = -invKsq .* qh

  q = irfft(qh, nx)
  u = -irfft(im.*L.*psih, nx)
  v = irfft(im.*K.*psih, nx)

  rhs = (-im).*K.*rfft(u.*q) .- im*L.*rfft(v.*q) .- nu*Ksq.*qh

  qh .+= dt.*rhs
  t += dt
end
