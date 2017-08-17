using PyPlot

nu = 2e-5
Lx = 2.0*pi
nx = 256
dt = 1e-1
nsteps = 10000


# Construct the grid
ny, Ly = nx, Lx
dx = dy = Lx/nx
nk, nl = Int(nx/2+1), ny
x = y = 0.0:dx:Lx-dx
k = 2.0*pi/Lx * (0:nk)
l = 2.0*pi/Ly * cat(1, 0:nl/2, -nl/2+1:-1)

X = broadcast(*, ones(nx, ny), reshape(x, nx, 1))
Y = broadcast(*, ones(nx, ny), reshape(y, 1, ny))
K = broadcast(*, ones(nk, nl), reshape(k, nk, 1))
L = broadcast(*, ones(nk, nl), reshape(l, 1, nl))

Ksq = K.^2.0 + L.^2.0
invKsq = 1.0 ./ Ksq
invKsq[1, 1] = 0.0


# Initial condition
q = rand(nx, nx)
qh = fft2(q)

# Step forward
fig = figure()
for step = 1:nsteps

  if step-1 % 100 == 0
    clf(); imshow(q); axis("square")
    xlabel(L"x"); ylabel(L"y"); pause(0.01)
  end

  psih = - invKsq .* qh

  q = irfft(qh)
  u = irfft(im.*L.*psih)
  v = -irfft(im.*K.*psih)

  rhs = (-im).*K.*fft(u.*q) .- 1i*L.*fft2(v.*q) .- nu*Ksq.*qh

  qh .+= dt.*rhs
  t += dt
end
