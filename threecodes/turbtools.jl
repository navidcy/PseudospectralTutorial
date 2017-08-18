
function makegrid(nx, Lx, ny, Ly)

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

  return dx, dy, nk, nl, x, y, k, l, K, L, Ksq, invKsq

end

function expannulus(Ksq, k0, dk)
  nk, nl = size(Ksq)
  exp.(im*2*pi*rand(nk, nl) - (sqrt.(Ksq)-k0).^2.0/(2*dk^2) )
end

function allocateturbfields(nx, ny, nk, nl)

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

  u, v, uq, vq, qsh, psih, rhs, uh, vh, uqh, vqh
end

function makefftplans(;effort=FFTW.MEASURE)

  rfftplan  = plan_rfft(Array{Float64,2}(nx, ny); flags=effort)
  irfftplan = plan_irfft(
    Array{Complex{Float64},2}(nk, nl), nx; flags=effort)

  rfftplan, irfftplan
end
