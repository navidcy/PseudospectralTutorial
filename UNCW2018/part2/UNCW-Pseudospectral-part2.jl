using FourierFlows, PyPlot
import FourierFlows.TwoDTurb

# Parameters
 n, ν, dt, nsubs, maxc = 64, 3.2e-4, 0.02, 15, 7
 # n, ν, dt, nsubs, maxc = 128, 8.0e-5, 0.01, 30, 10
 # n, ν, dt, nsubs, maxc = 256, 0.0e-5, 0.005, 60, 20
 # n, ν, dt, nsubs, maxc = 512, 0.0e-5, 0.0025, 120, 25
 # n, ν, dt, nsubs, maxc = 1024, 0.0e-5, 0.001, 240, 28

 L = 2π
 nν = 0

# Time-stepping

tfin = 120
nsteps = Int(tfin/dt)

# Initialize problem
prob = TwoDTurb.InitialValueProblem(;nx = n, Lx = L, ny = n, Ly = L, ν = ν,
                                     nν = nν, dt = dt, stepper = "FilteredRK4")

g = prob.grid

# Initial condition closely following pyqg barotropic example
# that reproduces the results of the paper by McWilliams (1984)
srand(1234)
k0, E0 = 6, 0.5
modk = sqrt.(g.KKrsq)
psik = zeros(g.nk, g.nl)
psik =  (modk.^2 .* (1 + (modk/k0).^4)).^(-0.5)
psik[1, 1] = 0.0
psik[g.KKrsq .> 20^2 ] = 0.0
psih = (randn(g.nkr, g.nl)+im*randn(g.nkr, g.nl)).*psik
psih = psih.*prob.ts.filter
Ein = real(sum(g.KKrsq.*abs2.(psih)/(g.nx*g.ny)^2))
psih = psih*sqrt(E0/Ein)
qi = -irfft(g.KKrsq.*psih, g.nx)
E0 = FourierFlows.parsevalsum(g.KKrsq.*abs2.(psih), g)

TwoDTurb.set_q!(prob, qi)


function plot_output(prob, fig, axs; drawcolorbar=false)
  # Plot the vorticity field and the evolution of energy and enstrophy.
  clf()
  TwoDTurb.updatevars!(prob)
  # sca(axs[1])
  pcolormesh(prob.grid.X, prob.grid.Y, prob.vars.q)
  clim(-maxc, maxc)
  axis("off")
  axis("square")
  # if drawcolorbar==true
  #   colorbar()
  # end
  draw()

  pause(0.02)
end




# Step forward
startwalltime = time()


# fig, axs = subplots(ncols=2, nrows=1, figsize=(12, 4))
fig, axs = subplots(ncols=1, nrows=1, figsize=(7, 6))
plot_output(prob, fig, axs; drawcolorbar=true)

asfasfd

while prob.step < nsteps
  stepforward!(prob, nsubs)

  # Message
  log = @sprintf("step: %04d, t: %d, τ: %.2f min",
    prob.step, prob.t, (time()-startwalltime)/60)

  println(log)

  plot_output(prob, fig, axs; drawcolorbar=false)

end

plot_output(prob, fig, axs; drawcolorbar=true)
