clear all;


% Time step the 2d vorticity equation on a doubly-periodic
% domain using a pseudospectral method


% Parameters
nu = 3e-4;          % Viscosity
Lx = 2*pi;          % Domain size
nx = 64;           % Resolution
dt = 2e-2;          % Time step
nsteps = 60000;      % Number of time steps


% Construct square physical and wavenumber grid
dx = Lx/nx;
x = 0:dx:Lx-dx;
k = 2*pi/Lx * [ 0:nx/2, -nx/2+1:-1 ];

y=x; ny=nx; Ly=Lx; dy=dx; l=k;    % Square domain

[X, Y] = meshgrid(x, y);
[K, L] = meshgrid(k, l);

Ksq = K.^2 + L.^2;         % Square wavenumber
invKsq = 1./Ksq;           % Inverse square wavenumber
invKsq(1, 1) = 0;          % for vorticity inversion


% Initial condition
t = 0.0;
zeta = rand(nx, nx);
zetah = fft2(zeta);


% Step forward
fig = figure(1);
t1 = tic;
for step = 1:nsteps

    % Quick plot
    if mod(step, 1000) == 0 && step > 1
        disp([ ...
            'step = '      num2str(step,    '%04d'),   ', ', ...
            't = '         num2str(t,       '%06.1f'), ' s, ', ...
            'wall time = ' num2str(toc(t1), '%0.3f')]), t1=tic;
        zeta = real(ifft2(zetah));
        clf, pcolor(X, Y, zeta); shading flat, axis xy, axis square, pause(0.01)
    end


    % Calculate right hand side
    psih = -zetah .* invKsq;
    zetax = real(ifft2(1i*K.*zetah));
    zetay = real(ifft2(1i*L.*zetah));
    u = -real(ifft2(1i*L.*psih));
    v =  real(ifft2(1i*K.*psih));
    rhs = -fft2( u.*zetax + v.*zetay ) - nu*Ksq.*zetah;

    % Forward Euler timestep
    zetah = zetah + dt*rhs;
    t = t + dt;
end
