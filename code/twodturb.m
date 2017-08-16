% Time step the 2d vorticity equation on a doubly-periodic
% domain using a pseudospectral method


% Parameters
nu = 2e-5;          % Viscosity
Lx = 2*pi;          % Domain size
nx = 256;           % Resolution
dt = 1e-1;          % Time step
nsteps = 10000;     % Number of time steps


% Construct square physical and wavenumber grid
dx = Lx/nx;
x = 0:dx:Lx-dx;
k = 2*pi/Lx * [ 0:nx/2, -nx/2+1:-1 ];

[X, Y] = meshgrid(x, x);
[K, L] = meshgrid(k, k);

Ksq = K.^2 + L.^2;         % Square wavenumber
invKsq = 1./Ksq;           % Inverse square wavenumber
invKsq(1, 1) = 0;          % for vorticity inversion


% Initial condition
t = 0.0;
q = rand(nx, nx);
qh = fft2(q);


% Step forward
fig = figure(1);
t1 = tic;
for step = 1:nsteps

    % Quick plot
    if mod(step-1, 100) == 0
        disp([ ...
            'step = '      num2str(step,    '%04d'),   ', ', ...
            't = '         num2str(t,       '%06.1f'), ' s, ', ...
            'wall time = ' num2str(toc(t1), '%0.3f')]), t1=tic;

        clf, imagesc(q), axis xy, axis square, pause(0.01)
    end


    % Calculate right hand side
    psih = -qh .* invKsq;

    q =  real(ifft2(qh));
    u =  real(ifft2(1i*L.*psih));
    v = -real(ifft2(1i*K.*psih));

    rhs = -1i*K.*fft2(u.*q) - 1i*L.*fft2(v.*q) - nu*Ksq.*qh;


    % Forward Euler timestep
    qh = qh + dt*rhs;
    t = t + dt;
end
