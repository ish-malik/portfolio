%% Final Optics Project: Space vs Ground Telescope Imaging
% The objective of this project is to use real Hubble galaxy data to
% simulate and compare diffraction-limited (space-based) and seeing-limited
% (ground-based) telescope imaging, and to quantify how atmospheric seeing
% degrades image resolution and fine structure.
%%

% steps are oulined above
clear

%% Step 1: load Hubble galaxy image (M51, F814W)

file = 'h_m51_i_s05_drz_sci.fits';
I = fitsread(file);
I = double(I);

% we are working with intensity and not color - colormap gray

% Figure 1: Original HST image
figure;
set(gcf,'Position',[1 1 4.0 5.0]);  % (original image is in portrait layout)
set(gcf,'PaperPositionMode','auto');
imagesc(log10(I+1));
axis image off;
colormap gray;
title('Figure 1: Original Hubble Image (log display)');
drawnow; snapnow;

%% Step 2: Crop ROI (define field of view)
% define the region of interest (ROI)
% store the cropped image for subseqsuent processing
 
[nr,nc] = size(I);

% Set ROI size N = 2048 (2^11) to optimize FFT-based convolution while
% retaining spiral-arm detail for the imaging comparison.
N = 2048;  

% center coordinates
r0 = floor(nr/2) - floor(N/2);
c0 = floor(nc/2) - floor(N/2);

r0 = r0 - 1200;   % move crop UP
c0 = c0 - 150;    % move crop LEFT

% coordinates for the region of interest
Icrop = I(r0:r0+N-1, c0:c0+N-1);
Icrop_raw = Icrop;

fprintf('ROI parameters: r0 = %d, c0 = %d, N = %d\n', r0, c0, N);

%% Step 3: Normalize data (take care of NaNs/Inf)
% we MUST normalize the data to ensure that differences in the simulated images
% arise solely from the imaging PSF, not from arbitrary brightness scaling
% in the input data. 

% for NaN or Inf values
Icrop(~isfinite(Icrop)) = 0;
Icrop = Icrop / max(Icrop(:));

% Figure 2: (a) Cropped ROI, (b) Normalized ROI
figure;
set(gcf,'Units','inches','Position',[1 1 7.5 3.8]);   % shorter than before to take care of white spaces
set(gcf,'PaperPositionMode','auto');
tiledlayout(1,2,'TileSpacing','compact','Padding','compact');

nexttile;
imagesc(log10(Icrop_raw+1));
axis image off; colormap gray;
title('(a) Cropped ROI (log)');

nexttile;
imagesc(log10(Icrop + 1e-4));
axis image off; colormap gray;
title('(b) Normalized ROI (log)');
drawnow; snapnow;

%% Step 4: Build PSFs (space diffraction-limited vs ground seeing-limited)

% we model the observed image as: I_obs = I_true * PSF for both cases, where * denotes 2D convolution.

% initial conditions
% lambda for the band (F814W ~ 0.814 microns)
lambda = 0.814e-6;   % [m]

% aperture diameter: same for both cases
D = 2.4;             % [m] 2.4 m is the HST's primary mirror diameter.

% pixel scale for the simulation 
% This sets the angular size of each pixel in our cropped image.
pixscale = 0.05;     % [arcsec/pixel]

% coordinate grid in arcsec centered at 0 ---
% We use angular coordinates so the Airy PFSs (space) and seeing (ground) PSFs have a physical width.
[xpix, ypix] = meshgrid( (-N/2):(N/2-1), (-N/2):(N/2-1) );
x_arcsec = xpix * pixscale;
y_arcsec = ypix * pixscale;

% Convert arcseconds to radians for diffraction formulas
arcsec2rad = (pi/180) / 3600;
r_rad = sqrt( (x_arcsec*arcsec2rad).^2 + (y_arcsec*arcsec2rad).^2 );

% --- (A) Space PSF: Airy pattern for a circular aperture ---
% The Airy pattern is the Fraunhofer diffraction pattern of a circular aperture.
k = pi * D / lambda;                         % so that arg = k * theta
arg = k * r_rad;                             % argument of the Bessel function that appears in diffraction from a circular aperture.

PSF_space = (2*besselj(1,arg)./(arg)).^2;    
PSF_space(arg==0) = 1;                       % define center value by limit
PSF_space = PSF_space / sum(PSF_space(:));   % normalize to unit energy

% --- (B) Ground PSF: seeing-limited (we use gaussian optics) ---
% trying different FWHM in arcseconds (try 0.6, 1.0, 2.0)
seeing_FWHM = 1.2;                           % [arcsec]
sigma = seeing_FWHM / (2*sqrt(2*log(2)));    % convert FWHM -> sigma

r_arcsec = sqrt(x_arcsec.^2 + y_arcsec.^2);
PSF_ground = exp(-(r_arcsec.^2)/(2*sigma^2));
PSF_ground = PSF_ground / sum(PSF_ground(:));  

% Figure 3: (a) Space PSF, (b) Ground PSF
figure;
set(gcf,'Units','inches','Position',[1 1 7.5 3.8]);   % shorter than before to take care of white spaces
set(gcf,'PaperPositionMode','auto');
tiledlayout(1,2,'TileSpacing','compact','Padding','compact');

nexttile;
imagesc(log10(PSF_space + 1e-12));
axis image off; colormap gray; 
title('(a) Space PSF (Airy)');

nexttile;
imagesc(log10(PSF_ground + 1e-12));
axis image off; colormap gray; 
title(sprintf('(b) Ground PSF (seeing), FWHM = %.1f"', seeing_FWHM));
drawnow; snapnow;

%% Step 5: Simulate imaging by convolution
% use FFT-based convolution (consistent with your N=2^11=2048 choice).
% shift PSFs so that their center is at (1,1) for FFT convolution

PSF_space_shifted  = fftshift(PSF_space);
PSF_ground_shifted = fftshift(PSF_ground);

I_space  = real(ifft2( fft2(Icrop) .* fft2(PSF_space_shifted) ));
I_ground = real(ifft2( fft2(Icrop) .* fft2(PSF_ground_shifted) ));

% shift small numerical negatives to zero (can happen from roundoff)
I_space(I_space<0)   = 0;
I_ground(I_ground<0) = 0;

% Figure 4: (a) Simulated space image, (b) Simulated ground image
figure;
set(gcf,'Units','inches','Position',[1 1 7.5 3.8]);   % shorter than before to take care of white spaces
set(gcf,'PaperPositionMode','auto');
tiledlayout(1,2,'TileSpacing','compact','Padding','compact');

nexttile;
imagesc(log10(I_space + 1e-4));
axis image off; colormap gray;
title('(a) Simulated space image (log)');

nexttile;
imagesc(log10(I_ground + 1e-4));
axis image off; colormap gray;
title('(b) Simulated ground image (log)');
drawnow; snapnow;

%% Step 6: quantify resolution loss and plot radially averaged power spectrum (quantitative comparison)

% fourier transforms of images
F_space  = fftshift(fft2(I_space));
F_ground = fftshift(fft2(I_ground));

% power spectra
P_space  = abs(F_space).^2;
P_ground = abs(F_ground).^2;

% radial spatial-frequency coordinate
[kx, ky] = meshgrid( (-N/2):(N/2-1), (-N/2):(N/2-1) );
k_radius = sqrt(kx.^2 + ky.^2);

% radial averaging collapses the 2D power spectrum into a 1D function P(k),
% providing a direct measure of how much power survives at each spatial scale.

kmax = floor(max(k_radius(:)));

Pspace = zeros(kmax,1);
Pground = zeros(kmax,1);

for k = 1:kmax
    mask = (k_radius >= (k-1)) & (k_radius < k);
    Pspace(k) = mean(P_space(mask));
    Pground(k) = mean(P_ground(mask));
end

% normalize spatial frequency
kvalues = (1:kmax)';
k_norm = kvalues / kmax;

% Figure 5: Radially averaged power spectra
figure;
loglog(k_norm, Pspace, 'k-', k_norm, Pground, 'r-', 'LineWidth', 1.5);
grid on;
xlabel('Normalized spatial frequency k / k_{max}');
ylabel('Radially averaged power P(k)');
legend('Space (diffraction-limited)','Ground (seeing-limited)','Location','southwest');
title('Figure 5: Radially averaged power spectra');
drawnow; snapnow;

