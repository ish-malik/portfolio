  function [dataout] = Acoustic_demo
%ACOUSTIC_DEMO 
%   A demo of Helmholtz resonance for acoustic class
%
%   r = ACOUSTIC_DEMO collect sounds from microphone and subsequently plots 
%   the spectrogram and spectrum.
%
%   The output r is a audiorecorder object which can be used later for
%   playing the sounds
%
%    r.play/pause/stop;
%   

% Configure
fs = 22050;          % sample frequency fs = 22050 Hz
nbits = 16;         % bits per sample = 16 bits
ncha = 1;           % number of channels
devid = -1;         % device id (by default -1)
flim = [0, 4000];    % limit of frequency axis for results display
fsm = 5;            % smooth psd for every 5 Hz

% Record signal
r = audiorecorder(fs, nbits, ncha, devid);
record(r);
fprintf('Press any key to stop recording ...\n');
pause;
stop(r);

% Collect t-domain signal, plot spectrogram
sig = r.getaudiodata;
taxis = 0:1/fs:(length(sig)-1)/fs;
%[~,f,t,p] = spectrogram(sig, powerval_gt(0.05*fs), ...
%    powerval_gt(0.05*fs)-powerval_lt(0.01*fs), powerval_gt(fs), fs);
[~,f,t,p] = spectrogram(sig, 2^10, 2^10-2^8, ...
    2^14, fs);
in = f>=flim(1) & f<=flim(2);
f = f(in);
p = p(in,:);

% Plot spectrogram
f1 = figure;
set(f1, 'color', 'white', 'Position', [1000, 300, 1024, 768]);
ax(1) = subplot(1,1,1);
imagesc(taxis, f, 10*log10(p)); axis xy; hold on; grid on;
%caxis([-80, -40]);
xlabel('Time (s)');
ylabel('Freq (Hz)');
set(ax(1), 'FontSize', 16);
title('Spectrogram of Recorded Signal','FontSize', 18)
colorbar

% Select time window by mouse
fprintf('Press any key to select time ...\n');
pause;

%tmp = getrect(ax(1));
%tlim = [tmp(1), tmp(1)+tmp(2)];

[xa,ya] = ginput(1);
[xb,yb] = ginput(1);


in=[round(xa*fs):round(xb*fs)];

%in = taxis>=tlim(1) & taxis<=tlim(2);
sig2 = sig(in);

%===============

[~,f2,t2,p2] = spectrogram(sig2, 2^10, 2^10-2^8, ...
    2^14, fs);
in = f2>=flim(1) & f2<=flim(2);
f2 = f2(in);
p2 = p2(in,:);

% Plot spectrogram
fb = figure;
set(fb, 'color', 'white', 'Position', [1000, 300, 1024, 768]);
ax(1) = subplot(2,1,1);
imagesc(t2, f2, 10*log10(p2)); axis xy; hold on; grid on;
caxis([-80, -40]);
xlabel('Time (s)');
ylabel('Freq (Hz)');
set(ax(1), 'FontSize', 16);
title('Spectrogram of Recorded Signal','FontSize', 18)
colorbar

%=================

% Plot spectrum
psd_dB = 20*log10(abs(fftshift(fft(sig2))));
psd_dB = imfilter(psd_dB, fspecial('average', [round(length(sig2)/fs*fsm), 1]), 'symmetric');
faxis2 = linspace(-fs/2, fs/2, length(psd_dB));
ax(2) = subplot(2,1,2);
psd_dB2=psd_dB-max(psd_dB);
plot(faxis2, psd_dB2); hold on; grid on; box on;
xlim([-flim(2), flim(2)]);
xlabel('Freq (Hz)');
ylabel('Normalized PSD (dB)');
set(ax(2), 'FontSize', 16);
% Find frequency of peak spectra for chosen time window 
kk=find(psd_dB2==max(psd_dB2));
dataout=[abs(faxis2(kk).'), psd_dB2(kk)];   
fprintf('%f Hz maximum frequency, and %f dB re 1 unit maximum value',abs(faxis2(kk(1))),psd_dB2(kk(1)))
end

