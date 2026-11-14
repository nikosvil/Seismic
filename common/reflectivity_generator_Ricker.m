% Title: reflectivity_generator_Ricker.m
% Author: N. Economou
%
% A progressive multi-output 1-D convolutional network for low frequency
% seismic trace reconstruction
% N. D. Vilanakis, N. Economou, A. Vafidis
% Computers and Geosciences
%
% User guide
%   Purpose : Builds the synthetic training set. A reflectivity series is drawn
%             for each trace and convolved with the band wavelets, so that all
%             eleven bands share the same reflectivity.
%             Wavelet family: Ricker.
%   Inputs  : None. The parameters are set at the top of the file:
%             freq_list, that is the 6400 Hz input and the ten targets at
%             5500, 4500, 3300, 2600, 2000, 1500, 1100, 800, 600 and 500 Hz,
%             dt = 1e-5 s, nt = 2001, k_initial_value and g_final_value,
%             which control the spikiness and the number of reflectivity groups.
%   Outputs : dataset_multi_freq_Ricker.mat, holding the reflectivity series rmat,
%             the band traces freqmat_all, freq_list, the time axis t and dt.

% reflectivity_generator_Ricker.m
% Clean, physically-consistent generator of frequency-dependent traces.
% Produces the band traces freqmat_all for 6400, 5500, 4500, 3300, 2600, 2000,
% 1500, 1100, 800, 600 and 500 Hz, together with rmat, freq_list, t and dt.
clear; close all; clc;
k_initial_value = 20;
g_final_value= 3001;
%% Time axis
dt = 1e-5; % sample interval (s)
nt = 2001; % number of samples per trace (the recording length of the seismograph)
t = (0:nt-1)'*dt;
%% Frequencies used
%freq_list = [500 600 700 800 1100 1500 1900 3000 3800 5000 6400];
%freq_list = [500 600 700 800 1050 1400 1900 2600 3500 4700 6400];
freq_list = [500 600 800 1100 1500 2000 2600 3300 4500 5500 6400];
%freq_list = [800 6400];
nFreq = numel(freq_list);
%% Ricker wavelets (same length for all freqs, correctly centered)
% Choose an odd number of samples for the wavelet length.
% 511 samples at dt=1e-5 is 5.11 ms total length – plenty for these freqs.
% nw must be odd and enabled for ricker_fixed, disabled for rickers
nw = 511;
W = zeros(nw, nFreq); % each column: one Ricker wavelet
for i = 1:nFreq
    [W(:,i), tw] = ricker_fixed(freq_list(i), dt, nw);
end
%% Sunisoidal wavelets - same length for all freqs
%clear alphaw sizew
%Real Wavelet_________________________________1
nw=1023;
wavle=78;
alphaval=20;

%clear alphaw sizew
for k=1:nFreq
    sizew(k,1)=freq_list(nFreq)/freq_list(k)*wavle;
end
for k=1:nFreq
    alphaw(k,1)=alphaval*(sizew(k)/sizew(nFreq));
end
sizew=floor(sizew);
%clear Wr W
count2=0;
for r=1:nFreq
    count2=count2+1;
    t1=0:dt:sizew(count2)*dt-dt; % Time vector
    % cos of signle frequency full CW (continuous wavelet)
    cosu1=-cos(2*pi.*freq_list(count2).*t1);cosu1=cosu1(:); % invert for frequency?

    % Time window gauss to reduce side lobs
    alphagauss1 =alphaw(count2);
    gausswin1=gausswin(sizew(count2),alphagauss1);% invert for gauss alpha?
    % Assign the reference wavelet to the synthetic one--which is fitted on the real ref wvelet
    wavtemp=gausswin1.*cosu1;
    wavtemp1=zeros(nw,1);wavtemp1(1:length(wavtemp))=wavtemp;wavtemp1=wavtemp1(:);
    txtt=(['wavelet1synth', num2str(freq_list(r)),'k=wavtemp1;']);
    %eval(txtt)
    Wr(:,r)=wavtemp1;
end

%Real wavelet END__________________________________1

%% Reflectivity ensemble parameters
k_values = k_initial_value:-2:2; % "spikiness" / nonlinear exponent m
g_values = 2:g_final_value; % seeds for random generator
nK = numel(k_values);
nG = numel(g_values);
nTraces = nK * nG;
fprintf('Total traces to generate: %d\n', nTraces);

%% Allocate arrays
rmat = zeros(nt, nTraces); % reflectivity
freqmat_all = zeros(nt, nTraces, nFreq); % all frequencies

%% Generate reflectivity and convolve with wavelets
k_trace = zeros(nTraces,1);
g_trace = zeros(nTraces,1);
trace_idx = 0;
for ik = 1:nK
    k = k_values(ik);
    for ig = 1:nG
        g = g_values(ig);
        trace_idx = trace_idx + 1;
        k_trace(trace_idx) = k;
        g_trace(trace_idx) = g;
        [r, ~] = reflec(t(end), dt, 1, k, g);
        r = flipud(r);
        r(1001:end)=0;
        rmat(:, trace_idx) = r;
        for ifreq = 1:nFreq
            w = W(:, ifreq); % Ricker
            %w = Wr(:, ifreq) ;% Synthetic UHRS wavelet
            %figure;plot(Wr);pause;
            conv_trace = conv(r, w);
            conv_trace = conv_trace(1:length(r));
            freqmat_all(:, trace_idx, ifreq) = conv_trace; %conv(r, w, 'same');
        end
    end
end
%% Also create separate freqmatXXXX matrices (for backwards compatibility)
for ifreq = 1:nFreq
f = freq_list(ifreq);
varname = sprintf('freqmat%d', f);
eval([varname ' = squeeze(freqmat_all(:,:,ifreq));']);
end
%% Quick visual sanity check for one trace
j = 28; % index of the trace to plot
figure;
for ifreq = 1:nFreq
    subplot(nFreq+1,1,ifreq);
    plot(t, freqmat_all(:, j, ifreq));
    ylabel(sprintf('%d Hz', freq_list(ifreq)));
    if ifreq == 1
        title(sprintf('Trace #%d at different frequencies', j));
    end
end
subplot(nFreq+1,1,nFreq+1);
plot(t, rmat(:, j));
ylabel('Refl');
xlabel('Time (s)');
title('Reflectivity');
%% Save dataset multi
save('dataset_multi_freq_Ricker.mat', ...
     'freq_list','dt','t','rmat','freqmat_all', ...
     'g_trace','k_trace', ...
     'freqmat500', 'freqmat600', 'freqmat800', 'freqmat1100',...
     'freqmat1500','freqmat2000','freqmat2600','freqmat3300',...
     'freqmat4500','freqmat5500','freqmat6400', '-v7.3');
fprintf('Saved dataset_multi_freq_Ricker.mat\n');
%% --------- Helper functions -------------------------------------------
function [w, t] = ricker_fixed(f, dt, nw)
% Optimized version based on NE initial script
%RICKER_FIXED Proper Ricker wavelet centered in time.
%
% [w, t] = ricker_fixed(f, dt, nw)
%
% f - dominant frequency (Hz)
% dt - time sampling interval (s)
% nw - number of samples (must be odd)
%
% w - wavelet samples (column vector)
% t - time vector (column), centered around 0
    if mod(nw,2) == 0
        error('nw (wavelet length) must be odd.');
    end
% n0 = (nw-1)/2;
% t = ((0:nw-1) - n0) * dt; % centered around 0
% pf = pi * f;
% w = (1 - 2*(pf*t).^2) .* exp(-(pf*t).^2);
% w = w(:); % ensure column vector


    npt=nw*dt;
    t=(-npt/2):dt:npt/2;
    rick1=(1-t .*t * f^2 *pi^2  ) .*exp(- t.^2 * pi^2 * f^2 ) ;
    rick=rick1(round(nw/2)-round(1/f/dt)+1:nw);
    l=length(rick);
    ricktemp=zeros(nw,1);ricktemp(1:length(rick))=rick;rick=ricktemp;
    w=ricktemp;w=w(:);
end

function [r,t] = reflec(tmax,dt,amp,m,n)
    if(nargin<3)
        amp=.2;
    end
    if(nargin<4)
        m=3;
    end
    if(nargin<5)
        c=clock;
        n=c(6);
    end

    m=round(m);
    t=(0.:dt:tmax)';
    randn('seed',abs(n))

    if(floor(m/2)*2==m)
        tmp=randn(size(t));
        r=(tmp.^m).*sign(tmp);
        else
        r=randn(size(t)).^m;
    end

    r=amp*r/max(abs(r));
end

function [w,tw,nw] = rickers(f,dt) % New version sent by NE at 11/12/2025
    nw=2.2/f/dt;
    nw=2*floor(nw/2)+1;
    nc=floor(nw/2);
    w = zeros(nw,1);
    k=[1:1:nw]';
    alpha = (nc-k+1).*f*dt*pi;
    beta=alpha.^2;
    w = (1.-beta.*2).*exp(-beta);
    if nargout>1;
        tw = -(nc+1-[1:1:nw])*dt;
    end
end


function gw=gausswin(n,hwidth)
    no = n/2;
    x=1:n;
    gw= exp( -((x-no)/hwidth).^2)';
end