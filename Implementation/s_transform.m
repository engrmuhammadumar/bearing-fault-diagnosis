function [st_matrix, f, t] = s_transform(signal, fs)
    % S-transform implementation in MATLAB
    %
    % signal : Input signal (1D array)
    % fs     : Sampling frequency
    %
    % st_matrix : S-transform matrix (Time-Frequency representation)
    % f         : Frequency vector
    % t         : Time vector

    N = length(signal); % Length of the signal
    t = (0:N-1) / fs;   % Time vector
    f = (0:N-1) * (fs / N); % Frequency vector

    % Perform FFT of the signal
    signal_fft = fft(signal);

    % Preallocate S-transform matrix
    st_matrix = zeros(N, N);

    % Loop through frequencies to calculate the S-transform
    for k = 1:N
        % Frequency for this step
        freq = f(k);

        % Gaussian window width based on frequency
        width = 1 / (2 * pi * abs(freq + eps));

        % Create the Gaussian window
        gaussian_window = exp(-(t - t(round(N/2))).^2 / (2 * width^2));

        % Perform the inverse FFT of the windowed Fourier coefficients
        st_matrix(:, k) = ifft(signal_fft .* fftshift(gaussian_window), N);
    end
end

% Test the S-transform function
fs = 1000; % Example: Sampling frequency of 1000 Hz
t = 0:1/fs:1; % Time vector for 1 second
signal = sin(2 * pi * 50 * t) + sin(2 * pi * 120 * t); % Test signal with two frequencies

% Apply S-transform
[st_matrix, f, t] = s_transform(signal, fs);

% Plot the scalogram
figure;
imagesc(t, f, abs(st_matrix));
axis xy;
title('S-transform Scalogram');
xlabel('Time (s)');
ylabel('Frequency (Hz)');
colorbar;
