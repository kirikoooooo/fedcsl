import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import stft
from utils.dataset_utils import LoadDataset_HAR

# Load the HAR dataset
# Assuming LoadDataset_HAR is available and works as in the provided snippet
# We only need X_all for this validation task.
X_all, _, _, _, _, _ = LoadDataset_HAR(1, 0.1)
har_data = X_all

# Select a sample to analyze. Let's pick the first sample.
# The shape of har_data is (num_samples, num_dimensions, sequence_length)
sample_to_analyze = har_data[0]
# Let's select the first dimension of this sample
time_series = sample_to_analyze[0, :]
L = len(time_series)
fs = 1.0 # Assuming a sampling frequency of 1.0 Hz for simplicity, as we are interested in periods in terms of samples.

# STFT parameters
# These are chosen to be similar to the original script for consistency
window_size = L // 2
n_fft = 8 * window_size
hop_size = window_size // 4

# --- STFT Calculation and Period Extraction ---

# Calculate STFT
f, t, Zxx = stft(time_series, fs=fs, nperseg=window_size, noverlap=window_size - hop_size, nfft=n_fft)

# Calculate the power spectrum (spectrogram)
power_spectrum = np.abs(Zxx)**2

# To find the dominant period, we look for the frequency with the most power.
# We can average the power spectrum over time to get a single power-vs-frequency profile.
mean_power_spectrum = np.mean(power_spectrum, axis=1)

# Find the frequency with the maximum power.
# We skip the zero-frequency component (f[0]) as it represents the DC offset.
dominant_frequency_index = np.argmax(mean_power_spectrum[1:]) + 1
dominant_frequency = f[dominant_frequency_index]

# Calculate the corresponding period in number of samples.
# Period = 1 / frequency.
if dominant_frequency > 0:
    extracted_period = 1 / dominant_frequency
else:
    extracted_period = np.inf # Handle the case of zero frequency


# --- Visualization ---

fig, axs = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [1, 2]})

# 1. Plot the Power Spectrum
axs[0].plot(f[1:], mean_power_spectrum[1:]) # Plot without DC component for better scaling
axs[0].axvline(x=dominant_frequency, color='r', linestyle='--', label=f'Dominant Frequency: {dominant_frequency:.4f} Hz')
axs[0].set_title('Average Power Spectrum (from STFT)')
axs[0].set_xlabel('Frequency (Hz)')
axs[0].set_ylabel('Power')
axs[0].legend()
axs[0].grid(True, alpha=0.5)

# 2. Plot the Original Time Series with the extracted period
axs[1].plot(time_series, label='Original Time Series')
# Mark the period on the plot
if np.isfinite(extracted_period):
    for i in range(0, L, int(round(extracted_period))):
        axs[1].axvline(x=i, color='g', linestyle='--', alpha=0.7)
    # Add a final line for the legend
    axs[1].axvline(x=-1, color='g', linestyle='--', alpha=0.7, label=f'Extracted Period: {extracted_period:.2f} samples')

axs[1].set_title('Original Time Series with Extracted Period')
axs[1].set_xlabel('Time (samples)')
axs[1].set_ylabel('Amplitude')
axs[1].legend()
axs[1].grid(True, alpha=0.5)
axs[1].set_xlim(0, L)


plt.tight_layout()
plt.savefig("stft_period_validation.png")
plt.close()

print(f"STFT validation plot saved as 'stft_period_validation.png'")
print(f"Original signal length: {L} samples")
print(f"Dominant frequency found: {dominant_frequency:.4f} Hz")
print(f"Extracted period: {extracted_period:.2f} samples")