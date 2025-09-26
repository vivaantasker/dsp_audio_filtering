import sounddevice as sd
import soundfile as sf
import numpy as np
from scipy import signal
import matplotlib.pyplot as plt
import os

def record_audio(duration, fs, filename="original_audio.wav"):
    print("Recording for 5 seconds...")
   
    # Record
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    
    sd.wait()  # Wait for the recording to finish
    sf.write(filename, recording, fs)
    print(f"Recording finished and saved as '{filename}'")
    return recording, fs

def play_audio(filename):
    if not os.path.exists(filename):
        print(f"Error: The file '{filename}' was not found.")
        return
    
    # Read the audio file
    data, fs = sf.read(filename, dtype='float32')

    print(f"Playing '{filename}'...")
    sd.play(data, fs)
    sd.wait()  # Wait for the playback to finish
    
    print("Playback finished.")

def add_noise_to_audio(original_audio, noise_source_file="noisewave.wav", filename="noisy_audio.wav"):
    # Check if the noise source file exists
    if not os.path.exists(noise_source_file):
        print(f"Error: Noise source file '{noise_source_file}' not found.")
        print("Skipping noise addition. The 'noisy' file will be a copy of the original.")
        # To avoid downstream errors, save the original audio as the "noisy" one
        fs = 44000 # Assuming a fixed sampling rate
        sf.write(filename, original_audio, fs)
        return original_audio

    # Read the noise audio file
    print(f"Reading noise from '{noise_source_file}'...")
    noise_signal, fs_noise = sf.read(noise_source_file, dtype='float32')

    # If the recorded audio or noise has more than one channel, flatten to mono
    if original_audio.ndim > 1:
        original_audio = original_audio.flatten()
    if noise_signal.ndim > 1:
        noise_signal = noise_signal.flatten()

    len_original = len(original_audio)
    len_noise = len(noise_signal)

    # Adjust the length of the noise signal to match the original audio
    if len_noise < len_original:
        # Repeat the noise signal if it's shorter than the original
        num_repeats = int(np.ceil(len_original / len_noise))
        adjusted_noise = np.tile(noise_signal, num_repeats)[:len_original]
    else:
        # Truncate the noise signal if it's longer than the original
        adjusted_noise = noise_signal[:len_original]

    # Add the adjusted noise to the original audio signal
    noisy_audio = original_audio + adjusted_noise

    # Ensure the combined audio is clipped between -1 and 1 to prevent distortion
    noisy_audio = np.clip(noisy_audio, -1, 1)
    
    fs = 44000  # Assuming a fixed sampling rate consistent with the rest of the script
    sf.write(filename, noisy_audio, fs)
    print(f"Noise from '{noise_source_file}' was added and saved as '{filename}'")
    return noisy_audio

def design_fir_lowpass_filter(fs, fpass, fstop):
    nyq_rate = fs / 2.0
    # The cutoff frequency is the midpoint of the transition band
    cutoff_hz = (fpass + fstop) / 2.0

    # The transition width
    trans_width = fstop - fpass

    # Number of taps
    numtaps = int(4 / (trans_width / fs))
    if numtaps % 2 == 0:
        numtaps += 1 # Ensure an odd number of taps

    # Create the filter coefficients 
    taps = signal.firwin(numtaps, cutoff_hz / nyq_rate)
    print("FIR low-pass filter designed.")
    return taps

def apply_filter_convolution(input_signal, filter_coeffs, filename="filtered_audio.wav"):
    # Flatten the signal to 1D if it's not already
    if input_signal.ndim > 1:
        input_signal = input_signal.flatten()
        
    # Lengths
    N = len(input_signal)
    M = len(filter_coeffs)

    # perform linear convolution
    full_length = N + M - 1
    conv_result = np.zeros(full_length)

    # Loop to perform cnvolution
    for n in range(full_length):
        for k in range(M):
            if 0 <= n - k < N:
                conv_result[n] += input_signal[n - k] * filter_coeffs[k]

    # equate size to orignal
    start = (M - 1) // 2
    end = start + N
    filtered_signal = conv_result[start:end]

    fs = 44000 # Assuming a fixed sampling rate
    sf.write(filename, filtered_signal, fs)
    print(f"Filtering complete and saved as '{filename}'")
    return filtered_signal

def calculate_mse(original_signal, processed_signal):
    # Ensure signals are of the same length
    min_len = min(len(original_signal), len(processed_signal))

    #make signal length mimimum
    original_signal = original_signal[:min_len]
    processed_signal = processed_signal[:min_len]
    error = original_signal - processed_signal
    return np.mean(error**2)

def calculate_psnr(mse, max_possible_value=1.0):
    if mse == 0:
        return float('inf')
    return 20 * np.log10(max_possible_value / np.sqrt(mse))


def plot_all_sound_waves(original_signal, noisy_signal, filtered_signal, fs):
   
    # Ensure all signals are 1D for plotting
    original_signal = original_signal.flatten()
    noisy_signal = noisy_signal.flatten()
    filtered_signal = filtered_signal.flatten()

    # Create a time axis based on the length of the original signal
    duration = len(original_signal) / fs
    time = np.linspace(0., duration, len(original_signal))

    # Create a figure with 3 subplots stacked vertically
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True, sharey=True)
    fig.suptitle('Comparison of Audio Waveforms', fontsize=16)

    # Plot Original Signal
    axes[0].plot(time, original_signal)
    axes[0].set_title("Original Signal (x[n])")
    axes[0].set_ylabel("Amplitude")
    axes[0].grid(True)

    # Plot Noisy Signal
    # Ensure the time vector matches the signal length if they differ
    time_noisy = np.linspace(0., len(noisy_signal) / fs, len(noisy_signal))
    axes[1].plot(time_noisy, noisy_signal, color='r')
    axes[1].set_title("Noisy Signal (a[n])")
    axes[1].set_ylabel("Amplitude")
    axes[1].grid(True)

    # Plot Filtered Signal
    # Ensure the time vector matches the signal length if they differ
    time_filtered = np.linspace(0., len(filtered_signal) / fs, len(filtered_signal))
    axes[2].plot(time_filtered, filtered_signal, color='g')
    axes[2].set_title("Filtered Signal (y[n])")
    axes[2].set_xlabel("Time [s]")
    axes[2].set_ylabel("Amplitude")
    axes[2].grid(True)

    # Adjust layout and display the plot
    plt.tight_layout(rect=[0, 0.03, 1, 0.95]) 
    plt.show()



def main():
    # Define parameters
    fs = 44000       # Sampling frequency in Hz
    duration = 5     # Duration of recording in seconds
    fpass = 4000     # Passband edge frequency in Hz
    fstop = 6000     # Stopband edge frequency in Hz

    # Name Files
    original_filename = "x_n_original.wav"
    noisy_filename = "a_n_noisy.wav"
    filtered_filename = "y_n_filtered.wav"

    # Record and play audio
    xn, fs_rec = record_audio(duration, fs, original_filename)
    play_audio(original_filename)

    # Add noise and play
    an = add_noise_to_audio(xn, filename=noisy_filename)
    play_audio(noisy_filename)

    # Design FIR LPF
    fir_coeffs = design_fir_lowpass_filter(fs, fpass, fstop)

    # Use linear convolution to filter and play the audio
    yn = apply_filter_convolution(an, fir_coeffs, filename=filtered_filename)
    play_audio(filtered_filename)

    # Calculate MSE and PSNR to measure sound quality
    mse_value = calculate_mse(xn.flatten(), yn)
    psnr_value = calculate_psnr(mse_value)

    print("\n--- Audio Quality Assessment ---")
    print(f"Mean Squared Error (MSE): {mse_value:.6f}")
    print(f"Peak Signal-to-Noise Ratio (PSNR): {psnr_value:.2f} dB")

    # Plot all audio signals
    plot_all_sound_waves(xn, an, yn, fs)


if __name__ == "__main__":
    main()