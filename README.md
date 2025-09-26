**Audio Noise Filtering with Python (DSP Project)**
This project demonstrates how Digital Signal Processing (DSP) can be used to clean noisy audio using a Finite Impulse Response (FIR) Low-Pass Filter.
It records an audio signal, adds artificial noise, filters the noisy signal using convolution, and then evaluates the quality using MSE and PSNR metrics.

**I. Features**
•	Record audio in real time (5 seconds at 44 kHz).
•	Play back original, noisy, and filtered audio.
•	Add a pre-recorded noise file (noisewave.wav) to the original audio.
•	Design an FIR Low-Pass Filter using scipy.signal.firwin.
•	Apply the filter via linear convolution (implemented manually in Python).
•	Evaluate results using Mean Squared Error (MSE) and Peak Signal-to-Noise Ratio (PSNR).
•	Plot waveforms (original, noisy, filtered) for visual comparison.

**Requirements**
Install the following Python libraries before running the code:
pip install numpy scipy matplotlib sounddevice soundfile

**Usage**
1.	Clone/download this repository.
2.	Place a noise file named noisewave.wav in the same folder (or modify the code to use your file).
3.	Run the program:
4.	python audio_filter.py
5.	The script will:
  •	Record 5 seconds of audio and save it as x_n_original.wav.
  •	Add noise and save it as a_n_noisy.wav.
  •	Filter the noisy signal and save it as y_n_filtered.wav.
  •	Print MSE and PSNR values.
  •	Plot waveforms of all three signals.

**Example Output**
--- Audio Quality Assessment ---
Mean Squared Error (MSE): 0.001136
Peak Signal-to-Noise Ratio (PSNR): 29.45 dB
Waveform plots will show how the noisy signal is smoothed after filtering.

**How It Works**
1.	Record: Capture audio at 44 kHz.
2.	Add Noise: Overlay external noise (or repeat/truncate noise to match signal length).
3.	Filter Design: Use scipy.signal.firwin to create a low-pass filter with:
  o	Passband = 4 kHz
  o	Stopband = 6 kHz
4.	Convolution: Apply the FIR filter manually through linear convolution.
5.	Quality Metrics: Compare filtered signal against original using MSE & PSNR.
6.	Visualization: Plot all signals for analysis.

**Notes**
•	Ensure your microphone and speakers are accessible (required by sounddevice).
•	The program assumes a fixed sampling frequency of 44 kHz.
•	The filter performance depends on the type of noise (hissing noise is harder to remove than sharp tapping).

**License**
This project is open-source and free to use for learning and experimentation.

