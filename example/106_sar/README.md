# Question

Assume a 2D problem. The platform shall move with constant speed along
a straight line. The target shall be 3 scatter points. Which steps are
required to simulate SAR data?

# Answer

To simulate Synthetic Aperture Radar (SAR) data for a 2D problem with
a platform moving along a straight line at a constant speed and a
target consisting of three scatter points, follow these steps:

1. Define the simulation parameters: Set up the necessary parameters
   for the simulation, including the radar system's operating
   frequency, wavelength, pulse repetition frequency (PRF), range and
   azimuth resolution, antenna gain, and platform altitude. Also,
   define the platform's speed and trajectory, and specify the
   positions of the three scatter points.

2. Calculate the slant range: For each scatter point, compute the
   slant range, which is the line-of-sight distance from the radar
   antenna to the scatter point at each position along the platform's
   flight path.

3. Compute the round-trip time delay: Calculate the round-trip time
   delay for the radar signal to travel from the antenna to each
   scatter point and back. This can be obtained by dividing the slant
   range by the speed of light and then multiplying by two.

4. Generate the transmitted signal: Create the transmitted radar
   signal, which is typically a frequency-modulated linear chirp
   signal. Define the signal's duration, bandwidth, and chirp rate
   based on the desired range resolution and system parameters.

5. Calculate the Doppler shift: For each scatter point and position
   along the flight path, compute the Doppler shift due to the
   relative motion between the platform and the scatter point. This
   can be done using the Doppler equation: Δf = 2 * (velocity_radial /
   wavelength), where Δf is the Doppler shift, velocity_radial is the
   radial component of the platform's velocity relative to the scatter
   point, and wavelength is the radar signal's wavelength.

6. Generate the received signal: For each scatter point, modulate the
   transmitted signal with the round-trip time delay and Doppler shift
   calculated in steps 3 and 5. Then, sum the modulated signals from
   all three scatter points to generate the total received signal at
   each position along the flight path.

7. Add noise: Optionally, add noise to the received signal to simulate
   the presence of thermal noise, clutter, and other interference in
   the SAR data.

8. Create the raw SAR data matrix: Organize the received signals in a
   matrix format, with each row corresponding to a different position
   along the flight path and each column representing a range cell (a
   specific time delay). This matrix represents the raw SAR data that
   would be acquired by the radar system.

After completing these steps, you will have a simulated SAR data
matrix for a 2D problem with a platform moving along a straight line
at a constant speed and a target consisting of three scatter
points. This simulated data can be used for further processing and
analysis, such as image reconstruction and target detection.

# Show code

```
import numpy as np

# Simulation parameters
c = 3e8  # Speed of light (m/s)
fc = 5.4e9  # Center frequency (Hz)
wavelength = c / fc
platform_speed = 100  # Platform speed (m/s)
altitude = 5000  # Platform altitude (m)
duration = 10  # Simulation duration (s)
sample_rate = 1000  # Samples per second

# Transmitted signal parameters
bw = 50e6  # Bandwidth (Hz)
T_p = 10e-6  # Pulse duration (s)
K = bw / T_p  # Chirp rate (Hz/s)

# Scatter points [x, y] (m)
scatter_points = np.array([[3000, 5000], [5000, 8000], [8000, 15000]])

# Time and platform position arrays
t = np.arange(0, duration, 1 / sample_rate)
platform_positions = platform_speed * t

# Calculate slant range for each scatter point
slant_ranges = np.sqrt(altitude ** 2 + (scatter_points[:, 0] - platform_positions[:, np.newaxis]) ** 2)

# Calculate round-trip time delay
time_delays = 2 * slant_ranges / c

# Generate transmitted signal
t_chirp = np.arange(0, T_p, 1 / sample_rate)
transmitted_signal = np.exp(1j * np.pi * K * t_chirp ** 2)

# Calculate Doppler shift
v_radial = -platform_speed * (scatter_points[:, 0] - platform_positions[:, np.newaxis]) / slant_ranges
doppler_shifts = 2 * v_radial / wavelength

# Generate received signal
received_signal = np.zeros((len(t), len(transmitted_signal)), dtype=complex)
for scatter_idx in range(scatter_points.shape[0]):
    delay_samples = np.round(time_delays[:, scatter_idx] * sample_rate).astype(int)
    for idx, delay in enumerate(delay_samples):
        if 0 <= delay < len(transmitted_signal):
            received_signal[idx] += np.exp(1j * 2 * np.pi * doppler_shifts[idx, scatter_idx] * t_chirp) * \
                                    transmitted_signal[np.newaxis, 0:len(transmitted_signal) - delay]

# Add noise (optional)
noise_level = 1e-5
received_signal += noise_level * (np.random.randn(*received_signal.shape) + 1j * np.random.randn(*received_signal.shape))

# Create raw SAR data matrix
raw_sar_data = received_signal

```
