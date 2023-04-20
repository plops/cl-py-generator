#!/usr/bin/python
import time
import numpy as np
start_time=time.time()
debug=True
_code_git_version="fe3342f0a6244db71e585d67179e4c782a7c67e7"
_code_repository="https://github.com/plops/cl-py-generator/tree/master/example/106_sar/source/"
_code_generation_time="22:23:37 of Thursday, 2023-04-20 (GMT+1)"
# speed of light (m/s)
c=(3.0e+8)
# center frequency (Hz)
fc=(5.5e+9)
# wavelength (m)
wavelength=((c)/(fc))
# platform_speed (m/s)
platform_speed=100
# platform altitude (m)
altitude=5000
# simulation duration (s)
duration=10
# sample_rate (Hz)
sample_rate=1000
# bandwidth (Hz)
bw=(5.0e+7)
# pulse duration (s)
T_p=(1.00e-5)
# chirp rate (Hz/s)
K=((bw)/(T_p))
# scattering targets on the ground plane (m)
scatter_points=np.array([[3000, 5000], [5000, 8000], [8000, 15000]])
# time array (s)
tt=np.arange(0, duration, ((1)/(sample_rate)))
# position array (m)
platform_positions=((platform_speed)*(tt))
# slant ranges for each scatter point (m)
slant_ranges=np.sqrt(((((altitude)**(2)))+(((((scatter_points[:,0])-(platform_positions[:,np.newaxis])))**(2)))))
# round-trip time delay (s)
time_delays=((2)*(((slant_ranges)/(c))))
# time axis for pulse (s)
t_chirp=np.arange(0, T_p, ((1)/(sample_rate)))
# chirped radar pulse amplitude (amplitude)
transmitted_signal=np.exp(((1j)*(np.pi)*(K)*(((t_chirp)**(2)))))
# radial speed difference between platform and target (m/s)
v_radial=((((-platform_speed)*(((scatter_points[:,0])-(platform_positions[:,np.newaxis])))))/(slant_ranges))
# doppler_shifts (Hz)
doppler_shifts=((2)*(((v_radial)/(wavelength))))
# received_signal (amplitude)
received_signal=np.zeros((len(tt),len(transmitted_signal),), dtype=complex)
for scatter_idx in range(scatter_points.shape[0]):
    delay_samples=np.round(((time_delays[:,scatter_idx])*(sample_rate))).astype(int)
    for idx, delay in enumerate(delay_samples):
        if ( ((((0)<=(delay))) and (((delay)<(len(transmitted_signal))))) ):
            received_signal[idx] += ((np.exp(((1j)*(2)*(np.pi)*(doppler_shifts[idx,scatter_idx])*(t_chirp))))*(transmitted_signal[delay:]))
# noise_level (amplitude)
noise_level=(1.00e-5)
# received_signal (-)
received_signal=((received_signal)+(((noise_level)*(((np.random.randn(*received_signal.shape))+(((1j)*(np.random.randn(*received_signal.shape)))))))))