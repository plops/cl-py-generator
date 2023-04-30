#!/usr/bin/env python3
# python3 -m pip install --user scipy
import matplotlib
import matplotlib.pyplot as plt
plt.ioff()
from matplotlib.pyplot import plot, imshow, tight_layout, xlabel, ylabel, title, subplot, subplot2grid, grid, text, legend, figure, gcf, xlim, ylim
import time
import numpy as np
import scipy.fftpack as fft
start_time=time.time()
debug=True
_code_git_version="adb2b1749ed2dd5bbed501fc2db50ad45cbad2bf"
_code_repository="https://github.com/plops/cl-py-generator/tree/master/example/107_ofdm/source/"
_code_generation_time="16:05:23 of Sunday, 2023-04-30 (GMT+1)"
class OFDMTransmitter():
    def __init__(self, n_subcarriers, data_size):
        self.n_subcarriers=n_subcarriers
        self.data_size=data_size
    def _generate_random_data(self):
        data=np.random.randint(0, 2, (self.n_subcarriers,self.data_size,))
        return ((((2)*(data)))-(1))
    def _ifft(self, data):
        return fft.ifft(data, axis=0)
    def _create_schmidl_cox_training_sequence(self):
        random_symbols=self._generate_random_data()[:,0]
        training_sequence=np.vstack((random_symbols,random_symbols,)).T
        return training_sequence
    def modulate(self):
        global data_symbols, ifft_data, ofdm_frame
        data_symbols=self._generate_random_data()
        fig=figure()
        imshow(data_symbols)
        plt.show()
        ifft_data=self._ifft(data_symbols)
        training_sequence=self._create_schmidl_cox_training_sequence()
        ofdm_frame=np.hstack((training_sequence,ifft_data,))
        serialized_data=ofdm_frame.T.flatten()
        return serialized_data
class OFDMReceiver():
    def __init__(self, n_subcarriers, data_size):
        self.n_subcarriers=n_subcarriers
        self.data_size=data_size
    def _fft(self, data):
        return fft.fft(data, axis=0)
    def _schmidl_cox_time_sync(self, received_signal):
        half_len=self.n_subcarriers
        R=np.zeros(((received_signal.size)-(((2)*(half_len)))))
        for i in range(R.size):
            first_half=received_signal[i:((i)+(half_len))]
            second_half=received_signal[((i)+(half_len)):((i)+(((2)*(half_len))))]
            R[i]=np.abs(((((np.sum(((np.conj(first_half))*(second_half))))**(2)))/(np.sum(((((np.abs(first_half))**(2)))*(((np.abs(second_half))**(2))))))))
        frame_start=np.argmax(R)
        return frame_start
    def _schmidl_cox_frequency_sync(self, received_signal, frame_start):
        half_len=self.n_subcarriers
        first_half=received_signal[frame_start:((frame_start)+(half_len))]
        second_half=received_signal[((frame_start)+(half_len)):((frame_start)+(((2)*(half_len))))]
        angle_sum=np.angle(np.sum(((np.conj(first_half))*(second_half))))
        cfo_est=((-angle_sum)/(((2)*(np.pi)*(half_len))))
        return cfo_est
    def demodulate(self, received_signal):
        frame_start=self._schmidl_cox_time_sync(received_signal)
        cfo_est=self._schmidl_cox_frequency_sync(received_signal, frame_start)
        received_signal=((np.exp(((-1j)*(2)*(np.pi)*(cfo_est)*(np.arange(received_signal.size)))))*(received_signal))
        ofdm_data=np.reshape(received_signal[frame_start:], (self.n_subcarriers,-1,))[:,:((2)+(self.data_size))]
        fft_data=self._fft(ofdm_data)
        return fft_data
n_subcarriers=64
data_size=100
ofdm_tx=OFDMTransmitter(n_subcarriers, data_size)
ofdm_data=ofdm_tx.modulate()
print("{} nil ofdm_data={}".format(((time.time())-(start_time)), ofdm_data))
fig=figure()
plot(np.real(ofdm_data), label="real")
plot(np.imag(ofdm_data), label="imag")
plot(np.abs(ofdm_data), label="abs")
plot(np.angle(ofdm_data), alpha=(0.30    ))
legend()
plt.show()
fig=figure()
imshow(np.abs(ofdm_frame))
plt.show()
received_signal=ofdm_data
ofdm_rx=OFDMReceiver(n_subcarriers, data_size)
demodulated_data=ofdm_rx.demodulate(received_signal)
print("{} nil demodulated_data={}".format(((time.time())-(start_time)), demodulated_data))
fig=figure()
imshow(np.abs(demodulated_data))
plt.show()