#!/usr/bin/python
import time
import numpy as np
import scipy.fftpack as fft
start_time=time.time()
debug=True
_code_git_version="e85a3ffc8f7fd95775a0927fb894d8f6750fdee4"
_code_repository="https://github.com/plops/cl-py-generator/tree/master/example/107_ofdm/source/"
_code_generation_time="00:24:54 of Sunday, 2023-04-30 (GMT+1)"
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
        random_symbols=self._generate_random_data()
        training_sequence=np.vstack((random_symbols[:,0],random_symbols[:,0],))
        return training_sequence
    def modulate(self):
        data_symbols=self._generate_random_data()
        ifft_data=self._ifft(data_symbols)
        training_sequence=self._create_schmidl_cox_training_sequence()
        ofdm_frame=np.hstack((training_sequence,ifft_data,))
        serialized_data=np.reshape(ofdm_frame, ((self.n_subcarriers)*(((2)+(self.data_size)))))
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
        frame_start=self._schmidl_cox_time_sync(received)