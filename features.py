import pandas as pd
import numpy as np
import glob
import matplotlib.pyplot as plt

from scipy.fftpack import fft, ifft
from scipy import signal

# get frequency of data chunk
def create_FFT_feature(x, fs=4000000):
    fft = abs(np.fft.fft(x))
    timestep = len(x)/fs
    freq = np.fft.fftfreq(len(x), d=timestep)
    i = int(len(x)/2)
    freq = freq[1:i]
    fft = fft[1:i]    
    ind = fft.argmax()
    frequency = freq[ind]
    return frequency, freq, fft

