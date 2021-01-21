import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
from tfdwt import *


def signal(frequencies,length=1024,sample_rate=10**-3,overlap=False,scale=True):
    """
    The output signal is obtained 'composing' sinusoids, where the ith sinusoid has
    frequency frequencies[i]. If overlap is True the sinusoids are summed, so each frequence
    is present in every moment. If overlap is false the sinusoids are concatenated.
    -> length is the number of samples
    -> sample_rate is the sampling rate
    -> scale, if True the ith sinusoid is scaled by a factor i
    """
    X = np.array([sample_rate*i for i in range(length)])
    Y = np.zeros_like(X)
    scales = np.ones_like(X,dtype=float)
    for i in range(len(scales)):
        scales[i] = scales[i] / (i+1)
    if overlap:
        for i,f in enumerate(frequencies):
            if not scale:
                Y = Y + np.sin(2*math.pi*f*X)
            else:
                Y = Y + scales[i]*np.sin(2*math.pi*f*X)
    else:
        split_size=int(length/len(frequencies))
        for i,f in enumerate(frequencies):
            Z = np.sin(2*math.pi*f*X)
            Z = Z[i*split_size:(i+1)*split_size]
            Y[i*split_size:(i+1)*split_size] = Z
    return X,Y

if __name__=="__main__":
    dec = MultivariateWaveDec(max_level=-1,wavelet="db4")
    rec = MultivariateWaveRec(max_level=-1,wavelet="db4")
    s1 = tf.convert_to_tensor(signal([5,50])[1])
    s2 = tf.convert_to_tensor(signal([10,50],overlap=True)[1])
    s = tf.stack([s1,s2],axis=-1)
    s = tf.reshape(s,(1,1024,2))
    fig,ax = plt.subplots(4,2)
    for i in range(s.shape[2]):
        plt.sca(ax[0,i])
        plt.plot(s[0,:,i].numpy())
        plt.title("Original ch {}".format(i))
    
    o = dec(s)
    for i in range(o.shape[2]):
        plt.sca(ax[1,i])
        plt.plot(o[0,:,i].numpy())
        plt.title("Coeffs ch {}".format(i))
    for i in range(o.shape[2]):
        plt.sca(ax[2,i])
        plt.title("time freq plot")
        time_freq_plot(o[0,:,i],wavelet='db4',max_level=-1,ax=ax[2,i])
    o = rec(o)
    for i in range(o.shape[2]):
        plt.sca(ax[3,i])
        plt.plot(o[0,:,i].numpy())
        plt.title("Reconstructed ch {}".format(i))
    plt.show()