import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt
from tfdwt import *
import pywt


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
    rec = MultivariateWaveRec(max_level=3)
    dec = MultivariateWaveDec(max_level=3)
    s = signal([5,50])[1]
    plt.plot(s)
    plt.title("Original Signale")
    plt.show()

    s = tf.reshape(tf.convert_to_tensor(s),(1,1024,1))
    c = dec(s)
    """
    c = pywt.wavedec(s,'db4',mode='periodization',level=3)
    c = tf.concat(c,axis=0)
    c = tf.reshape(c,(1,1024,1))
    """

    r = rec(c)
    r = r[0,:,0].numpy()
    plt.plot(r)
    plt.title("Reconstructed Signal")
    plt.show()
