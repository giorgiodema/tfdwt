import tensorflow as tf
import numpy as np
import math
from tensorflow.python.ops.gen_array_ops import transpose
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
    dec = WaveDec()
    rec = WaveRec()
    s = tf.convert_to_tensor(signal([5,50])[1])
    plt.plot(s.numpy())
    plt.show()
    s = tf.reshape(s,(1,1024,1))
    o = dec(s)
    plt.plot(o[0,:,0].numpy())
    plt.show()
    o = rec(s)
    o = o[0,:,0].numpy()
    plt.plot(o)
    plt.show()
    print()