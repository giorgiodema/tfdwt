import tensorflow as tf
import numpy as np
import math
import matplotlib.pyplot as plt


# Coefficients of the Mother
# Wavelet Functions
coeffs = {
    "db4":[0.4829629131445341,0.8365163037378079,0.2241438680420134, -0.1294095225512604]
}

def interleave(t1,t2):
    """
    Produce a new tensor t which elements are the elements
    of t1 and t2 interleaved along the first axis. t1 and
    t2 must have same rank
    """
    tf.assert_equal(t1.shape,t2.shape)
    even_pos = tf.convert_to_tensor(list(range(0,2*t1.shape[0],2)),dtype=tf.int32)
    odd_pos = tf.convert_to_tensor(list(range(1,2*t1.shape[0],2)),dtype=tf.int32)
    out = tf.dynamic_stitch([even_pos,odd_pos],[t1,t2])
    return out

def build_matrix(coeffs,signal_len,transpose=False):
    """
    build the transformation matrix of the given
    coefficients, if transpose is False build the
    decomposition matrix, otherwise the reconstruction
    matrix
    """
    m = np.zeros((signal_len,signal_len))
    m[-2:,0:2] = np.array([[coeffs[2],coeffs[3]],[coeffs[1],-coeffs[0]]])
    m[-2:,-2:] = np.array([[coeffs[0],coeffs[1]],[coeffs[3],-coeffs[2]]])
    shift = 0

    for i in range(0,m.shape[0]-2,2):
        m[i,shift:shift+len(coeffs)] = np.array(coeffs)
        m[i+1,shift:shift+len(coeffs)] = np.array([coeffs[3],-coeffs[2],coeffs[1],-coeffs[0]])
        shift+=2
    if transpose:
        m = np.transpose(m)
    return m



class DWT(tf.keras.layers.Layer):
    """
    Compute the Discrete Wavelet Transform or the Inverse Discrete Wavelet Transform
    untill the maximum level of decomposition.
    """
    def __init__(self,wavelet='db4'):
        super(DWT, self).__init__(name='dwt_')
        if wavelet!='db4':
            raise NotImplementedError()
        self.coeffs = coeffs[wavelet]

    def build(self,input_shape):
        self.w = tf.Variable(
            initial_value = build_matrix(self.coeffs,input_shape[1],transpose=True),
            dtype=tf.float32,
            trainable=False
        )

    def call(self,input):
        input = (tf.matmul(input,self.w))
        a = input[:,::2]
        d = input[:,1::2]
        input = tf.concat([a,d],axis=1)
        return input

class IDWT(tf.keras.layers.Layer):
    """
    Compute the Discrete Wavelet Transform or the Inverse Discrete Wavelet Transform
    untill the maximum level of decomposition.
    """
    def __init__(self,wavelet='db4'):
        super(IDWT, self).__init__(name='dwt_')
        if wavelet!='db4':
            raise NotImplementedError()
        self.coeffs = coeffs[wavelet]

    def build(self,input_shape):
        self.w = tf.Variable(
            initial_value = build_matrix(self.coeffs,input_shape[1],transpose=False),
            dtype=tf.float32,
            trainable=False
        )

    def call(self,input):
        input = tf.transpose(interleave(tf.transpose(input[:,0:input.shape[1]//2],[1,0]),tf.transpose(input[:,input.shape[1]//2:],[1,0])),[1,0])
        input = (tf.matmul(input,self.w))
        return input


class WaveDec(tf.keras.layers.Layer):
    def __init__(self,wavelet='db4',max_level=-1):
        super(WaveDec,self).__init__()
        self.wavelet=wavelet
        self.max_level=max_level
        self.coeffs = coeffs[wavelet]
    def build(self,input_shape):
        if self.max_level < 0:
            self.max_level = int(math.log2(input_shape[1])) - 2
        self.input_var = tf.Variable(tf.zeros(input_shape,dtype=tf.float32),trainable=False)
        self.dwt_layers = []
        for i in range(self.max_level):
            self.dwt_layers.append(DWT(wavelet=self.wavelet))
            self.dwt_layers[i].build((input_shape[0],input_shape[1]//2**i))
    def call(self,input):
        self.input_var.assign(input)
        for i in range(self.max_level):
            self.input_var[0:input.shape[1]//2**i].assign(self.dwt_layers[i](self.input_var[0:input.shape[1]//2**i]))
        return self.input_var

class WaveRec(tf.keras.layers.Layer):
    def __init__(self,wavelet='db4',max_level=-1):
        super(WaveRec,self).__init__()
        self.wavelet=wavelet
        self.max_level=max_level
        self.coeffs = coeffs[wavelet]
    def build(self,input_shape):
        if self.max_level < 0:
            self.max_level = int(math.log2(input_shape[1])) - 2
        self.input_var = tf.Variable(tf.zeros(input_shape,dtype=tf.float32),trainable=False)
        self.idwt_layers = []
        for i in range(self.max_level):
            self.idwt_layers.append(IDWT(wavelet=self.wavelet))
            self.idwt_layers[i].build((input_shape[0],input_shape[1]//2**i))
    def call(self,input):
        self.input_var.assign(input)
        for i in range(self.max_level):
            self.input_var[0:input.shape[1]//2**(self.max_level-i-1)].assign(self.idwt_layers[i](self.input_var[0:input.shape[1]//2**(self.max_level-i-1)]))
        return self.input_var




#DEBUG
def print_matrix(m):
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            print('{s:{c}^{n}}'.format(s=str(m[i,j]),c=' ',n=4),end=" ")
        print()

def check_identity(coeffs):
    m1 = tf.convert_to_tensor(build_matrix(coeffs,1024,transpose=False),dtype=tf.float32)
    m2 = tf.convert_to_tensor(build_matrix(coeffs,1024,transpose=True),dtype=tf.float32)
    n_it = int(math.log2(1024))-1
    for i in range(n_it):
        m1 = m1[0:1024//2**i,0:1024//2**i]
        m2 = m2[0:1024//2**i,0:1024//2**i]
        prod = tf.matmul(m1,m2)
        diag = tf.linalg.diag_part(prod)
        print("shape={},diag_sum={},prod_sum={}".format(
            prod.shape,
            tf.reduce_sum(diag,axis=0),
            tf.reduce_sum(prod,axis=[0,1])
        ))


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
    dwt = WaveDec(max_level=-1)
    idwt = WaveRec(max_level=-1)
    s = tf.convert_to_tensor(signal([5,50])[1])
    plt.plot(s.numpy())
    plt.show()
    s = tf.reshape(s,(1,1024))
    o = dwt(s)
    o = idwt(o)
    plt.plot(o[0].numpy())
    plt.show()
    print()
    """
    check_identity(coeffs['db4'])
    dec = DWT()
    rec = IDWT()
    s = tf.convert_to_tensor(signal([5,50])[1])
    plt.plot(s.numpy())
    plt.show()
    s = tf.reshape(s,(1,1024))
    o = dec(s)
    o = rec(o)
    plt.plot(o[0].numpy())
    plt.show()
    print()
    """