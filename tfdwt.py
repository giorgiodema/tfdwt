import tensorflow as tf
import numpy as np
import math
from coefficients import coeffs


def get_wavelets():
    """
    list all the wavelet families available
    """
    return list(coeffs.keys())

def split_coeffs(coefficients,wavelet='db4',max_level=-1):
    """
    receive as input the output of the decomposition with shape (BS,SEQ_LEN) or
    (BS,SEQ_LEN,NFEATURE) and produce a tensorarray of coefficients with the form
    [CAn,CDn,CDn-1,...,CD1]
    """
    shape_len = len(coefficients.shape)
    if shape_len==2:
        coefficients = tf.reshape(coefficients,(coefficients.shape[0],coefficients.shape[1],1))
    if max_level < 0:
        max_level = 0
        inp = coefficients.shape[1]//2
        while inp >= len(coeffs[wavelet]):
            inp = inp//2
            max_level+=1
    a = tf.TensorArray(
        tf.float32,
        infer_shape=False,
        element_shape=tf.TensorShape([coefficients.shape[0],None,coefficients.shape[2]]),
        size=max_level+1)
    for i in range(max_level-1):
        a.write(max_level-i,coefficients[:,coefficients.shape[1]//2:,:])
        coefficients = coefficients[:,0:coefficients.shape[1]//2,:]
    a.write(1,coefficients[:,coefficients.shape[1]//2:,:])
    a.write(0,coefficients[:,0:coefficients.shape[1]//2,:])
    return a


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
    low_pass = np.array(coeffs)
    high_pass = low_pass[::-1].copy()
    high_pass[0::2] = - high_pass[0::2].copy()

    m = np.zeros((signal_len,signal_len))
    shift = 0

    for i in range(0,m.shape[0],2):
        if len(m[i,shift:shift+len(coeffs)]) == len(coeffs):
            m[i,shift:shift+len(coeffs)] = low_pass.copy()
            m[i+1,shift:shift+len(coeffs)] = high_pass.copy()
        else:
            #  last rows wrap around
            # like convolutions with periodic boundary conditions
            low = low_pass[0:len(m[i,shift:shift+len(coeffs)])].copy()
            low_wrap = low_pass[len(m[i,shift:shift+len(coeffs)]):].copy()
            high = high_pass[0:len(m[i,shift:shift+len(coeffs)])].copy()
            high_wrap = high_pass[len(m[i,shift:shift+len(coeffs)]):].copy()
            m[i,shift:shift+len(low)] = low
            m[i+1,shift:shift+len(high)] = high
            m[i,0:len(low_wrap)] = low_wrap
            m[i+1,0:len(high_wrap)] = high_wrap

        shift+=2
    if transpose:
        m = np.transpose(m)
    return m



class DWT(tf.keras.layers.Layer):
    """
    Compute The Discrete Wavelet Transform for univariate signals,
    (for signals with multiple features see MultivariateDWT)
    Constructor Parameters:
        -> wavelet: wavelet function to use (up to now only db4 is supported)
    Call Parameters:
        -> input: the input signal, with shape (BS,SEQ_LEN), if the signal has 
                  multiple features then use MultivariateDWT
    Output:
        -> output: the DWT coefficients with shape (BS,SEQ_LEN), where
                   output[:,0:output.shape[1]//2] contains the approximation
                   coefficients while output[:,output.shape[1]//2:] contains the
                   detail coefficients
    """
    def __init__(self,wavelet='db4'):
        super(DWT, self).__init__(name='dwt_')
        if wavelet not in coeffs.keys():
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
    Compute the Inverse Discrete Wavelet Transform for univariate signals. (For
    signals with multiple features see MultivariateIDWT)
    Constructor Parameters:
        -> wavelet: wavelet function to use (up to now only db4 is supported)
    Call Parameters:
        -> input: the DWT coefficients with shape (BS,SEQ_LEN), where
                   output[:,0:output.shape[1]//2] contains the approximation
                   coefficients while output[:,output.shape[1]//2:] contains the
                   detail coefficients
    Output:
        -> output: the output signal, with shape (BS,SEQ_LEN), if the signal has 
                  multiple features then use MultivariateDWT

    """
    def __init__(self,wavelet='db4'):
        super(IDWT, self).__init__(name='dwt_')
        if wavelet not in coeffs.keys():
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
    """
    Compute the Wavelet Decomposition for univariate signals (for signal with multiple features
    see MultivariateWaveDec)
    Constructor Parameters:
        -> wavelet: wavelet function to use (up to now only db4 is supported)
        -> max_level: the maximum level of decomposition, if max_level=-1
                      then the max_level is the maximum level of decomposition
                      for the specified wavelet
    Call Parameters:
        -> input: the input signal, with shape (BS,SEQ_LEN), if the signal has 
                  multiple features then use MultivariateDWT
    Output:
        -> output: the DWT coefficients with shape (BS,SEQ_LEN), where
                   output[:,0:output.shape[1]//2**(max_level-1)] contains the approximation
                   coefficients while output[:,output.shape[1]//2**(i+1):output.shape[1]//2**(i)] 
                   for i = [1,...] contains the detail coefficients of the level i
    """

    def __init__(self,wavelet='db4',max_level=-1):
        super(WaveDec,self).__init__()
        self.wavelet=wavelet
        self.max_level=max_level
        self.coeffs = coeffs[wavelet]
    def build(self,input_shape):
        if self.max_level < 0:
            self.max_level = 0
            inp = input_shape[1]//2
            while inp >= len(coeffs[self.wavelet]):
                inp = inp//2
                self.max_level+=1

        self.input_var = tf.Variable(tf.zeros(input_shape,dtype=tf.float32),trainable=False)
        self.dwt_layers = []
        for i in range(self.max_level):
            self.dwt_layers.append(DWT(wavelet=self.wavelet))
            self.dwt_layers[i].build((input_shape[0],input_shape[1]//2**i))
    def call(self,input):
        self.input_var.assign(input)
        for i in range(self.max_level):
            self.input_var[:,0:input.shape[1]//2**i].assign(self.dwt_layers[i](self.input_var[:,0:input.shape[1]//2**i]))
        return tf.convert_to_tensor(self.input_var)

    def get_max_decomposition_level(self):
        return self.max_level

class WaveRec(tf.keras.layers.Layer):
    """
    Compute the Wavelet Reconstruction for univariate signals (for signal with multiple features
    see MultivariateWaveDec)
    Constructor Parameters:
        -> wavelet: wavelet function to use (up to now only db4 is supported)
        -> max_level: the maximum level of decomposition, if max_level=-1
                      then max_level is the maximum level
                      of decomposition for the specified wavelet
    Call Parameters:
        -> input: the DWT coefficients with shape (BS,SEQ_LEN), where
                   output[:,0:output.shape[1]//2**(max_level-1)] contains the approximation
                   coefficients while output[:,output.shape[1]//2**(i+1):output.shape[1]//2**(i)] 
                   for i = [1,...] contains the detail coefficients of the level i
    Output:
        -> output: the output signal, with shape (BS,SEQ_LEN), if the signal has 
                  multiple features then use MultivariateDWT

    """
    def __init__(self,wavelet='db4',max_level=-1):
        super(WaveRec,self).__init__()
        self.wavelet=wavelet
        self.max_level=max_level
        self.coeffs = coeffs[wavelet]
    def build(self,input_shape):
        if self.max_level < 0:
            self.max_level = 0
            inp = input_shape[1]//2
            while inp >= len(coeffs[self.wavelet]):
                inp = inp//2
                self.max_level+=1
        self.input_var = tf.Variable(tf.zeros(input_shape,dtype=tf.float32),trainable=False)
        self.idwt_layers = []
        for i in range(self.max_level):
            self.idwt_layers.append(IDWT(wavelet=self.wavelet))
            self.idwt_layers[i].build((input_shape[0],input_shape[1]//2**i))
    def call(self,input):
        self.input_var.assign(input)
        for i in range(self.max_level):
            self.input_var[:,0:input.shape[1]//2**(self.max_level-i-1)].assign(self.idwt_layers[i](self.input_var[:,0:input.shape[1]//2**(self.max_level-i-1)]))
        return tf.convert_to_tensor(self.input_var)

    def get_max_decomposition_level(self):
        return self.max_level

class MultivariateDWT(tf.keras.layers.Layer):
    """
    Multivariate Verion of the DWT, input_shape
    and output shape are (BS,SEQ_LEN,N_FEATURES)
    """
    def __init__(self,wavelet='db4'):
        super(MultivariateDWT,self).__init__()
        self.wavelet=wavelet
    def build(self,input_shape):
        self.dwt = DWT(self.wavelet)
        self.dwt.build((input_shape[0],input_shape[1]))
        self.td = tf.keras.layers.TimeDistributed(self.dwt)
    def call(self,input):
        input = tf.transpose(input,[0,2,1])
        input = self.td(input)
        input = tf.transpose(input,[0,2,1])
        return input

class MultivariateIDWT(tf.keras.layers.Layer):
    """
    Multivariate Verion of the IDWT, input_shape
    and output shape are (BS,SEQ_LEN,N_FEATURES)
    """

    def __init__(self,wavelet='db4'):
        super(MultivariateIDWT,self).__init__()
        self.wavelet=wavelet
    def build(self,input_shape):
        self.idwt = IDWT(self.wavelet)
        self.idwt.build((input_shape[0],input_shape[1]))
        self.td = tf.keras.layers.TimeDistributed(self.idwt)
    def call(self,input):
        input = tf.transpose(input,[0,2,1])
        input = self.td(input)
        input = tf.transpose(input,[0,2,1])
        return input


class MultivariateWaveDec(tf.keras.layers.Layer):
    """
    Multivariate Verion of the WaveDec, input_shape
    and output shape are (BS,SEQ_LEN,N_FEATURES)
    """
    def __init__(self,wavelet='db4',max_level=-1):
        super(MultivariateWaveDec,self).__init__()
        self.wavelet=wavelet
        self.max_level=max_level
    def build(self,input_shape):
        self.wavedec = WaveDec(self.wavelet,self.max_level)
        self.wavedec.build((input_shape[0],input_shape[1]))
        self.td = tf.keras.layers.TimeDistributed(self.wavedec)
    def call(self,input):
        input = tf.transpose(input,[0,2,1])
        input = self.td(input)
        input = tf.transpose(input,[0,2,1])
        return input
    def get_max_decomposition_level(self):
        return self.wavedec.max_level

class MultivariateWaveRec(tf.keras.layers.Layer):
    """
    Multivariate Verion of the WaveRec, input_shape
    and output shape are (BS,SEQ_LEN,N_FEATURES)
    """
    def __init__(self,wavelet='db4',max_level=-1):
        super(MultivariateWaveRec,self).__init__()
        self.wavelet=wavelet
        self.max_level=max_level
    def build(self,input_shape):
        self.waverec = WaveRec(self.wavelet,self.max_level)
        self.waverec.build((input_shape[0],input_shape[1]))
        self.td = tf.keras.layers.TimeDistributed(self.waverec)
    def call(self,input):
        input = tf.transpose(input,[0,2,1])
        input = self.td(input)
        input = tf.transpose(input,[0,2,1])
        return input
    def get_max_decomposition_level(self):
        return self.waverec.max_level



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

def print_summary(input_shape,batch_size):
    inp = tf.keras.layers.Input(input_shape,batch_size=batch_size)
    o = MultivariateWaveDec()(inp)
    o = MultivariateWaveRec()(o)
    m = tf.keras.Model(inputs=inp,outputs=o)
    m.summary()