import tensorflow as tf
import numpy as np
import math
from tfdwt.coefficients import coeffs
import matplotlib.pyplot as plt


def get_wavelets():
    """
    list all the wavelet families available
    """
    return list(coeffs.keys())

def filter_len(wavelet):
    """
    return the length of the filter of the given wavelet
    """
    return len(coeffs[wavelet])

def dwt_max_level(data_len,wavelet):
    """
    return the maximum decomposition level for the specified
    lenght of the input signal (data_len) and the specified
    wavelet function (wavelet).
    The maximum decomposition level is the maximum level where at least 
    one coefficient in the output is uncorrupted by edge effects caused by 
    signal extension. Put another way, decomposition stops when the 
    signal becomes shorter than the FIR filter length for a given wavelet
    """
    return int(math.log2(data_len/(filter_len(wavelet)-1)))

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
        a = a.write(max_level-i,coefficients[:,coefficients.shape[1]//2:,:])
        coefficients = coefficients[:,0:coefficients.shape[1]//2,:]
    a = a.write(1,coefficients[:,coefficients.shape[1]//2:,:])
    a = a.write(0,coefficients[:,0:coefficients.shape[1]//2,:])
    return a

def mask_coeffs(coefficients,mask,wavelet='db4',max_level=-1):
    """
    mask the coefficient setting to zeros the coefficients of the selected bands starting from
    the lower bands. mask is a boolean tensor which length is equal to the number of frequency bands
    (max_level+1)
    """
    coeffs_len=coefficients.shape[1]
    shape_len = len(coefficients.shape)
    if shape_len==2:
        coefficients = tf.reshape(coefficients,(coefficients.shape[0],coefficients.shape[1],1))
    if max_level<0:
        max_level=dwt_max_level(coeffs_len,wavelet)
    assert(len(mask)==max_level+1)
    new_coeffs = tf.Variable(initial_value=coefficients)
    prev = 0
    for i in range(max_level+1):
        if mask[i]:
            new_coeffs[:,prev:coeffs_len//2**(max_level-i),:].assign(tf.zeros((coefficients.shape[0],coeffs_len//2**(max_level-i)-prev,coefficients.shape[2])))
        prev = coeffs_len//2**(max_level-i)
    new_coeffs = tf.convert_to_tensor(new_coeffs)
    if shape_len==2:
        new_coeffs = tf.reshape(new_coeffs,(coefficients.shape[0],coefficients.shape[1]))
    return new_coeffs


def time_freq_plot(coefficients,wavelet='db4',max_level=-1,drop_low=0,drop_high=0,ax=None):
    """
    plot the time-frequency decomposition for the coefficients received as input. The input should 
    have shape = (SEQ_LEN), so it is not batched and it has a single channel.
    -> wavelet: the wavelet function used for decomposition
    -> maxlevel: the level of decomposition (-1 for the maximum level)
    -> drop_low: how many low frequency bands should be not included in the plot (starting from the lowest)
    -> drop_high: how many high frequency bands should be not included in the plot (starting from the highest)
    """
    if len(coefficients.shape)!=1:
        raise ValueError("coefficients must have a single dimension (sequence length)")
    coefficients = tf.reshape(coefficients,(1,coefficients.shape[0]))
    coefficients = split_coeffs(coefficients,wavelet,max_level)
    y_labels = []
    coeffs = []
    for i in range(coefficients.size()):
        coeffs.append(coefficients.read(i)[0].numpy())
    for k in range(len(coeffs)-1):
        y_labels.append("f/{}".format(2**(k+1)))
    y_labels.reverse()
    y_labels = ['0'] + y_labels + ['f']
    for k in range(len(coeffs)):
        coeffs[k] = np.repeat(coeffs[k],len(coeffs[-1])//len(coeffs[k]))
    coeffs = coeffs[drop_low:len(coeffs)-drop_high]
    y_ticks = range(len(coeffs)+1)
    y_labels = y_labels[drop_low:len(y_labels)-drop_high]

    coeffs = np.stack(coeffs)
    coeffs = np.abs(coeffs)
    if ax:
        plt.sca(ax)
    plt.imshow(coeffs,aspect='auto',interpolation='nearest',origin='lower',extent=[0,len(coeffs[0]),0,len(coeffs)])
    plt.yticks(y_ticks,y_labels)
    plt.colorbar()
    if not ax:
        plt.show()


def interleave(t1,t2):
    """
    Produce a new tensor t which elements are the elements
    of t1 and t2 interleaved along the first axis. t1 and
    t2 must have same rank
    """
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
    low_pass = np.array(coeffs[::-1])
    high_pass = low_pass[::-1].copy()
    high_pass[1::2] = - high_pass[1::2].copy()

    m = np.zeros((signal_len,signal_len),dtype=np.float32)
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



class SplitCoefficients(tf.keras.layers.Layer):
    
    def __init__(self,wavelet='db4',max_level=-1):
        super().__init__()
        self.wavelet=wavelet
        self.max_level=max_level

    def build(self,input_shape):
        if self.max_level<0:
            self.max_level = dwt_max_level(input_shape[1],self.wavelet)
        self.shape_len=len(input_shape)

    def call(self,input):
        shape=tf.TensorShape([None,None,input.shape[2]]) if self.shape_len==3 else tf.TensorShape([None,None])
        a = tf.TensorArray(
            tf.float32,
            infer_shape=False,
            element_shape=shape,
            size=self.max_level+1)
        for i in range(self.max_level-1):
            a = a.write(self.max_level-i,input[:,input.shape[1]//2:,:]) if self.shape_len==3 else a.write(self.max_level-i,input[:,input.shape[1]//2:])
            input = input[:,0:input.shape[1]//2,:] if self.shape_len==3 else input[:,0:input.shape[1]//2]
        a = a.write(1,input[:,input.shape[1]//2:,:]) if self.shape_len==3 else a.write(1,input[:,input.shape[1]//2])
        a = a.write(0,input[:,0:input.shape[1]//2,:]) if self.shape_len==3 else a.write(0,input[:,0:input.shape[1]//2])
        return a

def soft_treshold(x,tau):
    cond = tf.cast(tf.math.abs(x)>tau,tf.float32)
    h = (x/x) * (x-tau)
    h = h*cond
    return h

def hybrid_treshold(x,tau):
    cond = tf.cast(tf.math.abs(x)>tau,tf.float32)
    h = x - (tf.math.square(tau)/x)
    h = h*cond
    return h

class Treshold(tf.keras.layers.Layer):
    def __init__(self,wavelet='db4',max_level=-1,treshold="soft"):
        super().__init__()
        if treshold=="soft":
            self.treshold = soft_treshold
        elif treshold=="hybrid":
            self.treshold= hybrid_treshold
        else:
            raise ValueError("treshold can be one of [soft,hybrid]")
        self.wavelet=wavelet
        self.max_level=max_level
        self.split = SplitCoefficients(wavelet,max_level)

    def build(self,input_shape):
        self.shape_len=len(input_shape)
        self.reshape = tf.keras.layers.Reshape((input_shape[1])) if self.shape_len==2 else tf.keras.layers.Reshape((input_shape[1],input_shape[2]))
    
    def call(self,coeffs):
        shape=tf.TensorShape([None,None,coeffs.shape[2]]) if self.shape_len==3 else tf.TensorShape([None,None])
        coeffs = self.split(coeffs)
        new_coeffs = tf.TensorArray(
            coeffs.dtype,
            size=coeffs.size(),
            infer_shape=False,
            element_shape=shape)
        finest = coeffs.read(coeffs.size()-1)
        std = tf.math.reduce_std(finest,axis=tf.range(1, len(finest.shape), delta=1, dtype=tf.int32, name='range'))
        tau = tf.math.sqrt(3.*tf.math.square(std))
        finest = self.treshold(finest,tau)
        finest = tf.transpose(finest,[1,0]) if self.shape_len==2 else tf.transpose(finest,[1,0,2])
        new_coeffs=new_coeffs.write(coeffs.size()-1,finest)

        coarsest = coeffs.read(0)
        coarsest = tf.transpose(coarsest,[1,0]) if self.shape_len==2 else tf.transpose(coarsest,[1,0,2])
        new_coeffs = new_coeffs.write(0,coarsest)
        for i in range(1,coeffs.size()-1):
            e = coeffs.read(i)
            e = self.treshold(e,tau)
            e = tf.transpose(e,[1,0]) if self.shape_len==2 else tf.transpose(e,[1,0,2])
            new_coeffs = new_coeffs.write(i,e)
        new_coeffs = new_coeffs.concat()
        new_coeffs = tf.transpose(new_coeffs,[1,0]) if self.shape_len==2 else tf.transpose(new_coeffs,[1,0,2])
        new_coeffs = self.reshape(new_coeffs)
        return new_coeffs



class MaskCoefficients(tf.keras.layers.Layer):
    def __init__(self,mask,wavelet='db4',max_level=-1):
        super().__init__()
        self.wavelet=wavelet
        self.max_level=max_level
        self.mask = tf.convert_to_tensor(mask)
        self.split_coeffs = SplitCoefficients(wavelet,max_level)

    def build(self,input_shape):
        self.seqlen = input_shape[1]
        if self.max_level<0:
            self.max_level = dwt_max_level(input_shape[1],self.wavelet)
        assert(len(self.mask)==self.max_level+1)
        self.shape_len=len(input_shape)
        self.reshape = tf.keras.layers.Reshape((input_shape[1])) if self.shape_len==2 else tf.keras.layers.Reshape((input_shape[1],input_shape[2]))

    def call(self,input):
        shape=tf.TensorShape([None,None,input.shape[2]]) if self.shape_len==3 else tf.TensorShape([None,None])
        a = tf.TensorArray(
            tf.float32,
            infer_shape=False,
            element_shape=shape,
            size=self.max_level+1)
        s = self.split_coeffs(input)
        for i in range(s.size()):
            e = s.read(i)
            e = tf.transpose(e,[1,0]) if self.shape_len==2 else tf.transpose(e,[1,0,2])
            if self.mask[i]:
                a = a.write(i,tf.zeros_like(e))
            else:
                a = a.write(i,e)
        filtered = a.concat()
        filtered = tf.transpose(filtered,[1,0]) if self.shape_len==2 else tf.transpose(filtered,[1,0,2])
        filtered = self.reshape(filtered)
        return filtered
                



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
        maxl = dwt_max_level(input_shape[1],self.wavelet)
        if self.max_level > maxl:
            raise ValueError("maximum decomposition level is {}, received {}".format(maxl,self.max_level))
        if self.max_level < 0:
            self.max_level = maxl
        self.dwt_layers = []
        for i in range(self.max_level):
            self.dwt_layers.append(DWT(wavelet=self.wavelet))
            self.dwt_layers[i].build((input_shape[0],input_shape[1]//2**i))
        self.reshape = tf.keras.layers.Reshape((input_shape[1],))
    def call(self,input):
        outputs = tf.TensorArray(
            tf.float32, size=self.max_level+1, dynamic_size=False, clear_after_read=False,
            infer_shape=False,
            element_shape=tf.TensorShape([None,None])
        )
        for i in range(self.max_level):
            o = self.dwt_layers[i](input)
            a = o[:,0:o.shape[1]//2]
            d = o[:,o.shape[1]//2:]
            outputs = outputs.write(self.max_level-i,tf.transpose(d,[1,0]))
            input = a
        outputs = outputs.write(0,tf.transpose(input,[1,0]))
        o = outputs.concat()
        o = tf.transpose(o,[1,0])
        return self.reshape(o)

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
        maxl = dwt_max_level(input_shape[1],self.wavelet)
        if self.max_level > maxl:
            raise ValueError("maximum decomposition level is {}, received {}".format(maxl,self.max_level))
        if self.max_level < 0:
            self.max_level = maxl
        self.idwt_layers = []
        for i in range(self.max_level):
            self.idwt_layers.append(IDWT(wavelet=self.wavelet))
            self.idwt_layers[i].build((input_shape[0],input_shape[1]//2**i))
    def call(self,input):
        a = input[:,0:input.shape[1]//2**(self.max_level-1)]
        a = self.idwt_layers[0](a)
        for i in range(self.max_level-1):
            a = tf.concat([
                a,
                input[:,input.shape[1]//2**(self.max_level-i-1):input.shape[1]//2**(self.max_level-i-2)]
            ],axis=1)
            a = self.idwt_layers[i+1](a)
        return a

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
    def call(self,input):
        o = tf.TensorArray(
            tf.float32,
            size=input.shape[2]
        )
        for i in range(input.shape[2]):
            o = o.write(i,self.wavedec(input[:,:,i]))
        o = o.stack()
        o = tf.transpose(o,perm=[1,2,0])
        return o
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
    def call(self,input):
        o = tf.TensorArray(
            tf.float32,
            size=input.shape[2]
        )
        for i in range(input.shape[2]):
            o = o.write(i,self.waverec(input[:,:,i]))
        o = o.stack()
        o = tf.transpose(o,perm=[1,2,0])
        return o
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