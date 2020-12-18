import tensorflow as tf
import numpy as np
import math
from tensorflow.python.ops.gen_array_ops import transpose
from wavelet import signal
import matplotlib.pyplot as plt

coeffs = {
    "db4":[0.4829629131445341,0.8365163037378079,0.2241438680420134, -0.1294095225512604]
}

def interleave(t1,t2):
    tf.assert_equal(t1.shape,t2.shape)
    even_pos = tf.convert_to_tensor(list(range(0,2*t1.shape[0],2)),dtype=tf.int32)
    odd_pos = tf.convert_to_tensor(list(range(1,2*t1.shape[0],2)),dtype=tf.int32)
    out = tf.dynamic_stitch([even_pos,odd_pos],[t1,t2])
    return out

def build_matrix(coeffs,signal_len,transpose=False):
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

def print_matrix(m):
    for i in range(m.shape[0]):
        for j in range(m.shape[1]):
            print('{s:{c}^{n}}'.format(s=str(m[i,j]),c=' ',n=4),end=" ")
        print()

class DWTDec(tf.keras.layers.Layer):
    def __init__(self,wavelet='db4'):
        super(DWTDec, self).__init__(name='dwt_dec_layer')
        if wavelet!='db4':
            raise NotImplemented()
        self.coeffs = coeffs[wavelet]

    def build(self,input_shape):
        self.w = tf.Variable(
            initial_value = build_matrix(self.coeffs,input_shape[1],transpose=True),
            dtype=tf.float32,
            trainable=False
        )
        self.input_var = tf.Variable(
            initial_value = tf.zeros(input_shape,dtype=tf.float32),
            trainable=False
        )
        self.n_it = int(math.log2(input_shape[1]))-1

    def call(self,input):
        self.input_var.assign(input)
        w = tf.convert_to_tensor(self.w)
        for i in range(self.n_it):
            w = w[0:input.shape[1]//2**i,0:input.shape[1]//2**i]
            self.input_var[:,0:input.shape[1]//2**i].assign(tf.matmul(self.input_var[:,0:input.shape[1]//2**i],w))
            a = self.input_var[:,0:input.shape[1]//2**i][:,::2]
            d = self.input_var[:,0:input.shape[1]//2**i][:,1::2]
            self.input_var[:,0:(input.shape[1]//2**i)//2].assign(a)
            self.input_var[:,(input.shape[1]//2**i)//2:input.shape[1]//2**i].assign(d)
        return tf.convert_to_tensor(self.input_var)

class DWTRec(tf.keras.layers.Layer):
    def __init__(self,wavelet='db4'):
        super(DWTRec, self).__init__(name='dwt_rec_layer')
        if wavelet!='db4':
            raise NotImplemented()
        self.coeffs = coeffs[wavelet]

    def build(self,input_shape):
        self.w = tf.Variable(
            initial_value = build_matrix(self.coeffs,input_shape[1],transpose=False),
            dtype=tf.float32,
            trainable=False
        )
        self.input_var = tf.Variable(
            initial_value = tf.zeros(input_shape,dtype=tf.float32),
            trainable=False
        )
        self.n_it = int(math.log2(input_shape[1]))-1

    def call(self,input):
        self.input_var.assign(input)
        w = tf.convert_to_tensor(self.w)
        for i in range(self.n_it):
            w = w[0:input.shape[1]//2**i,0:input.shape[1]//2**i]
            self.input_var[:,0:input.shape[1]//2**i].assign(tf.matmul(self.input_var[:,0:input.shape[1]//2**i],w))
            a = self.input_var[:,0:input.shape[1]//2**i][:,::2]
            d = self.input_var[:,0:input.shape[1]//2**i][:,1::2]
            self.input_var[:,0:(input.shape[1]//2**i)//2].assign(a)
            self.input_var[:,(input.shape[1]//2**i)//2:input.shape[1]//2**i].assign(d)
        return tf.convert_to_tensor(self.input_var)



if __name__=="__main__":
    dec = DWTDec()
    rec = DWTRec()
    s = tf.convert_to_tensor(signal([5,50])[1])
    plt.plot(s.numpy())
    plt.show()
    s = tf.reshape(s,(1,1024))
    o = dec(s)
    o = rec(s)
    s = s[0].numpy()
    plt.plot(s)
    plt.show()
    print()