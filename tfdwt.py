import tensorflow as tf
from wavelet import *
import pywt


coefficients = {
    "db4":{
        "dec":{
            "lpf":tf.constant([-0.0105974018, 0.0328830117, 0.0308413818,-0.1870348117,-0.0279837694, 0.6308807679, 0.7148465706, 0.2303778133]),
            "hpf":tf.constant([-0.2303778133, 0.7148465706,-0.6308807679,-0.0279837694, 0.1870348117, 0.0308413818,-0.0328830117,-0.0105974018])
        },
        "rec":{
            "lpf":tf.constant([0.2303778133,0.7148465706,0.6308807679,0.0279837694,0.1870348117,0.0308413818,0.0328830117,0.0105974018]),
            "hpf":tf.constant([-0.0105974018,-0.0328830117, 0.0308413818, 0.1870348117,-0.0279837694,-0.6308807679, 0.7148465706,-0.2303778133])
        }
    }
}


def conv(x,filter):
    filter = tf.reshape(filter,(filter.shape[0],1,1,1))
    #x = tf.transpose(x,[0,2,1])
    x = tf.reshape(x,(x.shape[0],x.shape[1],x.shape[2],1))
    o = tf.nn.conv2d(x,filter,(2,1),'SAME')
    o = tf.reshape(o,(o.shape[0],o.shape[1],o.shape[2]))
    #o = tf.transpose(o,[0,2,1])
    return o


def log2(x):
    return tf.math.log(tf.cast(x,tf.float32))/tf.math.log(2.)


def dwt(x,wavelet="db4",multilevel=False):
    """
    Params:
        -> x:   the input signal, with shape [BS,SEQ_LEN,N_DIM],
                SEQ_LEN must be a power of 2
        -> wavelet [str]: the mother wavelet name
        -> multilevel [bool]:   if true the layer computes the multilevel decomposition,
                                otherwise it computes the single level discrete wavelet transform
    """
    lpf = coefficients[wavelet]["dec"]["lpf"]
    hpf = coefficients[wavelet]["dec"]["hpf"]

    detail_coeffs = conv(x,hpf)
    approx_coeffs = conv(x,lpf)
    if not multilevel:
        return tf.concat([approx_coeffs,detail_coeffs],axis=1)
    levels = tf.cast(log2(x.shape[1]),tf.int32)
    for l in range(1,levels+1):
        pass


def idwt():
    """
    Params:
        -> x: the coefficients of the dwt decomposition
        -> wavelet [str]: the mother wavelet name
        -> multilevel [bool]:   if true the layer computes the multilevel decomposition,
                                otherwise it computes the single level discrete wavelet transform
        -> mask:
    """
    pass



if __name__=='__main__':
    X,Y = signal([16,96],length=1024,overlap=True)
    """
    y = tf.ones((1,1024,1))
    f = tf.convert_to_tensor([.5,.5,.5,.5])
    o = conv(y,f)
    pass
    """

    yt = tf.reshape(tf.convert_to_tensor(Y,dtype=tf.float32),(1,Y.shape[0],1))
    pcoeffs = dwt(yt)
    tcoeffs = pywt.dwt(Y,'db4')
    pass
