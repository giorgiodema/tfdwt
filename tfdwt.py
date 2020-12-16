import tensorflow as tf
from wavelet import *
import pywt
import matplotlib.pyplot as plt


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
    x = tf.reshape(x,(x.shape[0],x.shape[1],x.shape[2],1))
    o = tf.nn.conv2d(x,filter,(2,1),'SAME')
    o = tf.reshape(o,(o.shape[0],o.shape[1],o.shape[2]))
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
    
    outputs = tf.TensorArray(tf.float32,size=0, dynamic_size=True,infer_shape=False,element_shape=tf.TensorShape([None,None,None]))
    outputs.write(0,tf.transpose(detail_coeffs,[1,2,0]))
    idx = tf.constant(1)
    while approx_coeffs.shape[1] > lpf.shape[0]:
        detail_coeffs = conv(approx_coeffs,hpf)
        approx_coeffs = conv(approx_coeffs,lpf)
        outputs.write(idx,tf.transpose(detail_coeffs,[1,2,0]))
        idx = tf.add(idx,1)
    outputs.write(idx,tf.transpose(approx_coeffs,[1,2,0]))
    coeffs = outputs.concat()
    coeffs = tf.reverse(coeffs,[0])
    coeffs = tf.transpose(coeffs,[2,0,1])
    return coeffs
    


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


def split_coeffs(coeffs,filter_length):
    output = []
    while len(coeffs) > filter_length:
        output.append(coeffs[len(coeffs)//2:])
        coeffs = coeffs[0:len(coeffs)//2]
    output.append(coeffs)
    output.reverse()
    return output


if __name__=='__main__':
    X,Y = signal([16,96],length=1024,overlap=True)
    yt = tf.reshape(tf.convert_to_tensor(Y,dtype=tf.float32),(1,Y.shape[0],1))
    pcoeffs = dwt(yt)
    coeffs = split_coeffs(tf.reshape(pcoeffs,(1024)).numpy(),8)
    for i in range(len(coeffs)-1):
        coeffs[-i] = np.zeros_like(coeffs[-i])
        y_r = pywt.waverec(coeffs,'db4',mode='periodization')
        plt.plot(y_r)
        plt.show()


    """
    y = tf.ones((1,1024,1))
    f = tf.convert_to_tensor([.5,.5,.5,.5])
    o = conv(y,f)
    pass
    """
    """
    yt = tf.reshape(tf.convert_to_tensor(Y,dtype=tf.float32),(1,Y.shape[0],1))
    plt.plot(Y)
    plt.title("Original")
    plt.show()
    pcoeffs = dwt(yt)
    tcoeffs = pywt.dwt(Y,'db4')

    pcoeffs = tf.reshape(pcoeffs,(1024))
    ac = pcoeffs[0:512].numpy()
    dc = pcoeffs[512:].numpy()
    yr = pywt.idwt(ac,dc,'db4')
    plt.plot(yr)
    plt.title("Reconstructed")
    plt.show()
    pass
    """
