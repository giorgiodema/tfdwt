import tensorflow as tf
from tensorflow.python.ops.gen_array_ops import transpose
from wavelet import *
import pywt
import matplotlib.pyplot as plt
import math


coefficients = {
    "db4":{
        "dec":{
            "lpf":tf.constant([0.4829629131445341,0.8365163037378077,0.2241438680420134,-0.12940952255126034]),
            "hpf":tf.constant([-0.12940952255126034,-0.2241438680420134,0.8365163037378077,-0.4829629131445341])
        },
        "rec":{
            "lpf":tf.constant([0.2241438680420134,0.8365163037378077,0.4829629131445341,-0.12940952255126034]),
            "hpf":tf.constant([-0.12940952255126034,-0.4829629131445341,0.8365163037378077,-0.2241438680420134])

        }
    }
}


def conv(x,filter):
    filter = tf.reshape(filter,(filter.shape[0],1,1,1))
    x = tf.reshape(x,(x.shape[0],x.shape[1],x.shape[2],1))
    #padding = filter.shape[0]-2
    #pad = x[:,0:padding,:]#tf.repeat(x[:,-1:,:],repeats=[padding],axis=1)
    #x = tf.concat([x,pad],axis=1)
    padding=filter.shape[0]
    lpad = x[:,0:padding,:]
    rpad = x[:,-padding:,:]
    x = tf.concat([lpad,x,rpad],axis=1)
    o = tf.nn.conv2d(x,filter,(2,1),'SAME')
    o = o[:,padding//2:-padding//2,:,:]
    o = tf.reshape(o,(o.shape[0],o.shape[1],o.shape[2]))
    return o

def interleave(t1,t2):
    tf.assert_equal(t1.shape,t2.shape)
    even_pos = tf.convert_to_tensor(list(range(0,2*t1.shape[0],2)),dtype=tf.int32)
    odd_pos = tf.convert_to_tensor(list(range(1,2*t1.shape[0],2)),dtype=tf.int32)
    out = tf.dynamic_stitch([even_pos,odd_pos],[t1,t2])
    return out


def log2(x):
    return tf.math.log(tf.cast(x,tf.float32))/tf.math.log(2.)


def dwt(x,wavelet="db4",maxlevel=-1):
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
    if maxlevel==1:
        return tf.concat([approx_coeffs,detail_coeffs],axis=1)
    
    outputs = tf.TensorArray(tf.float32,size=0, dynamic_size=True,infer_shape=False,element_shape=tf.TensorShape([None,None,None]))
    outputs.write(0,tf.transpose(detail_coeffs,[1,2,0]))
    idx = tf.constant(1)
    while approx_coeffs.shape[1] > lpf.shape[0]:
        detail_coeffs = conv(approx_coeffs,hpf)
        approx_coeffs = conv(approx_coeffs,lpf)
        outputs.write(idx,tf.transpose(detail_coeffs,[1,2,0]))
        idx = tf.add(idx,1)
        if maxlevel>0 and idx >= maxlevel:
            break
    outputs.write(idx,tf.transpose(approx_coeffs,[1,2,0]))
    coeffs = outputs.concat()
    coeffs = tf.reverse(coeffs,[0])
    coeffs = tf.transpose(coeffs,[2,0,1])
    return coeffs
    


def idwt(coeffs,wavelet="db4",maxlevel=-1):
    """
    Params:
        -> x: the coefficients of the dwt decomposition
        -> wavelet [str]: the mother wavelet name
        -> multilevel [bool]:   if true the layer computes the multilevel decomposition,
                                otherwise it computes the single level discrete wavelet transform
        -> mask:
    """
    lpf = coefficients[wavelet]["rec"]["lpf"]
    hpf = coefficients[wavelet]["rec"]["hpf"]

    signal_len = coeffs.shape[1]
    current_len = lpf.shape[0] if maxlevel<0 else signal_len//2**(maxlevel)
    while current_len <= signal_len//2:
        lpin = coeffs[:,0:current_len,:]
        hpin = coeffs[:,current_len:2*current_len,:]
        interleaved = tf.transpose(interleave(
            tf.transpose(lpin,[1,2,0]),
            tf.transpose(hpin,[1,2,0])),[2,0,1])
        lpout = tf.reverse(conv(tf.reverse(interleaved,axis=[1]),lpf),axis=[1])
        hpout = tf.reverse(conv(tf.reverse(interleaved,axis=[1]),hpf),axis=[1])
        out = tf.transpose(interleave(
            tf.transpose(lpout,[1,2,0]),
            tf.transpose(hpout,[1,2,0])),[2,0,1])
        right = coeffs[:,2*current_len:,:]
        coeffs = tf.concat([out,right],axis=1)
        current_len*=2
    return coeffs


def split_coeffs(coeffs,filter_length):
    output = []
    while len(coeffs) > filter_length:
        output.append(coeffs[len(coeffs)//2:])
        coeffs = coeffs[0:len(coeffs)//2]
    output.append(coeffs)
    output.reverse()
    return output


if __name__=='__main__':
    X,Y = signal([5],length=1024,overlap=True)
    plt.plot(Y)
    plt.show()
    yt = tf.reshape(tf.convert_to_tensor(Y,dtype=tf.float32),(1,Y.shape[0],1))
    maxlevel=4
    pcoeffs = dwt(yt,maxlevel=maxlevel)
    tcoeffs = pywt.wavedec(Y,'db4',mode='periodization')
    py_r = idwt(pcoeffs,maxlevel=maxlevel)
    py_r = tf.reshape(py_r,(py_r.shape[1]))
    plt.plot(py_r.numpy())
    plt.show()

    """
    pcoeffs = split_coeffs(tf.reshape(pcoeffs,(1024)).numpy(),4)
    y_r = pywt.waverec(pcoeffs,'db4',mode='periodization')
    plt.plot(y_r)
    plt.show()
    """

    """
    for i in range(len(coeffs)-1):
        coeffs[-i] = np.zeros_like(coeffs[-i])
        y_r = pywt.waverec(coeffs,'db4',mode='periodization')
        plt.plot(y_r)
        plt.show()
    """


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
