import tensorflow as tf


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


def dwt(x,wavelet="db4",multilevel=False):
    """
    Params:
        -> x:   the input signal, with shape [BS,SEQ_LEN,N_DIM],
                SEQ_LEN must be a power of 2
        -> wavelet [str]: the mother wavelet name
        -> multilevel [bool]:   if true the layer computes the multilevel decomposition,
                                otherwise it computes the single level discrete wavelet transform
    """

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
    pass