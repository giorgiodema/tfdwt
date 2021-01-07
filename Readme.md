# tfDiscreteWaveletTransform
tfdwt is a tensorflow module that contains layers to compute Discrete Wavelet Transforms, as well as
wavelet decomposition and reconstruction for both univariate and multivariate input signals.
NB: to build keras model with some of these layers the batch size must be specified in the
input layer

## Class DWT
Compute The Discrete Wavelet Transform for univariate signals,
(for signals with multiple features see MultivariateDWT)
### Constructor Parameters:
- wavelet: wavelet function to use (up to now only db4 is supported)
### Call Parameters:
- input: the input signal, with shape (BS,SEQ_LEN), if the signal has 
                multiple features then use MultivariateDWT
### Output:
- output: the DWT coefficients with shape `(BS,SEQ_LEN)`, where
                `output[:,0:output.shape[1]//2]` contains the approximation
                coefficients while `output[:,output.shape[1]//2:]` contains the
                detail coefficients

## Class IDWT
Compute the Inverse Discrete Wavelet Transform for univariate signals. (For
signals with multiple features see MultivariateIDWT)
### Constructor Parameters:
- wavelet: wavelet function to use (up to now only db4 is supported)
### Call Parameters:
- input: the DWT coefficients with shape `(BS,SEQ_LEN)`, where
                `output[:,0:output.shape[1]//2]` contains the approximation
                coefficients while `output[:,output.shape[1]//2:]` contains the
                detail coefficients
### Output:
- output: the output signal, with shape `(BS,SEQ_LEN)`, if the signal has 
                multiple features then use MultivariateDWT

## Class WaveDec
Compute the Wavelet Decomposition for univariate signals (for signal with multiple features
see MultivariateWaveDec)
### Constructor Parameters:
- wavelet: wavelet function to use (up to now only db4 is supported)
- max_level: the maximum level of decomposition, if max_level=-1
                    then max_level is the maximum level of decomposition
                    for the chosen wavelet family
### Call Parameters:
- input: the input signal, with shape (BS,SEQ_LEN), if the signal has 
                multiple features then use MultivariateDWT
### Output:
- output: the DWT coefficients with shape `(BS,SEQ_LEN)`, where
                `output[:,0:output.shape[1]//2**(max_level-1)]` contains the approximation
                coefficients while `output[:,output.shape[1]//2**(i+1):output.shape[1]//2**(i)] `
                for `i = [1,...]` contains the detail coefficients of the level i

## Class WaveRec
Compute the Wavelet Reconstruction for univariate signals (for signal with multiple features
see MultivariateWaveDec)
### Constructor Parameters:
- wavelet: wavelet function to use (up to now only db4 is supported)
- max_level: the maximum level of decomposition, if `max_level=-1`
                    then max_level is the maximum level of 
                    decomposition for the chosen wavelet family
### Call Parameters:
- input: the DWT coefficients with shape `(BS,SEQ_LEN)`, where
                `output[:,0:output.shape[1]//2**(max_level-1)]` contains the approximation
                coefficients while `output[:,output.shape[1]//2**(i+1):output.shape[1]//2**(i)]` 
                for `i = [1,...]` contains the detail coefficients of the level i
### Output:
- output: the output signal, with shape `(BS,SEQ_LEN)`, if the signal has 
                multiple features then use MultivariateDWT

## Class MultivariateDWT
Multivariate Verion of the DWT, input_shape
and output shape are `(BS,SEQ_LEN,N_FEATURES)`

## Class MultivariateIDWT
Multivariate Verion of the IDWT, input_shape
and output shape are `(BS,SEQ_LEN,N_FEATURES)`

## Class MultivariateWaveDec
Multivariate Verion of the WaveDec, input_shape
and output shape are `(BS,SEQ_LEN,N_FEATURES)`

## Class MultivariateWaveRec
Multivariate Verion of the WaveRec, input_shape
and output shape are `(BS,SEQ_LEN,N_FEATURES)`
