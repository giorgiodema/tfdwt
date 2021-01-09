# tfdwt
tfdwt is a tensorflow module that contains layers to compute Discrete Wavelet Transforms, as well as
wavelet decomposition and reconstruction for both univariate and multivariate input signals.
NB: to build keras model with some of these layers the batch size must be specified in the
input layer

## get_wavelets
Return a list with the names of the available wavelet functions

## Class DWT
Compute The Discrete Wavelet Transform for univariate signals,
(for signals with multiple features see MultivariateDWT)
### Constructor Parameters:
- wavelet: wavelet function to use (check 'get_wavelets' to see which wavelet functions are supported)
### Call Parameters:
- input: the input signal, with shape (BS,SEQ_LEN), if the signal has 
                multiple features then use MultivariateDWT
### Output:
- output: the DWT coefficients with shape `(BS,SEQ_LEN)`, where
                `output = [Ca,Cd]` where Ca and Cd are the approximation and detail coefficients 

## Class IDWT
Compute the Inverse Discrete Wavelet Transform for univariate signals. (For
signals with multiple features see MultivariateIDWT)
### Constructor Parameters:
- wavelet: wavelet function to use (check 'get_wavelets' to see which wavelet functions are supported)
### Call Parameters:
- input: the DWT coefficients with shape `(BS,SEQ_LEN)`, where
                `input = [Ca,Cd]` where Ca and Cd are the approximation and detail coefficients
### Output:
- output: the output signal, with shape `(BS,SEQ_LEN)`, if the signal has 
                multiple features then use MultivariateDWT

## Class WaveDec
Compute the Wavelet Decomposition for univariate signals (for signal with multiple features
see MultivariateWaveDec)
### Constructor Parameters:
- wavelet: wavelet function to use (check 'get_wavelets' to see which wavelet functions are supported)
- max_level: the maximum level of decomposition, if max_level=-1
                    then max_level is the maximum level of decomposition
                    for the chosen wavelet family. The signal is decomposed untill
                    the length of the output of the decomposition is greater than the
                    length of the filter.
### Call Parameters:
- input: the input signal, with shape (BS,SEQ_LEN), if the signal has 
                multiple features then use MultivariateDWT
### Output:
- output: the DWT coefficients with shape `(BS,SEQ_LEN)`, where
                `output = [Ca_n,Cd_n,Cd_n-1,...,Cd_1]` where the length of
                Cd_n is one half of the length of Cd_n-1

## Class WaveRec
Compute the Wavelet Reconstruction for univariate signals (for signal with multiple features
see MultivariateWaveDec)
### Constructor Parameters:
- wavelet: wavelet function to use (check 'get_wavelets' to see which wavelet functions are supported)
- max_level: the maximum level of decomposition, if `max_level=-1`
                    then max_level is the maximum level of 
                    decomposition for the chosen wavelet family
### Call Parameters:
- input: the DWT coefficients with shape `(BS,SEQ_LEN)`, where
                `input = [Ca_n,Cd_n,Cd_n-1,...,Cd_1]` where the length of
                Cd_n is one half of the length of Cd_n-1
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
