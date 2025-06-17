from typing import List, Optional, Tuple
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Concatenate, Bidirectional, Input, Dense, LSTM, Reshape, Lambda
from tensorflow.keras.models import Model


def initialize_network(network_name: str,
                       layer_shapes: List[int],
                       activations: List[str],
                       layer_B1_constraint_incorporation: Optional[int] = None
                       ) -> Model:
    """
    Initializes a neural network based on the specified architecture.
    
    Parameters
    ----------
    network_name: str
        Name of the network architecture to be initialized.
    layer_shapes: List[int]
        List of integers representing the number of units in each layer.
    activations: List[str]
        List of activation functions for each layer, excluding the output layer.
    layer_B1_constraint_incorporation: Optional[int]
        If provided, the network will include an additional input for B1 constraint incorporation.
    
    Returns
    -------
    Model: tensorflow.keras.Model
        The initialized neural network model.
    """
    if network_name == 'BiLSTM':
        NN = MRF_BiLSTM(layer_shapes, activations)

    if network_name == 'BiLSTM_complex':
        NN = MRF_BiLSTM_complex(layer_shapes, activations)

    if layer_B1_constraint_incorporation is None:
        NN.build(input_shape=(None, layer_shapes[0]))
    else:
        NN.build(input_shape=[(None, layer_shapes[0]), (None, 1)])

    return NN


def MRF_BiLSTM(layer_shapes: List[int], activations: List[str]) -> Model:
    """ 
    Builds a BiLSTM-based recurrent neural network for quantitative maps prediction from MRF signal.
    This network processes the input signal through a bidirectional LSTM layer followed by dense layers. 

    Parameters
    ----------
    layer_shapes: List[int]
        List of lengths of the layers. 
    activations: List[str]
        List of activation functions (has 2 less items than layer_shapes). 

    Returns
    -------
    Model: tensorflow.keras.Model
        The BiLSTM neural network. 
    """

    input_layer = Input(shape=(layer_shapes[0],))
    x = Reshape((layer_shapes[0], 1))(input_layer)
    lstm = Bidirectional(LSTM(layer_shapes[1], activation=activations[0], return_sequences=False))(x)

    for units, activation in zip(layer_shapes[2:-1], activations[1:]):
        lstm = Dense(units, activation=activation)(lstm)

    output_layer = Dense(layer_shapes[-1])(lstm)

    return Model(inputs=input_layer, outputs=output_layer)
  

def MRF_BiLSTM_complex(layer_shapes: List[int], activations: List[str]) -> Model:
    """ 
    Builds a complex-valued BiLSTM-based recurrent neural network for quantitative maps prediction from MRF signal.
    This network processes complex-valued input by separating the real and imaginary parts,
    applying bidirectional LSTM layers to each part, and then concatenating the results.

    Parameters
    ----------
    layer_shapes: List[int]
        List of lengths of the layers. 
    activations: List[str]
        List of activation functions (has 2 less items than layer_shapes). 

    Returns
    -------
    model : tensorflow.keras.Model
        The complex BiLSTM neural network. 
    """
    input_layer = Input(shape=(layer_shapes[0]*2,))
    
    magn = Lambda(lambda x: x[:, :260])(input_layer)
    phase = Lambda(lambda x: x[:, 260:])(input_layer)

    lstm_magn = Reshape((layer_shapes[0], 1))(magn)
    lstm_magn = Bidirectional(LSTM(layer_shapes[1], activation=activations[0], return_sequences=False))(lstm_magn)

    lstm_phase = Reshape((layer_shapes[0], 1))(phase)
    lstm_phase = Bidirectional(LSTM(layer_shapes[1], activation=activations[0], return_sequences=False))(lstm_phase)

    complex_nb = Concatenate()([lstm_magn, lstm_phase])

    for layer_shape, activation in zip(layer_shapes[2:-1], activations[1:]):
        complex_nb = Dense(layer_shape, activation=activation)(complex_nb)
    
    output = Dense(layer_shapes[-1])(complex_nb)

    return Model(input_layer, output)
