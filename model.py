import tensorflow as tf
import keras
from keras import backend as K
from keras.callbacks import ModelCheckpoint, EarlyStopping

def rmse (y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true)))

def create_model():
    
    keras.backend.clear_session()

    layers = [128, 128] # Number of hidden neuros in each layer of the encoder and decoder

    learning_rate = 0.001
    decay = 0 # Learning rate decay
    optimiser = keras.optimizers.Adam(lr=learning_rate, decay=decay) # Other possible optimiser "sgd" (Stochastic Gradient Descent)

    num_input_features = 2 # The dimensionality of the input at each time step. In this case a 1D signal.
    num_output_features = 1 # The dimensionality of the output at each time step. In this case a 1D signal.
    
    loss = rmse # Other loss functions are possible, see Keras documentation.

    # Regularisation isn't really needed for this application
    lambda_regulariser = 0.000001 # Will not be used if regulariser is None
    regulariser = None # Possible regulariser: keras.regularizers.l2(lambda_regulariser)
    
    # Define an input sequence.
    encoder_inputs = keras.layers.Input(shape=(None, num_input_features))

    # Create a list of RNN Cells, these are then concatenated into a single layer
    # with the RNN layer.
    encoder_cells = []
    for hidden_neurons in layers:
        encoder_cells.append(keras.layers.LSTMCell(hidden_neurons,
                                                  kernel_regularizer=regulariser,
                                                  recurrent_regularizer=regulariser,
                                                  bias_regularizer=regulariser))

    encoder = keras.layers.RNN(encoder_cells, return_state=True)

    encoder_outputs_and_states = encoder(encoder_inputs)

    # Discard encoder outputs and only keep the states.
    # The outputs are of no interest to us, the encoder's
    # job is to create a state describing the input sequence.
    encoder_states = encoder_outputs_and_states[1:]
    
    # The decoder input will be set to zero (see random_sine function of the utils module).
    # Do not worry about the input size being 1, I will explain that in the next cell.
    decoder_inputs = keras.layers.Input(shape=(None, 1))

    decoder_cells = []
    for hidden_neurons in layers:
        decoder_cells.append(keras.layers.LSTMCell(hidden_neurons,
                                                  kernel_regularizer=regulariser,
                                                  recurrent_regularizer=regulariser,
                                                  bias_regularizer=regulariser))

    decoder = keras.layers.RNN(decoder_cells, return_sequences=True, return_state=True)

    # Set the initial state of the decoder to be the ouput state of the encoder.
    # This is the fundamental part of the encoder-decoder.
    decoder_outputs_and_states = decoder(decoder_inputs, initial_state=encoder_states)

    # Only select the output of the decoder (not the states)
    decoder_outputs = decoder_outputs_and_states[0]

    # Apply a dense layer with linear activation to set output to correct dimension
    # and scale (tanh is default activation for GRU in Keras, our output sine function can be larger then 1)
    decoder_dense = keras.layers.Dense(num_output_features,
                                       activation='linear',
                                       kernel_regularizer=regulariser,
                                       bias_regularizer=regulariser)

    decoder_outputs = decoder_dense(decoder_outputs)
    
    # Create a model using the functional API provided by Keras.
    # The functional API is great, it gives an amazing amount of freedom in architecture of your NN.
    # A read worth your time: https://keras.io/getting-started/functional-api-guide/ 
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=2)
    ]
    model = keras.models.Model(inputs=[encoder_inputs, decoder_inputs], outputs=decoder_outputs)
    model.compile(optimizer=optimiser, loss=loss)
    
    return model