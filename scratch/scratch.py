import keras.backend as K
import numpy as np
from keras.layers.embeddings import Embedding
from keras.layers.convolutional import Convolution1D, MaxPooling1D
from keras.layers.core import Dense, Dropout, Activation, Flatten, Reshape
from keras.layers.recurrent import LSTM
from keras.layers.embeddings import Embedding
from keras.models import Model
from keras.layers import Convolution2D, Dense, Flatten, Input, merge, Lambda, TimeDistributed
from keras.optimizers import RMSprop, Adadelta, Adam
from rlofc.ofc_environment import OFCEnv


def simple_pgnet(input_dim,
                 action_space,
                 hidden_size=200,
                 dropout=0.5,
                 learning_rate=1e-4):
    """Karpathy-approved PGNet. From kerlym."""
    S = Input(shape=[input_dim])
    h = Dense(hidden_size, activation='relu', init='he_normal')(S)
    h = Dropout(dropout)(h)
    V = Dense(action_space, activation='sigmoid',init='zero')(h)
    model = Model(S,V)
    model.compile(loss='mse', optimizer=RMSprop(lr=learning_rate))
    return model


env = OFCEnv([])

my_board, opponent_board, current_card, remaining_cards = env.observe()

my_board.pretty()
opponent_board.pretty()
current_card
remaining_cards

env.step(0)
