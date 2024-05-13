import tensorflow as tf
import numpy as np


class Model(tf.keras.Model):
    def __init__(self):
        super().__init__()

        self.mlayers = [ # [None, 8]    
            tf.keras.layers.Dense(units=16, activation='swish', activity_regularizer='l1'),
            tf.keras.layers.Dense(units=32, activation='swish', activity_regularizer='l1'),
            tf.keras.layers.Dense(units=2, activation='softmax')
        ]

    