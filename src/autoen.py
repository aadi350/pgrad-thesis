import tensorflow as tf
from tensorflow.keras import Model, layers, losses
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from tensorflow.keras import layers, losses
from tensorflow.keras.datasets import fashion_mnist
from tensorflow.keras.models import Model
latent_dim = 64


class Autoencoder(Model):
    def __init__(self, ):
        super(Autoencoder, self).__init__()
        self.encoder = tf.keras.Sequential([
            layers.Flatten(),
            layers.Dense(64, activation='relu')
        ])
        self.decoder = tf.keras.Sequential([
            layers.Dense(65536, activation='sigmoid'),
            layers.Reshape((256, 256))
        ])

    def call(self, x):
        enc = self.encoder(x)
        dec = self.decoder(enc)
        return dec
