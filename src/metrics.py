import tensorflow as tf


class DiceMetric(tf.keras.metrics.Metric):
    def __init__(self, name='dice_loss_metric', smooth=1e-6, gama=2, **kwargs):
        super(DiceMetric, self).__init__(name=name, **kwargs)
        self.dice = self.add_weight(name='dice', initializer='zeros')
        self.smooth = smooth
        self.gama = gama

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)
        nominator = 2 * \
            tf.reduce_sum(tf.multiply(y_pred, y_true)) + self.smooth
        denominator = tf.reduce_sum(
            y_pred ** self.gama) + tf.reduce_sum(y_true ** self.gama) + self.smooth
        result = 1 - tf.divide(nominator, denominator)
        self.dice.assign(tf.divide(nominator, denominator))

    def result(self):
        return self.dice
