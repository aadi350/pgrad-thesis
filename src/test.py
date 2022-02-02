from models.segnet import build_segnet
import tensorflow as tf
import logging
model = build_segnet(input_shape=(224, 224, 3))

logging.info(model.summary())
dot_img_file = './tmp/model_1.png'
tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)
