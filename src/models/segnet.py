import tensorflow as tf
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from models.utils import *
from tensorflow.keras.utils import plot_model

IMAGE_ORDERING = "channels_last"
INPUT_SHAPE = (256, 256, 3)
IMAGE_H_W = (256, 256)
pretrained_url = "https://github.com/fchollet/deep-learning-models/releases/download/v0.1/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5"


def from_data(func):
    def wrapper(*args, **kwargs):
        os.chdir(DATA_PATH)
        ret = func(*args, **kwargs)
        os.chdir(PROJ_PATH)
        return ret
    return wrapper


def from_res(func):
    def wrapper(*args, **kwargs):
        os.chdir(RES_PATH)
        ret = func(*args, **kwargs)
        os.chdir(PROJ_PATH)
        return ret
    return wrapper


def show_progress(it, milestones=1):
    for i, x in enumerate(it):
        yield x
        processed = i + 1
        if processed % milestones == 0:
            logging.info('Processed %s elements' % processed)


def to_rgb(img: np.array):
    '''Converts given image to RHB and normalized to [0,1]'''
    raise NotImplementedError


def vgg_encoder(input_shape, channels=3):
    '''VGG16-style encoder
        assumes channels-last image ordering and input-size = (224, 224, 3)
    '''

    input_height, input_width, channels = input_shape
    img_input = Input(shape=(input_height, input_width, channels))

    x = Conv2D(64, (3, 3), activation='relu', padding='same',
               name='block1_conv1', data_format=IMAGE_ORDERING)(img_input)
    x = Conv2D(64, (3, 3), activation='relu', padding='same',
               name='block1_conv2', data_format=IMAGE_ORDERING)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool',
                     data_format=IMAGE_ORDERING)(x)
    f1 = x
    # Block 2
    x = Conv2D(128, (3, 3), activation='relu', padding='same',
               name='block2_conv1', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(128, (3, 3), activation='relu', padding='same',
               name='block2_conv2', data_format=IMAGE_ORDERING)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool',
                     data_format=IMAGE_ORDERING)(x)
    f2 = x

    # Block 3
    x = Conv2D(256, (3, 3), activation='relu', padding='same',
               name='block3_conv1', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same',
               name='block3_conv2', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(256, (3, 3), activation='relu', padding='same',
               name='block3_conv3', data_format=IMAGE_ORDERING)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool',
                     data_format=IMAGE_ORDERING)(x)
    f3 = x

    # Block 4
    x = Conv2D(512, (3, 3), activation='relu', padding='same',
               name='block4_conv1', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same',
               name='block4_conv2', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same',
               name='block4_conv3', data_format=IMAGE_ORDERING)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool',
                     data_format=IMAGE_ORDERING)(x)
    f4 = x

    # Block 5
    x = Conv2D(512, (3, 3), activation='relu', padding='same',
               name='block5_conv1', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same',
               name='block5_conv2', data_format=IMAGE_ORDERING)(x)
    x = Conv2D(512, (3, 3), activation='relu', padding='same',
               name='block5_conv3', data_format=IMAGE_ORDERING)(x)
    x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool',
                     data_format=IMAGE_ORDERING)(x)

    f5 = x

    VGG_weights_path = tf.keras.utils.get_file(
        pretrained_url.split('/')[-1], pretrained_url)
    Model(img_input, x).load_weights(
        VGG_weights_path, by_name=True, skip_mismatch=True)

    return img_input, (f1, f2, f3, f4, f5)


def decoder(f, n):

    output = f

    # adds rows/columns of zeros on edges of input
    output = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(output)
    output = (Conv2D(512, (3, 3), padding='valid',
              data_format=IMAGE_ORDERING))(output)
    output = (BatchNormalization())(output)

    output = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(output)
    output = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(output)
    output = (Conv2D(256, (3, 3), padding='valid',
              data_format=IMAGE_ORDERING))(output)
    output = (BatchNormalization())(output)

    output = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(output)
    output = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(output)
    output = (Conv2D(128, (3, 3), padding='valid',
                     data_format=IMAGE_ORDERING))(output)
    output = (BatchNormalization())(output)

    output = (UpSampling2D((2, 2), data_format=IMAGE_ORDERING))(output)
    output = (ZeroPadding2D((1, 1), data_format=IMAGE_ORDERING))(output)
    output = (Conv2D(64, (3, 3), padding='valid',
                     data_format=IMAGE_ORDERING, name="seg_feats"))(output)
    output = (BatchNormalization())(output)

    output = Conv2D(n, (3, 3), padding='same',
                    data_format=IMAGE_ORDERING)(output)

    return output

# Apply softmax activation to output
# Need to use inverse feature weighting for class imbalance ref: J. Shotton, M. Johnson, and R. Cipolla. Semantic texton
#   forests for image categorization and segmentation. In CVPR, 2008


# from Segnet Paper->
'''SegNet uses a “flat” architecture, i.e, the number of features in each layer remains the same (64 in our case) but
with full connectivity. This choice is motivated by two reasons. First, it avoids parameter explosion, unlike an expanding deep encoder network with full feature connectivity (same for decoder). Second, the training time remains
the same (in our experiments it slightly decreases) for each
additional/deeper encoder-decoder pair as the feature map
resolution is smaller which makes convolutions faster - 

Local Contrast Normalization done prior: improves convergence by decorrelating input dimensions, corrects for non-uniform scene illumination (reduces dynamic range), highlights edges

A fixed pooling window of
2×2 with a stride of non-overlapping 2 pixels is used.

Smaller kernels decrease context and larger ones potentially destroy thin structures'''


def build_segnet(encoder=None, input_shape=INPUT_SHAPE, enc_level=3, channels=3):
    if not encoder:
        encoder = vgg_encoder

    input, stages = encoder(
        input_shape=input_shape, channels=channels
    )

    # choosed what level to take output at
    features = stages[enc_level]
    output = decoder(features, 2)

    output = Activation('softmax')(output)

    model = tf.keras.models.Model(input, output)

    return model


if __name__ == '__main__':
    model = build_segnet(vgg_encoder, input_shape=(224, 224, 3))

    logging.info(model.summary())
    dot_img_file = './tmp/model_1.png'
    tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)
