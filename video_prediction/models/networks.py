import tensorflow as tf
from tensorflow.python.util import nest

from video_prediction import ops
from video_prediction.ops import conv2d
from video_prediction.ops import dense
from video_prediction.ops import lrelu
from video_prediction.ops import pool2d
from video_prediction.utils import tf_utils


def encoder(inputs, nef=64, n_layers=3, norm_layer='instance'):
    norm_layer = ops.get_norm_layer(norm_layer)
    layers = []
    paddings = [[0, 0], [1, 1], [1, 1], [0, 0]]

    with tf.variable_scope("layer_1"):
        convolved = conv2d(tf.pad(inputs, paddings), nef, kernel_size=4, strides=2, padding='VALID')
        rectified = lrelu(convolved, 0.2)
        layers.append(rectified)

    for i in range(1, n_layers):
        with tf.variable_scope("layer_%d" % (len(layers) + 1)):
            out_channels = nef * min(2**i, 4)
            convolved = conv2d(tf.pad(layers[-1], paddings), out_channels, kernel_size=4, strides=2, padding='VALID')
            normalized = norm_layer(convolved)
            rectified = lrelu(normalized, 0.2)
            layers.append(rectified)

    pooled = pool2d(rectified, rectified.shape.as_list()[1:3], padding='VALID', pool_mode='avg')
    squeezed = tf.squeeze(pooled, [1, 2])
    return squeezed


def image_sn_discriminator(images, ndf=64):
    batch_size = images.shape[0].value
    layers = []
    paddings = [[0, 0], [1, 1], [1, 1], [0, 0]]

    def conv2d(inputs, *args, **kwargs):
        kwargs.setdefault('padding', 'VALID')
        kwargs.setdefault('use_spectral_norm', True)
        return ops.conv2d(tf.pad(inputs, paddings), *args, **kwargs)

    with tf.variable_scope("sn_conv0_0"):
        layers.append(lrelu(conv2d(images, ndf, kernel_size=3, strides=1), 0.1))

    with tf.variable_scope("sn_conv0_1"):
        layers.append(lrelu(conv2d(layers[-1], ndf * 2, kernel_size=4, strides=2), 0.1))

    with tf.variable_scope("sn_conv1_0"):
        layers.append(lrelu(conv2d(layers[-1], ndf * 2, kernel_size=3, strides=1), 0.1))

    with tf.variable_scope("sn_conv1_1"):
        layers.append(lrelu(conv2d(layers[-1], ndf * 4, kernel_size=4, strides=2), 0.1))

    with tf.variable_scope("sn_conv2_0"):
        layers.append(lrelu(conv2d(layers[-1], ndf * 4, kernel_size=3, strides=1), 0.1))

    with tf.variable_scope("sn_conv2_1"):
        layers.append(lrelu(conv2d(layers[-1], ndf * 8, kernel_size=4, strides=2), 0.1))

    with tf.variable_scope("sn_conv3_0"):
        layers.append(lrelu(conv2d(layers[-1], ndf * 8, kernel_size=3, strides=1), 0.1))

    with tf.variable_scope("sn_fc4"):
        logits = dense(tf.reshape(layers[-1], [batch_size, -1]), 1, use_spectral_norm=True)
        layers.append(logits)
    return layers


def video_sn_discriminator(clips, ndf=64):
    clips = tf_utils.transpose_batch_time(clips)
    batch_size = clips.shape[0].value
    layers = []
    paddings = [[0, 0], [1, 1], [1, 1], [1, 1], [0, 0]]

    def conv3d(inputs, *args, **kwargs):
        kwargs.setdefault('padding', 'VALID')
        kwargs.setdefault('use_spectral_norm', True)
        return ops.conv3d(tf.pad(inputs, paddings), *args, **kwargs)

    with tf.variable_scope("sn_conv0_0"):
        layers.append(lrelu(conv3d(clips, ndf, kernel_size=3, strides=1), 0.1))

    with tf.variable_scope("sn_conv0_1"):
        layers.append(lrelu(conv3d(layers[-1], ndf * 2, kernel_size=4, strides=(1, 2, 2)), 0.1))

    with tf.variable_scope("sn_conv1_0"):
        layers.append(lrelu(conv3d(layers[-1], ndf * 2, kernel_size=3, strides=1), 0.1))

    with tf.variable_scope("sn_conv1_1"):
        layers.append(lrelu(conv3d(layers[-1], ndf * 4, kernel_size=4, strides=(1, 2, 2)), 0.1))

    with tf.variable_scope("sn_conv2_0"):
        layers.append(lrelu(conv3d(layers[-1], ndf * 4, kernel_size=3, strides=1), 0.1))

    with tf.variable_scope("sn_conv2_1"):
        layers.append(lrelu(conv3d(layers[-1], ndf * 8, kernel_size=4, strides=2), 0.1))

    with tf.variable_scope("sn_conv3_0"):
        layers.append(lrelu(conv3d(layers[-1], ndf * 8, kernel_size=3, strides=1), 0.1))

    with tf.variable_scope("sn_fc4"):
        logits = dense(tf.reshape(layers[-1], [batch_size, -1]), 1, use_spectral_norm=True)
        layers.append(logits)
    layers = nest.map_structure(tf_utils.transpose_batch_time, layers)
    return layers
