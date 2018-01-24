import tensorflow as tf

from video_prediction import ops
from video_prediction.ops import conv2d, conv3d, lrelu


def create_image_discriminator(inputs,
                               ndf=64,
                               norm_layer='instance'):
    norm_layer = ops.get_norm_layer(norm_layer)
    layers = []
    paddings = [[0, 0], [1, 1], [1, 1], [0, 0]]

    with tf.variable_scope("image_layer_1"):
        h1 = conv2d(tf.pad(inputs, paddings), ndf, kernel_size=4, strides=2, padding='VALID')
        h1 = lrelu(h1, 0.2)
        layers.append(h1)

    with tf.variable_scope("image_layer_2"):
        h2 = conv2d(tf.pad(h1, paddings), ndf * 2, kernel_size=4, strides=2, padding='VALID')
        h2 = norm_layer(h2)
        h2 = lrelu(h2, 0.2)
        layers.append(h2)

    with tf.variable_scope("image_layer_3"):
        h3 = conv2d(tf.pad(h2, paddings), ndf * 4, kernel_size=4, strides=2, padding='VALID')
        h3 = norm_layer(h3)
        h3 = lrelu(h3, 0.2)
        layers.append(h3)

    with tf.variable_scope("image_layer_4"):
        logits = conv2d(h3, 1, kernel_size=4, strides=1, padding='VALID')
        layers.append(logits)
    return layers[-1]


def create_video_discriminator(inputs,
                               ndf=64,
                               norm_layer='instance'):
    inputs = tf.transpose(inputs, [1, 0] + list(range(2, inputs.shape.ndims)))
    norm_layer = ops.get_norm_layer(norm_layer)
    layers = []
    paddings = [[0, 0], [0, 0], [1, 1], [1, 1], [0, 0]]

    with tf.variable_scope("video_layer_1"):
        h1 = conv3d(tf.pad(inputs, paddings), ndf, kernel_size=4, strides=(1, 2, 2), padding='VALID')
        h1 = lrelu(h1, 0.2)
        layers.append(h1)

    with tf.variable_scope("video_layer_2"):
        h2 = conv3d(tf.pad(h1, paddings), ndf * 2, kernel_size=4, strides=(1, 2, 2), padding='VALID')
        h2 = norm_layer(h2)
        h2 = lrelu(h2, 0.2)
        layers.append(h2)

    with tf.variable_scope("video_layer_3"):
        h3 = conv3d(tf.pad(h2, paddings), ndf * 4, kernel_size=4, strides=(1, 2, 2), padding='VALID')
        h3 = norm_layer(h3)
        h3 = lrelu(h3, 0.2)
        layers.append(h3)

    with tf.variable_scope("video_layer_4"):
        logits = conv3d(h3, 1, kernel_size=(1, 4, 4), strides=1, padding='VALID')
        layers.append(logits)
    return tf.transpose(layers[-1], [1, 0] + list(range(2, inputs.shape.ndims)))


def discriminator_fn(targets, inputs=None, hparams=None):
    batch_size = targets.shape[0].value
    t_sample = tf.random_uniform([batch_size], minval=0, maxval=targets.shape[0].value, dtype=tf.int32)
    images_sample = tf.gather_nd(targets, tf.stack([t_sample, tf.range(batch_size)], axis=1))
    # assume that the clips have the same length as the targets
    clips_sample = targets

    outputs = {}
    if hparams.image_gan_weight or hparams.image_vae_gan_weight:
        image_logits = create_image_discriminator(images_sample, ndf=hparams.ndf, norm_layer=hparams.norm_layer)
        outputs['discrim_image_logits'] = image_logits
    if hparams.video_gan_weight or hparams.video_vae_gan_weight:
        video_logits = create_video_discriminator(clips_sample, ndf=hparams.ndf, norm_layer=hparams.norm_layer)
        outputs['discrim_video_logits'] = video_logits
    return None, outputs
