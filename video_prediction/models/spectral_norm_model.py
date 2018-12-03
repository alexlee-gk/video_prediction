import tensorflow as tf
from tensorflow.python.util import nest

from video_prediction import ops
from video_prediction.ops import dense, lrelu, flatten
from video_prediction.utils import tf_utils


def create_image_sn_discriminator(images,
                                  ndf=64):
    batch_size = images.shape[0].value
    layers = []
    paddings = [[0, 0], [1, 1], [1, 1], [0, 0]]

    def conv2d(inputs, *args, **kwargs):
        kwargs.setdefault('padding', 'VALID')
        kwargs.setdefault('use_spectral_norm', True)
        return ops.conv2d(tf.pad(inputs, paddings), *args, **kwargs)

    with tf.variable_scope("image_sn_conv0_0"):
        layers.append(lrelu(conv2d(images, ndf, kernel_size=3, strides=1), 0.1))

    with tf.variable_scope("image_sn_conv0_1"):
        layers.append(lrelu(conv2d(layers[-1], ndf * 2, kernel_size=4, strides=2), 0.1))

    with tf.variable_scope("image_sn_conv1_0"):
        layers.append(lrelu(conv2d(layers[-1], ndf * 2, kernel_size=3, strides=1), 0.1))

    with tf.variable_scope("image_sn_conv1_1"):
        layers.append(lrelu(conv2d(layers[-1], ndf * 4, kernel_size=4, strides=2), 0.1))

    with tf.variable_scope("image_sn_conv2_0"):
        layers.append(lrelu(conv2d(layers[-1], ndf * 4, kernel_size=3, strides=1), 0.1))

    with tf.variable_scope("image_sn_conv2_1"):
        layers.append(lrelu(conv2d(layers[-1], ndf * 8, kernel_size=4, strides=2), 0.1))

    with tf.variable_scope("image_sn_conv3_0"):
        layers.append(lrelu(conv2d(layers[-1], ndf * 8, kernel_size=3, strides=1), 0.1))

    with tf.variable_scope("image_sn_fc4"):
        logits = dense(tf.reshape(layers[-1], [batch_size, -1]), 1, use_spectral_norm=True)
        layers.append(logits)
    return layers


def create_video_sn_discriminator(clips,
                                  ndf=64):
    clips = tf_utils.transpose_batch_time(clips)
    batch_size = clips.shape[0].value
    layers = []
    paddings = [[0, 0], [1, 1], [1, 1], [1, 1], [0, 0]]

    def conv3d(inputs, *args, **kwargs):
        kwargs.setdefault('padding', 'VALID')
        kwargs.setdefault('use_spectral_norm', True)
        return ops.conv3d(tf.pad(inputs, paddings), *args, **kwargs)

    with tf.variable_scope("video_sn_conv0_0"):
        layers.append(lrelu(conv3d(clips, ndf, kernel_size=3, strides=1), 0.1))

    with tf.variable_scope("video_sn_conv0_1"):
        layers.append(lrelu(conv3d(layers[-1], ndf * 2, kernel_size=4, strides=(1, 2, 2)), 0.1))

    with tf.variable_scope("video_sn_conv1_0"):
        layers.append(lrelu(conv3d(layers[-1], ndf * 2, kernel_size=3, strides=1), 0.1))

    with tf.variable_scope("video_sn_conv1_1"):
        layers.append(lrelu(conv3d(layers[-1], ndf * 4, kernel_size=4, strides=(1, 2, 2)), 0.1))

    with tf.variable_scope("video_sn_conv2_0"):
        layers.append(lrelu(conv3d(layers[-1], ndf * 4, kernel_size=3, strides=1), 0.1))

    with tf.variable_scope("video_sn_conv2_1"):
        layers.append(lrelu(conv3d(layers[-1], ndf * 8, kernel_size=4, strides=2), 0.1))

    with tf.variable_scope("video_sn_conv3_0"):
        layers.append(lrelu(conv3d(layers[-1], ndf * 8, kernel_size=3, strides=1), 0.1))

    with tf.variable_scope("video_sn_fc4"):
        logits = dense(tf.reshape(layers[-1], [batch_size, -1]), 1, use_spectral_norm=True)
        layers.append(logits)
    layers = nest.map_structure(tf_utils.transpose_batch_time, layers)
    return layers


def discriminator_fn(targets, inputs=None, hparams=None):
    batch_size = targets.shape[1].value
    # sort of hack to ensure that the same t_sample is used for all the
    # discriminators that are given the same inputs
    if 't_sample' in inputs:
        t_sample = inputs['t_sample']
    else:
        t_sample = tf.random_uniform([batch_size], minval=0, maxval=targets.shape[0].value, dtype=tf.int32)
        inputs['t_sample'] = t_sample
    image_sample = tf.gather_nd(targets, tf.stack([t_sample, tf.range(batch_size)], axis=1))

    if 't_start' in inputs:
        t_start = inputs['t_start']
    else:
        t_start = tf.random_uniform([batch_size], minval=0, maxval=targets.shape[0].value - hparams.clip_length + 1, dtype=tf.int32)
        inputs['t_start'] = t_start
    t_start_indices = tf.stack([t_start, tf.range(batch_size)], axis=1)
    t_offset_indices = tf.stack([tf.range(hparams.clip_length), tf.zeros(hparams.clip_length, dtype=tf.int32)], axis=1)
    indices = tf.expand_dims(t_start_indices, axis=0) + tf.expand_dims(t_offset_indices, axis=1)
    clip_sample = tf.reshape(tf.gather_nd(targets, flatten(indices, 0, 1)), [hparams.clip_length] + targets.shape.as_list()[1:])

    outputs = {}
    if hparams.image_sn_gan_weight or hparams.image_sn_vae_gan_weight:
        image_features = create_image_sn_discriminator(image_sample, ndf=hparams.ndf)
        image_features, image_logits = image_features[:-1], image_features[-1]
        outputs['discrim_image_sn_logits'] = tf.expand_dims(image_logits, axis=0)  # expand dims for the time dimension
        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            images_features = create_image_sn_discriminator(flatten(targets, 0, 1), ndf=hparams.ndf)
        images_features = images_features[:-1]
        for i, images_feature in enumerate(images_features):
            images_feature = tf.reshape(images_feature, targets.shape[:2].as_list() + images_feature.shape[1:].as_list())
            outputs['discrim_image_sn_feature%d' % i] = images_feature
    if hparams.video_sn_gan_weight or hparams.video_sn_vae_gan_weight:
        video_features = create_video_sn_discriminator(clip_sample, ndf=hparams.ndf)
        video_features, video_logits = video_features[:-1], video_features[-1]
        outputs['discrim_video_sn_logits'] = video_logits
        for i, video_feature in enumerate(video_features):
            outputs['discrim_video_sn_feature%d' % i] = video_feature
    if hparams.images_sn_gan_weight or hparams.images_sn_vae_gan_weight:
        # assume single-image discriminator is not used since otherwise variable scopes would collide
        assert not (hparams.image_sn_gan_weight or hparams.image_sn_vae_gan_weight)
        images_sample = tf.reshape(clip_sample, [batch_size * hparams.clip_length] + clip_sample.shape.as_list()[2:])
        images_features = create_image_sn_discriminator(images_sample, ndf=hparams.ndf)
        images_features, images_logits = images_features[:-1], images_features[-1]
        outputs['discrim_images_sn_logits'] = tf.reshape(images_logits, [batch_size, hparams.clip_length] + images_logits.shape.as_list()[1:])
    return None, outputs
