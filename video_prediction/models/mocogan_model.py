import tensorflow as tf
from tensorflow.python.util import nest

from video_prediction import ops
from video_prediction.ops import conv2d, conv3d, lrelu, tile_concat, flatten
from video_prediction.utils import tf_utils


def create_image_discriminator(images,
                               ndf=64,
                               norm_layer='instance'):
    norm_layer = ops.get_norm_layer(norm_layer)
    layers = []
    paddings = [[0, 0], [1, 1], [1, 1], [0, 0]]

    with tf.variable_scope("image_layer_1"):
        h1 = conv2d(tf.pad(images, paddings), ndf, kernel_size=4, strides=2, padding='VALID')
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
    return layers


def create_video_discriminator(clips,
                               ndf=64,
                               norm_layer='instance'):
    norm_layer = ops.get_norm_layer(norm_layer)
    layers = []
    paddings = [[0, 0], [0, 0], [1, 1], [1, 1], [0, 0]]

    clips = tf_utils.transpose_batch_time(clips)

    with tf.variable_scope("video_layer_1"):
        h1 = conv3d(tf.pad(clips, paddings), ndf, kernel_size=4, strides=(1, 2, 2), padding='VALID')
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
        if h3.shape[1].value < 4:
            kernel_size = (h3.shape[1].value, 4, 4)
        else:
            kernel_size = 4
        logits = conv3d(h3, 1, kernel_size=kernel_size, strides=1, padding='VALID')
        layers.append(logits)
    return nest.map_structure(tf_utils.transpose_batch_time, layers)


def create_acvideo_discriminator(clips,
                                 actions,
                                 ndf=64,
                                 norm_layer='instance'):
    norm_layer = ops.get_norm_layer(norm_layer)
    layers = []
    paddings = [[0, 0], [0, 0], [1, 1], [1, 1], [0, 0]]

    clip_pairs = tf.concat([clips[:-1], clips[1:]], axis=-1)
    clip_pairs = tile_concat([clip_pairs, actions[..., None, None, :]], axis=-1)
    clip_pairs = tf_utils.transpose_batch_time(clip_pairs)

    with tf.variable_scope("acvideo_layer_1"):
        h1 = conv3d(tf.pad(clip_pairs, paddings), ndf, kernel_size=(3, 4, 4), strides=(1, 2, 2), padding='VALID')
        h1 = lrelu(h1, 0.2)
        layers.append(h1)

    with tf.variable_scope("acvideo_layer_2"):
        h2 = conv3d(tf.pad(h1, paddings), ndf * 2, kernel_size=(3, 4, 4), strides=(1, 2, 2), padding='VALID')
        h2 = norm_layer(h2)
        h2 = lrelu(h2, 0.2)
        layers.append(h2)

    with tf.variable_scope("acvideo_layer_3"):
        h3 = conv3d(tf.pad(h2, paddings), ndf * 4, kernel_size=(3, 4, 4), strides=(1, 2, 2), padding='VALID')
        h3 = norm_layer(h3)
        h3 = lrelu(h3, 0.2)
        layers.append(h3)

    with tf.variable_scope("acvideo_layer_4"):
        if h3.shape[1].value < 4:
            kernel_size = (h3.shape[1].value, 4, 4)
        else:
            kernel_size = 4
        logits = conv3d(h3, 1, kernel_size=kernel_size, strides=1, padding='VALID')
        layers.append(logits)
    return nest.map_structure(tf_utils.transpose_batch_time, layers)


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
    # assume that the clips have the same length as the targets
    clip_sample = targets

    outputs = {}
    if hparams.image_gan_weight or hparams.image_vae_gan_weight:
        image_features = create_image_discriminator(image_sample, ndf=hparams.ndf, norm_layer=hparams.norm_layer)
        image_features, image_logits = image_features[:-1], image_features[-1]
        outputs['discrim_image_logits'] = tf.expand_dims(image_logits, axis=0)  # expand dims for the time dimension
        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            images_features = create_image_discriminator(flatten(targets, 0, 1), ndf=hparams.ndf, norm_layer=hparams.norm_layer)
        images_features = images_features[:-1]
        for i, images_feature in enumerate(images_features):
            images_feature = tf.reshape(images_feature, targets.shape[:2].as_list() + images_feature.shape[1:].as_list())
            outputs['discrim_images_feature%d' % i] = images_feature
    if hparams.video_gan_weight or hparams.video_vae_gan_weight:
        video_features = create_video_discriminator(clip_sample, ndf=hparams.ndf, norm_layer=hparams.norm_layer)
        video_features, video_logits = video_features[:-1], video_features[-1]
        outputs['discrim_video_logits'] = video_logits
        for i, video_feature in enumerate(video_features):
            outputs['discrim_video_feature%d' % i] = video_feature
    if hparams.acvideo_gan_weight or hparams.acvideo_vae_gan_weight:
        actions_sample = inputs['actions'][hparams.context_frames:]
        acvideo_features = create_acvideo_discriminator(clip_sample, actions_sample, ndf=hparams.ndf, norm_layer=hparams.norm_layer)
        acvideo_features, acvideo_logits = acvideo_features[:-1], acvideo_features[-1]
        outputs['discrim_acvideo_logits'] = acvideo_logits
        for i, acvideo_feature in enumerate(acvideo_features):
            outputs['discrim_acvideo_feature%d' % i] = acvideo_feature
    return None, outputs
