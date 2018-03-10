import itertools

import tensorflow as tf
from tensorflow.python.util import nest

from video_prediction import ops
from video_prediction.models import VideoPredictionModel
from video_prediction.ops import conv3d
from video_prediction.ops import lrelu, conv2d, flatten, tile_concat, pool2d, deconv2d
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
    if hparams.image_gan_weight or hparams.image_vae_gan_weight:
        image_features = create_image_discriminator(image_sample, ndf=hparams.ndf, norm_layer=hparams.norm_layer)
        image_features, image_logits = image_features[:-1], image_features[-1]
        outputs['discrim_image_logits'] = tf.expand_dims(image_logits, axis=0)  # expand dims for the time dimension
        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            images_features = create_image_discriminator(flatten(targets, 0, 1), ndf=hparams.ndf, norm_layer=hparams.norm_layer)
        images_features = images_features[:-1]
        for i, images_feature in enumerate(images_features):
            images_feature = tf.reshape(images_feature, targets.shape[:2].as_list() + images_feature.shape[1:].as_list())
            outputs['discrim_image_feature%d' % i] = images_feature
    if hparams.video_gan_weight or hparams.video_vae_gan_weight:
        video_features = create_video_discriminator(clip_sample, ndf=hparams.ndf, norm_layer=hparams.norm_layer)
        video_features, video_logits = video_features[:-1], video_features[-1]
        outputs['discrim_video_logits'] = video_logits
        for i, video_feature in enumerate(video_features):
            outputs['discrim_video_feature%d' % i] = video_feature
    if hparams.acvideo_gan_weight or hparams.acvideo_vae_gan_weight:
        t_offset_indices = tf.stack([tf.range(hparams.clip_length - 1), tf.zeros(hparams.clip_length - 1, dtype=tf.int32)], axis=1)
        indices = tf.expand_dims(t_start_indices, axis=0) + tf.expand_dims(t_offset_indices, axis=1)
        actions = inputs['actions'][hparams.context_frames:]
        actions_sample = tf.reshape(tf.gather_nd(actions, flatten(indices, 0, 1)), [hparams.clip_length - 1] + actions.shape.as_list()[1:])
        acvideo_features = create_acvideo_discriminator(clip_sample, actions_sample, ndf=hparams.ndf, norm_layer=hparams.norm_layer)
        acvideo_features, acvideo_logits = acvideo_features[:-1], acvideo_features[-1]
        outputs['discrim_acvideo_logits'] = acvideo_logits
        for i, acvideo_feature in enumerate(acvideo_features):
            outputs['discrim_acvideo_feature%d' % i] = acvideo_feature
    return None, outputs


def create_encoder(image,
                   nef=64,
                   norm_layer='instance',
                   dim_z=10):
    norm_layer = ops.get_norm_layer(norm_layer)
    layers = []
    paddings = [[0, 0], [1, 1], [1, 1], [0, 0]]

    with tf.variable_scope("layer_1"):
        h0 = conv2d(tf.pad(image, paddings), nef, kernel_size=4, strides=2, padding='VALID')
        h0 = norm_layer(h0)
        h0 = lrelu(h0, 0.2)
        layers.append(h0)

    with tf.variable_scope("layer_2"):
        h1 = conv2d(tf.pad(h0, paddings), nef * 2, kernel_size=4, strides=2, padding='VALID')
        h1 = norm_layer(h1)
        h1 = lrelu(h1, 0.2)
        layers.append(h1)

    with tf.variable_scope("layer_3"):
        h2 = conv2d(tf.pad(h1, paddings), nef * 4, kernel_size=4, strides=2, padding='VALID')
        h2 = norm_layer(h2)
        h2 = lrelu(h2, 0.2)
        layers.append(h2)

    with tf.variable_scope("layer_4"):
        h3 = conv2d(tf.pad(h2, paddings), nef * 8, kernel_size=4, strides=2, padding='VALID')
        h3 = norm_layer(h3)
        h3 = lrelu(h3, 0.2)
        layers.append(h3)

    with tf.variable_scope("layer_5"):
        h4 = conv2d(tf.pad(h3, paddings), dim_z, kernel_size=4, strides=2, padding='VALID')
        layers.append(h4)

    pooled = pool2d(h4, h4.shape[1:3].as_list(), padding='VALID', pool_mode='avg')
    squeezed = tf.squeeze(pooled, [1, 2])
    return squeezed


def create_generator(z,
                     ngf=64,
                     norm_layer='instance',
                     n_channels=3):
    norm_layer = ops.get_norm_layer(norm_layer)
    layers = []

    with tf.variable_scope("layer_1"):
        h0 = deconv2d(z, ngf * 8, kernel_size=4, strides=1, padding='VALID')
        h0 = norm_layer(h0)
        h0 = tf.nn.relu(h0)
        layers.append(h0)

    with tf.variable_scope("layer_2"):
        h1 = deconv2d(h0, ngf * 4, kernel_size=4, strides=2)
        h1 = norm_layer(h1)
        h1 = tf.nn.relu(h1)
        layers.append(h1)

    with tf.variable_scope("layer_3"):
        h2 = deconv2d(h1, ngf * 2, kernel_size=4, strides=2)
        h2 = norm_layer(h2)
        h2 = tf.nn.relu(h2)
        layers.append(h2)

    with tf.variable_scope("layer_4"):
        h3 = deconv2d(h2, ngf, kernel_size=4, strides=2)
        h3 = norm_layer(h3)
        h3 = tf.nn.relu(h3)
        layers.append(h3)

    with tf.variable_scope("layer_5"):
        h4 = deconv2d(h3, n_channels, kernel_size=4, strides=2)
        h4 = tf.nn.tanh(h4)
        layers.append(h4)
    return h4


def generator_fn(inputs, hparams=None):
    batch_size = inputs['images'].shape[1].value
    inputs = {name: tf_utils.maybe_pad_or_slice(input, hparams.sequence_length - 1)
              for name, input in inputs.items()}

    with tf.variable_scope('gru'):
        gru_cell = tf.nn.rnn_cell.GRUCell(hparams.dim_z_motion)

    if hparams.context_frames:
        with tf.variable_scope('content_encoder'):
            z_c = create_encoder(inputs['images'][0],  # first context image for content encoder
                                 nef=hparams.nef, norm_layer=hparams.norm_layer, dim_z=hparams.dim_z_content)
        with tf.variable_scope('initial_motion_encoder'):
            h_0 = create_encoder(inputs['images'][hparams.context_frames - 1],  # last context image for motion encoder
                                 nef=hparams.nef, norm_layer=hparams.norm_layer, dim_z=hparams.dim_z_motion)
    else:  # unconditional case
        z_c = tf.random_normal([batch_size, hparams.dim_z_content])
        h_0 = gru_cell.zero_state(batch_size, tf.float32)

    h_t = [h_0]
    for t in range(hparams.context_frames - 1, hparams.sequence_length - 1):
        with tf.variable_scope('gru', reuse=t > hparams.context_frames - 1):
            e_t = tf.random_normal([batch_size, hparams.dim_z_motion])
            if 'actions' in inputs:
                e_t = tf.concat(inputs['actions'][t], axis=-1)
            h_t.append(gru_cell(e_t, h_t[-1])[1])  # the output and state is the same in GRUs
    z_m = tf.stack(h_t[1:], axis=0)
    z = tf.concat([tf.tile(z_c[None, :, :], [hparams.sequence_length - hparams.context_frames, 1, 1]), z_m], axis=-1)

    z_flatten = flatten(z[:, :, None, None, :], 0, 1)
    gen_images_flatten = create_generator(z_flatten, ngf=hparams.ngf, norm_layer=hparams.norm_layer)
    gen_images = tf.reshape(gen_images_flatten, [-1, batch_size] + gen_images_flatten.shape.as_list()[1:])

    outputs = {'gen_images': gen_images}
    return gen_images, outputs


class MoCoGANVideoPredictionModel(VideoPredictionModel):
    def __init__(self, *args, **kwargs):
        super(MoCoGANVideoPredictionModel, self).__init__(
            generator_fn, discriminator_fn, *args, **kwargs)
        self.deterministic = False

    def get_default_hparams_dict(self):
        default_hparams = super(MoCoGANVideoPredictionModel, self).get_default_hparams_dict()
        hparams = dict(
            l1_weight=10.0,
            l2_weight=0.0,
            image_gan_weight=1.0,
            video_gan_weight=1.0,
            ndf=64,
            ngf=64,
            nef=64,
            dim_z_content=50,
            dim_z_motion=10,
            norm_layer='batch',
            clip_length=10,
            lr=0.0002,
            beta1=0.5,
            beta2=0.999,
        )
        return dict(itertools.chain(default_hparams.items(), hparams.items()))
