import itertools
from collections import OrderedDict

import numpy as np
import tensorflow as tf

from video_prediction import ops
from video_prediction import tf_utils
from video_prediction.models import VideoPredictionModel
from video_prediction.ops import conv2d, lrelu


def create_generator(generator_inputs,
                     output_nc=3,
                     ngf=64,
                     norm_layer='instance',
                     downsample_layer='conv_pool2d',
                     upsample_layer='upsample_conv2d'):
    norm_layer = ops.get_norm_layer(norm_layer)
    downsample_layer = ops.get_downsample_layer(downsample_layer)
    upsample_layer = ops.get_upsample_layer(upsample_layer)

    layers = []
    inputs = generator_inputs

    scale_size = min(*inputs.shape.as_list()[1:3])
    if scale_size == 256:
        layer_specs = [
            (ngf, 2),      # encoder_1: [batch, 256, 256, in_channels] => [batch, 128, 128, ngf]
            (ngf * 2, 2),  # encoder_2: [batch, 128, 128, ngf] => [batch, 64, 64, ngf * 2]
            (ngf * 4, 2),  # encoder_3: [batch, 64, 64, ngf * 2] => [batch, 32, 32, ngf * 4]
            (ngf * 8, 2),  # encoder_4: [batch, 32, 32, ngf * 4] => [batch, 16, 16, ngf * 8]
            (ngf * 8, 2),  # encoder_5: [batch, 16, 16, ngf * 8] => [batch, 8, 8, ngf * 8]
            (ngf * 8, 2),  # encoder_6: [batch, 8, 8, ngf * 8] => [batch, 4, 4, ngf * 8]
            (ngf * 8, 2),  # encoder_7: [batch, 4, 4, ngf * 8] => [batch, 2, 2, ngf * 8]
            (ngf * 8, 2),  # encoder_8: [batch, 2, 2, ngf * 8] => [batch, 1, 1, ngf * 8]
        ]
    elif scale_size == 128:
        layer_specs = [
            (ngf, 2),
            (ngf * 2, 2),
            (ngf * 4, 2),
            (ngf * 8, 2),
            (ngf * 8, 2),
            (ngf * 8, 2),
            (ngf * 8, 2),
        ]
    elif scale_size == 64:
        layer_specs = [
            (ngf, 2),
            (ngf * 2, 2),
            (ngf * 4, 2),
            (ngf * 8, 2),
            (ngf * 8, 2),
            (ngf * 8, 2),
        ]
    else:
        raise NotImplementedError

    with tf.variable_scope("encoder_1"):
        out_channels, strides = layer_specs[0]
        if strides == 1:
            output = conv2d(inputs, out_channels, kernel_size=4)
        else:
            output = downsample_layer(inputs, out_channels, kernel_size=4, strides=strides)
        layers.append(output)

    for out_channels, strides in layer_specs[1:]:
        with tf.variable_scope("encoder_%d" % (len(layers) + 1)):
            rectified = lrelu(layers[-1], 0.2)
            # [batch, in_height, in_width, in_channels] => [batch, in_height/2, in_width/2, out_channels]
            if strides == 1:
                convolved = conv2d(rectified, out_channels, kernel_size=4)
            else:
                convolved = downsample_layer(rectified, out_channels, kernel_size=4, strides=strides)
            output = norm_layer(convolved)
            layers.append(output)

    if scale_size == 256:
        layer_specs = [
            (ngf * 8, 2, 0.5),    # decoder_8: [batch, 1, 1, ngf * 8] => [batch, 2, 2, ngf * 8 * 2]
            (ngf * 8, 2, 0.5),    # decoder_7: [batch, 2, 2, ngf * 8 * 2] => [batch, 4, 4, ngf * 8 * 2]
            (ngf * 8, 2, 0.5),    # decoder_6: [batch, 4, 4, ngf * 8 * 2] => [batch, 8, 8, ngf * 8 * 2]
            (ngf * 8, 2, 0.0),    # decoder_5: [batch, 8, 8, ngf * 8 * 2] => [batch, 16, 16, ngf * 8 * 2]
            (ngf * 4, 2, 0.0),    # decoder_4: [batch, 16, 16, ngf * 8 * 2] => [batch, 32, 32, ngf * 4 * 2]
            (ngf * 2, 2, 0.0),    # decoder_3: [batch, 32, 32, ngf * 4 * 2] => [batch, 64, 64, ngf * 2 * 2]
            (ngf, 2, 0.0),        # decoder_2: [batch, 64, 64, ngf * 2 * 2] => [batch, 128, 128, ngf * 2]
            (output_nc, 2, 0.0),  # decoder_1: [batch, 128, 128, ngf * 2] => [batch, 256, 256, generator_outputs_channels]
        ]
    elif scale_size == 128:
        layer_specs = [
            (ngf * 8, 2, 0.5),
            (ngf * 8, 2, 0.5),
            (ngf * 8, 2, 0.5),
            (ngf * 4, 2, 0.0),
            (ngf * 2, 2, 0.0),
            (ngf, 2, 0.0),
            (output_nc, 2, 0.0),
        ]
    elif scale_size == 64:
        layer_specs = [
            (ngf * 8, 2, 0.5),
            (ngf * 8, 2, 0.5),
            (ngf * 4, 2, 0.0),
            (ngf * 2, 2, 0.0),
            (ngf, 2, 0.0),
            (output_nc, 2, 0.0),
        ]
    else:
        raise NotImplementedError

    num_encoder_layers = len(layers)
    for decoder_layer, (out_channels, stride, dropout) in enumerate(layer_specs[:-1]):
        skip_layer = num_encoder_layers - decoder_layer - 1
        with tf.variable_scope("decoder_%d" % (skip_layer + 1)):
            if decoder_layer == 0:
                # first decoder layer doesn't have skip connections
                # since it is directly connected to the skip_layer
                input = layers[-1]
            else:
                input = tf.concat([layers[-1], layers[skip_layer]], axis=3)

            rectified = tf.nn.relu(input)
            # [batch, in_height, in_width, in_channels] => [batch, in_height*2, in_width*2, out_channels]
            if stride == 1:
                output = conv2d(rectified, out_channels, kernel_size=4)
            else:
                output = upsample_layer(rectified, out_channels, kernel_size=4, strides=strides)
            output = norm_layer(output)

            if dropout > 0.0:
                output = tf.nn.dropout(output, keep_prob=1 - dropout)

            layers.append(output)

    with tf.variable_scope("decoder_1"):
        out_channels, stride, dropout = layer_specs[-1]
        assert dropout == 0.0  # no dropout at the last layer
        input = tf.concat([layers[-1], layers[0]], axis=3)
        rectified = tf.nn.relu(input)
        if stride == 1:
            output = conv2d(rectified, out_channels, kernel_size=4)
        else:
            output = upsample_layer(rectified, out_channels, kernel_size=4, strides=strides)
        output = tf.tanh(output)
        output = (output + 1) / 2
        layers.append(output)

    return layers[-1]


def create_s_layer_discriminator(discrim_targets, discrim_inputs=None,
                                 ndf=64,
                                 norm_layer='instance',
                                 downsample_layer='conv_pool2d'):
    norm_layer = ops.get_norm_layer(norm_layer)
    downsample_layer = ops.get_downsample_layer(downsample_layer)

    layers = []
    inputs = [discrim_targets]
    if discrim_inputs is not None:
        inputs.append(discrim_inputs)
    inputs = tf.concat(inputs, axis=-1)

    scale_size = min(*inputs.shape.as_list()[1:3])
    if scale_size == 256:
        layer_specs = [
            (ndf, 2),      # layer_1: [batch, 256, 256, in_channels * 2] => [batch, 128, 128, ndf]
            (ndf * 2, 2),  # layer_2: [batch, 128, 128, ndf] => [batch, 64, 64, ndf * 2]
            (ndf * 4, 2),  # layer_3: [batch, 64, 64, ndf * 2] => [batch, 32, 32, ndf * 4]
            (ndf * 8, 1),  # layer_4: [batch, 32, 32, ndf * 4] => [batch, 32, 32, ndf * 8]
            (1, 1),        # layer_5: [batch, 32, 32, ndf * 8] => [batch, 32, 32, 1]
        ]
    elif scale_size == 128:
        layer_specs = [
            (ndf, 2),
            (ndf * 2, 2),
            (ndf * 4, 1),
            (ndf * 8, 1),
            (1, 1),
        ]
    elif scale_size == 64:
        layer_specs = [
            (ndf, 2),
            (ndf * 2, 1),
            (ndf * 4, 1),
            (ndf * 8, 1),
            (1, 1),
        ]
    else:
        raise NotImplementedError

    with tf.variable_scope("layer_1"):
        out_channels, strides = layer_specs[0]
        convolved = downsample_layer(inputs, out_channels, kernel_size=4, strides=strides)
        rectified = lrelu(convolved, 0.2)
        layers.append(rectified)

    for out_channels, strides in layer_specs[1:-1]:
        with tf.variable_scope("layer_%d" % (len(layers) + 1)):
            if strides == 1:
                convolved = conv2d(layers[-1], out_channels, kernel_size=4)
            else:
                convolved = downsample_layer(layers[-1], out_channels, kernel_size=4, strides=strides)
            normalized = norm_layer(convolved)
            rectified = lrelu(normalized, 0.2)
            layers.append(rectified)

    with tf.variable_scope("layer_%d" % (len(layers) + 1)):
        out_channels, strides = layer_specs[-1]
        if strides == 1:
            logits = conv2d(rectified, out_channels, kernel_size=4)
        else:
            logits = downsample_layer(rectified, out_channels, kernel_size=4, strides=strides)
        layers.append(logits)  # don't apply sigmoid to the logits in case we want to use LSGAN

    return layers[-1]


def create_n_layer_discriminator(discrim_targets, discrim_inputs=None,
                                 ndf=64,
                                 n_layers=3,
                                 norm_layer='instance'):
    norm_layer = ops.get_norm_layer(norm_layer)

    layers = []
    inputs = [discrim_targets]
    if discrim_inputs is not None:
        inputs.append(discrim_inputs)
    inputs = tf.concat(inputs, axis=-1)

    paddings = [[0, 0], [1, 1], [1, 1], [0, 0]]

    # layer_1: [batch, 256, 256, in_channels * 2] => [batch, 128, 128, ndf]
    with tf.variable_scope("layer_1"):
        convolved = conv2d(tf.pad(inputs, paddings), ndf, kernel_size=4, strides=2, padding='VALID')
        rectified = lrelu(convolved, 0.2)
        layers.append(rectified)

    # layer_2: [batch, 128, 128, ndf] => [batch, 64, 64, ndf * 2]
    # layer_3: [batch, 64, 64, ndf * 2] => [batch, 32, 32, ndf * 4]
    # layer_4: [batch, 32, 32, ndf * 4] => [batch, 31, 31, ndf * 8]
    for i in range(n_layers):
        with tf.variable_scope("layer_%d" % (len(layers) + 1)):
            out_channels = ndf * min(2**(i+1), 8)
            stride = 1 if i == n_layers - 1 else 2  # last layer here has stride 1
            convolved = conv2d(tf.pad(layers[-1], paddings), out_channels, kernel_size=4, strides=stride, padding='VALID')
            normalized = norm_layer(convolved)
            rectified = lrelu(normalized, 0.2)
            layers.append(rectified)

    # layer_5: [batch, 31, 31, ndf * 8] => [batch, 30, 30, 1]
    with tf.variable_scope("layer_%d" % (len(layers) + 1)):
        logits = conv2d(tf.pad(rectified, paddings), 1, kernel_size=4, strides=1, padding='VALID')
        layers.append(logits)  # don't apply sigmoid to the logits in case we want to use LSGAN
    return layers[-1]


def create_discriminator(discrim_targets, discrim_inputs=None, d_net='s_layer', **kwargs):
    if d_net == 's_layer':
        kwargs.pop('n_layers', None)  # unused
        logits = create_s_layer_discriminator(discrim_targets, discrim_inputs, **kwargs)
    elif d_net == 'n_layer':
        kwargs.pop('downsample_layer', None)  # unused
        n_layers = kwargs.pop('n_layers', None)
        if not n_layers:
            scale_size = min(*discrim_targets.shape.as_list()[1:3])
            n_layers = int(np.log2(scale_size // 32))
        logits = create_n_layer_discriminator(discrim_targets, discrim_inputs, n_layers=n_layers, **kwargs)
    else:
        raise NotImplementedError
    return logits


class Pix2PixCell(tf.nn.rnn_cell.RNNCell):
    def __init__(self, inputs, hparams, reuse=None):
        super(Pix2PixCell, self).__init__(_reuse=reuse)
        self.inputs = inputs
        self.hparams = hparams
        _, batch_size, *image_shape = inputs['images'].shape.as_list()
        if 'actions' in inputs:
            _, _, *action_shape = inputs['actions'].shape.as_list()
            gen_input_shape = list(image_shape[:-1]) + [image_shape[-1] + action_shape[-1]]
        else:
            gen_input_shape = image_shape
        self._output_size = (tf.TensorShape(image_shape),  # gen_image
                             tf.TensorShape(gen_input_shape))  # gen_input
        self._state_size = (tf.TensorShape([]),  # time
                            tf.TensorShape(image_shape))  # gen_image

        if self.hparams.schedule_sampling_k == -1:
            self.ground_truth_conds = tf.constant(False)
        else:
            # Scheduled sampling
            k = self.hparams.schedule_sampling_k
            iter_num = tf.to_float(tf.train.get_or_create_global_step())
            prob = (k / (k + tf.exp(iter_num / k)))
            log_probs = tf.log([1 - prob, prob])
            ground_truth_conds = tf.multinomial([log_probs] * batch_size,
                                                self.hparams.sequence_length - self.hparams.context_frames)
            ground_truth_conds = tf.cast(tf.transpose(ground_truth_conds, [1, 0]), dtype=tf.bool)
            # Ensure that eventually, the model is deterministically
            # autoregressive (as opposed to autoregressive with very high probability).
            self.ground_truth_conds = tf.cond(tf.less(prob, 0.001),
                                              lambda: tf.constant(False, shape=ground_truth_conds.shape),
                                              lambda: ground_truth_conds)

    @property
    def output_size(self):
        return self._output_size

    @property
    def state_size(self):
        return self._state_size

    def call(self, inputs, states):
        image = inputs['images']
        time, gen_image = states

        with tf.control_dependencies([tf.assert_equal(time[1:], time[0])]):
            t = tf.identity(time[0])
            t = tf.to_int32(t)
        done_warm_start = tf.greater_equal(t, self.hparams.context_frames)
        if self.hparams.schedule_sampling_k == -1:
            image = tf.cond(done_warm_start,
                            lambda: gen_image,  # feed in generated image
                            lambda: image)  # feed in ground_truth
        else:
            ground_truth_cond = self.ground_truth_conds[t - self.hparams.context_frames]
            image = tf.cond(done_warm_start,
                            lambda: tf.where(ground_truth_cond, image, gen_image),  # schedule sampling
                            lambda: image)  # feed in ground_truth
        if 'actions' in inputs:
            action = inputs['actions']
            gen_input = ops.tile_concat([image, action[:, None, None, :]], axis=-1)
        else:
            gen_input = image
        gen_image = create_generator(gen_input,
                                     output_nc=self.hparams.output_nc,
                                     ngf=self.hparams.ngf,
                                     norm_layer=self.hparams.norm_layer,
                                     downsample_layer=self.hparams.downsample_layer,
                                     upsample_layer=self.hparams.upsample_layer)
        outputs = (gen_image, gen_input)
        new_states = (time + 1, gen_image)
        return outputs, new_states


def generator_fn(inputs, hparams=None):
    batch_size = inputs['images'].shape.as_list()[1]
    cell = Pix2PixCell(inputs, hparams)
    inputs = OrderedDict([(name, tf_utils.maybe_pad_or_slice(input, hparams.sequence_length - 1))
                          for name, input in inputs.items() if name in ('images', 'actions')])
    (gen_images, gen_inputs), _ = \
        tf.nn.dynamic_rnn(cell, inputs, sequence_length=[hparams.sequence_length - 1] * batch_size,
                          dtype=tf.float32, swap_memory=False, time_major=True)
    # the RNN outputs generated images from time step 1 to sequence_length,
    # but generator_fn should only return images past context_frames
    gen_images = gen_images[hparams.context_frames - 1:]
    gen_inputs = gen_inputs[hparams.context_frames - 1:]
    return gen_images, {'gen_images': gen_images, 'gen_inputs': gen_inputs,
                        'ground_truth_conds_mean': tf.reduce_mean(tf.to_float(cell.ground_truth_conds))}


def discriminator_fn(targets, inputs=None, hparams=None):
    if targets.shape.ndims == 4:
        logits = create_discriminator(targets, inputs,
                                      d_net=hparams.d_net,
                                      n_layers=hparams.n_layers,
                                      ndf=hparams.ndf,
                                      norm_layer=hparams.norm_layer,
                                      downsample_layer=hparams.d_downsample_layer)
    else:
        if inputs is None:
            targets_and_inputs = (targets,)
        else:
            targets_and_inputs = (targets, inputs['gen_inputs'])
        logits = tf.map_fn(lambda args: discriminator_fn(*args, hparams=hparams)[0],
                           targets_and_inputs,
                           dtype=tf.float32, swap_memory=False)
    return logits, {'discrim_logits': logits}


class Pix2PixVideoPredictionModel(VideoPredictionModel):
    def __init__(self, *args, **kwargs):
        super(Pix2PixVideoPredictionModel, self).__init__(
            generator_fn, discriminator_fn, *args, **kwargs)

    def get_default_hparams_dict(self):
        default_hparams = super(Pix2PixVideoPredictionModel, self).get_default_hparams_dict()
        hparams = dict(
            d_net='s_layer',
            n_layers=0,
            output_nc=3,
            ngf=64,
            ndf=64,
            norm_layer='instance',
            d_downsample_layer='conv_pool2d',
            downsample_layer='conv_pool2d',
            upsample_layer='upsample_conv2d',
            schedule_sampling_k=900.0,
        )
        return dict(itertools.chain(default_hparams.items(), hparams.items()))
