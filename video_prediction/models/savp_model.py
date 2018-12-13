import collections
import functools
import itertools
from collections import OrderedDict

import numpy as np
import tensorflow as tf
from tensorflow.python.util import nest

from video_prediction import ops, flow_ops
from video_prediction.models import VideoPredictionModel
from video_prediction.models import networks
from video_prediction.ops import dense, pad2d, conv2d, flatten, tile_concat
from video_prediction.rnn_ops import BasicConv2DLSTMCell, Conv2DGRUCell
from video_prediction.utils import tf_utils

# Amount to use when lower bounding tensors
RELU_SHIFT = 1e-12


def posterior_fn(inputs, hparams):
    images = inputs['images']
    image_pairs = tf.concat([images[:-1], images[1:]], axis=-1)
    if 'actions' in inputs:
        image_pairs = tile_concat(
            [image_pairs, inputs['actions'][..., None, None, :]], axis=-1)

    h = tf_utils.with_flat_batch(networks.encoder)(
        image_pairs, nef=hparams.nef, n_layers=hparams.n_layers, norm_layer=hparams.norm_layer)

    if hparams.use_e_rnn:
        with tf.variable_scope('layer_%d' % (hparams.n_layers + 1)):
            h = tf_utils.with_flat_batch(dense, 2)(h, hparams.nef * 4)

        if hparams.rnn == 'lstm':
            RNNCell = tf.contrib.rnn.BasicLSTMCell
        elif hparams.rnn == 'gru':
            RNNCell = tf.contrib.rnn.GRUCell
        else:
            raise NotImplementedError
        with tf.variable_scope('%s' % hparams.rnn):
            rnn_cell = RNNCell(hparams.nef * 4)
            h, _ = tf_utils.unroll_rnn(rnn_cell, h)

    with tf.variable_scope('z_mu'):
        z_mu = tf_utils.with_flat_batch(dense, 2)(h, hparams.nz)
    with tf.variable_scope('z_log_sigma_sq'):
        z_log_sigma_sq = tf_utils.with_flat_batch(dense, 2)(h, hparams.nz)
        z_log_sigma_sq = tf.clip_by_value(z_log_sigma_sq, -10, 10)
    outputs = {'zs_mu': z_mu, 'zs_log_sigma_sq': z_log_sigma_sq}
    return outputs


def prior_fn(inputs, hparams):
    images = inputs['images']
    image_pairs = tf.concat([images[:hparams.context_frames - 1], images[1:hparams.context_frames]], axis=-1)
    if 'actions' in inputs:
        image_pairs = tile_concat(
            [image_pairs, inputs['actions'][..., None, None, :]], axis=-1)

    h = tf_utils.with_flat_batch(networks.encoder)(
        image_pairs, nef=hparams.nef, n_layers=hparams.n_layers, norm_layer=hparams.norm_layer)
    h_zeros = tf.zeros(tf.concat([[hparams.sequence_length - hparams.context_frames], tf.shape(h)[1:]], axis=0))
    h = tf.concat([h, h_zeros], axis=0)

    with tf.variable_scope('layer_%d' % (hparams.n_layers + 1)):
        h = tf_utils.with_flat_batch(dense, 2)(h, hparams.nef * 4)

    if hparams.rnn == 'lstm':
        RNNCell = tf.contrib.rnn.BasicLSTMCell
    elif hparams.rnn == 'gru':
        RNNCell = tf.contrib.rnn.GRUCell
    else:
        raise NotImplementedError
    with tf.variable_scope('%s' % hparams.rnn):
        rnn_cell = RNNCell(hparams.nef * 4)
        h, _ = tf_utils.unroll_rnn(rnn_cell, h)

    with tf.variable_scope('z_mu'):
        z_mu = tf_utils.with_flat_batch(dense, 2)(h, hparams.nz)
    with tf.variable_scope('z_log_sigma_sq'):
        z_log_sigma_sq = tf_utils.with_flat_batch(dense, 2)(h, hparams.nz)
        z_log_sigma_sq = tf.clip_by_value(z_log_sigma_sq, -10, 10)
    outputs = {'zs_mu': z_mu, 'zs_log_sigma_sq': z_log_sigma_sq}
    return outputs


def discriminator_given_video_fn(targets, hparams):
    sequence_length, batch_size = targets.shape.as_list()[:2]
    clip_length = hparams.clip_length

    # sample an image and apply the image distriminator on that frame
    t_sample = tf.random_uniform([batch_size], minval=0, maxval=sequence_length, dtype=tf.int32)
    image_sample = tf.gather_nd(targets, tf.stack([t_sample, tf.range(batch_size)], axis=1))

    # sample a subsequence of length clip_length and apply the images/video discriminators on those frames
    t_start = tf.random_uniform([batch_size], minval=0, maxval=sequence_length - clip_length + 1, dtype=tf.int32)
    t_start_indices = tf.stack([t_start, tf.range(batch_size)], axis=1)
    t_offset_indices = tf.stack([tf.range(clip_length), tf.zeros(clip_length, dtype=tf.int32)], axis=1)
    indices = t_start_indices[None] + t_offset_indices[:, None]
    clip_sample = tf.gather_nd(targets, flatten(indices, 0, 1))
    clip_sample = tf.reshape(clip_sample, [clip_length] + targets.shape.as_list()[1:])

    outputs = {}
    if hparams.image_sn_gan_weight or hparams.image_sn_vae_gan_weight:
        with tf.variable_scope('image'):
            image_features = networks.image_sn_discriminator(image_sample, ndf=hparams.ndf)
            image_features, image_logits = image_features[:-1], image_features[-1]
            outputs['discrim_image_sn_logits'] = image_logits
            for i, image_feature in enumerate(image_features):
                outputs['discrim_image_sn_feature%d' % i] = image_feature
    if hparams.video_sn_gan_weight or hparams.video_sn_vae_gan_weight:
        with tf.variable_scope('video'):
            video_features = networks.video_sn_discriminator(clip_sample, ndf=hparams.ndf)
            video_features, video_logits = video_features[:-1], video_features[-1]
            outputs['discrim_video_sn_logits'] = video_logits
            for i, video_feature in enumerate(video_features):
                outputs['discrim_video_sn_feature%d' % i] = video_feature
    if hparams.images_sn_gan_weight or hparams.images_sn_vae_gan_weight:
        with tf.variable_scope('images'):
            images_features = tf_utils.with_flat_batch(networks.image_sn_discriminator)(clip_sample, ndf=hparams.ndf)
            images_features, images_logits = images_features[:-1], images_features[-1]
            outputs['discrim_images_sn_logits'] = images_logits
            for i, images_feature in enumerate(images_features):
                outputs['discrim_images_sn_feature%d' % i] = images_feature
    return outputs


def discriminator_fn(inputs, outputs, mode, hparams):
    # do the encoder version first so that it isn't affected by the reuse_variables() call
    if hparams.nz == 0:
        discrim_outputs_enc_real = collections.OrderedDict()
        discrim_outputs_enc_fake = collections.OrderedDict()
    else:
        images_enc_real = inputs['images'][1:]
        images_enc_fake = outputs['gen_images_enc']
        if hparams.use_same_discriminator:
            with tf.name_scope("real"):
                discrim_outputs_enc_real = discriminator_given_video_fn(images_enc_real, hparams)
            tf.get_variable_scope().reuse_variables()
            with tf.name_scope("fake"):
                discrim_outputs_enc_fake = discriminator_given_video_fn(images_enc_fake, hparams)
        else:
            with tf.variable_scope('encoder'), tf.name_scope("real"):
                discrim_outputs_enc_real = discriminator_given_video_fn(images_enc_real, hparams)
            with tf.variable_scope('encoder', reuse=True), tf.name_scope("fake"):
                discrim_outputs_enc_fake = discriminator_given_video_fn(images_enc_fake, hparams)

    images_real = inputs['images'][1:]
    images_fake = outputs['gen_images']
    with tf.name_scope("real"):
        discrim_outputs_real = discriminator_given_video_fn(images_real, hparams)
    tf.get_variable_scope().reuse_variables()
    with tf.name_scope("fake"):
        discrim_outputs_fake = discriminator_given_video_fn(images_fake, hparams)

    discrim_outputs_real = OrderedDict([(k + '_real', v) for k, v in discrim_outputs_real.items()])
    discrim_outputs_fake = OrderedDict([(k + '_fake', v) for k, v in discrim_outputs_fake.items()])
    discrim_outputs_enc_real = OrderedDict([(k + '_enc_real', v) for k, v in discrim_outputs_enc_real.items()])
    discrim_outputs_enc_fake = OrderedDict([(k + '_enc_fake', v) for k, v in discrim_outputs_enc_fake.items()])
    outputs = [discrim_outputs_real, discrim_outputs_fake,
               discrim_outputs_enc_real, discrim_outputs_enc_fake]
    total_num_outputs = sum([len(output) for output in outputs])
    outputs = collections.OrderedDict(itertools.chain(*[output.items() for output in outputs]))
    assert len(outputs) == total_num_outputs  # ensure no output is lost because of repeated keys
    return outputs


class SAVPCell(tf.nn.rnn_cell.RNNCell):
    def __init__(self, inputs, mode, hparams, reuse=None):
        super(SAVPCell, self).__init__(_reuse=reuse)
        self.inputs = inputs
        self.mode = mode
        self.hparams = hparams

        if self.hparams.where_add not in ('input', 'all', 'middle'):
            raise ValueError('Invalid where_add %s' % self.hparams.where_add)

        batch_size = inputs['images'].shape[1].value
        image_shape = inputs['images'].shape.as_list()[2:]
        height, width, _ = image_shape
        scale_size = min(height, width)
        if scale_size >= 256:
            self.encoder_layer_specs = [
                (self.hparams.ngf, False),
                (self.hparams.ngf * 2, False),
                (self.hparams.ngf * 4, True),
                (self.hparams.ngf * 8, True),
                (self.hparams.ngf * 8, True),
            ]
            self.decoder_layer_specs = [
                (self.hparams.ngf * 8, True),
                (self.hparams.ngf * 4, True),
                (self.hparams.ngf * 2, False),
                (self.hparams.ngf, False),
                (self.hparams.ngf, False),
            ]
        elif scale_size >= 128:
            self.encoder_layer_specs = [
                (self.hparams.ngf, False),
                (self.hparams.ngf * 2, True),
                (self.hparams.ngf * 4, True),
                (self.hparams.ngf * 8, True),
            ]
            self.decoder_layer_specs = [
                (self.hparams.ngf * 8, True),
                (self.hparams.ngf * 4, True),
                (self.hparams.ngf * 2, False),
                (self.hparams.ngf, False),
            ]
        elif scale_size >= 64:
            self.encoder_layer_specs = [
                (self.hparams.ngf, True),
                (self.hparams.ngf * 2, True),
                (self.hparams.ngf * 4, True),
            ]
            self.decoder_layer_specs = [
                (self.hparams.ngf * 2, True),
                (self.hparams.ngf, True),
                (self.hparams.ngf, False),
            ]
        elif scale_size >= 32:
            self.encoder_layer_specs = [
                (self.hparams.ngf, True),
                (self.hparams.ngf * 2, True),
            ]
            self.decoder_layer_specs = [
                (self.hparams.ngf, True),
                (self.hparams.ngf, False),
            ]
        else:
            raise NotImplementedError
        assert len(self.encoder_layer_specs) == len(self.decoder_layer_specs)
        total_stride = 2 ** len(self.encoder_layer_specs)
        if (height % total_stride) or (width % total_stride):
            raise ValueError("The image has dimension (%d, %d), but it should be divisible "
                             "by the total stride, which is %d." % (height, width, total_stride))

        # output_size
        num_masks = self.hparams.last_frames * self.hparams.num_transformed_images + \
            int(bool(self.hparams.prev_image_background)) + \
            int(bool(self.hparams.first_image_background and not self.hparams.context_images_background)) + \
            int(bool(self.hparams.last_image_background and not self.hparams.context_images_background)) + \
            int(bool(self.hparams.last_context_image_background and not self.hparams.context_images_background)) + \
            (self.hparams.context_frames if self.hparams.context_images_background else 0) + \
            int(bool(self.hparams.generate_scratch_image))
        output_size = {
            'gen_images': tf.TensorShape(image_shape),
            'transformed_images': tf.TensorShape(image_shape + [num_masks]),
            'masks': tf.TensorShape([height, width, 1, num_masks]),
        }
        if 'pix_distribs' in inputs:
            num_motions = inputs['pix_distribs'].shape[-1].value
            output_size['gen_pix_distribs'] = tf.TensorShape([height, width, num_motions])
            output_size['transformed_pix_distribs'] = tf.TensorShape([height, width, num_motions, num_masks])
        if 'states' in inputs:
            output_size['gen_states'] = inputs['states'].shape[2:]
        if self.hparams.transformation == 'flow':
            output_size['gen_flows'] = tf.TensorShape([height, width, 2, self.hparams.last_frames * self.hparams.num_transformed_images])
            output_size['gen_flows_rgb'] = tf.TensorShape([height, width, 3, self.hparams.last_frames * self.hparams.num_transformed_images])
        self._output_size = output_size

        # state_size
        conv_rnn_state_sizes = []
        conv_rnn_height, conv_rnn_width = height, width
        for out_channels, use_conv_rnn in self.encoder_layer_specs:
            conv_rnn_height //= 2
            conv_rnn_width //= 2
            if use_conv_rnn and not self.hparams.ablation_rnn:
                conv_rnn_state_sizes.append(tf.TensorShape([conv_rnn_height, conv_rnn_width, out_channels]))
        for out_channels, use_conv_rnn in self.decoder_layer_specs:
            conv_rnn_height *= 2
            conv_rnn_width *= 2
            if use_conv_rnn and not self.hparams.ablation_rnn:
                conv_rnn_state_sizes.append(tf.TensorShape([conv_rnn_height, conv_rnn_width, out_channels]))
        if self.hparams.conv_rnn == 'lstm':
            conv_rnn_state_sizes = [tf.nn.rnn_cell.LSTMStateTuple(conv_rnn_state_size, conv_rnn_state_size)
                                    for conv_rnn_state_size in conv_rnn_state_sizes]
        state_size = {'time': tf.TensorShape([]),
                      'gen_image': tf.TensorShape(image_shape),
                      'last_images': [tf.TensorShape(image_shape)] * self.hparams.last_frames,
                      'conv_rnn_states': conv_rnn_state_sizes}
        if 'zs' in inputs and self.hparams.use_rnn_z and not self.hparams.ablation_rnn:
            rnn_z_state_size = tf.TensorShape([self.hparams.nz])
            if self.hparams.rnn == 'lstm':
                rnn_z_state_size = tf.nn.rnn_cell.LSTMStateTuple(rnn_z_state_size, rnn_z_state_size)
            state_size['rnn_z_state'] = rnn_z_state_size
        if 'pix_distribs' in inputs:
            state_size['gen_pix_distrib'] = tf.TensorShape([height, width, num_motions])
            state_size['last_pix_distribs'] = [tf.TensorShape([height, width, num_motions])] * self.hparams.last_frames
        if 'states' in inputs:
            state_size['gen_state'] = inputs['states'].shape[2:]
        self._state_size = state_size

        if self.hparams.learn_initial_state:
            learnable_initial_state_size = {k: v for k, v in state_size.items()
                                            if k in ('conv_rnn_states', 'rnn_z_state')}
        else:
            learnable_initial_state_size = {}
        learnable_initial_state_flat = []
        for i, size in enumerate(nest.flatten(learnable_initial_state_size)):
            with tf.variable_scope('initial_state_%d' % i):
                state = tf.get_variable('initial_state', size,
                                        dtype=tf.float32, initializer=tf.zeros_initializer())
                learnable_initial_state_flat.append(state)
        self._learnable_initial_state = nest.pack_sequence_as(
            learnable_initial_state_size, learnable_initial_state_flat)

        ground_truth_sampling_shape = [self.hparams.sequence_length - 1 - self.hparams.context_frames, batch_size]
        if self.hparams.schedule_sampling == 'none' or self.mode != 'train':
            ground_truth_sampling = tf.constant(False, dtype=tf.bool, shape=ground_truth_sampling_shape)
        elif self.hparams.schedule_sampling in ('inverse_sigmoid', 'linear'):
            if self.hparams.schedule_sampling == 'inverse_sigmoid':
                k = self.hparams.schedule_sampling_k
                start_step = self.hparams.schedule_sampling_steps[0]
                iter_num = tf.to_float(tf.train.get_or_create_global_step())
                prob = (k / (k + tf.exp((iter_num - start_step) / k)))
                prob = tf.cond(tf.less(iter_num, start_step), lambda: 1.0, lambda: prob)
            elif self.hparams.schedule_sampling == 'linear':
                start_step, end_step = self.hparams.schedule_sampling_steps
                step = tf.clip_by_value(tf.train.get_or_create_global_step(), start_step, end_step)
                prob = 1.0 - tf.to_float(step - start_step) / tf.to_float(end_step - start_step)
            log_probs = tf.log([1 - prob, prob])
            ground_truth_sampling = tf.multinomial([log_probs] * batch_size, ground_truth_sampling_shape[0])
            ground_truth_sampling = tf.cast(tf.transpose(ground_truth_sampling, [1, 0]), dtype=tf.bool)
            # Ensure that eventually, the model is deterministically
            # autoregressive (as opposed to autoregressive with very high probability).
            ground_truth_sampling = tf.cond(tf.less(prob, 0.001),
                                            lambda: tf.constant(False, dtype=tf.bool, shape=ground_truth_sampling_shape),
                                            lambda: ground_truth_sampling)
        else:
            raise NotImplementedError
        ground_truth_context = tf.constant(True, dtype=tf.bool, shape=[self.hparams.context_frames, batch_size])
        self.ground_truth = tf.concat([ground_truth_context, ground_truth_sampling], axis=0)

    @property
    def output_size(self):
        return self._output_size

    @property
    def state_size(self):
        return self._state_size

    def zero_state(self, batch_size, dtype):
        init_state = super(SAVPCell, self).zero_state(batch_size, dtype)
        learnable_init_state = nest.map_structure(
            lambda x: tf.tile(x[None], [batch_size] + [1] * x.shape.ndims), self._learnable_initial_state)
        init_state.update(learnable_init_state)
        init_state['last_images'] = [self.inputs['images'][0]] * self.hparams.last_frames
        if 'pix_distribs' in self.inputs:
            init_state['last_pix_distribs'] = [self.inputs['pix_distribs'][0]] * self.hparams.last_frames
        return init_state

    def _rnn_func(self, inputs, state, num_units):
        if self.hparams.rnn == 'lstm':
            RNNCell = functools.partial(tf.nn.rnn_cell.LSTMCell, name='basic_lstm_cell')
        elif self.hparams.rnn == 'gru':
            RNNCell = tf.contrib.rnn.GRUCell
        else:
            raise NotImplementedError
        rnn_cell = RNNCell(num_units, reuse=tf.get_variable_scope().reuse)
        return rnn_cell(inputs, state)

    def _conv_rnn_func(self, inputs, state, filters):
        if isinstance(inputs, (list, tuple)):
            inputs_shape = inputs[0].shape.as_list()
        else:
            inputs_shape = inputs.shape.as_list()
        input_shape = inputs_shape[1:]
        if self.hparams.conv_rnn_norm_layer == 'none':
            normalizer_fn = None
        else:
            normalizer_fn = ops.get_norm_layer(self.hparams.conv_rnn_norm_layer)
        if self.hparams.conv_rnn == 'lstm':
            Conv2DRNNCell = BasicConv2DLSTMCell
        elif self.hparams.conv_rnn == 'gru':
            Conv2DRNNCell = Conv2DGRUCell
        else:
            raise NotImplementedError
        if self.hparams.ablation_conv_rnn_norm:
            conv_rnn_cell = Conv2DRNNCell(input_shape, filters, kernel_size=(5, 5),
                                          reuse=tf.get_variable_scope().reuse)
            h, state = conv_rnn_cell(inputs, state)
            outputs = (normalizer_fn(h), state)
        else:
            conv_rnn_cell = Conv2DRNNCell(input_shape, filters, kernel_size=(5, 5),
                                          normalizer_fn=normalizer_fn,
                                          separate_norms=self.hparams.conv_rnn_norm_layer == 'layer',
                                          reuse=tf.get_variable_scope().reuse)
            outputs = conv_rnn_cell(inputs, state)
        return outputs

    def call(self, inputs, states):
        norm_layer = ops.get_norm_layer(self.hparams.norm_layer)
        downsample_layer = ops.get_downsample_layer(self.hparams.downsample_layer)
        upsample_layer = ops.get_upsample_layer(self.hparams.upsample_layer)
        activation_layer = ops.get_activation_layer(self.hparams.activation_layer)
        image_shape = inputs['images'].get_shape().as_list()
        batch_size, height, width, color_channels = image_shape
        conv_rnn_states = states['conv_rnn_states']

        time = states['time']
        with tf.control_dependencies([tf.assert_equal(time[1:], time[0])]):
            t = tf.to_int32(tf.identity(time[0]))

        image = tf.where(self.ground_truth[t], inputs['images'], states['gen_image'])  # schedule sampling (if any)
        last_images = states['last_images'][1:] + [image]
        if 'pix_distribs' in inputs:
            pix_distrib = tf.where(self.ground_truth[t], inputs['pix_distribs'], states['gen_pix_distrib'])
            last_pix_distribs = states['last_pix_distribs'][1:] + [pix_distrib]
        if 'states' in inputs:
            state = tf.where(self.ground_truth[t], inputs['states'], states['gen_state'])

        state_action = []
        state_action_z = []
        if 'actions' in inputs:
            state_action.append(inputs['actions'])
            state_action_z.append(inputs['actions'])
        if 'states' in inputs:
            state_action.append(state)
            # don't backpropagate the convnet through the state dynamics
            state_action_z.append(tf.stop_gradient(state))

        if 'zs' in inputs:
            if self.hparams.use_rnn_z:
                with tf.variable_scope('%s_z' % ('fc' if self.hparams.ablation_rnn else self.hparams.rnn)):
                    if self.hparams.ablation_rnn:
                        rnn_z = dense(inputs['zs'], self.hparams.nz)
                        rnn_z = tf.nn.tanh(rnn_z)
                    else:
                        rnn_z, rnn_z_state = self._rnn_func(inputs['zs'], states['rnn_z_state'], self.hparams.nz)
                state_action_z.append(rnn_z)
            else:
                state_action_z.append(inputs['zs'])

        def concat(tensors, axis):
            if len(tensors) == 0:
                return tf.zeros([batch_size, 0])
            elif len(tensors) == 1:
                return tensors[0]
            else:
                return tf.concat(tensors, axis=axis)
        state_action = concat(state_action, axis=-1)
        state_action_z = concat(state_action_z, axis=-1)

        layers = []
        new_conv_rnn_states = []
        for i, (out_channels, use_conv_rnn) in enumerate(self.encoder_layer_specs):
            with tf.variable_scope('h%d' % i):
                if i == 0:
                    h = tf.concat([image, self.inputs['images'][0]], axis=-1)
                    kernel_size = (5, 5)
                else:
                    h = layers[-1][-1]
                    kernel_size = (3, 3)
                if self.hparams.where_add == 'all' or (self.hparams.where_add == 'input' and i == 0):
                    if self.hparams.use_tile_concat:
                        h = tile_concat([h, state_action_z[:, None, None, :]], axis=-1)
                    else:
                        h = [h, state_action_z]
                h = _maybe_tile_concat_layer(downsample_layer)(
                    h, out_channels, kernel_size=kernel_size, strides=(2, 2))
                h = norm_layer(h)
                h = activation_layer(h)
            if use_conv_rnn:
                with tf.variable_scope('%s_h%d' % ('conv' if self.hparams.ablation_rnn else self.hparams.conv_rnn, i)):
                    if self.hparams.where_add == 'all':
                        if self.hparams.use_tile_concat:
                            conv_rnn_h = tile_concat([h, state_action_z[:, None, None, :]], axis=-1)
                        else:
                            conv_rnn_h = [h, state_action_z]
                    else:
                        conv_rnn_h = h
                    if self.hparams.ablation_rnn:
                        conv_rnn_h = _maybe_tile_concat_layer(conv2d)(
                            conv_rnn_h, out_channels, kernel_size=(5, 5))
                        conv_rnn_h = norm_layer(conv_rnn_h)
                        conv_rnn_h = activation_layer(conv_rnn_h)
                    else:
                        conv_rnn_state = conv_rnn_states[len(new_conv_rnn_states)]
                        conv_rnn_h, conv_rnn_state = self._conv_rnn_func(conv_rnn_h, conv_rnn_state, out_channels)
                        new_conv_rnn_states.append(conv_rnn_state)
            layers.append((h, conv_rnn_h) if use_conv_rnn else (h,))

        num_encoder_layers = len(layers)
        for i, (out_channels, use_conv_rnn) in enumerate(self.decoder_layer_specs):
            with tf.variable_scope('h%d' % len(layers)):
                if i == 0:
                    h = layers[-1][-1]
                else:
                    h = tf.concat([layers[-1][-1], layers[num_encoder_layers - i - 1][-1]], axis=-1)
                if self.hparams.where_add == 'all' or (self.hparams.where_add == 'middle' and i == 0):
                    if self.hparams.use_tile_concat:
                        h = tile_concat([h, state_action_z[:, None, None, :]], axis=-1)
                    else:
                        h = [h, state_action_z]
                h = _maybe_tile_concat_layer(upsample_layer)(
                    h, out_channels, kernel_size=(3, 3), strides=(2, 2))
                h = norm_layer(h)
                h = activation_layer(h)
            if use_conv_rnn:
                with tf.variable_scope('%s_h%d' % ('conv' if self.hparams.ablation_rnn else self.hparams.conv_rnn, len(layers))):
                    if self.hparams.where_add == 'all':
                        if self.hparams.use_tile_concat:
                            conv_rnn_h = tile_concat([h, state_action_z[:, None, None, :]], axis=-1)
                        else:
                            conv_rnn_h = [h, state_action_z]
                    else:
                        conv_rnn_h = h
                    if self.hparams.ablation_rnn:
                        conv_rnn_h = _maybe_tile_concat_layer(conv2d)(conv_rnn_h, out_channels, kernel_size=(5, 5))
                        conv_rnn_h = norm_layer(conv_rnn_h)
                        conv_rnn_h = activation_layer(conv_rnn_h)
                    else:
                        conv_rnn_state = conv_rnn_states[len(new_conv_rnn_states)]
                        conv_rnn_h, conv_rnn_state = self._conv_rnn_func(conv_rnn_h, conv_rnn_state, out_channels)
                        new_conv_rnn_states.append(conv_rnn_state)
            layers.append((h, conv_rnn_h) if use_conv_rnn else (h,))
        assert len(new_conv_rnn_states) == len(conv_rnn_states)

        if self.hparams.last_frames and self.hparams.num_transformed_images:
            if self.hparams.transformation == 'flow':
                with tf.variable_scope('h%d_flow' % len(layers)):
                    h_flow = conv2d(layers[-1][-1], self.hparams.ngf, kernel_size=(3, 3), strides=(1, 1))
                    h_flow = norm_layer(h_flow)
                    h_flow = activation_layer(h_flow)

                with tf.variable_scope('flows'):
                    flows = conv2d(h_flow, 2 * self.hparams.last_frames * self.hparams.num_transformed_images, kernel_size=(3, 3), strides=(1, 1))
                    flows = tf.reshape(flows, [batch_size, height, width, 2, self.hparams.last_frames * self.hparams.num_transformed_images])
            else:
                assert len(self.hparams.kernel_size) == 2
                kernel_shape = list(self.hparams.kernel_size) + [self.hparams.last_frames * self.hparams.num_transformed_images]
                if self.hparams.transformation == 'dna':
                    with tf.variable_scope('h%d_dna_kernel' % len(layers)):
                        h_dna_kernel = conv2d(layers[-1][-1], self.hparams.ngf, kernel_size=(3, 3), strides=(1, 1))
                        h_dna_kernel = norm_layer(h_dna_kernel)
                        h_dna_kernel = activation_layer(h_dna_kernel)

                    # Using largest hidden state for predicting untied conv kernels.
                    with tf.variable_scope('dna_kernels'):
                        kernels = conv2d(h_dna_kernel, np.prod(kernel_shape), kernel_size=(3, 3), strides=(1, 1))
                        kernels = tf.reshape(kernels, [batch_size, height, width] + kernel_shape)
                        kernels = kernels + identity_kernel(self.hparams.kernel_size)[None, None, None, :, :, None]
                    kernel_spatial_axes = [3, 4]
                elif self.hparams.transformation == 'cdna':
                    with tf.variable_scope('cdna_kernels'):
                        smallest_layer = layers[num_encoder_layers - 1][-1]
                        kernels = dense(flatten(smallest_layer), np.prod(kernel_shape))
                        kernels = tf.reshape(kernels, [batch_size] + kernel_shape)
                        kernels = kernels + identity_kernel(self.hparams.kernel_size)[None, :, :, None]
                    kernel_spatial_axes = [1, 2]
                else:
                    raise ValueError('Invalid transformation %s' % self.hparams.transformation)

            if self.hparams.transformation != 'flow':
                with tf.name_scope('kernel_normalization'):
                    kernels = tf.nn.relu(kernels - RELU_SHIFT) + RELU_SHIFT
                    kernels /= tf.reduce_sum(kernels, axis=kernel_spatial_axes, keepdims=True)

        if self.hparams.generate_scratch_image:
            with tf.variable_scope('h%d_scratch' % len(layers)):
                h_scratch = conv2d(layers[-1][-1], self.hparams.ngf, kernel_size=(3, 3), strides=(1, 1))
                h_scratch = norm_layer(h_scratch)
                h_scratch = activation_layer(h_scratch)

            # Using largest hidden state for predicting a new image layer.
            # This allows the network to also generate one image from scratch,
            # which is useful when regions of the image become unoccluded.
            with tf.variable_scope('scratch_image'):
                scratch_image = conv2d(h_scratch, color_channels, kernel_size=(3, 3), strides=(1, 1))
                scratch_image = tf.nn.sigmoid(scratch_image)

        with tf.name_scope('transformed_images'):
            transformed_images = []
            if self.hparams.last_frames and self.hparams.num_transformed_images:
                if self.hparams.transformation == 'flow':
                    transformed_images.extend(apply_flows(last_images, flows))
                else:
                    transformed_images.extend(apply_kernels(last_images, kernels, self.hparams.dilation_rate))
            if self.hparams.prev_image_background:
                transformed_images.append(image)
            if self.hparams.first_image_background and not self.hparams.context_images_background:
                transformed_images.append(self.inputs['images'][0])
            if self.hparams.last_image_background and not self.hparams.context_images_background:
                transformed_images.append(self.inputs['images'][self.hparams.context_frames - 1])
            if self.hparams.last_context_image_background and not self.hparams.context_images_background:
                last_context_image = tf.cond(
                    tf.less(t, self.hparams.context_frames),
                    lambda: self.inputs['images'][t],
                    lambda: self.inputs['images'][self.hparams.context_frames - 1])
                transformed_images.append(last_context_image)
            if self.hparams.context_images_background:
                transformed_images.extend(tf.unstack(self.inputs['images'][:self.hparams.context_frames]))
            if self.hparams.generate_scratch_image:
                transformed_images.append(scratch_image)

        if 'pix_distribs' in inputs:
            with tf.name_scope('transformed_pix_distribs'):
                transformed_pix_distribs = []
                if self.hparams.last_frames and self.hparams.num_transformed_images:
                    if self.hparams.transformation == 'flow':
                        transformed_pix_distribs.extend(apply_flows(last_pix_distribs, flows))
                    else:
                        transformed_pix_distribs.extend(apply_kernels(last_pix_distribs, kernels, self.hparams.dilation_rate))
                if self.hparams.prev_image_background:
                    transformed_pix_distribs.append(pix_distrib)
                if self.hparams.first_image_background and not self.hparams.context_images_background:
                    transformed_pix_distribs.append(self.inputs['pix_distribs'][0])
                if self.hparams.last_image_background and not self.hparams.context_images_background:
                    transformed_pix_distribs.append(self.inputs['pix_distribs'][self.hparams.context_frames - 1])
                if self.hparams.last_context_image_background and not self.hparams.context_images_background:
                    last_context_pix_distrib = tf.cond(
                        tf.less(t, self.hparams.context_frames),
                        lambda: self.inputs['pix_distribs'][t],
                        lambda: self.inputs['pix_distribs'][self.hparams.context_frames - 1])
                    transformed_pix_distribs.append(last_context_pix_distrib)
                if self.hparams.context_images_background:
                    transformed_pix_distribs.extend(tf.unstack(self.inputs['pix_distribs'][:self.hparams.context_frames]))
                if self.hparams.generate_scratch_image:
                    transformed_pix_distribs.append(pix_distrib)

        with tf.name_scope('masks'):
            if len(transformed_images) > 1:
                with tf.variable_scope('h%d_masks' % len(layers)):
                    h_masks = conv2d(layers[-1][-1], self.hparams.ngf, kernel_size=(3, 3), strides=(1, 1))
                    h_masks = norm_layer(h_masks)
                    h_masks = activation_layer(h_masks)

                with tf.variable_scope('masks'):
                    if self.hparams.dependent_mask:
                        h_masks = tf.concat([h_masks] + transformed_images, axis=-1)
                    masks = conv2d(h_masks, len(transformed_images), kernel_size=(3, 3), strides=(1, 1))
                    masks = tf.nn.softmax(masks)
                    masks = tf.split(masks, len(transformed_images), axis=-1)
            elif len(transformed_images) == 1:
                masks = [tf.ones([batch_size, height, width, 1])]
            else:
                raise ValueError("Either one of the following should be true: "
                                 "last_frames and num_transformed_images, first_image_background, "
                                 "prev_image_background, generate_scratch_image")

        with tf.name_scope('gen_images'):
            assert len(transformed_images) == len(masks)
            gen_image = tf.add_n([transformed_image * mask
                                  for transformed_image, mask in zip(transformed_images, masks)])

        if 'pix_distribs' in inputs:
            with tf.name_scope('gen_pix_distribs'):
                assert len(transformed_pix_distribs) == len(masks)
                gen_pix_distrib = tf.add_n([transformed_pix_distrib * mask
                                            for transformed_pix_distrib, mask in zip(transformed_pix_distribs, masks)])
                gen_pix_distrib /= tf.reduce_sum(gen_pix_distrib, axis=(1, 2), keepdims=True)

        if 'states' in inputs:
            with tf.name_scope('gen_states'):
                with tf.variable_scope('state_pred'):
                    gen_state = dense(state_action, inputs['states'].shape[-1].value)

        outputs = {'gen_images': gen_image,
                   'transformed_images': tf.stack(transformed_images, axis=-1),
                   'masks': tf.stack(masks, axis=-1)}
        if 'pix_distribs' in inputs:
            outputs['gen_pix_distribs'] = gen_pix_distrib
            outputs['transformed_pix_distribs'] = tf.stack(transformed_pix_distribs, axis=-1)
        if 'states' in inputs:
            outputs['gen_states'] = gen_state
        if self.hparams.transformation == 'flow':
            outputs['gen_flows'] = flows
            flows_transposed = tf.transpose(flows, [0, 1, 2, 4, 3])
            flows_rgb_transposed = tf_utils.flow_to_rgb(flows_transposed)
            flows_rgb = tf.transpose(flows_rgb_transposed, [0, 1, 2, 4, 3])
            outputs['gen_flows_rgb'] = flows_rgb

        new_states = {'time': time + 1,
                      'gen_image': gen_image,
                      'last_images': last_images,
                      'conv_rnn_states': new_conv_rnn_states}
        if 'zs' in inputs and self.hparams.use_rnn_z and not self.hparams.ablation_rnn:
            new_states['rnn_z_state'] = rnn_z_state
        if 'pix_distribs' in inputs:
            new_states['gen_pix_distrib'] = gen_pix_distrib
            new_states['last_pix_distribs'] = last_pix_distribs
        if 'states' in inputs:
            new_states['gen_state'] = gen_state
        return outputs, new_states


def generator_given_z_fn(inputs, mode, hparams):
    # all the inputs needs to have the same length for unrolling the rnn
    inputs = {name: tf_utils.maybe_pad_or_slice(input, hparams.sequence_length - 1)
              for name, input in inputs.items()}
    cell = SAVPCell(inputs, mode, hparams)
    outputs, _ = tf_utils.unroll_rnn(cell, inputs)
    outputs['ground_truth_sampling_mean'] = tf.reduce_mean(tf.to_float(cell.ground_truth[hparams.context_frames:]))
    return outputs


def generator_fn(inputs, mode, hparams):
    batch_size = tf.shape(inputs['images'])[1]

    if hparams.nz == 0:
        # no zs is given in inputs
        outputs = generator_given_z_fn(inputs, mode, hparams)
    else:
        zs_shape = [hparams.sequence_length - 1, batch_size, hparams.nz]

        # posterior
        with tf.variable_scope('encoder'):
            outputs_posterior = posterior_fn(inputs, hparams)
            eps = tf.random_normal(zs_shape, 0, 1)
            zs_posterior = outputs_posterior['zs_mu'] + tf.sqrt(tf.exp(outputs_posterior['zs_log_sigma_sq'])) * eps
        inputs_posterior = dict(inputs)
        inputs_posterior['zs'] = zs_posterior

        # prior
        if hparams.learn_prior:
            with tf.variable_scope('prior'):
                outputs_prior = prior_fn(inputs, hparams)
            eps = tf.random_normal(zs_shape, 0, 1)
            zs_prior = outputs_prior['zs_mu'] + tf.sqrt(tf.exp(outputs_prior['zs_log_sigma_sq'])) * eps
        else:
            outputs_prior = {}
            zs_prior = tf.random_normal([hparams.sequence_length - hparams.context_frames] + zs_shape[1:], 0, 1)
            zs_prior = tf.concat([zs_posterior[:hparams.context_frames - 1], zs_prior], axis=0)
        inputs_prior = dict(inputs)
        inputs_prior['zs'] = zs_prior

        # generate
        gen_outputs_posterior = generator_given_z_fn(inputs_posterior, mode, hparams)
        tf.get_variable_scope().reuse_variables()
        gen_outputs = generator_given_z_fn(inputs_prior, mode, hparams)

        # rename tensors to avoid name collisions
        output_prior = collections.OrderedDict([(k + '_prior', v) for k, v in outputs_prior.items()])
        outputs_posterior = collections.OrderedDict([(k + '_enc', v) for k, v in outputs_posterior.items()])
        gen_outputs_posterior = collections.OrderedDict([(k + '_enc', v) for k, v in gen_outputs_posterior.items()])

        outputs = [output_prior, gen_outputs, outputs_posterior, gen_outputs_posterior]
        total_num_outputs = sum([len(output) for output in outputs])
        outputs = collections.OrderedDict(itertools.chain(*[output.items() for output in outputs]))
        assert len(outputs) == total_num_outputs  # ensure no output is lost because of repeated keys

        # generate multiple samples from the prior for visualization purposes
        inputs_samples = {
            name: tf.tile(input[:, None], [1, hparams.num_samples] + [1] * (input.shape.ndims - 1))
            for name, input in inputs.items()}
        zs_samples_shape = [hparams.sequence_length - 1, hparams.num_samples, batch_size, hparams.nz]
        if hparams.learn_prior:
            eps = tf.random_normal(zs_samples_shape, 0, 1)
            zs_prior_samples = (outputs_prior['zs_mu'][:, None] +
                                tf.sqrt(tf.exp(outputs_prior['zs_log_sigma_sq']))[:, None] * eps)
        else:
            zs_prior_samples = tf.random_normal(
                [hparams.sequence_length - hparams.context_frames] + zs_samples_shape[1:], 0, 1)
            zs_prior_samples = tf.concat(
                [tf.tile(zs_posterior[:hparams.context_frames - 1][:, None], [1, hparams.num_samples, 1, 1]),
                 zs_prior_samples], axis=0)
        inputs_prior_samples = dict(inputs_samples)
        inputs_prior_samples['zs'] = zs_prior_samples
        inputs_prior_samples = {name: flatten(input, 1, 2) for name, input in inputs_prior_samples.items()}
        gen_outputs_samples = generator_given_z_fn(inputs_prior_samples, mode, hparams)
        gen_images_samples = gen_outputs_samples['gen_images']
        gen_images_samples = tf.stack(tf.split(gen_images_samples, hparams.num_samples, axis=1), axis=-1)
        gen_images_samples_avg = tf.reduce_mean(gen_images_samples, axis=-1)
        outputs['gen_images_samples'] = gen_images_samples
        outputs['gen_images_samples_avg'] = gen_images_samples_avg
    return outputs


class SAVPVideoPredictionModel(VideoPredictionModel):
    def __init__(self, *args, **kwargs):
        super(SAVPVideoPredictionModel, self).__init__(
            generator_fn, discriminator_fn, *args, **kwargs)
        if self.mode != 'train':
            self.discriminator_fn = None
        self.deterministic = not self.hparams.nz

    def get_default_hparams_dict(self):
        default_hparams = super(SAVPVideoPredictionModel, self).get_default_hparams_dict()
        hparams = dict(
            l1_weight=1.0,
            l2_weight=0.0,
            n_layers=3,
            ndf=32,
            norm_layer='instance',
            use_same_discriminator=False,
            ngf=32,
            downsample_layer='conv_pool2d',
            upsample_layer='upsample_conv2d',
            activation_layer='relu',  # for generator only
            transformation='cdna',
            kernel_size=(5, 5),
            dilation_rate=(1, 1),
            where_add='all',
            use_tile_concat=True,
            learn_initial_state=False,
            rnn='lstm',
            conv_rnn='lstm',
            conv_rnn_norm_layer='instance',
            num_transformed_images=4,
            last_frames=1,
            prev_image_background=True,
            first_image_background=True,
            last_image_background=False,
            last_context_image_background=False,
            context_images_background=False,
            generate_scratch_image=True,
            dependent_mask=True,
            schedule_sampling='inverse_sigmoid',
            schedule_sampling_k=900.0,
            schedule_sampling_steps=(0, 100000),
            use_e_rnn=False,
            learn_prior=False,
            nz=8,
            num_samples=8,
            nef=64,
            use_rnn_z=True,
            ablation_conv_rnn_norm=False,
            ablation_rnn=False,
        )
        return dict(itertools.chain(default_hparams.items(), hparams.items()))

    def parse_hparams(self, hparams_dict, hparams):
        # backwards compatibility
        deprecated_hparams_keys = [
            'num_gpus',
            'e_net',
            'd_conditional',
            'd_downsample_layer',
            'd_net',
            'd_use_gt_inputs',
            'acvideo_gan_weight',
            'acvideo_vae_gan_weight',
            'image_gan_weight',
            'image_vae_gan_weight',
            'tuple_gan_weight',
            'tuple_vae_gan_weight',
            'gan_weight',
            'vae_gan_weight',
            'video_gan_weight',
            'video_vae_gan_weight',
        ]
        for deprecated_hparams_key in deprecated_hparams_keys:
            hparams_dict.pop(deprecated_hparams_key, None)
        return super(SAVPVideoPredictionModel, self).parse_hparams(hparams_dict, hparams)

    def restore(self, sess, checkpoints, restore_to_checkpoint_mapping=None):
        def restore_to_checkpoint_mapping(restore_name, checkpoint_var_names):
            restore_name = restore_name.split(':')[0]
            if restore_name not in checkpoint_var_names:
                restore_name = restore_name.replace('savp_cell', 'dna_cell')
            return restore_name

        super(SAVPVideoPredictionModel, self).restore(sess, checkpoints, restore_to_checkpoint_mapping)


def apply_dna_kernels(image, kernels, dilation_rate=(1, 1)):
    """
    Args:
        image: A 4-D tensor of shape
            `[batch, in_height, in_width, in_channels]`.
        kernels: A 6-D of shape
            `[batch, in_height, in_width, kernel_size[0], kernel_size[1], num_transformed_images]`.

    Returns:
        A list of `num_transformed_images` 4-D tensors, each of shape
            `[batch, in_height, in_width, in_channels]`.
    """
    dilation_rate = list(dilation_rate) if isinstance(dilation_rate, (tuple, list)) else [dilation_rate] * 2
    batch_size, height, width, color_channels = image.get_shape().as_list()
    batch_size, height, width, kernel_height, kernel_width, num_transformed_images = kernels.get_shape().as_list()
    kernel_size = [kernel_height, kernel_width]

    # Flatten the spatial dimensions.
    kernels_reshaped = tf.reshape(kernels, [batch_size, height, width,
                                            kernel_size[0] * kernel_size[1], num_transformed_images])
    image_padded = pad2d(image, kernel_size, rate=dilation_rate, padding='SAME', mode='SYMMETRIC')
    # Combine channel and batch dimensions into the first dimension.
    image_transposed = tf.transpose(image_padded, [3, 0, 1, 2])
    image_reshaped = flatten(image_transposed, 0, 1)[..., None]
    patches_reshaped = tf.extract_image_patches(image_reshaped, ksizes=[1] + kernel_size + [1],
                                                strides=[1] * 4, rates=[1] + dilation_rate + [1], padding='VALID')
    # Separate channel and batch dimensions, and move channel dimension.
    patches_transposed = tf.reshape(patches_reshaped, [color_channels, batch_size, height, width, kernel_size[0] * kernel_size[1]])
    patches = tf.transpose(patches_transposed, [1, 2, 3, 0, 4])
    # Reduce along the spatial dimensions of the kernel.
    outputs = tf.matmul(patches, kernels_reshaped)
    outputs = tf.unstack(outputs, axis=-1)
    return outputs


def apply_cdna_kernels(image, kernels, dilation_rate=(1, 1)):
    """
    Args:
        image: A 4-D tensor of shape
            `[batch, in_height, in_width, in_channels]`.
        kernels: A 4-D of shape
            `[batch, kernel_size[0], kernel_size[1], num_transformed_images]`.

    Returns:
        A list of `num_transformed_images` 4-D tensors, each of shape
            `[batch, in_height, in_width, in_channels]`.
    """
    batch_size, height, width, color_channels = image.get_shape().as_list()
    batch_size, kernel_height, kernel_width, num_transformed_images = kernels.get_shape().as_list()
    kernel_size = [kernel_height, kernel_width]
    image_padded = pad2d(image, kernel_size, rate=dilation_rate, padding='SAME', mode='SYMMETRIC')
    # Treat the color channel dimension as the batch dimension since the same
    # transformation is applied to each color channel.
    # Treat the batch dimension as the channel dimension so that
    # depthwise_conv2d can apply a different transformation to each sample.
    kernels = tf.transpose(kernels, [1, 2, 0, 3])
    kernels = tf.reshape(kernels, [kernel_size[0], kernel_size[1], batch_size, num_transformed_images])
    # Swap the batch and channel dimensions.
    image_transposed = tf.transpose(image_padded, [3, 1, 2, 0])
    # Transform image.
    outputs = tf.nn.depthwise_conv2d(image_transposed, kernels, [1, 1, 1, 1], padding='VALID', rate=dilation_rate)
    # Transpose the dimensions to where they belong.
    outputs = tf.reshape(outputs, [color_channels, height, width, batch_size, num_transformed_images])
    outputs = tf.transpose(outputs, [4, 3, 1, 2, 0])
    outputs = tf.unstack(outputs, axis=0)
    return outputs


def apply_kernels(image, kernels, dilation_rate=(1, 1)):
    """
    Args:
        image: A 4-D tensor of shape
            `[batch, in_height, in_width, in_channels]`.
        kernels: A 4-D or 6-D tensor of shape
            `[batch, kernel_size[0], kernel_size[1], num_transformed_images]` or
            `[batch, in_height, in_width, kernel_size[0], kernel_size[1], num_transformed_images]`.

    Returns:
        A list of `num_transformed_images` 4-D tensors, each of shape
            `[batch, in_height, in_width, in_channels]`.
    """
    if isinstance(image, list):
        image_list = image
        kernels_list = tf.split(kernels, len(image_list), axis=-1)
        outputs = []
        for image, kernels in zip(image_list, kernels_list):
            outputs.extend(apply_kernels(image, kernels))
    else:
        if len(kernels.get_shape()) == 4:
            outputs = apply_cdna_kernels(image, kernels, dilation_rate=dilation_rate)
        elif len(kernels.get_shape()) == 6:
            outputs = apply_dna_kernels(image, kernels, dilation_rate=dilation_rate)
        else:
            raise ValueError
    return outputs


def apply_flows(image, flows):
    if isinstance(image, list):
        image_list = image
        flows_list = tf.split(flows, len(image_list), axis=-1)
        outputs = []
        for image, flows in zip(image_list, flows_list):
            outputs.extend(apply_flows(image, flows))
    else:
        flows = tf.unstack(flows, axis=-1)
        outputs = [flow_ops.image_warp(image, flow) for flow in flows]
    return outputs


def identity_kernel(kernel_size):
    kh, kw = kernel_size
    kernel = np.zeros(kernel_size)

    def center_slice(k):
        if k % 2 == 0:
            return slice(k // 2 - 1, k // 2 + 1)
        else:
            return slice(k // 2, k // 2 + 1)

    kernel[center_slice(kh), center_slice(kw)] = 1.0
    kernel /= np.sum(kernel)
    return kernel


def _maybe_tile_concat_layer(conv2d_layer):
    def layer(inputs, out_channels, *args, **kwargs):
        if isinstance(inputs, (list, tuple)):
            inputs_spatial, inputs_non_spatial = inputs
            outputs = (conv2d_layer(inputs_spatial, out_channels, *args, **kwargs) +
                       dense(inputs_non_spatial, out_channels, use_bias=False)[:, None, None, :])
        else:
            outputs = conv2d_layer(inputs, out_channels, *args, **kwargs)
        return outputs

    return layer
