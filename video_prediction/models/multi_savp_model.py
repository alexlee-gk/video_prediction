import itertools

import numpy as np
import tensorflow as tf

import video_prediction as vp
from video_prediction import ops
from video_prediction.models import VideoPredictionModel, SAVPVideoPredictionModel
from video_prediction.models import pix2pix_model, mocogan_model, spectral_norm_model
from video_prediction.models.savp_model import create_encoder, apply_kernels, apply_flows, identity_kernel
from video_prediction.ops import dense, conv2d, flatten, tile_concat
from video_prediction.rnn_ops import BasicConv2DLSTMCell, Conv2DGRUCell
from video_prediction.utils import tf_utils

# Amount to use when lower bounding tensors
RELU_SHIFT = 1e-12


def encoder_fn(inputs, hparams=None):
    image_pairs = []
    for i in range(hparams.num_views):
        suffix = '%d' % i if i > 0 else ''
        images = inputs['images' + suffix]
        image_pairs.append(images[:hparams.sequence_length - 1])
        image_pairs.append(images[1:hparams.sequence_length])
    image_pairs = tf.concat(image_pairs, axis=-1)
    if 'actions' in inputs:
        image_pairs = tile_concat([image_pairs,
                                   tf.expand_dims(tf.expand_dims(inputs['actions'], axis=-2), axis=-2)], axis=-1)
    outputs = create_encoder(image_pairs,
                             e_net=hparams.e_net,
                             use_e_rnn=hparams.use_e_rnn,
                             rnn=hparams.rnn,
                             nz=hparams.nz,
                             nef=hparams.nef,
                             n_layers=hparams.n_layers,
                             norm_layer=hparams.norm_layer)
    return outputs


def discriminator_fn(targets, inputs=None, hparams=None):
    outputs = {}
    if hparams.gan_weight or hparams.vae_gan_weight:
        _, pix2pix_outputs = pix2pix_model.discriminator_fn(targets, inputs=inputs, hparams=hparams)
        outputs.update(pix2pix_outputs)
    if hparams.image_gan_weight or hparams.image_vae_gan_weight or \
            hparams.video_gan_weight or hparams.video_vae_gan_weight or \
            hparams.acvideo_gan_weight or hparams.acvideo_vae_gan_weight:
        _, mocogan_outputs = mocogan_model.discriminator_fn(targets, inputs=inputs, hparams=hparams)
        outputs.update(mocogan_outputs)
    if hparams.image_sn_gan_weight or hparams.image_sn_vae_gan_weight or \
            hparams.video_sn_gan_weight or hparams.video_sn_vae_gan_weight:
        _, spectral_norm_outputs = spectral_norm_model.discriminator_fn(targets, inputs=inputs, hparams=hparams)
        outputs.update(spectral_norm_outputs)
    return None, outputs


class DNACell(tf.nn.rnn_cell.RNNCell):
    def __init__(self, inputs, hparams, reuse=None):
        super(DNACell, self).__init__(_reuse=reuse)
        self.inputs = inputs
        self.hparams = hparams

        if self.hparams.where_add not in ('input', 'all', 'middle'):
            raise ValueError('Invalid where_add %s' % self.hparams.where_add)

        batch_size = inputs['images'].shape[1].value
        image_shape = inputs['images'].shape.as_list()[2:]
        height, width, _ = image_shape
        scale_size = max(height, width)
        if scale_size == 256:
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
        elif scale_size == 64:
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
        elif scale_size == 32:
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

        # output_size
        num_masks = self.hparams.last_frames * self.hparams.num_transformed_images + \
            int(bool(self.hparams.prev_image_background)) + \
            int(bool(self.hparams.first_image_background and not self.hparams.context_images_background)) + \
            (self.hparams.context_frames if self.hparams.context_images_background else 0) + \
            int(bool(self.hparams.generate_scratch_image))
        output_size = {}
        for i in range(self.hparams.num_views):
            suffix = '%d' % i if i > 0 else ''
            output_size['gen_images' + suffix] = tf.TensorShape(image_shape)
            output_size['transformed_images' + suffix] = tf.TensorShape(image_shape + [num_masks])
            output_size['masks' + suffix] = tf.TensorShape([height, width, 1, num_masks])
        if 'pix_distribs' in inputs:
            for i in range(self.hparams.num_views):
                suffix = '%d' % i if i > 0 else ''
                num_motions = inputs['pix_distribs' + suffix].shape[-1].value
                output_size['gen_pix_distribs' + suffix] = tf.TensorShape([height, width, num_motions])
                output_size['transformed_pix_distribs' + suffix] = tf.TensorShape([height, width, num_motions, num_masks])
        if 'states' in inputs:
            output_size['gen_states'] = inputs['states'].shape[2:]
        if self.hparams.transformation == 'flow':
            for i in range(self.hparams.num_views):
                suffix = '%d' % i if i > 0 else ''
                output_size['gen_flows' + suffix] = tf.TensorShape([height, width, 2, self.hparams.last_frames * self.hparams.num_transformed_images])
                output_size['gen_flows_rgb' + suffix] = tf.TensorShape([height, width, 3, self.hparams.last_frames * self.hparams.num_transformed_images])
        self._output_size = output_size

        # state_size
        conv_rnn_state_sizes = []
        conv_rnn_height, conv_rnn_width = height, width
        for out_channels, use_conv_rnn in self.encoder_layer_specs:
            conv_rnn_height //= 2
            conv_rnn_width //= 2
            if use_conv_rnn:
                conv_rnn_state_sizes.append(tf.TensorShape([conv_rnn_height, conv_rnn_width, out_channels]))
        for out_channels, use_conv_rnn in self.decoder_layer_specs:
            conv_rnn_height *= 2
            conv_rnn_width *= 2
            if use_conv_rnn:
                conv_rnn_state_sizes.append(tf.TensorShape([conv_rnn_height, conv_rnn_width, out_channels]))
        if self.hparams.conv_rnn == 'lstm':
            conv_rnn_state_sizes = [tf.nn.rnn_cell.LSTMStateTuple(conv_rnn_state_size, conv_rnn_state_size)
                                    for conv_rnn_state_size in conv_rnn_state_sizes]
        state_size = {'time': tf.TensorShape([])}
        for i in range(self.hparams.num_views):
            suffix = '%d' % i if i > 0 else ''
            state_size['gen_image' + suffix] = tf.TensorShape(image_shape)
            state_size['last_images' + suffix] = [tf.TensorShape(image_shape)] * self.hparams.last_frames
        for i in range(self.hparams.num_views):
            suffix = '%d' % i if i > 0 else ''
            state_size['conv_rnn_states' + suffix] = conv_rnn_state_sizes
            if self.hparams.shared_views:
                break
        if 'zs' in inputs and self.hparams.use_rnn_z:
            rnn_z_state_size = tf.TensorShape([self.hparams.nz])
            if self.hparams.rnn == 'lstm':
                rnn_z_state_size = tf.nn.rnn_cell.LSTMStateTuple(rnn_z_state_size, rnn_z_state_size)
            state_size['rnn_z_state'] = rnn_z_state_size
        if 'pix_distribs' in inputs:
            for i in range(self.hparams.num_views):
                suffix = '%d' % i if i > 0 else ''
                state_size['gen_pix_distrib' + suffix] = tf.TensorShape([height, width, num_motions])
                state_size['last_pix_distribs' + suffix] = [tf.TensorShape([height, width, num_motions])] * self.hparams.last_frames
        if 'states' in inputs:
            state_size['gen_state'] = inputs['states'].shape[2:]
        self._state_size = state_size

        ground_truth_sampling_shape = [self.hparams.sequence_length - 1 - self.hparams.context_frames, batch_size]
        if self.hparams.schedule_sampling == 'none':
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
        init_state = super(DNACell, self).zero_state(batch_size, dtype)
        for i in range(self.hparams.num_views):
            suffix = '%d' % i if i > 0 else ''
            init_state['last_images' + suffix] = [self.inputs['images' + suffix][0]] * self.hparams.last_frames
            if 'pix_distribs' in self.inputs:
                init_state['last_pix_distribs' + suffix] = [self.inputs['pix_distribs' + suffix][0]] * self.hparams.last_frames
        return init_state

    def _rnn_func(self, inputs, state, num_units):
        if self.hparams.rnn == 'lstm':
            RNNCell = tf.contrib.rnn.BasicLSTMCell
        elif self.hparams.rnn == 'gru':
            RNNCell = tf.contrib.rnn.GRUCell
        else:
            raise NotImplementedError
        rnn_cell = RNNCell(num_units, reuse=tf.get_variable_scope().reuse)
        return rnn_cell(inputs, state)

    def _conv_rnn_func(self, inputs, state, filters):
        inputs_shape = inputs.get_shape().as_list()
        input_shape = inputs_shape[1:]
        if self.hparams.norm_layer == 'none':
            normalizer_fn = None
        else:
            normalizer_fn = ops.get_norm_layer(self.hparams.norm_layer)
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
                                          separate_norms=self.hparams.norm_layer == 'layer',
                                          reuse=tf.get_variable_scope().reuse)
            outputs = conv_rnn_cell(inputs, state)
        return outputs

    def call(self, inputs, states):
        norm_layer = ops.get_norm_layer(self.hparams.norm_layer)
        downsample_layer = ops.get_downsample_layer(self.hparams.downsample_layer)
        upsample_layer = ops.get_upsample_layer(self.hparams.upsample_layer)
        image_shape = inputs['images'].get_shape().as_list()
        batch_size, height, width, color_channels = image_shape

        time = states['time']
        with tf.control_dependencies([tf.assert_equal(time[1:], time[0])]):
            t = tf.to_int32(tf.identity(time[0]))

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
                with tf.variable_scope('%s_z' % self.hparams.rnn):
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

        image_views = []
        first_image_views = []
        if 'pix_distribs' in inputs:
            pix_distrib_views = []
        for i in range(self.hparams.num_views):
            suffix = '%d' % i if i > 0 else ''
            image_view = tf.where(self.ground_truth[t], inputs['images' + suffix], states['gen_image' + suffix])  # schedule sampling (if any)
            image_views.append(image_view)
            first_image_views.append(self.inputs['images' + suffix][0])
            if 'pix_distribs' in inputs:
                pix_distrib_view = tf.where(self.ground_truth[t], inputs['pix_distribs' + suffix], states['gen_pix_distrib' + suffix])
                pix_distrib_views.append(pix_distrib_view)

        outputs = {}
        new_states = {}
        all_layers = []
        for i in range(self.hparams.num_views):
            suffix = '%d' % i if i > 0 else ''
            conv_rnn_states = states['conv_rnn_states' + suffix]
            layers = []
            new_conv_rnn_states = []
            for i, (out_channels, use_conv_rnn) in enumerate(self.encoder_layer_specs):
                with tf.variable_scope('h%d' % i + suffix):
                    if i == 0:
                        # all image views and the first image corresponding to this view only
                        h = tf.concat(image_views + first_image_views, axis=-1)
                        kernel_size = (5, 5)
                    else:
                        h = layers[-1][-1]
                        kernel_size = (3, 3)
                    if self.hparams.where_add == 'all' or (self.hparams.where_add == 'input' and i == 0):
                        h = tile_concat([h, state_action_z[:, None, None, :]], axis=-1)
                    h = downsample_layer(h, out_channels, kernel_size=kernel_size, strides=(2, 2))
                    h = norm_layer(h)
                    h = tf.nn.relu(h)
                if use_conv_rnn:
                    conv_rnn_state = conv_rnn_states[len(new_conv_rnn_states)]
                    with tf.variable_scope('%s_h%d' % (self.hparams.conv_rnn, i) + suffix):
                        if self.hparams.where_add == 'all':
                            conv_rnn_h = tile_concat([h, state_action_z[:, None, None, :]], axis=-1)
                        else:
                            conv_rnn_h = h
                        conv_rnn_h, conv_rnn_state = self._conv_rnn_func(conv_rnn_h, conv_rnn_state, out_channels)
                    new_conv_rnn_states.append(conv_rnn_state)
                layers.append((h, conv_rnn_h) if use_conv_rnn else (h,))

            num_encoder_layers = len(layers)
            for i, (out_channels, use_conv_rnn) in enumerate(self.decoder_layer_specs):
                with tf.variable_scope('h%d' % len(layers) + suffix):
                    if i == 0:
                        h = layers[-1][-1]
                    else:
                        h = tf.concat([layers[-1][-1], layers[num_encoder_layers - i - 1][-1]], axis=-1)
                    if self.hparams.where_add == 'all' or (self.hparams.where_add == 'middle' and i == 0):
                        h = tile_concat([h, state_action_z[:, None, None, :]], axis=-1)
                    h = upsample_layer(h, out_channels, kernel_size=(3, 3), strides=(2, 2))
                    h = norm_layer(h)
                    h = tf.nn.relu(h)
                if use_conv_rnn:
                    conv_rnn_state = conv_rnn_states[len(new_conv_rnn_states)]
                    with tf.variable_scope('%s_h%d' % (self.hparams.conv_rnn, len(layers)) + suffix):
                        if self.hparams.where_add == 'all':
                            conv_rnn_h = tile_concat([h, state_action_z[:, None, None, :]], axis=-1)
                        else:
                            conv_rnn_h = h
                        conv_rnn_h, conv_rnn_state = self._conv_rnn_func(conv_rnn_h, conv_rnn_state, out_channels)
                    new_conv_rnn_states.append(conv_rnn_state)
                layers.append((h, conv_rnn_h) if use_conv_rnn else (h,))
            assert len(new_conv_rnn_states) == len(conv_rnn_states)

            new_states['conv_rnn_states' + suffix] = new_conv_rnn_states

            all_layers.append(layers)
            if self.hparams.shared_views:
                break

        for i in range(self.hparams.num_views):
            suffix = '%d' % i if i > 0 else ''
            if self.hparams.shared_views:
                layers, = all_layers
            else:
                layers = all_layers[i]

            image = image_views[i]
            last_images = states['last_images' + suffix][1:] + [image]
            if 'pix_distribs' in inputs:
                pix_distrib = pix_distrib_views[i]
                last_pix_distribs = states['last_pix_distribs' + suffix][1:] + [pix_distrib]

            if self.hparams.last_frames and self.hparams.num_transformed_images:
                if self.hparams.transformation == 'flow':
                    with tf.variable_scope('h%d_flow' % len(layers) + suffix):
                        h_flow = conv2d(layers[-1][-1], self.hparams.ngf, kernel_size=(3, 3), strides=(1, 1))
                        h_flow = norm_layer(h_flow)
                        h_flow = tf.nn.relu(h_flow)

                    with tf.variable_scope('flows' + suffix):
                        flows = conv2d(h_flow, 2 * self.hparams.last_frames * self.hparams.num_transformed_images, kernel_size=(3, 3), strides=(1, 1))
                        flows = tf.reshape(flows, [batch_size, height, width, 2, self.hparams.last_frames * self.hparams.num_transformed_images])
                else:
                    assert len(self.hparams.kernel_size) == 2
                    kernel_shape = list(self.hparams.kernel_size) + [self.hparams.last_frames * self.hparams.num_transformed_images]
                    if self.hparams.transformation == 'dna':
                        with tf.variable_scope('h%d_dna_kernel' % len(layers) + suffix):
                            h_dna_kernel = conv2d(layers[-1][-1], self.hparams.ngf, kernel_size=(3, 3), strides=(1, 1))
                            h_dna_kernel = norm_layer(h_dna_kernel)
                            h_dna_kernel = tf.nn.relu(h_dna_kernel)

                        # Using largest hidden state for predicting untied conv kernels.
                        with tf.variable_scope('dna_kernels' + suffix):
                            kernels = conv2d(h_dna_kernel, np.prod(kernel_shape), kernel_size=(3, 3), strides=(1, 1))
                            kernels = tf.reshape(kernels, [batch_size, height, width] + kernel_shape)
                            kernels = kernels + identity_kernel(self.hparams.kernel_size)[None, None, None, :, :, None]
                        kernel_spatial_axes = [3, 4]
                    elif self.hparams.transformation == 'cdna':
                        with tf.variable_scope('cdna_kernels' + suffix):
                            smallest_layer = layers[num_encoder_layers - 1][-1]
                            kernels = dense(flatten(smallest_layer), np.prod(kernel_shape))
                            kernels = tf.reshape(kernels, [batch_size] + kernel_shape)
                            kernels = kernels + identity_kernel(self.hparams.kernel_size)[None, :, :, None]
                        kernel_spatial_axes = [1, 2]
                    else:
                        raise ValueError('Invalid transformation %s' % self.hparams.transformation)

                if self.hparams.transformation != 'flow':
                    with tf.name_scope('kernel_normalization' + suffix):
                        kernels = tf.nn.relu(kernels - RELU_SHIFT) + RELU_SHIFT
                        kernels /= tf.reduce_sum(kernels, axis=kernel_spatial_axes, keepdims=True)

            if self.hparams.generate_scratch_image:
                with tf.variable_scope('h%d_scratch' % len(layers) + suffix):
                    h_scratch = conv2d(layers[-1][-1], self.hparams.ngf, kernel_size=(3, 3), strides=(1, 1))
                    h_scratch = norm_layer(h_scratch)
                    h_scratch = tf.nn.relu(h_scratch)

                # Using largest hidden state for predicting a new image layer.
                # This allows the network to also generate one image from scratch,
                # which is useful when regions of the image become unoccluded.
                with tf.variable_scope('scratch_image' + suffix):
                    scratch_image = conv2d(h_scratch, color_channels, kernel_size=(3, 3), strides=(1, 1))
                    scratch_image = tf.nn.sigmoid(scratch_image)

            with tf.name_scope('transformed_images' + suffix):
                transformed_images = []
                if self.hparams.last_frames and self.hparams.num_transformed_images:
                    if self.hparams.transformation == 'flow':
                        transformed_images.extend(apply_flows(last_images, flows))
                    else:
                        transformed_images.extend(apply_kernels(last_images, kernels, self.hparams.dilation_rate))
                if self.hparams.prev_image_background:
                    transformed_images.append(image)
                if self.hparams.first_image_background and not self.hparams.context_images_background:
                    transformed_images.append(self.inputs['images' + suffix][0])
                if self.hparams.context_images_background:
                    transformed_images.extend(tf.unstack(self.inputs['images' + suffix][:self.hparams.context_frames]))
                if self.hparams.generate_scratch_image:
                    transformed_images.append(scratch_image)

            if 'pix_distribs' in inputs:
                with tf.name_scope('transformed_pix_distribs' + suffix):
                    transformed_pix_distribs = []
                    if self.hparams.last_frames and self.hparams.num_transformed_images:
                        if self.hparams.transformation == 'flow':
                            transformed_pix_distribs.extend(apply_flows(last_pix_distribs, flows))
                        else:
                            transformed_pix_distribs.extend(apply_kernels(last_pix_distribs, kernels, self.hparams.dilation_rate))
                    if self.hparams.prev_image_background:
                        transformed_pix_distribs.append(pix_distrib)
                    if self.hparams.first_image_background and not self.hparams.context_images_background:
                        transformed_pix_distribs.append(self.inputs['pix_distribs' + suffix][0])
                    if self.hparams.context_images_background:
                        transformed_pix_distribs.extend(tf.unstack(self.inputs['pix_distribs' + suffix][:self.hparams.context_frames]))
                    if self.hparams.generate_scratch_image:
                        transformed_pix_distribs.append(pix_distrib)

            with tf.name_scope('masks' + suffix):
                if len(transformed_images) > 1:
                    with tf.variable_scope('h%d_masks' % len(layers) + suffix):
                        h_masks = conv2d(layers[-1][-1], self.hparams.ngf, kernel_size=(3, 3), strides=(1, 1))
                        h_masks = norm_layer(h_masks)
                        h_masks = tf.nn.relu(h_masks)

                    with tf.variable_scope('masks' + suffix):
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

            with tf.name_scope('gen_images' + suffix):
                assert len(transformed_images) == len(masks)
                gen_image = tf.add_n([transformed_image * mask
                                      for transformed_image, mask in zip(transformed_images, masks)])

            if 'pix_distribs' in inputs:
                with tf.name_scope('gen_pix_distribs' + suffix):
                    assert len(transformed_pix_distribs) == len(masks)
                    gen_pix_distrib = tf.add_n([transformed_pix_distrib * mask
                                                for transformed_pix_distrib, mask in zip(transformed_pix_distribs, masks)])
                    gen_pix_distrib /= tf.reduce_sum(gen_pix_distrib, axis=(1, 2), keepdims=True)

            outputs['gen_images' + suffix] = gen_image
            outputs['transformed_images' + suffix] = tf.stack(transformed_images, axis=-1)
            outputs['masks' + suffix] = tf.stack(masks, axis=-1)
            if 'pix_distribs' in inputs:
                outputs['gen_pix_distribs' + suffix] = gen_pix_distrib
                outputs['transformed_pix_distribs' + suffix] = tf.stack(transformed_pix_distribs, axis=-1)
            if self.hparams.transformation == 'flow':
                outputs['gen_flows' + suffix] = flows
                flows_transposed = tf.transpose(flows, [0, 1, 2, 4, 3])
                flows_rgb_transposed = tf_utils.flow_to_rgb(flows_transposed)
                flows_rgb = tf.transpose(flows_rgb_transposed, [0, 1, 2, 4, 3])
                outputs['gen_flows_rgb' + suffix] = flows_rgb

            new_states['gen_image' + suffix] = gen_image
            new_states['last_images' + suffix] = last_images
            if 'pix_distribs' in inputs:
                new_states['gen_pix_distrib' + suffix] = gen_pix_distrib
                new_states['last_pix_distribs' + suffix] = last_pix_distribs

        if 'states' in inputs:
            with tf.name_scope('gen_states'):
                with tf.variable_scope('state_pred'):
                    gen_state = dense(state_action, inputs['states'].shape[-1].value)

        if 'states' in inputs:
            outputs['gen_states'] = gen_state

        new_states['time'] = time + 1
        if 'zs' in inputs and self.hparams.use_rnn_z:
            new_states['rnn_z_state'] = rnn_z_state
        if 'states' in inputs:
            new_states['gen_state'] = gen_state
        return outputs, new_states


def generator_fn(inputs, outputs_enc=None, hparams=None):
    batch_size = inputs['images'].shape[1].value
    inputs = {name: tf_utils.maybe_pad_or_slice(input, hparams.sequence_length - 1)
              for name, input in inputs.items()}
    if hparams.nz:
        def sample_zs():
            if outputs_enc is None:
                zs = tf.random_normal([hparams.sequence_length - 1, batch_size, hparams.nz], 0, 1)
            else:
                enc_zs_mu = outputs_enc['enc_zs_mu']
                enc_zs_log_sigma_sq = outputs_enc['enc_zs_log_sigma_sq']
                eps = tf.random_normal([hparams.sequence_length - 1, batch_size, hparams.nz], 0, 1)
                zs = enc_zs_mu + tf.sqrt(tf.exp(enc_zs_log_sigma_sq)) * eps
            return zs
        inputs['zs'] = sample_zs()
    else:
        if outputs_enc is not None:
            raise ValueError('outputs_enc has to be None when nz is 0.')
    cell = DNACell(inputs, hparams)
    outputs, _ = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32,
                                   swap_memory=False, time_major=True)
    if hparams.nz:
        inputs_samples = {name: flatten(tf.tile(input[:, None], [1, hparams.num_samples] + [1] * (input.shape.ndims - 1)), 1, 2)
                          for name, input in inputs.items() if name != 'zs'}
        inputs_samples['zs'] = tf.concat([sample_zs() for _ in range(hparams.num_samples)], axis=1)
        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            cell_samples = DNACell(inputs_samples, hparams)
            outputs_samples, _ = tf.nn.dynamic_rnn(cell_samples, inputs_samples, dtype=tf.float32,
                                                   swap_memory=False, time_major=True)
        for i in range(hparams.num_views):
            suffix = '%d' % i if i > 0 else ''
            gen_images_samples = outputs_samples['gen_images' + suffix]
            gen_images_samples = tf.stack(tf.split(gen_images_samples, hparams.num_samples, axis=1), axis=-1)
            gen_images_samples_avg = tf.reduce_mean(gen_images_samples, axis=-1)
            outputs['gen_images_samples' + suffix] = gen_images_samples
            outputs['gen_images_samples_avg' + suffix] = gen_images_samples_avg
    # the RNN outputs generated images from time step 1 to sequence_length,
    # but generator_fn should only return images past context_frames
    outputs = {name: output[hparams.context_frames - 1:] for name, output in outputs.items()}
    gen_images = outputs['gen_images']
    outputs['ground_truth_sampling_mean'] = tf.reduce_mean(tf.to_float(cell.ground_truth[hparams.context_frames:]))
    return gen_images, outputs


class MultiSAVPVideoPredictionModel(SAVPVideoPredictionModel):
    def __init__(self, *args, **kwargs):
        VideoPredictionModel.__init__(self,
            generator_fn, discriminator_fn, encoder_fn, *args, **kwargs)
        if self.hparams.e_net == 'none' or self.hparams.nz == 0:
            self.encoder_fn = None
        if self.hparams.d_net == 'none':
            self.discriminator_fn = None
        self.deterministic = not self.hparams.nz

    def get_default_hparams_dict(self):
        default_hparams = super(MultiSAVPVideoPredictionModel, self).get_default_hparams_dict()
        hparams = dict(
            num_views=1,
            shared_views=False,
        )
        return dict(itertools.chain(default_hparams.items(), hparams.items()))

    def generator_loss_fn(self, inputs, outputs, targets):
        gen_losses = super(MultiSAVPVideoPredictionModel, self).generator_loss_fn(inputs, outputs, targets)
        hparams = self.hparams
        # TODO: support for other losses of the other views
        for i in range(1, hparams.num_views):  # skip i = 0 since it should have already been done by the superclass
            suffix = '%d' % i if i > 0 else ''
            if hparams.l1_weight or hparams.l2_weight:
                gen_images = outputs.get('gen_images%s_enc' % suffix, outputs['gen_images' + suffix])
                target_images = inputs['images' + suffix][self.hparams.context_frames:]
            if hparams.l1_weight:
                gen_l1_loss = vp.losses.l1_loss(gen_images, target_images)
                gen_losses["gen_l1_loss" + suffix] = (gen_l1_loss, hparams.l1_weight)
            if hparams.l2_weight:
                gen_l2_loss = vp.losses.l2_loss(gen_images, target_images)
                gen_losses["gen_l2_loss" + suffix] = (gen_l2_loss, hparams.l2_weight)
            if (hparams.l1_weight or hparams.l2_weight) and hparams.num_scales > 1:
                raise NotImplementedError
            if hparams.tv_weight:
                gen_flows = outputs.get('gen_flows%s_enc' % suffix, outputs['gen_flows' + suffix])
                flow_diff1 = gen_flows[..., 1:, :, :, :] - gen_flows[..., :-1, :, :, :]
                flow_diff2 = gen_flows[..., :, 1:, :, :] - gen_flows[..., :, :-1, :, :]
                # sum over the multiple transformations but take the mean for the other dimensions
                gen_tv_loss = tf.reduce_mean(tf.reduce_sum(tf.abs(flow_diff1), axis=(-2, -1))) + \
                              tf.reduce_mean(tf.reduce_sum(tf.abs(flow_diff2), axis=(-2, -1)))
                gen_losses['gen_tv_loss' + suffix] = (gen_tv_loss, hparams.tv_weight)
        return gen_losses
