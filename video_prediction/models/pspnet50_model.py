import itertools

import numpy as np
import tensorflow as tf

from video_prediction import ops, flow_ops
from video_prediction.models.improved_dna_model import apply_kernels, identity_kernel
from video_prediction.ops import conv2d, tile_concat, flatten
from video_prediction.ops import dense, upsample_conv2d, conv_pool2d
from video_prediction.rnn_ops import BasicConv2DLSTMCell, Conv2DGRUCell
from video_prediction.utils import tf_utils
from . import pspnet_network
from .base_model import VideoPredictionModel


def create_pspnet50_encoder(inputs):
    should_flatten = inputs.shape.ndims > 4
    if should_flatten:
        batch_shape = inputs.shape[:-3].as_list()
        inputs = flatten(inputs, 0, len(batch_shape) - 1)

    outputs = pspnet_network.pspnet(inputs, resnet_layers=50)

    if should_flatten:
        outputs = tf.reshape(outputs, batch_shape + outputs.shape.as_list()[1:])
    return outputs


def create_decoder(feature, output_nc=3):
    # TODO: use hparams.norm_layer?
    norm_layer = lambda x: tf.layers.batch_normalization(x, momentum=0.95, epsilon=1e-5)
    decoder_layer_specs = [256, 128, 64]

    layers = []
    h = feature
    for i, out_channels in enumerate(decoder_layer_specs):
        with tf.variable_scope('h%d' % len(layers)):
            h = upsample_conv2d(h, out_channels, kernel_size=(3, 3), strides=(2, 2))
            h = norm_layer(h)
            h = tf.nn.relu(h)
            layers.append(h)

    with tf.variable_scope('h%d' % len(layers)):
        image = conv2d(h, output_nc, kernel_size=(3, 3), strides=(1, 1))
        image = tf.nn.sigmoid(image)
    return image


class DynamicsCell(tf.nn.rnn_cell.RNNCell):
    def __init__(self, inputs, hparams, reuse=None):
        super(DynamicsCell, self).__init__(_reuse=reuse)
        self.inputs = inputs
        self.hparams = hparams

        batch_size = inputs['features'].shape[1].value
        feature_shape = inputs['features'].shape.as_list()[2:]
        height, width, feature_channels = feature_shape
        scale_size = min(height, width)
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
            ]
            self.decoder_layer_specs = [
                (self.hparams.ngf, True),
            ]
        # elif scale_size == 32:
        #     self.encoder_layer_specs = [
        #         (self.hparams.ngf, True),
        #         (self.hparams.ngf * 2, True),
        #     ]
        #     self.decoder_layer_specs = [
        #         (self.hparams.ngf, True),
        #         (self.hparams.ngf, False),
        #     ]
        else:
            raise NotImplementedError

        # output_size
        gen_input_shape = list(feature_shape)
        if 'actions' in inputs:
            gen_input_shape[-1] += inputs['actions'].shape[-1].value
        output_size = {
            'gen_features': tf.TensorShape(feature_shape),
            'gen_inputs': tf.TensorShape(gen_input_shape),
        }
        if 'states' in inputs:
            output_size['gen_states'] = inputs['states'].shape[2:]
        if self.hparams.transformation == 'flow':
            output_size['gen_flows'] = tf.TensorShape([height, width, 2, feature_channels])
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
        state_size = {
            'time': tf.TensorShape([]),
            'gen_feature': tf.TensorShape(feature_shape),
            'conv_rnn_states': conv_rnn_state_sizes,
        }
        if 'zs' in inputs and self.hparams.use_rnn_z:
            rnn_z_state_size = tf.TensorShape([self.hparams.nz])
            if self.hparams.rnn == 'lstm':
                rnn_z_state_size = tf.nn.rnn_cell.LSTMStateTuple(rnn_z_state_size, rnn_z_state_size)
            state_size['rnn_z_state'] = rnn_z_state_size
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
        conv_rnn_cell = Conv2DRNNCell(input_shape, filters, kernel_size=(3, 3),  # TODO: use kernel_size=(5, 5)?
                                      normalizer_fn=normalizer_fn,
                                      separate_norms=self.hparams.norm_layer == 'layer',
                                      reuse=tf.get_variable_scope().reuse)
        return conv_rnn_cell(inputs, state)

    def call(self, inputs, states):
        norm_layer = ops.get_norm_layer(self.hparams.norm_layer)
        feature_shape = inputs['features'].get_shape().as_list()
        batch_size, height, width, feature_channels = feature_shape
        conv_rnn_states = states['conv_rnn_states']

        time = states['time']
        with tf.control_dependencies([tf.assert_equal(time[1:], time[0])]):
            t = tf.to_int32(tf.identity(time[0]))

        feature = tf.where(self.ground_truth[t], inputs['features'], states['gen_feature'])  # schedule sampling (if any)
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
        if 'actions' in inputs:
            gen_input = tile_concat([feature, inputs['actions'][:, None, None, :]], axis=-1)
        else:
            gen_input = feature

        layers = []
        new_conv_rnn_states = []
        for i, (out_channels, use_conv_rnn) in enumerate(self.encoder_layer_specs):
            with tf.variable_scope('h%d' % i):
                if i == 0:
                    # h = tf.concat([feature, self.inputs['features'][0]], axis=-1)  # TODO: use first feature?
                    h = feature
                else:
                    h = layers[-1][-1]
                h = conv_pool2d(tile_concat([h, state_action_z[:, None, None, :]], axis=-1),
                                out_channels, kernel_size=(3, 3), strides=(2, 2))
                h = norm_layer(h)
                h = tf.nn.relu(h)
            if use_conv_rnn:
                conv_rnn_state = conv_rnn_states[len(new_conv_rnn_states)]
                with tf.variable_scope('%s_h%d' % (self.hparams.conv_rnn, i)):
                        conv_rnn_h, conv_rnn_state = self._conv_rnn_func(tile_concat([h, state_action_z[:, None, None, :]], axis=-1),
                                                                         conv_rnn_state, out_channels)
                new_conv_rnn_states.append(conv_rnn_state)
            layers.append((h, conv_rnn_h) if use_conv_rnn else (h,))

        num_encoder_layers = len(layers)
        for i, (out_channels, use_conv_rnn) in enumerate(self.decoder_layer_specs):
            with tf.variable_scope('h%d' % len(layers)):
                if i == 0:
                    h = layers[-1][-1]
                else:
                    h = tf.concat([layers[-1][-1], layers[num_encoder_layers - i - 1][-1]], axis=-1)
                h = upsample_conv2d(tile_concat([h, state_action_z[:, None, None, :]], axis=-1),
                                    out_channels, kernel_size=(3, 3), strides=(2, 2))
                h = norm_layer(h)
                h = tf.nn.relu(h)
            if use_conv_rnn:
                conv_rnn_state = conv_rnn_states[len(new_conv_rnn_states)]
                with tf.variable_scope('%s_h%d' % (self.hparams.conv_rnn, len(layers))):
                    conv_rnn_h, conv_rnn_state = self._conv_rnn_func(tile_concat([h, state_action_z[:, None, None, :]], axis=-1),
                                                                     conv_rnn_state, out_channels)
                new_conv_rnn_states.append(conv_rnn_state)
            layers.append((h, conv_rnn_h) if use_conv_rnn else (h,))
        assert len(new_conv_rnn_states) == len(conv_rnn_states)

        if self.hparams.transformation == 'direct':
            with tf.variable_scope('h%d_direct' % len(layers)):
                h_direct = conv2d(layers[-1][-1], self.hparams.ngf, kernel_size=(3, 3), strides=(1, 1))
                h_direct = norm_layer(h_direct)
                h_direct = tf.nn.relu(h_direct)

            with tf.variable_scope('direct'):
                gen_feature = conv2d(h_direct, feature_channels, kernel_size=(3, 3), strides=(1, 1))
        else:
            if self.hparams.transformation == 'flow':
                with tf.variable_scope('h%d_flow' % len(layers)):
                    h_flow = conv2d(layers[-1][-1], self.hparams.ngf, kernel_size=(3, 3), strides=(1, 1))
                    h_flow = norm_layer(h_flow)
                    h_flow = tf.nn.relu(h_flow)

                with tf.variable_scope('flows'):
                    flows = conv2d(h_flow, 2 * feature_channels, kernel_size=(3, 3), strides=(1, 1))
                    flows = tf.reshape(flows, [batch_size, height, width, 2, feature_channels])
                transformations = flows
            else:
                assert len(self.hparams.kernel_size) == 2
                kernel_shape = list(self.hparams.kernel_size) + [feature_channels]
                if self.hparams.transformation == 'local':
                    with tf.variable_scope('h%d_local_kernel' % len(layers)):
                        h_local_kernel = conv2d(layers[-1][-1], self.hparams.ngf, kernel_size=(3, 3), strides=(1, 1))
                        h_local_kernel = norm_layer(h_local_kernel)
                        h_local_kernel = tf.nn.relu(h_local_kernel)

                    # Using largest hidden state for predicting untied conv kernels.
                    with tf.variable_scope('local_kernels'):
                        kernels = conv2d(h_local_kernel, np.prod(kernel_shape), kernel_size=(3, 3), strides=(1, 1))
                        kernels = tf.reshape(kernels, [batch_size, height, width] + kernel_shape)
                        kernels = kernels + identity_kernel(self.hparams.kernel_size)[None, None, None, :, :, None]
                elif self.hparams.transformation == 'conv':
                    with tf.variable_scope('conv_kernels'):
                        smallest_layer = layers[num_encoder_layers - 1][-1]
                        kernels = dense(flatten(smallest_layer), np.prod(kernel_shape))
                        kernels = tf.reshape(kernels, [batch_size] + kernel_shape)
                        kernels = kernels + identity_kernel(self.hparams.kernel_size)[None, :, :, None]
                else:
                    raise ValueError('Invalid transformation %s' % self.hparams.transformation)
                transformations = kernels

            with tf.name_scope('gen_features'):
                if self.hparams.transformation == 'flow':
                    def apply_transformation(feature_and_flow):
                        feature, flow = feature_and_flow
                        return flow_ops.image_warp(feature[..., None], flow)
                else:
                    def apply_transformation(feature_and_kernel):
                        feature, kernel = feature_and_kernel
                        output, = apply_kernels(feature[..., None], kernel[..., None])
                        return tf.squeeze(output, axis=-1)
                gen_feature_transposed = tf.map_fn(apply_transformation,
                                                   (tf.stack(tf.unstack(feature, axis=-1)),
                                                    tf.stack(tf.unstack(transformations, axis=-1))),
                                                   dtype=tf.float32)
                gen_feature = tf.stack(tf.unstack(gen_feature_transposed), axis=-1)

        # TODO: use norm and relu for generated features?
        gen_feature = norm_layer(gen_feature)
        gen_feature = tf.nn.relu(gen_feature)

        if 'states' in inputs:
            with tf.name_scope('gen_states'):
                with tf.variable_scope('state_pred'):
                    gen_state = dense(state_action, inputs['states'].shape[-1].value)

        outputs = {
            'gen_features': gen_feature,
            'gen_inputs': gen_input,
        }
        if 'states' in inputs:
            outputs['gen_states'] = gen_state
        if self.hparams.transformation == 'flow':
            outputs['gen_flows'] = flows

        new_states = {
            'time': time + 1,
            'gen_feature': gen_feature,
            'conv_rnn_states': new_conv_rnn_states,
        }
        if 'zs' in inputs and self.hparams.use_rnn_z:
            new_states['rnn_z_state'] = rnn_z_state
        if 'states' in inputs:
            new_states['gen_state'] = gen_state
        return outputs, new_states


def generator_fn(inputs, hparams=None):
    images = inputs['images']
    with tf.variable_scope('encoder'):
        features = tf.map_fn(create_pspnet50_encoder, images)
    features = tf.stop_gradient(features)

    inputs = dict(inputs)
    inputs['features'] = features
    inputs = {name: tf_utils.maybe_pad_or_slice(input, hparams.sequence_length - 1)
              for name, input in inputs.items()}

    cell = DynamicsCell(inputs, hparams)
    outputs, _ = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32, swap_memory=False, time_major=True)
    # the RNN outputs generated images from time step 1 to sequence_length,
    # but generator_fn should only return images past context_frames
    outputs = {name: output[hparams.context_frames - 1:] for name, output in outputs.items()}
    outputs['ground_truth_sampling_mean'] = tf.reduce_mean(tf.to_float(cell.ground_truth[hparams.context_frames:]))

    gen_features = outputs['gen_features']
    with tf.variable_scope('decoder') as decoder_scope:
        gen_images = tf.map_fn(create_decoder, tf.stop_gradient(gen_features))  # TODO: stop gradient for decoder?
    with tf.variable_scope(decoder_scope, reuse=True):
        gen_images_dec = tf.map_fn(create_decoder, features)
    outputs['gen_images'] = gen_images
    outputs['gen_images_dec'] = gen_images_dec
    outputs['features'] = features
    return gen_images, outputs


class PSPNet50VideoPredictionModel(VideoPredictionModel):
    def __init__(self, *args, **kwargs):
        super(PSPNet50VideoPredictionModel, self).__init__(
            generator_fn, *args, **kwargs)

    def get_default_hparams_dict(self):
        default_hparams = super(PSPNet50VideoPredictionModel, self).get_default_hparams_dict()
        hparams = dict(
            l1_weight=1.0,
            l2_weight=0.0,
            norm_layer='instance',
            ngf=512,
            transformation='direct',
            kernel_size=(5, 5),
            dilation_rate=(1, 1),
            rnn='gru',
            conv_rnn='gru',
            schedule_sampling='inverse_sigmoid',
            schedule_sampling_k=900.0,
            schedule_sampling_steps=(0, 100000),
            e_net='none',
            nz=0,
            num_samples=8,
            nef=32,
            use_rnn_z=True,
        )
        return dict(itertools.chain(default_hparams.items(), hparams.items()))

    def parse_hparams(self, hparams_dict, hparams):
        hparams = super(PSPNet50VideoPredictionModel, self).parse_hparams(hparams_dict, hparams)
        if self.mode == 'test':
            def override_hparams_maybe(name, value):
                orig_value = hparams.values()[name]
                if orig_value != value:
                    print('Overriding hparams from %s=%r to %r for mode=%s.' %
                          (name, orig_value, value, self.mode))
                    hparams.set_hparam(name, value)
            override_hparams_maybe('schedule_sampling', 'none')
        return hparams

    def restore(self, sess, checkpoints):
        pspnet_network.pspnet50_assign_from_values_fn(var_name_prefix=self.generator_scope + '/encoder/')(sess)
        super(PSPNet50VideoPredictionModel, self).restore(sess, checkpoints)
