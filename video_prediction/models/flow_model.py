import itertools

import tensorflow as tf

from video_prediction import ops, flow_ops
from video_prediction.models import VideoPredictionModel
from video_prediction.models.improved_dna_model import encoder_fn, discriminator_fn
from video_prediction.ops import dense, conv2d, upsample_conv2d, conv_pool2d, flatten, tile_concat
from video_prediction.rnn_ops import BasicConv2DLSTMCell, Conv2DGRUCell
from video_prediction.utils import tf_utils


class FlowCell(tf.nn.rnn_cell.RNNCell):
    def __init__(self, inputs, hparams, reuse=None):
        super(FlowCell, self).__init__(_reuse=reuse)
        self.inputs = inputs
        self.hparams = hparams

        batch_size = inputs['images'].shape[1].value
        image_shape = inputs['images'].shape.as_list()[2:]
        height, width, _ = image_shape
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
                (self.hparams.ngf * 2, True),
            ]
            self.decoder_layer_specs = [
                (self.hparams.ngf, True),
                (self.hparams.ngf, False),
            ]
        else:
            raise NotImplementedError

        # output_size
        gen_input_shape = list(image_shape)
        gen_input_shape[-1] *= self.hparams.last_frames
        if 'actions' in inputs:
            gen_input_shape[-1] += inputs['actions'].shape[-1].value
        num_masks = self.hparams.last_frames
        output_size = {
            'gen_images': tf.TensorShape(image_shape),
            'gen_inputs': tf.TensorShape(gen_input_shape),
            'transformed_images': tf.TensorShape(image_shape + [num_masks]),
            'masks': tf.TensorShape([height, width, 1, num_masks]),
            'gen_flows': tf.TensorShape([height, width, 2, self.hparams.last_frames]),
        }
        if 'pix_distribs' in inputs:
            num_motions = inputs['pix_distribs'].shape[-1].value
            output_size['gen_pix_distribs'] = tf.TensorShape([height, width, num_motions])
            output_size['transformed_pix_distribs'] = tf.TensorShape([height, width, num_motions, num_masks])
        if 'states' in inputs:
            output_size['gen_states'] = inputs['states'].shape[2:]
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
        state_size = {'time': tf.TensorShape([]),
                      'last_gt_images': [tf.TensorShape(image_shape)] * self.hparams.last_frames,
                      'last_pred_flows': [tf.TensorShape([height, width, 2])] * self.hparams.last_frames,
                      'gen_image': tf.TensorShape(image_shape),
                      'last_images': [tf.TensorShape(image_shape)] * self.hparams.last_frames,
                      'last_base_images': [tf.TensorShape(image_shape)] * self.hparams.last_frames,
                      'last_base_flows': [tf.TensorShape([height, width, 2])] * self.hparams.last_frames,
                      'conv_rnn_states': conv_rnn_state_sizes}
        if 'zs' in inputs and self.hparams.use_rnn_z:
            rnn_z_state_size = tf.TensorShape([self.hparams.nz])
            if self.hparams.rnn == 'lstm':
                rnn_z_state_size = tf.nn.rnn_cell.LSTMStateTuple(rnn_z_state_size, rnn_z_state_size)
            state_size['rnn_z_state'] = rnn_z_state_size
        if 'pix_distribs' in inputs:
            state_size['last_gt_pix_distribs'] = [tf.TensorShape([height, width, num_motions])] * self.hparams.last_frames
            state_size['last_base_pix_distribs'] = [tf.TensorShape([height, width, num_motions])] * self.hparams.last_frames
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
        init_state = super(FlowCell, self).zero_state(batch_size, dtype)
        init_state['last_gt_images'] = [self.inputs['images'][0]] * self.hparams.last_frames
        init_state['last_images'] = [self.inputs['images'][0]] * self.hparams.last_frames
        init_state['last_base_images'] = [self.inputs['images'][0]] * self.hparams.last_frames
        if 'pix_distribs' in self.inputs:
            init_state['last_gt_pix_distribs'] = [self.inputs['pix_distribs'][0]] * self.hparams.last_frames
            init_state['last_base_pix_distribs'] = [self.inputs['pix_distribs'][0]] * self.hparams.last_frames
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
        conv_rnn_cell = Conv2DRNNCell(input_shape, filters, kernel_size=(5, 5),
                                      normalizer_fn=normalizer_fn,
                                      separate_norms=self.hparams.norm_layer == 'layer',
                                      reuse=tf.get_variable_scope().reuse)
        return conv_rnn_cell(inputs, state)

    def call(self, inputs, states):
        norm_layer = ops.get_norm_layer(self.hparams.norm_layer)
        image_shape = inputs['images'].get_shape().as_list()
        batch_size, height, width, color_channels = image_shape
        conv_rnn_states = states['conv_rnn_states']

        time = states['time']
        with tf.control_dependencies([tf.assert_equal(time[1:], time[0])]):
            t = tf.to_int32(tf.identity(time[0]))

        last_gt_images = states['last_gt_images'][1:] + [inputs['images']]
        last_pred_flows = states['last_pred_flows'][1:] + [tf.zeros_like(states['last_pred_flows'][-1])]
        image = tf.where(self.ground_truth[t], inputs['images'], states['gen_image'])
        last_images = states['last_images'][1:] + [image]
        last_base_images = [tf.where(self.ground_truth[t], last_gt_image, last_base_image)
                            for last_gt_image, last_base_image in zip(last_gt_images, states['last_base_images'])]
        last_base_flows = [tf.where(self.ground_truth[t], last_pred_flow, last_base_flow)
                           for last_pred_flow, last_base_flow in zip(last_pred_flows, states['last_base_flows'])]
        if 'pix_distribs' in inputs:
            last_gt_pix_distribs = states['last_gt_pix_distribs'][1:] + [inputs['pix_distribs']]
            last_base_pix_distribs = [tf.where(self.ground_truth[t], last_gt_pix_distrib, last_base_pix_distrib)
                                      for last_gt_pix_distrib, last_base_pix_distrib in zip(last_gt_pix_distribs, states['last_base_pix_distribs'])]
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
            gen_input = tile_concat(last_images + [inputs['actions'][:, None, None, :]], axis=-1)
        else:
            gen_input = tf.concat(last_images)

        layers = []
        new_conv_rnn_states = []
        for i, (out_channels, use_conv_rnn) in enumerate(self.encoder_layer_specs):
            with tf.variable_scope('h%d' % i):
                if i == 0:
                    h = tf.concat(last_images, axis=-1)
                    kernel_size = (5, 5)
                else:
                    h = layers[-1][-1]
                    kernel_size = (3, 3)
                h = conv_pool2d(tile_concat([h, state_action_z[:, None, None, :]], axis=-1),
                                out_channels, kernel_size=kernel_size, strides=(2, 2))
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

        with tf.variable_scope('h%d_flow' % len(layers)):
            h_flow = conv2d(layers[-1][-1], self.hparams.ngf, kernel_size=(3, 3), strides=(1, 1))
            h_flow = norm_layer(h_flow)
            h_flow = tf.nn.relu(h_flow)

        with tf.variable_scope('flows'):
            flows = conv2d(h_flow, 2 * self.hparams.last_frames, kernel_size=(3, 3), strides=(1, 1))
            flows = tf.reshape(flows, [batch_size, height, width, 2, self.hparams.last_frames])

        with tf.name_scope('transformed_images'):
            transformed_images = []
            last_pred_flows = [flow + flow_ops.image_warp(last_pred_flow, flow)
                               for last_pred_flow, flow in zip(last_pred_flows, tf.unstack(flows, axis=-1))]
            last_base_flows = [flow + flow_ops.image_warp(last_base_flow, flow)
                               for last_base_flow, flow in zip(last_base_flows, tf.unstack(flows, axis=-1))]
            for last_base_image, last_base_flow in zip(last_base_images, last_base_flows):
                transformed_images.append(flow_ops.image_warp(last_base_image, last_base_flow))

        if 'pix_distribs' in inputs:
            with tf.name_scope('transformed_pix_distribs'):
                transformed_pix_distribs = []
                for last_base_pix_distrib, last_base_flow in zip(last_base_pix_distribs, last_base_flows):
                    transformed_pix_distribs.append(flow_ops.image_warp(last_base_pix_distrib, last_base_flow))

        with tf.name_scope('masks'):
            if len(transformed_images) > 1:
                with tf.variable_scope('h%d_masks' % len(layers)):
                    h_masks = conv2d(layers[-1][-1], self.hparams.ngf, kernel_size=(3, 3), strides=(1, 1))
                    h_masks = norm_layer(h_masks)
                    h_masks = tf.nn.relu(h_masks)

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
                # TODO: is this needed?
                # gen_pix_distrib /= tf.reduce_sum(gen_pix_distrib, axis=(1, 2), keepdims=True)

        if 'states' in inputs:
            with tf.name_scope('gen_states'):
                with tf.variable_scope('state_pred'):
                    gen_state = dense(state_action, inputs['states'].shape[-1].value)

        outputs = {'gen_images': gen_image,
                   'gen_inputs': gen_input,
                   'transformed_images': tf.stack(transformed_images, axis=-1),
                   'masks': tf.stack(masks, axis=-1),
                   'gen_flow': flows}
        if 'pix_distribs' in inputs:
            outputs['gen_pix_distribs'] = gen_pix_distrib
            outputs['transformed_pix_distribs'] = tf.stack(transformed_pix_distribs, axis=-1)
        if 'states' in inputs:
            outputs['gen_states'] = gen_state

        new_states = {'time': time + 1,
                      'last_gt_images': last_gt_images,
                      'last_pred_flows': last_pred_flows,
                      'gen_image': gen_image,
                      'last_images': last_images,
                      'last_base_images': last_base_images,
                      'last_base_flows': last_base_flows,
                      'conv_rnn_states': new_conv_rnn_states}
        if 'zs' in inputs and self.hparams.use_rnn_z:
            new_states['rnn_z_state'] = rnn_z_state
        if 'pix_distribs' in inputs:
            new_states['last_gt_pix_distribs'] = last_gt_pix_distribs
            new_states['last_base_pix_distribs'] = last_base_pix_distribs
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
    cell = FlowCell(inputs, hparams)
    outputs, _ = tf.nn.dynamic_rnn(cell, inputs, dtype=tf.float32,
                                   swap_memory=False, time_major=True)
    if hparams.nz:
        inputs_samples = {name: flatten(tf.tile(input[:, None], [1, hparams.num_samples] + [1] * (input.shape.ndims - 1)), 1, 2)
                          for name, input in inputs.items() if name != 'zs'}
        inputs_samples['zs'] = tf.concat([sample_zs() for _ in range(hparams.num_samples)], axis=1)
        with tf.variable_scope(tf.get_variable_scope(), reuse=True):
            cell_samples = FlowCell(inputs_samples, hparams)
            outputs_samples, _ = tf.nn.dynamic_rnn(cell_samples, inputs_samples, dtype=tf.float32,
                                                   swap_memory=False, time_major=True)
        gen_images_samples = outputs_samples['gen_images']
        gen_images_samples = tf.stack(tf.split(gen_images_samples, hparams.num_samples, axis=1), axis=-1)
        gen_images_samples_avg = tf.reduce_mean(gen_images_samples, axis=-1)
        outputs['gen_images_samples'] = gen_images_samples
        outputs['gen_images_samples_avg'] = gen_images_samples_avg
    # the RNN outputs generated images from time step 1 to sequence_length,
    # but generator_fn should only return images past context_frames
    outputs = {name: output[hparams.context_frames - 1:] for name, output in outputs.items()}
    gen_images = outputs['gen_images']
    outputs['ground_truth_sampling_mean'] = tf.reduce_mean(tf.to_float(cell.ground_truth[hparams.context_frames:]))
    return gen_images, outputs


class FlowVideoPredictionModel(VideoPredictionModel):
    def __init__(self, *args, **kwargs):
        super(FlowVideoPredictionModel, self).__init__(
            generator_fn, discriminator_fn, encoder_fn, *args, **kwargs)
        if self.hparams.e_net == 'none' or self.hparams.nz == 0:
            self.encoder_fn = None
        if self.hparams.d_net == 'none':
            self.discriminator_fn = None

    def get_default_hparams_dict(self):
        default_hparams = super(FlowVideoPredictionModel, self).get_default_hparams_dict()
        hparams = dict(
            l1_weight=1.0,
            l2_weight=0.0,
            d_net='legacy',
            n_layers=3,
            ndf=32,
            norm_layer='instance',
            d_downsample_layer='conv_pool2d',
            d_conditional=True,
            d_use_gt_inputs=True,
            ngf=32,
            rnn='lstm',
            conv_rnn='lstm',
            last_frames=1,
            dependent_mask=True,
            schedule_sampling='inverse_sigmoid',
            schedule_sampling_k=900.0,
            schedule_sampling_steps=(0, 100000),
            e_net='n_layer',
            nz=8,
            num_samples=8,
            nef=32,
            use_rnn_z=True,
        )
        return dict(itertools.chain(default_hparams.items(), hparams.items()))

    def parse_hparams(self, hparams_dict, hparams):
        hparams = super(FlowVideoPredictionModel, self).parse_hparams(hparams_dict, hparams)
        if self.mode == 'test':
            def override_hparams_maybe(name, value):
                orig_value = hparams.values()[name]
                if orig_value != value:
                    print('Overriding hparams from %s=%r to %r for mode=%s.' %
                          (name, orig_value, value, self.mode))
                    hparams.set_hparam(name, value)
            override_hparams_maybe('d_net', 'none')
            override_hparams_maybe('e_net', 'none')
            override_hparams_maybe('schedule_sampling', 'none')
        return hparams
