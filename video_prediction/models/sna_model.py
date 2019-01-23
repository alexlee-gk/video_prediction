# Copyright 2016 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Model architecture for predictive model, including CDNA, DNA, and STP."""

import itertools

import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.contrib.layers.python import layers as tf_layers
from tensorflow.contrib.slim import add_arg_scope
from tensorflow.contrib.slim import layers

from video_prediction.models import VideoPredictionModel


# Amount to use when lower bounding tensors
RELU_SHIFT = 1e-12


@add_arg_scope
def basic_conv_lstm_cell(inputs,
                         state,
                         num_channels,
                         filter_size=5,
                         forget_bias=1.0,
                         scope=None,
                         reuse=None,
                         ):
    """Basic LSTM recurrent network cell, with 2D convolution connctions.
    We add forget_bias (default: 1) to the biases of the forget gate in order to
    reduce the scale of forgetting in the beginning of the training.
    It does not allow cell clipping, a projection layer, and does not
    use peep-hole connections: it is the basic baseline.
    Args:
        inputs: input Tensor, 4D, batch x height x width x channels.
        state: state Tensor, 4D, batch x height x width x channels.
        num_channels: the number of output channels in the layer.
        filter_size: the shape of the each convolution filter.
        forget_bias: the initial value of the forget biases.
        scope: Optional scope for variable_scope.
        reuse: whether or not the layer and the variables should be reused.
    Returns:
        a tuple of tensors representing output and the new state.
    """
    if state is None:
        state = tf.zeros(inputs.get_shape().as_list()[:3] + [2 * num_channels], name='init_state')

    with tf.variable_scope(scope,
                           'BasicConvLstmCell',
                           [inputs, state],
                           reuse=reuse):

        inputs.get_shape().assert_has_rank(4)
        state.get_shape().assert_has_rank(4)
        c, h = tf.split(axis=3, num_or_size_splits=2, value=state)
        inputs_h = tf.concat(values=[inputs, h], axis=3)
        # Parameters of gates are concatenated into one conv for efficiency.
        i_j_f_o = layers.conv2d(inputs_h,
                                4 * num_channels, [filter_size, filter_size],
                                stride=1,
                                activation_fn=None,
                                scope='Gates',
                                )

        # i = input_gate, j = new_input, f = forget_gate, o = output_gate
        i, j, f, o = tf.split(value=i_j_f_o, num_or_size_splits=4, axis=3)

        new_c = c * tf.sigmoid(f + forget_bias) + tf.sigmoid(i) * tf.tanh(j)
        new_h = tf.tanh(new_c) * tf.sigmoid(o)

        return new_h, tf.concat(values=[new_c, new_h], axis=3)


class Prediction_Model(object):

    def __init__(self,
                 images,
                 actions=None,
                 states=None,
                 iter_num=-1.0,
                 pix_distributions1=None,
                 pix_distributions2=None,
                 conf=None):

        self.pix_distributions1 = pix_distributions1
        self.pix_distributions2 = pix_distributions2
        self.actions = actions
        self.iter_num = iter_num
        self.conf = conf
        self.images = images

        self.cdna, self.stp, self.dna = False, False, False
        if self.conf['model'] == 'CDNA':
            self.cdna = True
        elif self.conf['model'] == 'DNA':
            self.dna = True
        elif self.conf['model'] == 'STP':
            self.stp = True
        if self.stp + self.cdna + self.dna != 1:
            raise ValueError("More than one option selected!")

        self.k = conf['schedsamp_k']
        self.use_state = conf['use_state']
        self.num_masks = conf['num_masks']
        self.context_frames = conf['context_frames']

        self.batch_size, self.img_height, self.img_width, self.color_channels = [int(i) for i in
                                                                                 images[0].get_shape()[0:4]]
        self.lstm_func = basic_conv_lstm_cell

        # Generated robot states and images.
        self.gen_states = []
        self.gen_images = []
        self.gen_masks = []

        self.moved_images = []

        self.moved_pix_distrib1 = []
        self.moved_pix_distrib2 = []

        self.states = states
        self.gen_distrib1 = []
        self.gen_distrib2 = []

        self.trafos = []

    def build(self):

        if 'kern_size' in self.conf.keys():
            KERN_SIZE = self.conf['kern_size']
        else:
            KERN_SIZE = 5

        batch_size, img_height, img_width, color_channels = self.images[0].get_shape()[0:4]
        lstm_func = basic_conv_lstm_cell


        if self.states != None:
            current_state = self.states[0]
        else:
            current_state = None

        if self.actions == None:
            self.actions = [None for _ in self.images]

        if self.k == -1:
            feedself = True
        else:
            # Scheduled sampling:
            # Calculate number of ground-truth frames to pass in.
            num_ground_truth = tf.to_int32(
                tf.round(tf.to_float(batch_size) * (self.k / (self.k + tf.exp(self.iter_num / self.k)))))
            feedself = False

        # LSTM state sizes and states.

        if 'lstm_size' in self.conf:
            lstm_size = self.conf['lstm_size']
            print('using lstm size', lstm_size)
        else:
            ngf = self.conf['ngf']
            lstm_size = np.int32(np.array([ngf, ngf * 2, ngf * 4, ngf * 2, ngf]))


        lstm_state1, lstm_state2, lstm_state3, lstm_state4 = None, None, None, None
        lstm_state5, lstm_state6, lstm_state7 = None, None, None

        for t, action in enumerate(self.actions):
            print(t)
            # Reuse variables after the first timestep.
            reuse = bool(self.gen_images)

            done_warm_start = len(self.gen_images) > self.context_frames - 1
            with slim.arg_scope(
                    [lstm_func, slim.layers.conv2d, slim.layers.fully_connected,
                     tf_layers.layer_norm, slim.layers.conv2d_transpose],
                    reuse=reuse):

                if feedself and done_warm_start:
                    # Feed in generated image.
                    prev_image = self.gen_images[-1]             # 64x64x6
                    if self.pix_distributions1 != None:
                        prev_pix_distrib1 = self.gen_distrib1[-1]
                        if 'ndesig' in self.conf:
                            prev_pix_distrib2 = self.gen_distrib2[-1]
                elif done_warm_start:
                    # Scheduled sampling
                    prev_image = scheduled_sample(self.images[t], self.gen_images[-1], batch_size,
                                                  num_ground_truth)
                else:
                    # Always feed in ground_truth
                    prev_image = self.images[t]
                    if self.pix_distributions1 != None:
                        prev_pix_distrib1 = self.pix_distributions1[t]
                        if 'ndesig' in self.conf:
                            prev_pix_distrib2 = self.pix_distributions2[t]
                        if len(prev_pix_distrib1.get_shape()) == 3:
                            prev_pix_distrib1 = tf.expand_dims(prev_pix_distrib1, -1)
                            if 'ndesig' in self.conf:
                                prev_pix_distrib2 = tf.expand_dims(prev_pix_distrib2, -1)

                if 'refeed_firstimage' in self.conf:
                    assert self.conf['model']=='STP'
                    if t > 1:
                        input_image = self.images[1]
                        print('refeed with image 1')
                    else:
                        input_image = prev_image
                else:
                    input_image = prev_image

                # Predicted state is always fed back in
                if not 'ignore_state_action' in self.conf:
                    state_action = tf.concat(axis=1, values=[action, current_state])

                enc0 = slim.layers.conv2d(    #32x32x32
                    input_image,
                    32, [5, 5],
                    stride=2,
                    scope='scale1_conv1',
                    normalizer_fn=tf_layers.layer_norm,
                    normalizer_params={'scope': 'layer_norm1'})

                hidden1, lstm_state1 = lstm_func(       # 32x32x16
                    enc0, lstm_state1, lstm_size[0], scope='state1')
                hidden1 = tf_layers.layer_norm(hidden1, scope='layer_norm2')

                enc1 = slim.layers.conv2d(     # 16x16x16
                    hidden1, hidden1.get_shape()[3], [3, 3], stride=2, scope='conv2')

                hidden3, lstm_state3 = lstm_func(   #16x16x32
                    enc1, lstm_state3, lstm_size[1], scope='state3')
                hidden3 = tf_layers.layer_norm(hidden3, scope='layer_norm4')

                enc2 = slim.layers.conv2d(  # 8x8x32
                    hidden3, hidden3.get_shape()[3], [3, 3], stride=2, scope='conv3')

                if not 'ignore_state_action' in self.conf:
                    # Pass in state and action.
                    if 'ignore_state' in self.conf:
                        lowdim = action
                        print('ignoring state')
                    else:
                        lowdim = state_action

                    smear = tf.reshape(
                        lowdim,
                        [int(batch_size), 1, 1, int(lowdim.get_shape()[1])])
                    smear = tf.tile(
                        smear, [1, int(enc2.get_shape()[1]), int(enc2.get_shape()[2]), 1])

                    enc2 = tf.concat(axis=3, values=[enc2, smear])
                else:
                    print('ignoring states and actions')

                enc3 = slim.layers.conv2d(   #8x8x32
                    enc2, hidden3.get_shape()[3], [1, 1], stride=1, scope='conv4')

                hidden5, lstm_state5 = lstm_func(  #8x8x64
                    enc3, lstm_state5, lstm_size[2], scope='state5')
                hidden5 = tf_layers.layer_norm(hidden5, scope='layer_norm6')
                enc4 = slim.layers.conv2d_transpose(  #16x16x64
                    hidden5, hidden5.get_shape()[3], 3, stride=2, scope='convt1')

                hidden6, lstm_state6 = lstm_func(  #16x16x32
                    enc4, lstm_state6, lstm_size[3], scope='state6')
                hidden6 = tf_layers.layer_norm(hidden6, scope='layer_norm7')

                if 'noskip' not in self.conf:
                    # Skip connection.
                    hidden6 = tf.concat(axis=3, values=[hidden6, enc1])  # both 16x16

                enc5 = slim.layers.conv2d_transpose(  #32x32x32
                    hidden6, hidden6.get_shape()[3], 3, stride=2, scope='convt2')
                hidden7, lstm_state7 = lstm_func( # 32x32x16
                    enc5, lstm_state7, lstm_size[4], scope='state7')
                hidden7 = tf_layers.layer_norm(hidden7, scope='layer_norm8')

                if not 'noskip' in self.conf:
                    # Skip connection.
                    hidden7 = tf.concat(axis=3, values=[hidden7, enc0])  # both 32x32

                enc6 = slim.layers.conv2d_transpose(   # 64x64x16
                    hidden7,
                    hidden7.get_shape()[3], 3, stride=2, scope='convt3',
                    normalizer_fn=tf_layers.layer_norm,
                    normalizer_params={'scope': 'layer_norm9'})

                if 'transform_from_firstimage' in self.conf:
                    prev_image = self.images[1]
                    if self.pix_distributions1 != None:
                        prev_pix_distrib1 = self.pix_distributions1[1]
                        prev_pix_distrib1 = tf.expand_dims(prev_pix_distrib1, -1)
                    print('transform from image 1')

                if self.conf['model'] == 'DNA':
                    # Using largest hidden state for predicting untied conv kernels.
                    trafo_input = slim.layers.conv2d_transpose(
                        enc6, KERN_SIZE ** 2, 1, stride=1, scope='convt4_cam2')

                    transformed_l = [self.dna_transformation(prev_image, trafo_input, self.conf['kern_size'])]
                    if self.pix_distributions1 != None:
                        transf_distrib_ndesig1 = [self.dna_transformation(prev_pix_distrib1, trafo_input, KERN_SIZE)]
                        if 'ndesig' in self.conf:
                            transf_distrib_ndesig2 = [
                                self.dna_transformation(prev_pix_distrib2, trafo_input, KERN_SIZE)]


                    extra_masks = 1  ## extra_masks = 2 is needed for running singleview_shifted!!
                    # print('using extra masks 2 because of single view shifted!!')
                    # extra_masks = 2

                if self.conf['model'] == 'CDNA':
                    if 'gen_pix' in self.conf:
                        # Using largest hidden state for predicting a new image layer.
                        enc7 = slim.layers.conv2d_transpose(
                            enc6, color_channels, 1, stride=1, scope='convt4', activation_fn=None)
                        # This allows the network to also generate one image from scratch,
                        # which is useful when regions of the image become unoccluded.
                        transformed_l = [tf.nn.sigmoid(enc7)]
                        extra_masks = 2
                    else:
                        transformed_l = []
                        extra_masks = 1

                    cdna_input = tf.reshape(hidden5, [int(batch_size), -1])
                    new_transformed, _ = self.cdna_transformation(prev_image,
                                                            cdna_input,
                                                            reuse_sc=reuse)
                    transformed_l += new_transformed
                    self.moved_images.append(transformed_l)

                    if self.pix_distributions1 != None:
                        transf_distrib_ndesig1, _ = self.cdna_transformation(prev_pix_distrib1,
                                                                       cdna_input,
                                                                         reuse_sc=True)
                        self.moved_pix_distrib1.append(transf_distrib_ndesig1)
                        if 'ndesig' in self.conf:
                            transf_distrib_ndesig2, _ = self.cdna_transformation(
                                                                               prev_pix_distrib2,
                                                                               cdna_input,
                                                                               reuse_sc=True)

                            self.moved_pix_distrib2.append(transf_distrib_ndesig2)

                if self.conf['model'] == 'STP':
                    enc7 = slim.layers.conv2d_transpose(enc6, color_channels, 1, stride=1, scope='convt5', activation_fn= None)
                    # This allows the network to also generate one image from scratch,
                    # which is useful when regions of the image become unoccluded.
                    if 'gen_pix' in self.conf:
                        transformed_l = [tf.nn.sigmoid(enc7)]
                        extra_masks = 2
                    else:
                        transformed_l = []
                        extra_masks = 1

                    enc_stp = tf.reshape(hidden5, [int(batch_size), -1])
                    stp_input = slim.layers.fully_connected(
                        enc_stp, 200, scope='fc_stp_cam2')

                    # disabling capability to generete pixels
                    reuse_stp = None
                    if reuse:
                        reuse_stp = reuse

                    # enable the generation of pixels:
                    transformed, trafo = self.stp_transformation(prev_image, stp_input, self.num_masks, reuse_stp, suffix='cam2')
                    transformed_l += transformed

                    self.trafos.append(trafo)
                    self.moved_images.append(transformed_l)

                    if self.pix_distributions1 != None:
                        transf_distrib_ndesig1, _ = self.stp_transformation(prev_pix_distrib1, stp_input, suffix='cam2', reuse=True)
                        self.moved_pix_distrib1.append(transf_distrib_ndesig1)

                if '1stimg_bckgd' in self.conf:
                    background = self.images[0]
                    print('using background from first image..')
                else: background = prev_image
                output, mask_list = self.fuse_trafos(enc6, background,
                                                     transformed_l,
                                                     scope='convt7_cam2',
                                                     extra_masks= extra_masks)
                self.gen_images.append(output)
                self.gen_masks.append(mask_list)

                if self.pix_distributions1!=None:
                    pix_distrib_output = self.fuse_pix_distrib(extra_masks,
                                                                mask_list,
                                                                self.pix_distributions1,
                                                                prev_pix_distrib1,
                                                                transf_distrib_ndesig1)

                    self.gen_distrib1.append(pix_distrib_output)
                    if 'ndesig' in self.conf:
                        pix_distrib_output = self.fuse_pix_distrib(extra_masks,
                                                                    mask_list,
                                                                    self.pix_distributions2,
                                                                    prev_pix_distrib2,
                                                                    transf_distrib_ndesig2)

                        self.gen_distrib2.append(pix_distrib_output)

                if int(current_state.get_shape()[1]) == 0:
                    current_state = tf.zeros_like(state_action)
                else:
                    current_state = slim.layers.fully_connected(
                        state_action,
                        int(current_state.get_shape()[1]),
                        scope='state_pred',
                        activation_fn=None)

                self.gen_states.append(current_state)

    def fuse_trafos(self, enc6, background_image, transformed, scope, extra_masks):
        masks = slim.layers.conv2d_transpose(
            enc6, (self.conf['num_masks']+ extra_masks), 1, stride=1, activation_fn=None, scope=scope)

        img_height = 64
        img_width = 64
        num_masks = self.conf['num_masks']

        if self.conf['model']=='DNA':
            if num_masks != 1:
                raise ValueError('Only one mask is supported for DNA model.')

        # the total number of masks is num_masks +extra_masks because of background and generated pixels!
        masks = tf.reshape(
            tf.nn.softmax(tf.reshape(masks, [-1, num_masks +extra_masks])),
            [int(self.batch_size), int(img_height), int(img_width), num_masks +extra_masks])
        mask_list = tf.split(axis=3, num_or_size_splits=num_masks +extra_masks, value=masks)
        output = mask_list[0] * background_image

        assert len(transformed) == len(mask_list[1:])
        for layer, mask in zip(transformed, mask_list[1:]):
            output += layer * mask

        return output, mask_list

    def fuse_pix_distrib(self, extra_masks, mask_list, pix_distributions, prev_pix_distrib,
                         transf_distrib):

        if '1stimg_bckgd' in self.conf:
            background_pix = pix_distributions[0]
            if len(background_pix.get_shape()) == 3:
                background_pix = tf.expand_dims(background_pix, -1)
            print('using pix_distrib-background from first image..')
        else:
            background_pix = prev_pix_distrib
        pix_distrib_output = mask_list[0] * background_pix
        if 'gen_pix' in self.conf:
            pix_distrib_output += mask_list[1] * prev_pix_distrib  # assume pixels don't when image is generated from scratch
        for i in range(self.num_masks):
            pix_distrib_output += transf_distrib[i] * mask_list[i + extra_masks]
        pix_distrib_output /= tf.reduce_sum(pix_distrib_output, axis=(1, 2), keepdims=True)
        return pix_distrib_output

    ## Utility functions
    def stp_transformation(self, prev_image, stp_input, num_masks, reuse= None, suffix = None):
        """Apply spatial transformer predictor (STP) to previous image.

        Args:
          prev_image: previous image to be transformed.
          stp_input: hidden layer to be used for computing STN parameters.
          num_masks: number of masks and hence the number of STP transformations.
        Returns:
          List of images transformed by the predicted STP parameters.
        """
        # Only import spatial transformer if needed.
        from spatial_transformer import transformer

        identity_params = tf.convert_to_tensor(
            np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0], np.float32))
        transformed = []
        trafos = []
        for i in range(num_masks):
            params = slim.layers.fully_connected(
                stp_input, 6, scope='stp_params' + str(i) + suffix,
                activation_fn=None,
                reuse= reuse) + identity_params
            outsize = (prev_image.get_shape()[1], prev_image.get_shape()[2])
            transformed.append(transformer(prev_image, params, outsize))
            trafos.append(params)

        return transformed, trafos

    def dna_transformation(self, prev_image, dna_input, DNA_KERN_SIZE):
        """Apply dynamic neural advection to previous image.

        Args:
          prev_image: previous image to be transformed.
          dna_input: hidden lyaer to be used for computing DNA transformation.
        Returns:
          List of images transformed by the predicted CDNA kernels.
        """
        # Construct translated images.
        pad_len = int(np.floor(DNA_KERN_SIZE / 2))
        prev_image_pad = tf.pad(prev_image, [[0, 0], [pad_len, pad_len], [pad_len, pad_len], [0, 0]])
        image_height = int(prev_image.get_shape()[1])
        image_width = int(prev_image.get_shape()[2])

        inputs = []
        for xkern in range(DNA_KERN_SIZE):
            for ykern in range(DNA_KERN_SIZE):
                inputs.append(
                    tf.expand_dims(
                        tf.slice(prev_image_pad, [0, xkern, ykern, 0],
                                 [-1, image_height, image_width, -1]), [3]))
        inputs = tf.concat(axis=3, values=inputs)

        # Normalize channels to 1.
        kernel = tf.nn.relu(dna_input - RELU_SHIFT) + RELU_SHIFT
        kernel = tf.expand_dims(
            kernel / tf.reduce_sum(
                kernel, [3], keepdims=True), [4])

        return tf.reduce_sum(kernel * inputs, [3], keepdims=False)

    def cdna_transformation(self, prev_image, cdna_input, reuse_sc=None):
        """Apply convolutional dynamic neural advection to previous image.

        Args:
          prev_image: previous image to be transformed.
          cdna_input: hidden lyaer to be used for computing CDNA kernels.
          num_masks: the number of masks and hence the number of CDNA transformations.
          color_channels: the number of color channels in the images.
        Returns:
          List of images transformed by the predicted CDNA kernels.
        """
        batch_size = int(cdna_input.get_shape()[0])
        height = int(prev_image.get_shape()[1])
        width = int(prev_image.get_shape()[2])

        DNA_KERN_SIZE = self.conf['kern_size']
        num_masks = self.conf['num_masks']
        color_channels = int(prev_image.get_shape()[3])

        # Predict kernels using linear function of last hidden layer.
        cdna_kerns = slim.layers.fully_connected(
            cdna_input,
            DNA_KERN_SIZE * DNA_KERN_SIZE * num_masks,
            scope='cdna_params',
            activation_fn=None,
            reuse = reuse_sc)

        # Reshape and normalize.
        cdna_kerns = tf.reshape(
            cdna_kerns, [batch_size, DNA_KERN_SIZE, DNA_KERN_SIZE, 1, num_masks])
        cdna_kerns = tf.nn.relu(cdna_kerns - RELU_SHIFT) + RELU_SHIFT
        norm_factor = tf.reduce_sum(cdna_kerns, [1, 2, 3], keepdims=True)
        cdna_kerns /= norm_factor
        cdna_kerns_summary = cdna_kerns

        # Transpose and reshape.
        cdna_kerns = tf.transpose(cdna_kerns, [1, 2, 0, 4, 3])
        cdna_kerns = tf.reshape(cdna_kerns, [DNA_KERN_SIZE, DNA_KERN_SIZE, batch_size, num_masks])
        prev_image = tf.transpose(prev_image, [3, 1, 2, 0])

        transformed = tf.nn.depthwise_conv2d(prev_image, cdna_kerns, [1, 1, 1, 1], 'SAME')

        # Transpose and reshape.
        transformed = tf.reshape(transformed, [color_channels, height, width, batch_size, num_masks])
        transformed = tf.transpose(transformed, [3, 1, 2, 0, 4])
        transformed = tf.unstack(value=transformed, axis=-1)

        return transformed, cdna_kerns_summary


def scheduled_sample(ground_truth_x, generated_x, batch_size, num_ground_truth):
    """Sample batch with specified mix of ground truth and generated data_files points.

    Args:
      ground_truth_x: tensor of ground-truth data_files points.
      generated_x: tensor of generated data_files points.
      batch_size: batch size
      num_ground_truth: number of ground-truth examples to include in batch.
    Returns:
      New batch with num_ground_truth sampled from ground_truth_x and the rest
      from generated_x.
    """
    idx = tf.random_shuffle(tf.range(int(batch_size)))
    ground_truth_idx = tf.gather(idx, tf.range(num_ground_truth))
    generated_idx = tf.gather(idx, tf.range(num_ground_truth, int(batch_size)))

    ground_truth_examps = tf.gather(ground_truth_x, ground_truth_idx)
    generated_examps = tf.gather(generated_x, generated_idx)
    return tf.dynamic_stitch([ground_truth_idx, generated_idx],
                             [ground_truth_examps, generated_examps])


def generator_fn(inputs, mode, hparams):
    images = tf.unstack(inputs['images'], axis=0)
    actions = tf.unstack(inputs['actions'], axis=0)
    states = tf.unstack(inputs['states'], axis=0)
    pix_distributions1 = tf.unstack(inputs['pix_distribs'], axis=0) if 'pix_distribs' in inputs else None
    iter_num = tf.to_float(tf.train.get_or_create_global_step())

    if isinstance(hparams.kernel_size, (tuple, list)):
        kernel_height, kernel_width = hparams.kernel_size
        assert kernel_height == kernel_width
        kern_size = kernel_height
    else:
        kern_size = hparams.kernel_size

    schedule_sampling_k = hparams.schedule_sampling_k if mode == 'train' else -1
    conf = {
        'context_frames': hparams.context_frames,  # of frames before predictions.' ,
        'use_state': 1,  # 'Whether or not to give the state+action to the model' ,
        'ngf': hparams.ngf,
        'model': hparams.transformation.upper(),  # 'model architecture to use - CDNA, DNA, or STP' ,
        'num_masks': hparams.num_masks,  # 'number of masks, usually 1 for DNA, 10 for CDNA, STN.' ,
        'schedsamp_k': schedule_sampling_k,  # 'The k hyperparameter for scheduled sampling -1 for no scheduled sampling.' ,
        'kern_size': kern_size,  # size of DNA kerns
    }
    if hparams.first_image_background:
        conf['1stimg_bckgd'] = ''
    if hparams.generate_scratch_image:
        conf['gen_pix'] = ''

    m = Prediction_Model(images, actions, states,
                         pix_distributions1=pix_distributions1,
                         iter_num=iter_num, conf=conf)
    m.build()
    outputs = {
        'gen_images': tf.stack(m.gen_images, axis=0),
        'gen_states': tf.stack(m.gen_states, axis=0),
    }
    if 'pix_distribs' in inputs:
        outputs['gen_pix_distribs'] = tf.stack(m.gen_distrib1, axis=0)
    return outputs


class SNAVideoPredictionModel(VideoPredictionModel):
    def __init__(self, *args, **kwargs):
        super(SNAVideoPredictionModel, self).__init__(
            generator_fn, *args, **kwargs)

    def get_default_hparams_dict(self):
        default_hparams = super(SNAVideoPredictionModel, self).get_default_hparams_dict()
        hparams = dict(
            batch_size=32,
            l1_weight=0.0,
            l2_weight=1.0,
            ngf=16,
            transformation='cdna',
            kernel_size=(5, 5),
            num_masks=10,
            first_image_background=True,
            generate_scratch_image=True,
            schedule_sampling_k=900.0,
        )
        return dict(itertools.chain(default_hparams.items(), hparams.items()))
