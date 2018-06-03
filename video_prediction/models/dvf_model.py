import functools
import itertools

import tensorflow as tf
import tensorflow.contrib.slim as slim

import video_prediction as vp
from video_prediction.models import VideoPredictionModel
from video_prediction.utils import tf_utils


def bilinear_interp(im, x, y, name):
    """Perform bilinear sampling on im given x, y coordinates

    This function implements the differentiable sampling mechanism with
    bilinear kernel. Introduced in https://arxiv.org/abs/1506.02025, equation
    (5).

    x,y are tensors specfying normalized coorindates [-1,1] to sample from im.
    (-1,1) means (0,0) coordinate in im. (1,1) means the most bottom right pixel.

    Args:
      im: Tensor of size [batch_size, height, width, depth]
      x: Tensor of size [batch_size, height, width, 1]
      y: Tensor of size [batch_size, height, width, 1]
      name: String for the name for this opt.
    Returns:
      Tensor of size [batch_size, height, width, depth]
    """
    with tf.variable_scope(name):
        x = tf.reshape(x, [-1])
        y = tf.reshape(y, [-1])

        # constants
        num_batch = tf.shape(im)[0]
        _, height, width, channels = im.get_shape().as_list()

        x = tf.to_float(x)
        y = tf.to_float(y)

        height_f = tf.cast(height, 'float32')
        width_f = tf.cast(width, 'float32')
        zero = tf.constant(0, dtype=tf.int32)

        max_x = tf.cast(tf.shape(im)[2] - 1, 'int32')
        max_y = tf.cast(tf.shape(im)[1] - 1, 'int32')
        x = (x + 1.0) * (width_f - 1.0) / 2.0
        y = (y + 1.0) * (height_f - 1.0) / 2.0

        # Sampling
        x0 = tf.cast(tf.floor(x), 'int32')
        x1 = x0 + 1
        y0 = tf.cast(tf.floor(y), 'int32')
        y1 = y0 + 1

        x0 = tf.clip_by_value(x0, zero, max_x)
        x1 = tf.clip_by_value(x1, zero, max_x)
        y0 = tf.clip_by_value(y0, zero, max_y)
        y1 = tf.clip_by_value(y1, zero, max_y)

        dim2 = width
        dim1 = width * height

        # Create base index
        base = tf.range(num_batch) * dim1
        base = tf.reshape(base, [-1, 1])
        base = tf.tile(base, [1, height * width])
        base = tf.reshape(base, [-1])

        base_y0 = base + y0 * dim2
        base_y1 = base + y1 * dim2
        idx_a = base_y0 + x0
        idx_b = base_y1 + x0
        idx_c = base_y0 + x1
        idx_d = base_y1 + x1

        # Use indices to look up pixels
        im_flat = tf.reshape(im, tf.stack([-1, channels]))
        im_flat = tf.to_float(im_flat)
        pixel_a = tf.gather(im_flat, idx_a)
        pixel_b = tf.gather(im_flat, idx_b)
        pixel_c = tf.gather(im_flat, idx_c)
        pixel_d = tf.gather(im_flat, idx_d)

        # Interpolate the values
        x1_f = tf.to_float(x1)
        y1_f = tf.to_float(y1)

        wa = tf.expand_dims((x1_f - x) * (y1_f - y), 1)
        wb = tf.expand_dims((x1_f - x) * (1.0 - (y1_f - y)), 1)
        wc = tf.expand_dims((1.0 - (x1_f - x)) * (y1_f - y), 1)
        wd = tf.expand_dims((1.0 - (x1_f - x)) * (1.0 - (y1_f - y)), 1)

        output = tf.add_n([wa * pixel_a, wb * pixel_b, wc * pixel_c, wd * pixel_d])
        output = tf.reshape(output, shape=tf.stack([num_batch, height, width, channels]))
        return output


def generator_fn(inputs, mode, hparams=None):
    inputs = {name: tf_utils.maybe_pad_or_slice(input, hparams.sequence_length - 1)
              for name, input in inputs.items()}
    images = inputs['images']
    input_images = tf.concat(tf.unstack(images[:hparams.context_frames], axis=0), axis=-1)

    with slim.arg_scope([slim.conv2d],
                        activation_fn=tf.nn.relu,
                        weights_initializer=tf.truncated_normal_initializer(0.0, 0.01),
                        weights_regularizer=slim.l2_regularizer(0.0001)):
        batch_norm_params = {
            'decay': 0.9997,
            'epsilon': 0.001,
            'is_training': mode == 'train',
        }
        with slim.arg_scope([slim.batch_norm], is_training=mode == 'train', updates_collections=None):
            with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm,
                                normalizer_params=batch_norm_params):
                h0 = slim.conv2d(input_images, 64, [5, 5], stride=1, scope='conv1')
                size0 = tf.shape(input_images)[-3:-1]

                h1 = slim.max_pool2d(h0, [2, 2], scope='pool1')
                h1 = slim.conv2d(h1, 128, [5, 5], stride=1, scope='conv2')
                size1 = tf.shape(h1)[-3:-1]

                h2 = slim.max_pool2d(h1, [2, 2], scope='pool2')
                h2 = slim.conv2d(h2, 256, [3, 3], stride=1, scope='conv3')
                size2 = tf.shape(h2)[-3:-1]

                h3 = slim.max_pool2d(h2, [2, 2], scope='pool3')
                h3 = slim.conv2d(h3, 256, [3, 3], stride=1, scope='conv4')

                h4 = tf.image.resize_bilinear(h3, size2)
                h4 = tf.concat([h4, h2], axis=-1)
                h4 = slim.conv2d(h4, 256, [3, 3], stride=1, scope='conv5')

                h5 = tf.image.resize_bilinear(h4, size1)
                h5 = tf.concat([h5, h1], axis=-1)
                h5 = slim.conv2d(h5, 128, [5, 5], stride=1, scope='conv6')

                h6 = tf.image.resize_bilinear(h5, size0)
                h6 = tf.concat([h6, h0], axis=-1)
                h6 = slim.conv2d(h6, 64, [5, 5], stride=1, scope='conv7')

    extrap_length = hparams.sequence_length - hparams.context_frames
    flows_masks = slim.conv2d(h6, 5 * extrap_length, [5, 5], stride=1, activation_fn=tf.tanh,
                              normalizer_fn=None, scope='conv8')
    flows_masks = tf.split(flows_masks, extrap_length, axis=-1)

    gen_images = []
    gen_flows_1 = []
    gen_flows_2 = []
    masks = []
    for flows_mask in flows_masks:
        flow_1, flow_2, mask = tf.split(flows_mask, [2, 2, 1], axis=-1)
        gen_flows_1.append(flow_1)
        gen_flows_2.append(flow_2)
        masks.append(mask)

        linspace_x = tf.linspace(-1.0, 1.0, size0[1])
        linspace_x.set_shape(input_images.shape[-2])
        linspace_y = tf.linspace(-1.0, 1.0, size0[0])
        linspace_y.set_shape(input_images.shape[-3])
        grid_x, grid_y = tf.meshgrid(linspace_x, linspace_y)

        coor_x_1 = grid_x[None, :, :] + flow_1[:, :, :, 0]
        coor_y_1 = grid_y[None, :, :] + flow_1[:, :, :, 1]

        coor_x_2 = grid_x[None, :, :] + flow_2[:, :, :, 0]
        coor_y_2 = grid_y[None, :, :] + flow_2[:, :, :, 1]

        output_1 = bilinear_interp(images[0], coor_x_1, coor_y_1, 'interpolate')
        output_2 = bilinear_interp(images[1], coor_x_2, coor_y_2, 'interpolate')

        mask = 0.5 * (1.0 + mask)
        gen_image = mask * output_1 + (1.0 - mask) * output_2
        gen_images.append(gen_image)
    gen_images = tf.stack(gen_images, axis=0)
    gen_flows_1 = tf.stack(gen_flows_1, axis=0)
    gen_flows_2 = tf.stack(gen_flows_2, axis=0)
    masks = tf.stack(masks, axis=0)

    outputs = {
        'gen_images': gen_images,
        'gen_flows_1': gen_flows_1,
        'gen_flows_2': gen_flows_2,
        'masks': masks,
    }
    return gen_images, outputs


class DVFVideoPredictionModel(VideoPredictionModel):
    def __init__(self, mode='train', *args, **kwargs):
        super(DVFVideoPredictionModel, self).__init__(
            functools.partial(generator_fn, mode=mode), *args, mode=mode, **kwargs)

    def get_default_hparams_dict(self):
        default_hparams = super(DVFVideoPredictionModel, self).get_default_hparams_dict()
        hparams = dict(
            batch_size=32,
            lr=0.00005,  # 0.0001
            l1_weight=0.0,
            l2_weight=0.0,
            charbonnier_weight=1.0,
            tv_charbonnier_weight=0.01,
            mask_charbonnier_weight=0.005,
        )
        return dict(itertools.chain(default_hparams.items(), hparams.items()))

    def generator_loss_fn(self, inputs, outputs, targets):
        hparams = self.hparams
        gen_losses = super(DVFVideoPredictionModel, self).generator_loss_fn(inputs, outputs, targets)

        def total_variation_charbonnier(x):
            diff1 = x[..., 1:, :, :] - x[..., :-1, :, :]
            diff2 = x[..., :, 1:, :] - x[..., :, :-1, :]
            return vp.losses.charbonnier_loss(diff1) + vp.losses.charbonnier_loss(diff2)

        if hparams.charbonnier_weight:
            gen_images = outputs['gen_images']
            target_images = targets
            gen_charbonnier_loss = vp.losses.charbonnier_loss(target_images - gen_images)
            gen_losses['gegen_charbonnier_loss'] = (gen_charbonnier_loss, hparams.charbonnier_weight)
        if hparams.tv_charbonnier_weight:
            gen_tv_charbonnier_loss = \
                total_variation_charbonnier(outputs['gen_flows_1']) + \
                total_variation_charbonnier(outputs['gen_flows_2'])
            gen_losses['gen_tv_charbonnier_loss'] = (gen_tv_charbonnier_loss, hparams.tv_charbonnier_weight)
        if hparams.mask_charbonnier_weight:
            gen_mask_charbonnier_loss = total_variation_charbonnier(outputs['masks'])
            gen_losses['gen_mask_charbonnier_loss'] = (gen_mask_charbonnier_loss, hparams.mask_charbonnier_weight)
        return gen_losses
