import numpy as np
import tensorflow as tf


def dense(inputs, units, use_spectral_norm=False, use_bias=True):
    with tf.variable_scope('dense'):
        input_shape = inputs.get_shape().as_list()
        kernel_shape = [input_shape[1], units]
        kernel = tf.get_variable('kernel', kernel_shape, dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
        if use_spectral_norm:
            kernel = spectral_normed_weight(kernel)
        outputs = tf.matmul(inputs, kernel)
        if use_bias:
            bias = tf.get_variable('bias', [units], dtype=tf.float32, initializer=tf.zeros_initializer())
            outputs = tf.nn.bias_add(outputs, bias)
        return outputs


def pad1d(inputs, size, strides=(1,), padding='SAME', mode='CONSTANT'):
    size = list(size) if isinstance(size, (tuple, list)) else [size]
    strides = list(strides) if isinstance(strides, (tuple, list)) else [strides]
    input_shape = inputs.get_shape().as_list()
    assert len(input_shape) == 3
    in_width = input_shape[1]
    if padding in ('SAME', 'FULL'):
        if in_width % strides[0] == 0:
            pad_along_width = max(size[0] - strides[0], 0)
        else:
            pad_along_width = max(size[0] - (in_width % strides[0]), 0)
        if padding == 'SAME':
            pad_left = pad_along_width // 2
            pad_right = pad_along_width - pad_left
        else:
            pad_left = pad_along_width
            pad_right = pad_along_width
        padding_pattern = [[0, 0],
                           [pad_left, pad_right],
                           [0, 0]]
        outputs = tf.pad(inputs, padding_pattern, mode=mode)
    elif padding == 'VALID':
        outputs = inputs
    else:
        raise ValueError("Invalid padding scheme %s" % padding)
    return outputs


def conv1d(inputs, filters, kernel_size, strides=(1,), padding='SAME', kernel=None, use_bias=True):
    kernel_size = list(kernel_size) if isinstance(kernel_size, (tuple, list)) else [kernel_size]
    strides = list(strides) if isinstance(strides, (tuple, list)) else [strides]
    input_shape = inputs.get_shape().as_list()
    kernel_shape = list(kernel_size) + [input_shape[-1], filters]
    if kernel is None:
        with tf.variable_scope('conv1d'):
            kernel = tf.get_variable('kernel', kernel_shape, dtype=tf.float32,
                                     initializer=tf.truncated_normal_initializer(stddev=0.02))
    else:
        if kernel_shape != kernel.get_shape().as_list():
            raise ValueError("Expecting kernel with shape %s but instead got kernel with shape %s" % (tuple(kernel_shape), tuple(kernel.get_shape().as_list())))
    if padding == 'FULL':
        inputs = pad1d(inputs, kernel_size, strides=strides, padding=padding, mode='CONSTANT')
        padding = 'VALID'
    stride, = strides
    outputs = tf.nn.conv1d(inputs, kernel, stride, padding=padding)
    if use_bias:
        with tf.variable_scope('conv1d'):
            bias = tf.get_variable('bias', [filters], dtype=tf.float32, initializer=tf.zeros_initializer())
            outputs = tf.nn.bias_add(outputs, bias)
    return outputs


def pad2d_paddings(inputs, size, strides=(1, 1), rate=(1, 1), padding='SAME'):
    """
    Computes the paddings for a 4-D tensor according to the convolution padding algorithm.

    See pad2d.

    Reference:
        https://www.tensorflow.org/api_guides/python/nn#convolution
        https://www.tensorflow.org/api_docs/python/tf/nn/with_space_to_batch
    """
    size = np.array(size) if isinstance(size, (tuple, list)) else np.array([size] * 2)
    strides = np.array(strides) if isinstance(strides, (tuple, list)) else np.array([strides] * 2)
    rate = np.array(rate) if isinstance(rate, (tuple, list)) else np.array([rate] * 2)
    if np.any(strides > 1) and np.any(rate > 1):
        raise ValueError("strides > 1 not supported in conjunction with rate > 1")
    input_shape = inputs.get_shape().as_list()
    assert len(input_shape) == 4
    input_size = np.array(input_shape[1:3])
    if padding in ('SAME', 'FULL'):
        if np.any(rate > 1):
            # We have two padding contributions. The first is used for converting "SAME"
            # to "VALID". The second is required so that the height and width of the
            # zero-padded value tensor are multiples of rate.

            # Spatial dimensions of the filters and the upsampled filters in which we
            # introduce (rate - 1) zeros between consecutive filter values.
            dilated_size = size + (size - 1) * (rate - 1)
            pad = dilated_size - 1
        else:
            pad = np.where(input_size % strides == 0,
                           np.maximum(size - strides, 0),
                           np.maximum(size - (input_size % strides), 0))
        if padding == 'SAME':
            # When full_padding_shape is odd, we pad more at end, following the same
            # convention as conv2d.
            pad_start = pad // 2
            pad_end = pad - pad_start
        else:
            pad_start = pad
            pad_end = pad
        if np.any(rate > 1):
            # More padding so that rate divides the height and width of the input.
            # TODO: not sure if this is correct when padding == 'FULL'
            orig_pad_end = pad_end
            full_input_size = input_size + pad_start + orig_pad_end
            pad_end_extra = (rate - full_input_size % rate) % rate
            pad_end = orig_pad_end + pad_end_extra
        paddings = [[0, 0],
                    [pad_start[0], pad_end[0]],
                    [pad_start[1], pad_end[1]],
                    [0, 0]]
    elif padding == 'VALID':
        paddings = [[0, 0]] * 4
    else:
        raise ValueError("Invalid padding scheme %s" % padding)
    return paddings


def pad2d(inputs, size, strides=(1, 1), rate=(1, 1), padding='SAME', mode='CONSTANT'):
    """
    Pads a 4-D tensor according to the convolution padding algorithm.

    Convolution with a padding scheme
        conv2d(..., padding=padding)
    is equivalent to zero-padding of the input with such scheme, followed by
    convolution with 'VALID' padding
        padded = pad2d(..., padding=padding, mode='CONSTANT')
        conv2d(padded, ..., padding='VALID')

    Args:
        inputs: A 4-D tensor of shape
            `[batch, in_height, in_width, in_channels]`.
        padding: A string, either 'VALID', 'SAME', or 'FULL'. The padding algorithm.
        mode: One of "CONSTANT", "REFLECT", or "SYMMETRIC" (case-insensitive).

    Returns:
        A 4-D tensor.

    Reference:
        https://www.tensorflow.org/api_guides/python/nn#convolution
    """
    paddings = pad2d_paddings(inputs, size, strides=strides, rate=rate, padding=padding)
    if paddings == [[0, 0]] * 4:
        outputs = inputs
    else:
        outputs = tf.pad(inputs, paddings, mode=mode)
    return outputs


def local2d(inputs, filters, kernel_size, strides=(1, 1), padding='SAME',
            kernel=None, flip_filters=False,
            use_bias=True, channelwise=False):
    """
    2-D locally connected operation.

    Works similarly to 2-D convolution except that the weights are unshared, that is, a different set of filters is
    applied at each different patch of the input.

    Args:
        inputs: A 4-D tensor of shape
            `[batch, in_height, in_width, in_channels]`.
        kernel: A 6-D or 7-D tensor of shape
            `[in_height, in_width, kernel_size[0], kernel_size[1], in_channels, filters]` or
            `[batch, in_height, in_width, kernel_size[0], kernel_size[1], in_channels, filters]`.

    Returns:
        A 4-D tensor.
    """
    kernel_size = list(kernel_size) if isinstance(kernel_size, (tuple, list)) else [kernel_size] * 2
    strides = list(strides) if isinstance(strides, (tuple, list)) else [strides] * 2
    if strides != [1, 1]:
        raise NotImplementedError
    if padding == 'FULL':
        inputs = pad2d(inputs, kernel_size, strides=strides, padding=padding, mode='CONSTANT')
        padding = 'VALID'
    input_shape = inputs.get_shape().as_list()
    if padding == 'SAME':
        output_shape = input_shape[:3] + [filters]
    elif padding == 'VALID':
        output_shape = [input_shape[0], input_shape[1] - kernel_size[0] + 1, input_shape[2] - kernel_size[1] + 1, filters]
    else:
        raise ValueError("Invalid padding scheme %s" % padding)

    if channelwise:
        if filters not in (input_shape[-1], 1):
            raise ValueError("Number of filters should match the number of input channels or be 1 when channelwise "
                             "is true, but got filters=%r and %d input channels" % (filters, input_shape[-1]))
        kernel_shape = output_shape[1:3] + kernel_size + [filters]
    else:
        kernel_shape = output_shape[1:3] + kernel_size + [input_shape[-1], filters]
    if kernel is None:
        with tf.variable_scope('local2d'):
            kernel = tf.get_variable('kernel', kernel_shape, dtype=tf.float32,
                                     initializer=tf.truncated_normal_initializer(stddev=0.02))
    else:
        if kernel.get_shape().as_list() not in (kernel_shape, [input_shape[0]] + kernel_shape):
            raise ValueError("Expecting kernel with shape %s or %s but instead got kernel with shape %s"
                             % (tuple(kernel_shape), tuple([input_shape[0]] + kernel_shape), tuple(kernel.get_shape().as_list())))

    outputs = []
    for i in range(kernel_size[0]):
        filter_h_ind = -i-1 if flip_filters else i
        if padding == 'VALID':
            ii = i
        else:
            ii = i - (kernel_size[0] // 2)
        input_h_slice = slice(max(ii, 0), min(ii + output_shape[1], input_shape[1]))
        output_h_slice = slice(input_h_slice.start - ii, input_h_slice.stop - ii)
        assert 0 <= output_h_slice.start < output_shape[1]
        assert 0 < output_h_slice.stop <= output_shape[1]

        for j in range(kernel_size[1]):
            filter_w_ind = -j-1 if flip_filters else j
            if padding == 'VALID':
                jj = j
            else:
                jj = j - (kernel_size[1] // 2)
            input_w_slice = slice(max(jj, 0), min(jj + output_shape[2], input_shape[2]))
            output_w_slice = slice(input_w_slice.start - jj, input_w_slice.stop - jj)
            assert 0 <= output_w_slice.start < output_shape[2]
            assert 0 < output_w_slice.stop <= output_shape[2]
            if channelwise:
                inc = inputs[:, input_h_slice, input_w_slice, :] * \
                      kernel[..., output_h_slice, output_w_slice, filter_h_ind, filter_w_ind, :]
            else:
                inc = tf.reduce_sum(inputs[:, input_h_slice, input_w_slice, :, None] *
                                    kernel[..., output_h_slice, output_w_slice, filter_h_ind, filter_w_ind, :, :], axis=-2)
            # equivalent to this
            # outputs[:, output_h_slice, output_w_slice, :] += inc
            paddings = [[0, 0], [output_h_slice.start, output_shape[1] - output_h_slice.stop],
                        [output_w_slice.start, output_shape[2] - output_w_slice.stop], [0, 0]]
            outputs.append(tf.pad(inc, paddings))
    outputs = tf.add_n(outputs)
    if use_bias:
        with tf.variable_scope('local2d'):
            bias = tf.get_variable('bias', output_shape[1:], dtype=tf.float32, initializer=tf.zeros_initializer())
            outputs = tf.nn.bias_add(outputs, bias)
    return outputs


def separable_local2d(inputs, filters, kernel_size, strides=(1, 1), padding='SAME',
                      vertical_kernel=None, horizontal_kernel=None, flip_filters=False,
                      use_bias=True, channelwise=False):
    """
    2-D locally connected operation with separable filters.

    Note that, unlike tf.nn.separable_conv2d, this is spatial separability between dimensions 1 and 2.

    Args:
        inputs: A 4-D tensor of shape
            `[batch, in_height, in_width, in_channels]`.
        vertical_kernel: A 5-D or 6-D tensor of shape
            `[in_height, in_width, kernel_size[0], in_channels, filters]` or
            `[batch, in_height, in_width, kernel_size[0], in_channels, filters]`.
        horizontal_kernel: A 5-D or 6-D tensor of shape
            `[in_height, in_width, kernel_size[1], in_channels, filters]` or
            `[batch, in_height, in_width, kernel_size[1], in_channels, filters]`.

    Returns:
        A 4-D tensor.
    """
    kernel_size = list(kernel_size) if isinstance(kernel_size, (tuple, list)) else [kernel_size] * 2
    strides = list(strides) if isinstance(strides, (tuple, list)) else [strides] * 2
    if strides != [1, 1]:
        raise NotImplementedError
    if padding == 'FULL':
        inputs = pad2d(inputs, kernel_size, strides=strides, padding=padding, mode='CONSTANT')
        padding = 'VALID'
    input_shape = inputs.get_shape().as_list()
    if padding == 'SAME':
        output_shape = input_shape[:3] + [filters]
    elif padding == 'VALID':
        output_shape = [input_shape[0], input_shape[1] - kernel_size[0] + 1, input_shape[2] - kernel_size[1] + 1, filters]
    else:
        raise ValueError("Invalid padding scheme %s" % padding)

    kernels = [vertical_kernel, horizontal_kernel]
    for i, (kernel_type, kernel_length, kernel) in enumerate(zip(['vertical', 'horizontal'], kernel_size, kernels)):
        if channelwise:
            if filters not in (input_shape[-1], 1):
                raise ValueError("Number of filters should match the number of input channels or be 1 when channelwise "
                                 "is true, but got filters=%r and %d input channels" % (filters, input_shape[-1]))
            kernel_shape = output_shape[1:3] + [kernel_length, filters]
        else:
            kernel_shape = output_shape[1:3] + [kernel_length, input_shape[-1], filters]
        if kernel is None:
            with tf.variable_scope('separable_local2d'):
                kernel = tf.get_variable('%s_kernel' % kernel_type, kernel_shape, dtype=tf.float32,
                                         initializer=tf.truncated_normal_initializer(stddev=0.02))
                kernels[i] = kernel
        else:
            if kernel.get_shape().as_list() not in (kernel_shape, [input_shape[0]] + kernel_shape):
                raise ValueError("Expecting %s kernel with shape %s or %s but instead got kernel with shape %s"
                                 % (kernel_type,
                                    tuple(kernel_shape), tuple([input_shape[0]] +kernel_shape),
                                    tuple(kernel.get_shape().as_list())))

    outputs = []
    for i in range(kernel_size[0]):
        filter_h_ind = -i-1 if flip_filters else i
        if padding == 'VALID':
            ii = i
        else:
            ii = i - (kernel_size[0] // 2)
        input_h_slice = slice(max(ii, 0), min(ii + output_shape[1], input_shape[1]))
        output_h_slice = slice(input_h_slice.start - ii, input_h_slice.stop - ii)
        assert 0 <= output_h_slice.start < output_shape[1]
        assert 0 < output_h_slice.stop <= output_shape[1]

        for j in range(kernel_size[1]):
            filter_w_ind = -j-1 if flip_filters else j
            if padding == 'VALID':
                jj = j
            else:
                jj = j - (kernel_size[1] // 2)
            input_w_slice = slice(max(jj, 0), min(jj + output_shape[2], input_shape[2]))
            output_w_slice = slice(input_w_slice.start - jj, input_w_slice.stop - jj)
            assert 0 <= output_w_slice.start < output_shape[2]
            assert 0 < output_w_slice.stop <= output_shape[2]
            if channelwise:
                inc = inputs[:, input_h_slice, input_w_slice, :] * \
                      kernels[0][..., output_h_slice, output_w_slice, filter_h_ind, :] * \
                      kernels[1][..., output_h_slice, output_w_slice, filter_w_ind, :]
            else:
                inc = tf.reduce_sum(inputs[:, input_h_slice, input_w_slice, :, None] *
                                    kernels[0][..., output_h_slice, output_w_slice, filter_h_ind, :, :] *
                                    kernels[1][..., output_h_slice, output_w_slice, filter_w_ind, :, :],
                                    axis=-2)
            # equivalent to this
            # outputs[:, output_h_slice, output_w_slice, :] += inc
            paddings = [[0, 0], [output_h_slice.start, output_shape[1] - output_h_slice.stop],
                        [output_w_slice.start, output_shape[2] - output_w_slice.stop], [0, 0]]
            outputs.append(tf.pad(inc, paddings))
    outputs = tf.add_n(outputs)
    if use_bias:
        with tf.variable_scope('separable_local2d'):
            bias = tf.get_variable('bias', output_shape[1:], dtype=tf.float32, initializer=tf.zeros_initializer())
            outputs = tf.nn.bias_add(outputs, bias)
    return outputs


def kronecker_local2d(inputs, filters, kernel_size, strides=(1, 1), padding='SAME',
                      kernels=None, flip_filters=False, use_bias=True, channelwise=False):
    """
    2-D locally connected operation with filters represented as a kronecker product of smaller filters

    Args:
        inputs: A 4-D tensor of shape
            `[batch, in_height, in_width, in_channels]`.
        kernel: A list of 6-D or 7-D tensors of shape
            `[in_height, in_width, kernel_size[i][0], kernel_size[i][1], in_channels, filters]` or
            `[batch, in_height, in_width, kernel_size[i][0], kernel_size[i][1], in_channels, filters]`.

    Returns:
        A 4-D tensor.
    """
    kernel_size = list(kernel_size) if isinstance(kernel_size, (tuple, list)) else [kernel_size] * 2
    strides = list(strides) if isinstance(strides, (tuple, list)) else [strides] * 2
    if strides != [1, 1]:
        raise NotImplementedError
    if padding == 'FULL':
        inputs = pad2d(inputs, kernel_size, strides=strides, padding=padding, mode='CONSTANT')
        padding = 'VALID'
    input_shape = inputs.get_shape().as_list()
    if padding == 'SAME':
        output_shape = input_shape[:3] + [filters]
    elif padding == 'VALID':
        output_shape = [input_shape[0], input_shape[1] - kernel_size[0] + 1, input_shape[2] - kernel_size[1] + 1, filters]
    else:
        raise ValueError("Invalid padding scheme %s" % padding)

    if channelwise:
        if filters not in (input_shape[-1], 1):
            raise ValueError("Number of filters should match the number of input channels or be 1 when channelwise "
                             "is true, but got filters=%r and %d input channels" % (filters, input_shape[-1]))
        kernel_shape = output_shape[1:3] + kernel_size + [filters]
        factor_kernel_shape = output_shape[1:3] + [None, None, filters]
    else:
        kernel_shape = output_shape[1:3] + kernel_size + [input_shape[-1], filters]
        factor_kernel_shape = output_shape[1:3] + [None, None, input_shape[-1], filters]
    if kernels is None:
        with tf.variable_scope('kronecker_local2d'):
            kernels = [tf.get_variable('kernel', kernel_shape, dtype=tf.float32,
                                       initializer=tf.truncated_normal_initializer(stddev=0.02))]
        filter_h_lengths = [kernel_size[0]]
        filter_w_lengths = [kernel_size[1]]
    else:
        for kernel in kernels:
            if not ((len(kernel.shape) == len(factor_kernel_shape) and
                    all(((k == f) or f is None) for k, f in zip(kernel.get_shape().as_list(), factor_kernel_shape))) or
                    (len(kernel.shape) == (len(factor_kernel_shape) + 1) and
                    all(((k == f) or f is None) for k, f in zip(kernel.get_shape().as_list(), [input_shape[0]] +factor_kernel_shape)))):
                raise ValueError("Expecting kernel with shape %s or %s but instead got kernel with shape %s"
                                 % (tuple(factor_kernel_shape), tuple([input_shape[0]] + factor_kernel_shape),
                                    tuple(kernel.get_shape().as_list())))
        if channelwise:
            filter_h_lengths, filter_w_lengths = zip(*[kernel.get_shape().as_list()[-3:-1] for kernel in kernels])
        else:
            filter_h_lengths, filter_w_lengths = zip(*[kernel.get_shape().as_list()[-4:-2] for kernel in kernels])
        if [np.prod(filter_h_lengths), np.prod(filter_w_lengths)] != kernel_size:
            raise ValueError("Expecting kernel size %s but instead got kernel size %s"
                             % (tuple(kernel_size), tuple([np.prod(filter_h_lengths), np.prod(filter_w_lengths)])))

    def get_inds(ind, lengths):
        inds = []
        for i in range(len(lengths)):
            curr_ind = int(ind)
            for j in range(len(lengths) - 1, i, -1):
                curr_ind //= lengths[j]
            curr_ind %= lengths[i]
            inds.append(curr_ind)
        return inds

    outputs = []
    for i in range(kernel_size[0]):
        if padding == 'VALID':
            ii = i
        else:
            ii = i - (kernel_size[0] // 2)
        input_h_slice = slice(max(ii, 0), min(ii + output_shape[1], input_shape[1]))
        output_h_slice = slice(input_h_slice.start - ii, input_h_slice.stop - ii)
        assert 0 <= output_h_slice.start < output_shape[1]
        assert 0 < output_h_slice.stop <= output_shape[1]

        for j in range(kernel_size[1]):
            if padding == 'VALID':
                jj = j
            else:
                jj = j - (kernel_size[1] // 2)
            input_w_slice = slice(max(jj, 0), min(jj + output_shape[2], input_shape[2]))
            output_w_slice = slice(input_w_slice.start - jj, input_w_slice.stop - jj)
            assert 0 <= output_w_slice.start < output_shape[2]
            assert 0 < output_w_slice.stop <= output_shape[2]
            kernel_slice = 1.0
            for filter_h_ind, filter_w_ind, kernel in zip(get_inds(i, filter_h_lengths), get_inds(j, filter_w_lengths), kernels):
                if flip_filters:
                    filter_h_ind = -filter_h_ind-1
                    filter_w_ind = -filter_w_ind-1
                if channelwise:
                    kernel_slice *= kernel[..., output_h_slice, output_w_slice, filter_h_ind, filter_w_ind, :]
                else:
                    kernel_slice *= kernel[..., output_h_slice, output_w_slice, filter_h_ind, filter_w_ind, :, :]
            if channelwise:
                inc = inputs[:, input_h_slice, input_w_slice, :] * kernel_slice
            else:
                inc = tf.reduce_sum(inputs[:, input_h_slice, input_w_slice, :, None] * kernel_slice, axis=-2)
            # equivalent to this
            # outputs[:, output_h_slice, output_w_slice, :] += inc
            paddings = [[0, 0], [output_h_slice.start, output_shape[1] - output_h_slice.stop],
                        [output_w_slice.start, output_shape[2] - output_w_slice.stop], [0, 0]]
            outputs.append(tf.pad(inc, paddings))
    outputs = tf.add_n(outputs)
    if use_bias:
        with tf.variable_scope('kronecker_local2d'):
            bias = tf.get_variable('bias', output_shape[1:], dtype=tf.float32, initializer=tf.zeros_initializer())
            outputs = tf.nn.bias_add(outputs, bias)
    return outputs


def depthwise_conv2d(inputs, channel_multiplier, kernel_size, strides=(1, 1), padding='SAME', kernel=None, use_bias=True):
    kernel_size = list(kernel_size) if isinstance(kernel_size, (tuple, list)) else [kernel_size] * 2
    strides = list(strides) if isinstance(strides, (tuple, list)) else [strides] * 2
    input_shape = inputs.get_shape().as_list()
    kernel_shape = kernel_size + [input_shape[-1], channel_multiplier]
    if kernel is None:
        with tf.variable_scope('depthwise_conv2d'):
            kernel = tf.get_variable('kernel', kernel_shape, dtype=tf.float32,
                                     initializer=tf.truncated_normal_initializer(stddev=0.02))
    else:
        if kernel_shape != kernel.get_shape().as_list():
            raise ValueError("Expecting kernel with shape %s but instead got kernel with shape %s"
                             % (tuple(kernel_shape), tuple(kernel.get_shape().as_list())))
    if padding == 'FULL':
        inputs = pad2d(inputs, kernel_size, strides=strides, padding=padding, mode='CONSTANT')
        padding = 'VALID'
    outputs = tf.nn.depthwise_conv2d(inputs, kernel, [1] + strides + [1], padding=padding)
    if use_bias:
        with tf.variable_scope('depthwise_conv2d'):
            bias = tf.get_variable('bias', [input_shape[-1] * channel_multiplier], dtype=tf.float32, initializer=tf.zeros_initializer())
            outputs = tf.nn.bias_add(outputs, bias)
    return outputs


def conv2d(inputs, filters, kernel_size, strides=(1, 1), padding='SAME', kernel=None, use_bias=True, bias=None, use_spectral_norm=False):
    """
    2-D convolution.

    Args:
        inputs: A 4-D tensor of shape
            `[batch, in_height, in_width, in_channels]`.
        kernel: A 4-D or 5-D tensor of shape
            `[kernel_size[0], kernel_size[1], in_channels, filters]` or
            `[batch, kernel_size[0], kernel_size[1], in_channels, filters]`.
        bias: A 1-D or 2-D tensor of shape
            `[filters]` or `[batch, filters]`.

    Returns:
        A 4-D tensor.
    """
    kernel_size = list(kernel_size) if isinstance(kernel_size, (tuple, list)) else [kernel_size] * 2
    strides = list(strides) if isinstance(strides, (tuple, list)) else [strides] * 2
    input_shape = inputs.get_shape().as_list()
    kernel_shape = list(kernel_size) + [input_shape[-1], filters]
    if kernel is None:
        with tf.variable_scope('conv2d'):
            kernel = tf.get_variable('kernel', kernel_shape, dtype=tf.float32,
                                     initializer=tf.truncated_normal_initializer(stddev=0.02))
            if use_spectral_norm:
                kernel = spectral_normed_weight(kernel)
    else:
        if kernel.get_shape().as_list() not in (kernel_shape, [input_shape[0]] + kernel_shape):
            raise ValueError("Expecting kernel with shape %s or %s but instead got kernel with shape %s"
                             % (tuple(kernel_shape), tuple([input_shape[0]] + kernel_shape), tuple(kernel.get_shape().as_list())))
    if padding == 'FULL':
        inputs = pad2d(inputs, kernel_size, strides=strides, padding=padding, mode='CONSTANT')
        padding = 'VALID'
    if kernel.get_shape().ndims == 4:
        outputs = tf.nn.conv2d(inputs, kernel, [1] + strides + [1], padding=padding)
    else:
        def conv2d_single_fn(args):
            input_, kernel_ = args
            input_ = tf.expand_dims(input_, axis=0)
            output = tf.nn.conv2d(input_, kernel_, [1] + strides + [1], padding=padding)
            output = tf.squeeze(output, axis=0)
            return output
        outputs = tf.map_fn(conv2d_single_fn, [inputs, kernel], dtype=tf.float32)
    if use_bias:
        bias_shape = [filters]
        if bias is None:
            with tf.variable_scope('conv2d'):
                bias = tf.get_variable('bias', [filters], dtype=tf.float32, initializer=tf.zeros_initializer())
        else:
            if bias.get_shape().as_list() not in (bias_shape, [input_shape[0]] + bias_shape):
                raise ValueError("Expecting bias with shape %s but instead got bias with shape %s"
                                 % (tuple(bias_shape), tuple(bias.get_shape().as_list())))
        if bias.get_shape().ndims == 1:
            outputs = tf.nn.bias_add(outputs, bias)
        else:
            outputs = tf.add(outputs, bias[:, None, None, :])
    return outputs


def deconv2d(inputs, filters, kernel_size, strides=(1, 1), padding='SAME', kernel=None, use_bias=True):
    """
    2-D transposed convolution.

    Notes on padding:
       The equivalent of transposed convolution with full padding is a convolution with valid padding, and
       the equivalent of transposed convolution with valid padding is a convolution with full padding.

    Reference:
        http://deeplearning.net/software/theano/tutorial/conv_arithmetic.html
    """
    kernel_size = list(kernel_size) if isinstance(kernel_size, (tuple, list)) else [kernel_size] * 2
    strides = list(strides) if isinstance(strides, (tuple, list)) else [strides] * 2
    input_shape = inputs.get_shape().as_list()
    kernel_shape = list(kernel_size) + [filters, input_shape[-1]]
    if kernel is None:
        with tf.variable_scope('deconv2d'):
            kernel = tf.get_variable('kernel', kernel_shape, dtype=tf.float32,
                                     initializer=tf.truncated_normal_initializer(stddev=0.02))
    else:
        if kernel_shape != kernel.get_shape().as_list():
            raise ValueError("Expecting kernel with shape %s but instead got kernel with shape %s" % (tuple(kernel_shape), tuple(kernel.get_shape().as_list())))
    if padding == 'FULL':
        output_h, output_w = [s * (i + 1) - k for (i, k, s) in zip(input_shape[1:3], kernel_size, strides)]
    elif padding == 'SAME':
        output_h, output_w = [s * i for (i, s) in zip(input_shape[1:3], strides)]
    elif padding == 'VALID':
        output_h, output_w = [s * (i - 1) + k for (i, k, s) in zip(input_shape[1:3], kernel_size, strides)]
    else:
        raise ValueError("Invalid padding scheme %s" % padding)
    output_shape = [input_shape[0], output_h, output_w, filters]
    outputs = tf.nn.conv2d_transpose(inputs, kernel, output_shape, [1] + strides + [1], padding=padding)
    if use_bias:
        with tf.variable_scope('deconv2d'):
            bias = tf.get_variable('bias', [filters], dtype=tf.float32, initializer=tf.zeros_initializer())
            outputs = tf.nn.bias_add(outputs, bias)
    return outputs


def get_bilinear_kernel(strides):
    strides = list(strides) if isinstance(strides, (tuple, list)) else [strides] * 2
    strides = np.array(strides)
    kernel_size = 2 * strides - strides % 2
    center = strides - (kernel_size % 2 == 1) - 0.5 * (kernel_size % 2 != 1)
    vertical_kernel = 1 - abs(np.arange(kernel_size[0]) - center[0]) / strides[0]
    horizontal_kernel = 1 - abs(np.arange(kernel_size[1]) - center[1]) / strides[1]
    kernel = vertical_kernel[:, None] * horizontal_kernel[None, :]
    return kernel


def upsample2d(inputs, strides, padding='SAME', upsample_mode='bilinear'):
    if upsample_mode == 'bilinear':
        single_bilinear_kernel = get_bilinear_kernel(strides).astype(np.float32)
        input_shape = inputs.get_shape().as_list()
        bilinear_kernel = tf.matrix_diag(tf.tile(tf.constant(single_bilinear_kernel)[..., None], (1, 1, input_shape[-1])))
        outputs = deconv2d(inputs, input_shape[-1], kernel_size=single_bilinear_kernel.shape,
                           strides=strides, kernel=bilinear_kernel, padding=padding, use_bias=False)
    elif upsample_mode == 'nearest':
        strides = list(strides) if isinstance(strides, (tuple, list)) else [strides] * 2
        input_shape = inputs.get_shape().as_list()
        inputs_tiled = tf.tile(inputs[:, :, None, :, None, :], [1, 1, strides[0], 1, strides[1], 1])
        outputs = tf.reshape(inputs_tiled, [input_shape[0], input_shape[1] * strides[0],
                                            input_shape[2] * strides[1], input_shape[3]])
    else:
        raise ValueError("Unknown upsample mode %s" % upsample_mode)
    return outputs


def upsample2d_v2(inputs, strides, padding='SAME', upsample_mode='bilinear'):
    """
    Possibly less computationally efficient but more memory efficent than upsampled2d.
    """
    if upsample_mode == 'bilinear':
        single_kernel = get_bilinear_kernel(strides).astype(np.float32)
    elif upsample_mode == 'nearest':
        strides = list(strides) if isinstance(strides, (tuple, list)) else [strides] * 2
        single_kernel = np.ones(strides, dtype=np.float32)
    else:
        raise ValueError("Unknown upsample mode %s" % upsample_mode)
    input_shape = inputs.get_shape().as_list()
    kernel = tf.constant(single_kernel)[:, :, None, None]
    inputs = tf.transpose(inputs, [3, 0, 1, 2])[..., None]
    outputs = tf.map_fn(lambda input: deconv2d(input, 1, kernel_size=single_kernel.shape,
                                               strides=strides, kernel=kernel,
                                               padding=padding, use_bias=False),
                        inputs, parallel_iterations=input_shape[-1])
    outputs = tf.transpose(tf.squeeze(outputs, axis=-1), [1, 2, 3, 0])
    return outputs


def upsample_conv2d(inputs, filters, kernel_size, strides=(1, 1), padding='SAME',
                    kernel=None, use_bias=True, bias=None, upsample_mode='bilinear'):
    """
    Upsamples the inputs by a factor using bilinear interpolation and the performs conv2d on the upsampled input. This
    function is more computationally and memory efficient than a naive implementation. Unlike a naive implementation
    that would upsample the input first, this implementation first convolves the bilinear kernel with the given kernel,
    and then performs the convolution (actually a deconv2d) with the combined kernel. As opposed to just using deconv2d
    directly, this function is less prone to checkerboard artifacts thanks to the implicit bilinear upsampling.

    Example:
        >>> import numpy as np
        >>> import tensorflow as tf
        >>> from video_prediction.ops import upsample_conv2d, upsample2d, conv2d, pad2d_paddings
        >>> inputs_shape = [4, 8, 8, 64]
        >>> kernel_size = [3, 3]  # for convolution
        >>> filters = 32  # for convolution
        >>> strides = [2, 2]  # for upsampling
        >>> inputs = tf.get_variable("inputs", inputs_shape)
        >>> kernel = tf.get_variable("kernel", (kernel_size[0], kernel_size[1], inputs_shape[-1], filters))
        >>> bias = tf.get_variable("bias", (filters,))
        >>> outputs = upsample_conv2d(inputs, filters, kernel_size=kernel_size, strides=strides, \
                                      kernel=kernel, bias=bias)
        >>> # upsample with bilinear interpolation
        >>> inputs_up = upsample2d(inputs, strides=strides, padding='VALID')
        >>> # convolve upsampled input with kernel
        >>> outputs_up = conv2d(inputs_up, filters, kernel_size=kernel_size, strides=(1, 1), \
                                kernel=kernel, bias=bias, padding='FULL')
        >>> # crop appropriately
        >>> same_paddings = pad2d_paddings(inputs, kernel_size, strides=(1, 1), padding='SAME')
        >>> full_paddings = pad2d_paddings(inputs, kernel_size, strides=(1, 1), padding='FULL')
        >>> crop_top = (strides[0] - strides[0] % 2) // 2 + full_paddings[1][1] - same_paddings[1][1]
        >>> crop_left = (strides[1] - strides[1] % 2) // 2 + full_paddings[2][1] - same_paddings[2][1]
        >>> outputs_up = outputs_up[:, crop_top:crop_top + strides[0] * inputs_shape[1], \
                                    crop_left:crop_left + strides[1] * inputs_shape[2], :]
        >>> sess = tf.Session()
        >>> sess.run(tf.global_variables_initializer())
        >>> assert np.allclose(*sess.run([outputs, outputs_up]), atol=1e-5)

    """
    kernel_size = list(kernel_size) if isinstance(kernel_size, (tuple, list)) else [kernel_size] * 2
    strides = list(strides) if isinstance(strides, (tuple, list)) else [strides] * 2
    if padding != 'SAME' or upsample_mode != 'bilinear':
        raise NotImplementedError
    input_shape = inputs.get_shape().as_list()
    kernel_shape = list(kernel_size) + [input_shape[-1], filters]
    if kernel is None:
        with tf.variable_scope('upsample_conv2d'):
            kernel = tf.get_variable('kernel', kernel_shape, dtype=tf.float32,
                                     initializer=tf.truncated_normal_initializer(stddev=0.02))
    else:
        if kernel_shape != kernel.get_shape().as_list():
            raise ValueError("Expecting kernel with shape %s but instead got kernel with shape %s" %
                             (tuple(kernel_shape), tuple(kernel.get_shape().as_list())))

    # convolve bilinear kernel with kernel
    single_bilinear_kernel = get_bilinear_kernel(strides).astype(np.float32)
    kernel_transposed = tf.transpose(kernel, (0, 1, 3, 2))
    kernel_reshaped = tf.reshape(kernel_transposed, kernel_size + [1, input_shape[-1] * filters])
    kernel_up_reshaped = conv2d(tf.constant(single_bilinear_kernel)[None, :, :, None], input_shape[-1] * filters,
                                kernel_size=kernel_size, kernel=kernel_reshaped, padding='FULL', use_bias=False)
    kernel_up = tf.reshape(kernel_up_reshaped,
                           kernel_up_reshaped.get_shape().as_list()[1:3] + [filters, input_shape[-1]])

    # deconvolve with the bilinearly convolved kernel
    outputs = deconv2d(inputs, filters, kernel_size=kernel_up.get_shape().as_list()[:2], strides=strides,
                       kernel=kernel_up, padding='SAME', use_bias=False)
    if use_bias:
        if bias is None:
            with tf.variable_scope('upsample_conv2d'):
                bias = tf.get_variable('bias', [filters], dtype=tf.float32, initializer=tf.zeros_initializer())
        else:
            bias_shape = [filters]
            if bias_shape != bias.get_shape().as_list():
                raise ValueError("Expecting bias with shape %s but instead got bias with shape %s" %
                                 (tuple(bias_shape), tuple(bias.get_shape().as_list())))
        outputs = tf.nn.bias_add(outputs, bias)
    return outputs


def upsample_conv2d_v2(inputs, filters, kernel_size, strides=(1, 1), padding='SAME',
                       kernel=None, use_bias=True, bias=None, upsample_mode='bilinear'):
    kernel_size = list(kernel_size) if isinstance(kernel_size, (tuple, list)) else [kernel_size] * 2
    strides = list(strides) if isinstance(strides, (tuple, list)) else [strides] * 2
    if padding != 'SAME':
        raise NotImplementedError
    input_shape = inputs.get_shape().as_list()
    kernel_shape = list(kernel_size) + [input_shape[-1], filters]
    if kernel is None:
        with tf.variable_scope('upsample_conv2d'):
            kernel = tf.get_variable('kernel', kernel_shape, dtype=tf.float32,
                                     initializer=tf.truncated_normal_initializer(stddev=0.02))
    else:
        if kernel_shape != kernel.get_shape().as_list():
            raise ValueError("Expecting kernel with shape %s but instead got kernel with shape %s" %
                             (tuple(kernel_shape), tuple(kernel.get_shape().as_list())))

    inputs_up = upsample2d_v2(inputs, strides=strides, padding='VALID', upsample_mode=upsample_mode)
    # convolve upsampled input with kernel
    outputs = conv2d(inputs_up, filters, kernel_size=kernel_size, strides=(1, 1),
                     kernel=kernel, bias=None, padding='FULL', use_bias=False)
    # crop appropriately
    same_paddings = pad2d_paddings(inputs, kernel_size, strides=(1, 1), padding='SAME')
    full_paddings = pad2d_paddings(inputs, kernel_size, strides=(1, 1), padding='FULL')
    crop_top = (strides[0] - strides[0] % 2) // 2 + full_paddings[1][1] - same_paddings[1][1]
    crop_left = (strides[1] - strides[1] % 2) // 2 + full_paddings[2][1] - same_paddings[2][1]
    outputs = outputs[:, crop_top:crop_top + strides[0] * input_shape[1],
              crop_left:crop_left + strides[1] * input_shape[2], :]

    if use_bias:
        if bias is None:
            with tf.variable_scope('upsample_conv2d'):
                bias = tf.get_variable('bias', [filters], dtype=tf.float32, initializer=tf.zeros_initializer())
        else:
            bias_shape = [filters]
            if bias_shape != bias.get_shape().as_list():
                raise ValueError("Expecting bias with shape %s but instead got bias with shape %s" %
                                 (tuple(bias_shape), tuple(bias.get_shape().as_list())))
        outputs = tf.nn.bias_add(outputs, bias)
    return outputs


def conv3d(inputs, filters, kernel_size, strides=(1, 1), padding='SAME', use_bias=True, use_spectral_norm=False):
    kernel_size = list(kernel_size) if isinstance(kernel_size, (tuple, list)) else [kernel_size] * 3
    strides = list(strides) if isinstance(strides, (tuple, list)) else [strides] * 3
    input_shape = inputs.get_shape().as_list()
    kernel_shape = list(kernel_size) + [input_shape[-1], filters]
    with tf.variable_scope('conv3d'):
        kernel = tf.get_variable('kernel', kernel_shape, dtype=tf.float32, initializer=tf.truncated_normal_initializer(stddev=0.02))
        if use_spectral_norm:
            kernel = spectral_normed_weight(kernel)
    outputs = tf.nn.conv3d(inputs, kernel, [1] + strides + [1], padding=padding)
    if use_bias:
        bias = tf.get_variable('bias', [filters], dtype=tf.float32, initializer=tf.zeros_initializer())
        outputs = tf.nn.bias_add(outputs, bias)
    return outputs


def pool2d(inputs, pool_size, strides=(1, 1), padding='SAME', pool_mode='avg'):
    pool_size = list(pool_size) if isinstance(pool_size, (tuple, list)) else [pool_size] * 2
    strides = list(strides) if isinstance(strides, (tuple, list)) else [strides] * 2
    if padding == 'FULL':
        inputs = pad2d(inputs, pool_size, strides=strides, padding=padding, mode='CONSTANT')
        padding = 'VALID'
    if pool_mode == 'max':
        outputs = tf.nn.max_pool(inputs, [1] + pool_size + [1], [1] + strides + [1], padding=padding)
    elif pool_mode == 'avg':
        outputs = tf.nn.avg_pool(inputs, [1] + pool_size + [1], [1] + strides + [1], padding=padding)
    else:
        raise ValueError('Invalid pooling mode:', pool_mode)
    return outputs


def conv_pool2d(inputs, filters, kernel_size, strides=(1, 1), padding='SAME', kernel=None, use_bias=True, bias=None, pool_mode='avg'):
    """
    Similar optimization as in upsample_conv2d

    Example:
        >>> import numpy as np
        >>> import tensorflow as tf
        >>> from video_prediction.ops import conv_pool2d, conv2d, pool2d
        >>> inputs_shape = [4, 16, 16, 32]
        >>> kernel_size = [3, 3]  # for convolution
        >>> filters = 64  # for convolution
        >>> strides = [2, 2]  # for pooling
        >>> inputs = tf.get_variable("inputs", inputs_shape)
        >>> kernel = tf.get_variable("kernel", (kernel_size[0], kernel_size[1], inputs_shape[-1], filters))
        >>> bias = tf.get_variable("bias", (filters,))
        >>> outputs = conv_pool2d(inputs, filters, kernel_size=kernel_size, strides=strides,
                                  kernel=kernel, bias=bias, pool_mode='avg')
        >>> inputs_conv = conv2d(inputs, filters, kernel_size=kernel_size, strides=(1, 1),
                                 kernel=kernel, bias=bias)
        >>> outputs_pool = pool2d(inputs_conv, pool_size=strides, strides=strides, pool_mode='avg')
        >>> sess = tf.Session()
        >>> sess.run(tf.global_variables_initializer())
        >>> assert np.allclose(*sess.run([outputs, outputs_pool]), atol=1e-5)

    """
    kernel_size = list(kernel_size) if isinstance(kernel_size, (tuple, list)) else [kernel_size] * 2
    strides = list(strides) if isinstance(strides, (tuple, list)) else [strides] * 2
    if padding != 'SAME' or pool_mode != 'avg':
        raise NotImplementedError
    input_shape = inputs.get_shape().as_list()
    if input_shape[1] % strides[0] or input_shape[2] % strides[1]:
        raise NotImplementedError("The height and width of the input should be "
                                  "an integer multiple of the respective stride.")
    kernel_shape = list(kernel_size) + [input_shape[-1], filters]
    if kernel is None:
        with tf.variable_scope('conv_pool2d'):
            kernel = tf.get_variable('kernel', kernel_shape, dtype=tf.float32,
                                     initializer=tf.truncated_normal_initializer(stddev=0.02))
    else:
        if kernel_shape != kernel.get_shape().as_list():
            raise ValueError("Expecting kernel with shape %s but instead got kernel with shape %s" %
                             (tuple(kernel_shape), tuple(kernel.get_shape().as_list())))

    # pool kernel
    kernel_reshaped = tf.reshape(kernel, [1] + kernel_size + [input_shape[-1] * filters])
    kernel_pool_reshaped = pool2d(kernel_reshaped, pool_size=strides, padding='FULL', pool_mode='avg')
    kernel_pool = tf.reshape(kernel_pool_reshaped,
                             kernel_pool_reshaped.get_shape().as_list()[1:3] + [input_shape[-1], filters])

    outputs = conv2d(inputs, filters, kernel_size=kernel_pool.get_shape().as_list()[:2], strides=strides,
                     kernel=kernel_pool, padding='SAME', use_bias=False)
    if use_bias:
        if bias is None:
            with tf.variable_scope('conv_pool2d'):
                bias = tf.get_variable('bias', [filters], dtype=tf.float32, initializer=tf.zeros_initializer())
        else:
            bias_shape = [filters]
            if bias_shape != bias.get_shape().as_list():
                raise ValueError("Expecting bias with shape %s but instead got bias with shape %s" %
                                 (tuple(bias_shape), tuple(bias.get_shape().as_list())))
        outputs = tf.nn.bias_add(outputs, bias)
    return outputs


def conv_pool2d_v2(inputs, filters, kernel_size, strides=(1, 1), padding='SAME', kernel=None, use_bias=True, bias=None, pool_mode='avg'):
    kernel_size = list(kernel_size) if isinstance(kernel_size, (tuple, list)) else [kernel_size] * 2
    strides = list(strides) if isinstance(strides, (tuple, list)) else [strides] * 2
    if padding != 'SAME' or pool_mode != 'avg':
        raise NotImplementedError
    input_shape = inputs.get_shape().as_list()
    if input_shape[1] % strides[0] or input_shape[2] % strides[1]:
        raise NotImplementedError("The height and width of the input should be "
                                  "an integer multiple of the respective stride.")
    kernel_shape = list(kernel_size) + [input_shape[-1], filters]
    if kernel is None:
        with tf.variable_scope('conv_pool2d'):
            kernel = tf.get_variable('kernel', kernel_shape, dtype=tf.float32,
                                     initializer=tf.truncated_normal_initializer(stddev=0.02))
    else:
        if kernel_shape != kernel.get_shape().as_list():
            raise ValueError("Expecting kernel with shape %s but instead got kernel with shape %s" %
                             (tuple(kernel_shape), tuple(kernel.get_shape().as_list())))

    inputs_conv = conv2d(inputs, filters, kernel_size=kernel_size, strides=(1, 1),
                         kernel=kernel, bias=None, use_bias=False)
    outputs = pool2d(inputs_conv, pool_size=strides, strides=strides, pool_mode='avg')

    if use_bias:
        if bias is None:
            with tf.variable_scope('conv_pool2d'):
                bias = tf.get_variable('bias', [filters], dtype=tf.float32, initializer=tf.zeros_initializer())
        else:
            bias_shape = [filters]
            if bias_shape != bias.get_shape().as_list():
                raise ValueError("Expecting bias with shape %s but instead got bias with shape %s" %
                                 (tuple(bias_shape), tuple(bias.get_shape().as_list())))
        outputs = tf.nn.bias_add(outputs, bias)
    return outputs


def lrelu(x, alpha):
    """
    Leaky ReLU activation function

    Reference:
        https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/ops/nn_ops.py
    """
    with tf.name_scope("lrelu"):
        return tf.maximum(alpha * x, x)


def batchnorm(input):
    with tf.variable_scope("batchnorm"):
        # this block looks like it has 3 inputs on the graph unless we do this
        input = tf.identity(input)

        channels = input.get_shape()[-1]
        offset = tf.get_variable("offset", [channels], dtype=tf.float32, initializer=tf.zeros_initializer())
        scale = tf.get_variable("scale", [channels], dtype=tf.float32, initializer=tf.truncated_normal_initializer(1.0, 0.02))
        mean, variance = tf.nn.moments(input, axes=list(range(len(input.get_shape()) - 1)), keepdims=False)
        variance_epsilon = 1e-5
        normalized = tf.nn.batch_normalization(input, mean, variance, offset, scale, variance_epsilon=variance_epsilon)
        return normalized


def instancenorm(input):
    with tf.variable_scope("instancenorm"):
        # this block looks like it has 3 inputs on the graph unless we do this
        input = tf.identity(input)

        channels = input.get_shape()[-1]
        offset = tf.get_variable("offset", [channels], dtype=tf.float32, initializer=tf.zeros_initializer())
        scale = tf.get_variable("scale", [channels], dtype=tf.float32,
                                initializer=tf.truncated_normal_initializer(1.0, 0.02))
        mean, variance = tf.nn.moments(input, axes=list(range(1, len(input.get_shape()) - 1)), keepdims=True)
        variance_epsilon = 1e-5
        normalized = tf.nn.batch_normalization(input, mean, variance, offset, scale,
                                               variance_epsilon=variance_epsilon)
        return normalized


def flatten(input, axis=1, end_axis=-1):
    """
    Caffe-style flatten.

    Args:
        inputs: An N-D tensor.
        axis: The first axis to flatten: all preceding axes are retained in the output.
            May be negative to index from the end (e.g., -1 for the last axis).
        end_axis: The last axis to flatten: all following axes are retained in the output.
            May be negative to index from the end (e.g., the default -1 for the last
            axis)

    Returns:
        A M-D tensor where M = N - (end_axis - axis)
    """
    input_shape = tf.shape(input)
    input_rank = tf.shape(input_shape)[0]
    if axis < 0:
        axis = input_rank + axis
    if end_axis < 0:
        end_axis = input_rank + end_axis
    output_shape = []
    if axis != 0:
        output_shape.append(input_shape[:axis])
    output_shape.append([tf.reduce_prod(input_shape[axis:end_axis + 1])])
    if end_axis + 1 != input_rank:
        output_shape.append(input_shape[end_axis + 1:])
    output_shape = tf.concat(output_shape, axis=0)
    output = tf.reshape(input, output_shape)
    return output


def tile_concat(values, axis):
    """
    Like concat except that first tiles the broadcastable dimensions if necessary
    """
    shapes = [value.get_shape() for value in values]
    # convert axis to positive form
    ndims = shapes[0].ndims
    for shape in shapes[1:]:
        assert ndims == shape.ndims
    if -ndims < axis < 0:
        axis += ndims
    # remove axis dimension
    shapes = [shape.as_list() for shape in shapes]
    dims = [shape.pop(axis) for shape in shapes]
    shapes = [tf.TensorShape(shape) for shape in shapes]
    # compute broadcasted shape
    b_shape = shapes[0]
    for shape in shapes[1:]:
        b_shape = tf.broadcast_static_shape(b_shape, shape)
    # add back axis dimension
    b_shapes = [b_shape.as_list() for _ in dims]
    for b_shape, dim in zip(b_shapes, dims):
        b_shape.insert(axis, dim)
    # tile values to match broadcasted shape, if necessary
    b_values = []
    for value, b_shape in zip(values, b_shapes):
        multiples = []
        for dim, b_dim in zip(value.get_shape().as_list(), b_shape):
            if dim == b_dim:
                multiples.append(1)
            else:
                assert dim == 1
                multiples.append(b_dim)
        if any(multiple != 1 for multiple in multiples):
            b_value = tf.tile(value, multiples)
        else:
            b_value = value
        b_values.append(b_value)
    return tf.concat(b_values, axis=axis)


def sigmoid_kl_with_logits(logits, targets):
    # broadcasts the same target value across the whole batch
    # this is implemented so awkwardly because tensorflow lacks an x log x op
    assert isinstance(targets, float)
    if targets in [0., 1.]:
        entropy = 0.
    else:
        entropy = - targets * np.log(targets) - (1. - targets) * np.log(1. - targets)
    return tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=tf.ones_like(logits) * targets) - entropy


def spectral_normed_weight(W, u=None, num_iters=1):
    SPECTRAL_NORMALIZATION_VARIABLES = 'spectral_normalization_variables'

    # Usually num_iters = 1 will be enough
    W_shape = W.shape.as_list()
    W_reshaped = tf.reshape(W, [-1, W_shape[-1]])
    if u is None:
        u = tf.get_variable("u", [1, W_shape[-1]], initializer=tf.truncated_normal_initializer(), trainable=False)

    def l2normalize(v, eps=1e-12):
        return v / (tf.norm(v) + eps)

    def power_iteration(i, u_i, v_i):
        v_ip1 = l2normalize(tf.matmul(u_i, tf.transpose(W_reshaped)))
        u_ip1 = l2normalize(tf.matmul(v_ip1, W_reshaped))
        return i + 1, u_ip1, v_ip1
    _, u_final, v_final = tf.while_loop(
        cond=lambda i, _1, _2: i < num_iters,
        body=power_iteration,
        loop_vars=(tf.constant(0, dtype=tf.int32),
                   u, tf.zeros(dtype=tf.float32, shape=[1, W_reshaped.shape.as_list()[0]]))
    )
    sigma = tf.squeeze(tf.matmul(tf.matmul(v_final, W_reshaped), tf.transpose(u_final)))
    W_bar_reshaped = W_reshaped / sigma
    W_bar = tf.reshape(W_bar_reshaped, W_shape)

    if u not in tf.get_collection(SPECTRAL_NORMALIZATION_VARIABLES):
        tf.add_to_collection(SPECTRAL_NORMALIZATION_VARIABLES, u)
        tf.add_to_collection(tf.GraphKeys.UPDATE_OPS, u.assign(u_final))
    return W_bar


def get_activation_layer(layer_type):
    if layer_type == 'relu':
        layer = tf.nn.relu
    elif layer_type == 'elu':
        layer = tf.nn.elu
    else:
        raise ValueError('Invalid activation layer %s' % layer_type)
    return layer


def get_norm_layer(layer_type):
    if layer_type == 'batch':
        layer = tf.layers.batch_normalization
    elif layer_type == 'layer':
        layer = tf.contrib.layers.layer_norm
    elif layer_type == 'instance':
        from video_prediction.layers import fused_instance_norm
        layer = fused_instance_norm
    elif layer_type == 'none':
        layer = tf.identity
    else:
        raise ValueError('Invalid normalization layer %s' % layer_type)
    return layer


def get_upsample_layer(layer_type):
    if layer_type == 'deconv2d':
        layer = deconv2d
    elif layer_type == 'upsample_conv2d':
        layer = upsample_conv2d
    elif layer_type == 'upsample_conv2d_v2':
        layer = upsample_conv2d_v2
    else:
        raise ValueError('Invalid upsampling layer %s' % layer_type)
    return layer


def get_downsample_layer(layer_type):
    if layer_type == 'conv2d':
        layer = conv2d
    elif layer_type == 'conv_pool2d':
        layer = conv_pool2d
    elif layer_type == 'conv_pool2d_v2':
        layer = conv_pool2d_v2
    else:
        raise ValueError('Invalid downsampling layer %s' % layer_type)
    return layer
