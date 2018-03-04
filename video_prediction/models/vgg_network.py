import numpy as np
import tensorflow as tf


def vgg_assign_from_values_fn(model='vgg16',
                              var_name_prefix='vgg/',
                              var_name_kernel_postfix='/kernel:0',
                              var_name_bias_postfix='/bias:0'):
    if model not in ('vgg16', 'vgg19'):
        raise ValueError('Invalid model %s' % model)
    import h5py
    WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.1/' \
                          '%s_weights_tf_dim_ordering_tf_kernels_notop.h5' % model
    weights_path = tf.keras.utils.get_file(
        '%s_weights_tf_dim_ordering_tf_kernels_notop.h5' % model,
        WEIGHTS_PATH_NO_TOP,
        cache_subdir='models')
    weights_file = h5py.File(weights_path, 'r')

    num_blocks = 5
    max_num_convs_in_block = 3 if model == 'vgg16' else 4

    weight_name_kernel_postfix = '_W_1:0'
    weight_name_bias_postfix = '_b_1:0'
    var_names_to_values = {}
    for block_id in range(num_blocks):
        for conv_id in range(max_num_convs_in_block):
            if block_id < 2 and conv_id >= 2:
                continue
            name = 'block%d_conv%d' % (block_id + 1, conv_id + 1)
            var_names_to_values[var_name_prefix + name + var_name_kernel_postfix] = \
                weights_file[name][name + weight_name_kernel_postfix][()]
            var_names_to_values[var_name_prefix + name + var_name_bias_postfix] = \
                weights_file[name][name + weight_name_bias_postfix][()]
    return tf.contrib.framework.assign_from_values_fn(var_names_to_values)


def vgg16(rgb_image):
    """
        rgb_image: 4-D tensor with pixel intensities between 0 and 1.
    """
    bgr_mean = np.array([103.939, 116.779, 123.68], np.float32)
    rgb_scaled_image = rgb_image * 255.0
    bgr_scaled_image = rgb_scaled_image[:, :, :, ::-1]
    bgr_centered_image = bgr_scaled_image - tf.convert_to_tensor(bgr_mean)

    x = bgr_centered_image
    tensors = [x]
    features = []

    # Block1
    x = tf.layers.conv2d(x, 64, (3, 3), padding='same', activation=tf.nn.relu, trainable=False, name='block1_conv1')
    tensors.append(x)
    x = tf.layers.conv2d(x, 64, (3, 3), padding='same', activation=tf.nn.relu, trainable=False, name='block1_conv2')
    tensors.append(x)
    features.append(x)
    x = tf.layers.max_pooling2d(x, (2, 2), (2, 2), padding='same', name='block1_pool')
    tensors.append(x)

    # Block2
    x = tf.layers.conv2d(x, 128, (3, 3), padding='same', activation=tf.nn.relu, trainable=False, name='block2_conv1')
    tensors.append(x)
    x = tf.layers.conv2d(x, 128, (3, 3), padding='same', activation=tf.nn.relu, trainable=False, name='block2_conv2')
    tensors.append(x)
    features.append(x)
    x = tf.layers.max_pooling2d(x, (2, 2), (2, 2), padding='same', name='block2_pool')
    tensors.append(x)

    # Block3
    x = tf.layers.conv2d(x, 256, (3, 3), padding='same', activation=tf.nn.relu, trainable=False, name='block3_conv1')
    tensors.append(x)
    x = tf.layers.conv2d(x, 256, (3, 3), padding='same', activation=tf.nn.relu, trainable=False, name='block3_conv2')
    tensors.append(x)
    x = tf.layers.conv2d(x, 256, (3, 3), padding='same', activation=tf.nn.relu, trainable=False, name='block3_conv3')
    tensors.append(x)
    features.append(x)
    x = tf.layers.max_pooling2d(x, (2, 2), (2, 2), padding='same', name='block3_pool')
    tensors.append(x)

    # Block4
    x = tf.layers.conv2d(x, 512, (3, 3), padding='same', activation=tf.nn.relu, trainable=False, name='block4_conv1')
    tensors.append(x)
    x = tf.layers.conv2d(x, 512, (3, 3), padding='same', activation=tf.nn.relu, trainable=False, name='block4_conv2')
    tensors.append(x)
    x = tf.layers.conv2d(x, 512, (3, 3), padding='same', activation=tf.nn.relu, trainable=False, name='block4_conv3')
    tensors.append(x)
    features.append(x)
    x = tf.layers.max_pooling2d(x, (2, 2), (2, 2), padding='same', name='block4_pool')
    tensors.append(x)

    # Block5
    x = tf.layers.conv2d(x, 512, (3, 3), padding='same', activation=tf.nn.relu, trainable=False, name='block5_conv1')
    tensors.append(x)
    x = tf.layers.conv2d(x, 512, (3, 3), padding='same', activation=tf.nn.relu, trainable=False, name='block5_conv2')
    tensors.append(x)
    x = tf.layers.conv2d(x, 512, (3, 3), padding='same', activation=tf.nn.relu, trainable=False, name='block5_conv3')
    tensors.append(x)
    features.append(x)
    x = tf.layers.max_pooling2d(x, (2, 2), (2, 2), padding='same', name='block5_pool')
    tensors.append(x)

    return tensors, features


def vgg19(rgb_image):
    """
        rgb_image: 4-D tensor with pixel intensities between 0 and 1.
    """
    bgr_mean = np.array([103.939, 116.779, 123.68], np.float32)
    rgb_scaled_image = rgb_image * 255.0
    bgr_scaled_image = rgb_scaled_image[:, :, :, ::-1]
    bgr_centered_image = bgr_scaled_image - tf.convert_to_tensor(bgr_mean)

    x = bgr_centered_image
    tensors = [x]
    features = []

    # Block1
    x = tf.layers.conv2d(x, 64, (3, 3), padding='same', activation=tf.nn.relu, trainable=False, name='block1_conv1')
    tensors.append(x)
    x = tf.layers.conv2d(x, 64, (3, 3), padding='same', activation=tf.nn.relu, trainable=False, name='block1_conv2')
    tensors.append(x)
    features.append(x)
    x = tf.layers.max_pooling2d(x, (2, 2), (2, 2), padding='same', name='block1_pool')
    tensors.append(x)

    # Block2
    x = tf.layers.conv2d(x, 128, (3, 3), padding='same', activation=tf.nn.relu, trainable=False, name='block2_conv1')
    tensors.append(x)
    x = tf.layers.conv2d(x, 128, (3, 3), padding='same', activation=tf.nn.relu, trainable=False, name='block2_conv2')
    tensors.append(x)
    features.append(x)
    x = tf.layers.max_pooling2d(x, (2, 2), (2, 2), padding='same', name='block2_pool')
    tensors.append(x)

    # Block3
    x = tf.layers.conv2d(x, 256, (3, 3), padding='same', activation=tf.nn.relu, trainable=False, name='block3_conv1')
    tensors.append(x)
    x = tf.layers.conv2d(x, 256, (3, 3), padding='same', activation=tf.nn.relu, trainable=False, name='block3_conv2')
    tensors.append(x)
    x = tf.layers.conv2d(x, 256, (3, 3), padding='same', activation=tf.nn.relu, trainable=False, name='block3_conv3')
    tensors.append(x)
    x = tf.layers.conv2d(x, 256, (3, 3), padding='same', activation=tf.nn.relu, trainable=False, name='block3_conv4')
    tensors.append(x)
    features.append(x)
    x = tf.layers.max_pooling2d(x, (2, 2), (2, 2), padding='same', name='block3_pool')
    tensors.append(x)

    # Block4
    x = tf.layers.conv2d(x, 512, (3, 3), padding='same', activation=tf.nn.relu, trainable=False, name='block4_conv1')
    tensors.append(x)
    x = tf.layers.conv2d(x, 512, (3, 3), padding='same', activation=tf.nn.relu, trainable=False, name='block4_conv2')
    tensors.append(x)
    x = tf.layers.conv2d(x, 512, (3, 3), padding='same', activation=tf.nn.relu, trainable=False, name='block4_conv3')
    tensors.append(x)
    x = tf.layers.conv2d(x, 512, (3, 3), padding='same', activation=tf.nn.relu, trainable=False, name='block4_conv4')
    tensors.append(x)
    features.append(x)
    x = tf.layers.max_pooling2d(x, (2, 2), (2, 2), padding='same', name='block4_pool')
    tensors.append(x)

    # Block5
    x = tf.layers.conv2d(x, 512, (3, 3), padding='same', activation=tf.nn.relu, trainable=False, name='block5_conv1')
    tensors.append(x)
    x = tf.layers.conv2d(x, 512, (3, 3), padding='same', activation=tf.nn.relu, trainable=False, name='block5_conv2')
    tensors.append(x)
    x = tf.layers.conv2d(x, 512, (3, 3), padding='same', activation=tf.nn.relu, trainable=False, name='block5_conv3')
    tensors.append(x)
    x = tf.layers.conv2d(x, 512, (3, 3), padding='same', activation=tf.nn.relu, trainable=False, name='block5_conv4')
    tensors.append(x)
    features.append(x)
    x = tf.layers.max_pooling2d(x, (2, 2), (2, 2), padding='same', name='block5_pool')
    tensors.append(x)

    return tensors, features
