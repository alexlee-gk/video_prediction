from math import ceil

import h5py
import tensorflow as tf
from keras.utils.data_utils import get_file


def residual_conv(prev, level, pad=1, lvl=1, sub_lvl=1, modify_stride=False):
    lvl = str(lvl)
    sub_lvl = str(sub_lvl)
    names = ["conv" + lvl + "_" + sub_lvl + "_1x1_reduce",
             "conv" + lvl + "_" + sub_lvl + "_1x1_reduce_bn",
             "conv" + lvl + "_" + sub_lvl + "_3x3",
             "conv" + lvl + "_" + sub_lvl + "_3x3_bn",
             "conv" + lvl + "_" + sub_lvl + "_1x1_increase",
             "conv" + lvl + "_" + sub_lvl + "_1x1_increase_bn"]
    if modify_stride is False:
        prev = tf.layers.conv2d(prev, 64 * level, (1, 1), strides=(1, 1), name=names[0],
                                use_bias=False)
    elif modify_stride is True:
        prev = tf.layers.conv2d(prev, 64 * level, (1, 1), strides=(2, 2), name=names[0],
                                use_bias=False)

    prev = tf.layers.batch_normalization(prev, momentum=0.95, epsilon=1e-5, name=names[1])
    prev = tf.nn.relu(prev)

    prev = tf.pad(prev, [[0, 0], [pad, pad], [pad, pad], [0, 0]])
    prev = tf.layers.conv2d(prev, 64 * level, (3, 3), strides=(1, 1), dilation_rate=pad,
                            name=names[2], use_bias=False)

    prev = tf.layers.batch_normalization(prev, momentum=0.95, epsilon=1e-5, name=names[3])
    prev = tf.nn.relu(prev)
    prev = tf.layers.conv2d(prev, 256 * level, (1, 1), strides=(1, 1), name=names[4],
                            use_bias=False)
    prev = tf.layers.batch_normalization(prev, momentum=0.95, epsilon=1e-5, name=names[5])
    return prev


def short_convolution_branch(prev, level, lvl=1, sub_lvl=1, modify_stride=False):
    lvl = str(lvl)
    sub_lvl = str(sub_lvl)
    names = ["conv" + lvl + "_" + sub_lvl + "_1x1_proj",
             "conv" + lvl + "_" + sub_lvl + "_1x1_proj_bn"]

    if modify_stride is False:
        prev = tf.layers.conv2d(prev, 256 * level, (1, 1), strides=(1, 1), name=names[0],
                                use_bias=False)
    elif modify_stride is True:
        prev = tf.layers.conv2d(prev, 256 * level, (1, 1), strides=(2, 2), name=names[0],
                                use_bias=False)

    prev = tf.layers.batch_normalization(prev, momentum=0.95, epsilon=1e-5, name=names[1])
    return prev


def empty_branch(prev):
    return prev


def residual_short(prev_layer, level, pad=1, lvl=1, sub_lvl=1, modify_stride=False):
    prev_layer = tf.nn.relu(prev_layer)
    block_1 = residual_conv(prev_layer, level,
                            pad=pad, lvl=lvl, sub_lvl=sub_lvl,
                            modify_stride=modify_stride)

    block_2 = short_convolution_branch(prev_layer, level,
                                       lvl=lvl, sub_lvl=sub_lvl,
                                       modify_stride=modify_stride)
    added = block_1 + block_2
    return added


def residual_empty(prev_layer, level, pad=1, lvl=1, sub_lvl=1):
    prev_layer = tf.nn.relu(prev_layer)

    block_1 = residual_conv(prev_layer, level, pad=pad,
                            lvl=lvl, sub_lvl=sub_lvl)
    block_2 = empty_branch(prev_layer)
    added = block_1 + block_2
    return added


def ResNet(inp, layers):
    # Names for the first couple layers of model
    names = ["conv1_1_3x3_s2",
             "conv1_1_3x3_s2_bn",
             "conv1_2_3x3",
             "conv1_2_3x3_bn",
             "conv1_3_3x3",
             "conv1_3_3x3_bn"]

    # Short branch(only start of network)

    cnv1 = tf.layers.conv2d(inp, 64, (3, 3), strides=(2, 2), padding='same', name=names[0],
                            use_bias=False)  # "conv1_1_3x3_s2"
    bn1 = tf.layers.batch_normalization(cnv1, momentum=0.95, epsilon=1e-5, name=names[1])  # "conv1_1_3x3_s2/bn"
    relu1 = tf.nn.relu(bn1)  # "conv1_1_3x3_s2/relu"

    cnv1 = tf.layers.conv2d(relu1, 64, (3, 3), strides=(1, 1), padding='same', name=names[2],
                            use_bias=False)  # "conv1_2_3x3"
    bn1 = tf.layers.batch_normalization(cnv1, momentum=0.95, epsilon=1e-5, name=names[3])  # "conv1_2_3x3/bn"
    relu1 = tf.nn.relu(bn1)  # "conv1_2_3x3/relu"

    cnv1 = tf.layers.conv2d(relu1, 128, (3, 3), strides=(1, 1), padding='same', name=names[4],
                            use_bias=False)  # "conv1_3_3x3"
    bn1 = tf.layers.batch_normalization(cnv1, momentum=0.95, epsilon=1e-5, name=names[5])  # "conv1_3_3x3/bn"
    relu1 = tf.nn.relu(bn1)  # "conv1_3_3x3/relu"

    res = tf.layers.max_pooling2d(relu1, pool_size=(3, 3), padding='same',
                                  strides=(2, 2))  # "pool1_3x3_s2"

    # ---Residual layers(body of network)

    """
    Modify_stride --Used only once in first 3_1 convolutions block.
    changes stride of first convolution from 1 -> 2
    """

    # 2_1- 2_3
    res = residual_short(res, 1, pad=1, lvl=2, sub_lvl=1)
    for i in range(2):
        res = residual_empty(res, 1, pad=1, lvl=2, sub_lvl=i + 2)

    # 3_1 - 3_3
    res = residual_short(res, 2, pad=1, lvl=3, sub_lvl=1, modify_stride=True)
    for i in range(3):
        res = residual_empty(res, 2, pad=1, lvl=3, sub_lvl=i + 2)
    if layers is 50:
        # 4_1 - 4_6
        res = residual_short(res, 4, pad=2, lvl=4, sub_lvl=1)
        for i in range(5):
            res = residual_empty(res, 4, pad=2, lvl=4, sub_lvl=i + 2)
    elif layers is 101:
        # 4_1 - 4_23
        res = residual_short(res, 4, pad=2, lvl=4, sub_lvl=1)
        for i in range(22):
            res = residual_empty(res, 4, pad=2, lvl=4, sub_lvl=i + 2)
    else:
        print("This ResNet is not implemented")

    # 5_1 - 5_3
    res = residual_short(res, 8, pad=4, lvl=5, sub_lvl=1)
    for i in range(2):
        res = residual_empty(res, 8, pad=4, lvl=5, sub_lvl=i + 2)

    res = tf.nn.relu(res)
    return res


def interp_block(prev_layer, level, feature_map_shape, input_shape):
    if input_shape == (256, 256):
        kernel_strides_map = {1: 30,
                              2: 15,
                              3: 10,
                              6: 5}
    elif input_shape == (473, 473):
        kernel_strides_map = {1: 60,
                              2: 30,
                              3: 20,
                              6: 10}
    elif input_shape == (713, 713):
        kernel_strides_map = {1: 90,
                              2: 45,
                              3: 30,
                              6: 15}
    else:
        print("Pooling parameters for input shape ",
              input_shape, " are not defined.")
        exit(1)

    names = [
        "conv5_3_pool" + str(level) + "_conv",
        "conv5_3_pool" + str(level) + "_conv_bn"
    ]
    kernel = (kernel_strides_map[level], kernel_strides_map[level])
    strides = (kernel_strides_map[level], kernel_strides_map[level])
    prev_layer = tf.layers.average_pooling2d(prev_layer, kernel, strides=strides)
    prev_layer = tf.layers.conv2d(prev_layer, 512, (1, 1), strides=(1, 1), name=names[0],
                                  use_bias=False)
    prev_layer = tf.layers.batch_normalization(prev_layer, momentum=0.95, epsilon=1e-5, name=names[1])
    prev_layer = tf.nn.relu(prev_layer)
    prev_layer = tf.image.resize_images(prev_layer, feature_map_shape, align_corners=True)
    return prev_layer


def build_pyramid_pooling_module(res, input_shape):
    """Build the Pyramid Pooling Module."""
    # ---PSPNet concat layers with Interpolation
    feature_map_size = tuple(int(ceil(input_dim / 8.0))
                             for input_dim in input_shape)
    print("PSP module will interpolate to a final feature map size of %s" %
          (feature_map_size, ))

    interp_block1 = interp_block(res, 1, feature_map_size, input_shape)
    interp_block2 = interp_block(res, 2, feature_map_size, input_shape)
    interp_block3 = interp_block(res, 3, feature_map_size, input_shape)
    interp_block6 = interp_block(res, 6, feature_map_size, input_shape)

    # concat all these layers. resulted
    # shape=(1,feature_map_size_x,feature_map_size_y,4096)
    res = tf.concat([res,
                     interp_block6,
                     interp_block3,
                     interp_block2,
                     interp_block1], axis=-1)
    return res


def pspnet(inputs, resnet_layers=50, nb_classes=150, include_top=False):
    """Build PSPNet."""
    print("Building a PSPNet based on ResNet %i" % resnet_layers)
    input_shape = tuple(inputs.shape.as_list()[-3:-1])

    res = ResNet(inputs, layers=resnet_layers)  # (?, 32, 32, 2048)
    psp = build_pyramid_pooling_module(res, input_shape)  # (?, 32, 32, 4096)

    x = tf.layers.conv2d(psp, 512, (3, 3), strides=(1, 1), padding="same", name="conv5_4",
                         use_bias=False)  # (?, 32, 32, 512)
    x = tf.layers.batch_normalization(x, momentum=0.95, epsilon=1e-5, name="conv5_4_bn")
    x = tf.nn.relu(x)

    if include_top:
        x = tf.nn.dropout(x, 0.1)

        x = tf.layers.conv2d(x, nb_classes, (1, 1), strides=(1, 1), name="conv6")
        x = tf.image.resize_images(x, [input_shape[0], input_shape[1]], align_corners=True)
        x = tf.nn.softmax(x)
    return x


def pspnet50_assign_from_values_fn(var_name_prefix='generator/', include_top=False):
    weights_path = get_file('pspnet50_ade20k.h5',
                            'https://www.dropbox.com/s/0uxn14y26jcui4v/pspnet50_ade20k.h5?dl=1',
                            cache_subdir='weights')
    f = h5py.File(weights_path, 'r')
    var_names_to_values = {}
    layer_names = [n.decode('utf8') for n in f.attrs['layer_names']]
    for name in layer_names:
        g = f[name]
        weight_names = [n.decode('utf8') for n in g.attrs['weight_names']]
        for weight_name in weight_names:
            var_names_to_values[var_name_prefix + weight_name] = g[weight_name][()]
    if not include_top:
        var_names_to_values.pop(var_name_prefix + 'conv6/kernel:0')
        var_names_to_values.pop(var_name_prefix + 'conv6/bias:0')
    return tf.contrib.framework.assign_from_values_fn(var_names_to_values)
