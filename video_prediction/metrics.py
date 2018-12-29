import tensorflow as tf
import lpips_tf


def mse(a, b):
    return tf.reduce_mean(tf.squared_difference(a, b), [-3, -2, -1])


def psnr(a, b):
    return tf.image.psnr(a, b, 1.0)


def ssim(a, b):
    return tf.image.ssim(a, b, 1.0)


def lpips(input0, input1):
    if input0.shape[-1].value == 1:
        input0 = tf.tile(input0, [1] * (input0.shape.ndims - 1) + [3])
    if input1.shape[-1].value == 1:
        input1 = tf.tile(input1, [1] * (input1.shape.ndims - 1) + [3])

    distance = lpips_tf.lpips(input0, input1)
    return -distance
