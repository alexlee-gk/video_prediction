import numpy as np
import tensorflow as tf

from video_prediction.models import vgg_network


def _axis(keep_axis, ndims):
    if keep_axis is None:
        axis = None
    else:
        axis = list(range(ndims))
        try:
            for keep_axis_ in keep_axis:
                axis.remove(keep_axis_)
        except TypeError:
            axis.remove(keep_axis)
        axis = tuple(axis)
    return axis


def peak_signal_to_noise_ratio_np(true, pred, keep_axis=None):
    ndims = max(true.ndim, pred.ndim)
    mse = mean_squared_error_np(true, pred, keep_axis=list(range(ndims))[:-3])
    psnr = 10.0 * np.log(1.0 / mse) / np.log(10.0)
    return np.mean(psnr, axis=_axis(keep_axis, psnr.ndim))


def mean_squared_error_np(true, pred, keep_axis=None):
    error = true - pred
    return np.mean(np.square(error), axis=_axis(keep_axis, error.ndim))


def structural_similarity_np(true, pred, k1=0.01, k2=0.03, kernel_size=7,
                             data_range=1.0, use_sample_covariance=True,
                             keep_axis=None):
    from skimage.measure import compare_ssim
    kwargs = dict(K1=k1, K2=k2,
                  win_size=kernel_size,
                  data_range=data_range,
                  multichannel=True,
                  use_sample_covariance=use_sample_covariance)
    assert true.shape == pred.shape
    shape = true.shape
    true = true.reshape((-1,) + shape[-3:])
    pred = pred.reshape((-1,) + shape[-3:])
    ssim = []
    for true_y, pred_y in zip(true, pred):
        ssim.append(compare_ssim(true_y, pred_y, **kwargs))
    ssim = np.reshape(ssim, shape[:-3])
    return np.mean(ssim, axis=_axis(keep_axis, ssim.ndim))


def expected_pixel_distribution_np(pix_distrib):
    pix_distrib = pix_distrib / np.sum(pix_distrib, axis=(-3, -2), keepdims=True)
    height, width = pix_distrib.shape[-3:-1]
    xv, yv = np.meshgrid(np.arange(width), np.arange(height))
    return np.stack([np.sum(yv[:, :, None] * pix_distrib, axis=(-3, -2, -1)),
                     np.sum(xv[:, :, None] * pix_distrib, axis=(-3, -2, -1))], axis=-1)


def expected_pixel_distance_np(true_pix_distrib, pred_pix_distribs, keep_axis=None):
    error = expected_pixel_distribution_np(true_pix_distrib) - \
            expected_pixel_distribution_np(pred_pix_distribs)
    return np.linalg.norm(error, axis=_axis(keep_axis, error.ndim))


def peak_signal_to_noise_ratio(true, pred, keep_axis=None):
    """
    Image quality metric based on maximal signal power vs. power of the noise.

    Args:
        true: the ground truth image.
        pred: the predicted image.
        keep_axis: None or int or iterable of ints (all non-negative).

    Returns:
        peak signal to noise ratio (PSNR).
    """
    true = tf.convert_to_tensor(true)
    pred = tf.convert_to_tensor(pred)
    ndims = max(true.shape.ndims, pred.shape.ndims)
    mse = mean_squared_error(true, pred, keep_axis=list(range(ndims))[:-3])
    psnr = 10.0 * tf.log(1.0 / mse) / tf.cast(tf.log(10.0), mse.dtype)
    return tf.reduce_mean(psnr, axis=_axis(keep_axis, psnr.shape.ndims))


def mean_squared_error(true, pred, keep_axis=None):
    """
    L2 distance between tensors true and pred.

    Args:
        true: the ground truth image.
        pred: the predicted image.
        keep_axis: None or int or iterable of ints (all non-negative).

    Returns:
        mean squared error between ground truth and predicted image.
    """
    true = tf.convert_to_tensor(true)
    pred = tf.convert_to_tensor(pred)
    error = true - pred
    return tf.reduce_mean(tf.square(error), axis=_axis(keep_axis, error.shape.ndims))


def structural_similarity(true, pred, k1=0.01, k2=0.03, kernel_size=7,
                          data_range=1.0, use_sample_covariance=True,
                          keep_axis=None):
    """
    Structural SIMilarity (SSIM) index between two images

    Args:
        true: the ground truth image.
        pred: the predicted image.
        keep_axis: None or int or iterable of ints (all non-negative).

    Returns:
        The SSIM between ground truth and predicted image.

    Reference:
        https://github.com/scikit-image/scikit-image/blob/master/skimage/measure/_structural_similarity.py
    """
    def _structural_similarity(true, pred):
        assert true.shape.ndims == 4
        assert pred.shape.ndims == 4
        channels = pred.get_shape().as_list()[-1]
        c1 = (k1 * data_range) ** 2
        c2 = (k2 * data_range) ** 2
        # compute patches independently per channel as in the reference implementation
        patches_true = []
        patches_pred = []
        for true_single_channel, pred_single_channel in zip(tf.split(true, channels, axis=-1),
                                                            tf.split(pred, channels, axis=-1)):
            # use no padding (i.e. valid padding) so we don't compute values near the borders
            patches_true_single_channel = \
                tf.extract_image_patches(true_single_channel, [1] + [kernel_size] * 2 + [1],
                                         strides=[1] * 4, rates=[1] * 4, padding="VALID")
            patches_pred_single_channel = \
                tf.extract_image_patches(pred_single_channel, [1] + [kernel_size] * 2 + [1],
                                         strides=[1] * 4, rates=[1] * 4, padding="VALID")
            patches_true.append(patches_true_single_channel)
            patches_pred.append(patches_pred_single_channel)
        patches_true = tf.stack(patches_true, axis=-2)
        patches_pred = tf.stack(patches_pred, axis=-2)

        mean_true, var_true = tf.nn.moments(patches_true, axes=[-1])
        mean_pred, var_pred = tf.nn.moments(patches_pred, axes=[-1])
        cov_true_pred = tf.reduce_mean(patches_true * patches_pred, axis=-1) - mean_true * mean_pred

        if use_sample_covariance:
            NP = kernel_size ** 2  # 2 spatial dimensions
            cov_norm = NP / (NP - 1)  # sample covariance
        else:
            cov_norm = 1.0  # population covariance to match Wang et. al. 2004
        var_true *= cov_norm
        var_pred *= cov_norm
        cov_true_pred *= cov_norm

        ssim = (2 * mean_true * mean_pred + c1) * (2 * cov_true_pred + c2)
        denom = (tf.square(mean_true) + tf.square(mean_pred) + c1) * (var_pred + var_true + c2)
        ssim /= denom
        return tf.reduce_mean(ssim, axis=(1, 2, 3))

    true = tf.convert_to_tensor(true)
    pred = tf.convert_to_tensor(pred)
    shape = true.shape
    if shape.ndims == 3:
        ssim = tf.squeeze(_structural_similarity(
            tf.expand_dims(true, 0), tf.expand_dims(pred, 0)), 0)
    elif shape.ndims == 4:
        ssim = _structural_similarity(true, pred)
    elif shape.ndims > 4:
        true = tf.reshape(true, tf.concat([[-1], shape[-4:]], axis=0))
        pred = tf.reshape(pred, tf.concat([[-1], shape[-4:]], axis=0))
        ssim = tf.map_fn(lambda args: _structural_similarity(*args),
                         (true, pred), dtype=true.dtype, parallel_iterations=1)
        ssim = tf.reshape(ssim, shape[:-3])
    else:
        raise ValueError
    return tf.reduce_mean(ssim, axis=_axis(keep_axis, ssim.shape.ndims))


def normalize_tensor(tensor, eps=1e-10):
    norm_factor = tf.norm(tensor, axis=-1, keep_dims=True)
    return tensor / (norm_factor + eps)


def cosine_similarity(tensor0, tensor1, keep_axis=None):
    tensor0 = normalize_tensor(tensor0)
    tensor1 = normalize_tensor(tensor1)
    csim = tf.reduce_sum(tensor0 * tensor1, axis=-1)
    return tf.reduce_mean(csim, axis=_axis(keep_axis, csim.shape.ndims))


def cosine_distance(tensor0, tensor1, keep_axis=None):
    """
    Equivalent to:
        tensor0 = normalize_tensor(tensor0)
        tensor1 = normalize_tensor(tensor1)
        return tf.reduce_mean(tf.reduce_sum(tf.square(tensor0 - tensor1), axis=-1)) / 2.0
    """
    return 1.0 - cosine_similarity(tensor0, tensor1, keep_axis=keep_axis)


def vgg_cosine_distance(image0, image1, keep_axis=None):
    def _vgg_cosine_distance(image0, image1):
        assert image0.shape.ndims == 4
        assert image1.shape.ndims == 4
        with tf.variable_scope('vgg', reuse=tf.AUTO_REUSE):
            _, features0 = vgg_network.vgg16(image0)
        with tf.variable_scope('vgg', reuse=tf.AUTO_REUSE):
            _, features1 = vgg_network.vgg16(image1)
        cdist = 0.0
        for feature0, feature1 in zip(features0, features1):
            cdist += cosine_distance(feature0, feature1, keep_axis=0)
        return cdist

    image0 = tf.convert_to_tensor(image0, dtype=tf.float32)
    image1 = tf.convert_to_tensor(image1, dtype=tf.float32)
    shape = image0.shape
    if shape.ndims == 3:
        cdist = tf.squeeze(_vgg_cosine_distance(
            tf.expand_dims(image0, 0), tf.expand_dims(image1, 0)), 0)
    elif shape.ndims == 4:
        cdist = _vgg_cosine_distance(image0, image1)
    elif shape.ndims > 4:
        image0 = tf.reshape(image0, tf.concat([[-1], shape[-4:]], axis=0))
        image1 = tf.reshape(image1, tf.concat([[-1], shape[-4:]], axis=0))
        cdist = tf.map_fn(lambda args: _vgg_cosine_distance(*args),
                          (image0, image1), dtype=image0.dtype)
        cdist = tf.reshape(cdist, shape[:-3])
    else:
        raise ValueError
    return tf.reduce_mean(cdist, axis=_axis(keep_axis, cdist.shape.ndims))


def normalize_tensor_np(tensor, eps=1e-10):
    norm_factor = np.linalg.norm(tensor, axis=-1, keep_dims=True)
    return tensor / (norm_factor + eps)


def cosine_similarity_np(tensor0, tensor1, keep_axis=None):
    tensor0 = normalize_tensor_np(tensor0)
    tensor1 = normalize_tensor_np(tensor1)
    csim = np.sum(tensor0 * tensor1, axis=-1)
    return np.mean(csim, axis=_axis(keep_axis, csim.ndim))


def cosine_distance_np(tensor0, tensor1, keep_axis=None):
    """
    Equivalent to:
        tensor0 = normalize_tensor_np(tensor0)
        tensor1 = normalize_tensor_np(tensor1)
        return np.mean(np.sum(np.square(tensor0 - tensor1), axis=-1)) / 2.0
    """
    return 1.0 - cosine_similarity_np(tensor0, tensor1, keep_axis=keep_axis)


def vgg_cosine_distance_np(image0, image1, keep_axis=None, sess=None):
    if sess is None:
        sess = tf.Session()
        cdist = vgg_cosine_distance(image0, image1, keep_axis=keep_axis)
        sess.run(tf.global_variables_initializer())
        vgg_network.vgg_assign_from_values_fn(var_name_prefix='vgg/')(sess)
    else:
        cdist = vgg_cosine_distance(image0, image1, keep_axis=keep_axis)
    cdist = sess.run(cdist)
    return cdist


def test_ssim():
    import numpy as np
    from skimage.measure import compare_ssim

    batch_size = 4
    image_shape = (64, 64, 3)
    true = np.random.random((batch_size,) + image_shape)
    pred = np.random.random((batch_size,) + image_shape)

    sess = tf.Session()
    ssim_tf = structural_similarity(tf.constant(true), tf.constant(pred))
    ssim_tf = sess.run(ssim_tf)
    ssim_np = structural_similarity_np(true, pred)
    ssim = np.mean([compare_ssim(true_y, pred_y, data_range=1.0, multichannel=True)
                    for true_y, pred_y in zip(true, pred)])
    print(ssim_tf, ssim_np, ssim)


def test_metrics_equivalence():
    import numpy as np

    a = np.random.random((10, 16, 64, 64, 3))
    b = np.random.random((10, 16, 64, 64, 3))
    metrics = [mean_squared_error,
               peak_signal_to_noise_ratio,
               structural_similarity]
    metrics_np = [mean_squared_error_np,
                  peak_signal_to_noise_ratio_np,
                  structural_similarity_np]
    sess = tf.Session()
    with tf.variable_scope('vgg'):
        vgg_network.vgg16(tf.placeholder(tf.float32, shape=[None] * 4))
    sess.run(tf.global_variables_initializer())
    vgg_network.vgg_assign_from_values_fn(var_name_prefix='vgg/')(sess)

    for keep_axis in (None, 0, 1, (0, 1)):
        for metric, metric_np in zip(metrics, metrics_np):
            m = metric(a, b, keep_axis=keep_axis)
            m_np = metric_np(a, b, keep_axis=keep_axis)
            assert np.allclose(sess.run(m), m_np)

        m = vgg_cosine_distance(a, b, keep_axis=keep_axis)
        m_np = vgg_cosine_distance_np(a, b, keep_axis=keep_axis, sess=sess)
        assert np.allclose(sess.run(m), m_np)
    print('The test metrics returned the same values.')


if __name__ == '__main__':
    test_ssim()
    test_metrics_equivalence()
