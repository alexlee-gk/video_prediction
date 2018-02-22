import numpy as np
import tensorflow as tf

from video_prediction.ops import flatten


def peak_signal_to_noise_ratio_np(true, pred, axis=None):
    return 10.0 * np.log(1.0 / mean_squared_error_np(true, pred, axis=axis)) / np.log(10.0)


def mean_squared_error_np(true, pred, axis=None):
    return np.mean(np.square(true - pred), axis=axis)


def structural_similarity_np(true, pred, k1=0.01, k2=0.03, kernel_size=7, data_range=1.0, use_sample_covariance=True, axis=None):
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
    ssim = np.array(ssim).reshape(shape[:-3])
    return np.mean(ssim, axis=axis)


def expected_pixel_distribution_np(pix_distrib):
    pix_distrib = pix_distrib / np.sum(pix_distrib, axis=(-3, -2), keep_dims=True)
    height, width = pix_distrib.shape[-3:-1]
    xv, yv = np.meshgrid(np.arange(width), np.arange(height))
    return np.stack([np.sum(yv[:, :, None] * pix_distrib, axis=(-3, -2, -1)),
                     np.sum(xv[:, :, None] * pix_distrib, axis=(-3, -2, -1))], axis=-1)


def expected_pixel_distance_np(true_pix_distrib, pred_pix_distribs, axis=None):
    return np.linalg.norm(expected_pixel_distribution_np(true_pix_distrib) -
                          expected_pixel_distribution_np(pred_pix_distribs),
                          axis=axis)


def peak_signal_to_noise_ratio(true, pred):
    """
    Image quality metric based on maximal signal power vs. power of the noise.

    Args:
        true: the ground truth image.
        pred: the predicted image.

    Returns:
        peak signal to noise ratio (PSNR).
    """
    return 10.0 * tf.log(1.0 / mean_squared_error(true, pred)) / tf.log(10.0)


def mean_squared_error(true, pred):
    """
    L2 distance between tensors true and pred.

    Args:
        true: the ground truth image.
        pred: the predicted image.

    Returns:
        mean squared error between ground truth and predicted image.
    """
    return tf.reduce_sum(tf.square(true - pred)) / tf.to_float(tf.size(pred))


def structural_similarity(true, pred, k1=0.01, k2=0.03, kernel_size=7, data_range=1.0, use_sample_covariance=True):
    """
    Structural SIMilarity (SSIM) index between two images

    Args:
        true: The ground truth image. A 4-D tensor of shape
            `[batch, height, width, channels]`.
        pred: The predicted image. A 4-D tensor of shape
            `[batch, height, width, channels]`.

    Returns:
        The SSIM between ground truth and predicted image.

    Reference:
        https://github.com/scikit-image/scikit-image/blob/master/skimage/measure/_structural_similarity.py
    """
    if true.shape.ndims == 4 and pred.shape.ndims == 4:
        channels = pred.get_shape().as_list()[-1]
        c1 = (k1 * data_range) ** 2
        c2 = (k2 * data_range) ** 2
        # compute patches independently per channel as in the reference implementation
        patches_true = []
        patches_pred = []
        for true_single_channel, pred_single_channel in zip(tf.split(true, channels, axis=-1),
                                                            tf.split(pred, channels, axis=-1)):
            # use no padding (i.e. valid padding) so we don't compute values near the borders.
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
            NP = kernel_size ** (len(true.shape) - 2)  # substract batch and channel dimensions
            cov_norm = NP / (NP - 1)  # sample covariance
        else:
            cov_norm = 1.0  # population covariance to match Wang et. al. 2004
        var_true *= cov_norm
        var_pred *= cov_norm
        cov_true_pred *= cov_norm

        ssim = (2 * mean_true * mean_pred + c1) * (2 * cov_true_pred + c2)
        denom = (tf.square(mean_true) + tf.square(mean_pred) + c1) * (var_pred + var_true + c2)
        ssim /= denom
        ssim = tf.reduce_mean(ssim)
    else:
        kwargs = dict(k1=k1, k2=k2, kernel_size=kernel_size, data_range=data_range,
                      use_sample_covariance=use_sample_covariance)
        ssim = tf.map_fn(lambda args: structural_similarity(*args, **kwargs),
                         (true, pred), dtype=tf.float32, back_prop=False)
        ssim = tf.reduce_mean(ssim)
    return ssim


def normalize_tensor(tensor, eps=1e-10):
    norm_factor = tf.norm(tensor, axis=-1, keep_dims=True)
    return tensor / (norm_factor + eps)


def cosine_similarity(tensor0, tensor1):
    tensor0 = normalize_tensor(tensor0)
    tensor1 = normalize_tensor(tensor1)
    return tf.reduce_mean(tf.reduce_sum(tensor0 * tensor1, axis=-1))


def cosine_distance(tensor0, tensor1):
    """
    Equivalent to:
        tensor0 = normalize_tensor(tensor0)
        tensor1 = normalize_tensor(tensor1)
        return tf.reduce_mean(tf.reduce_sum(tf.square(tensor0 - tensor1), axis=-1)) / 2.0
    """
    return 1.0 - cosine_similarity(tensor0, tensor1)


def vgg_cosine_distance(image0, image1, model):
    if image0.shape.ndims == 5:
        image0 = flatten(image0, 0, 1)
    if image1.shape.ndims == 5:
        image1 = flatten(image1, 0, 1)
    image0 = image0 * 255.0
    image1 = image1 * 255.0
    image0 = tf.keras.applications.vgg16.preprocess_input(image0)
    image1 = tf.keras.applications.vgg16.preprocess_input(image1)
    features0 = model(image0)
    features1 = model(image1)
    cdist = 0.0
    for feature0, feature1 in zip(features0, features1):
        cdist += cosine_distance(feature0, feature1)
    return cdist


def normalize_tensor_np(tensor, eps=1e-10):
    norm_factor = np.linalg.norm(tensor, axis=-1, keep_dims=True)
    return tensor / (norm_factor + eps)


def cosine_similarity_np(tensor0, tensor1):
    tensor0 = normalize_tensor_np(tensor0)
    tensor1 = normalize_tensor_np(tensor1)
    return np.mean(np.sum(tensor0 * tensor1, axis=-1))


def cosine_distance_np(tensor0, tensor1):
    """
    Equivalent to:
        tensor0 = normalize_tensor_np(tensor0)
        tensor1 = normalize_tensor_np(tensor1)
        return np.mean(np.sum(np.square(tensor0 - tensor1), axis=-1)) / 2.0
    """
    return 1.0 - cosine_similarity_np(tensor0, tensor1)


def main():
    import numpy as np
    from skimage.measure import compare_ssim

    batch_size = 4
    image_shape = (64, 64, 3)
    true = np.random.random((batch_size,) + image_shape)
    pred = np.random.random((batch_size,) + image_shape)

    sess = tf.Session()
    ssim_tf = structural_similarity(tf.constant(true), tf.constant(pred))
    ssim_tf = sess.run(ssim_tf)
    ssim = np.mean([compare_ssim(true_y, pred_y, data_range=1.0, multichannel=True)
                    for true_y, pred_y in zip(true, pred)])
    print(ssim_tf, ssim)


if __name__ == '__main__':
    main()
