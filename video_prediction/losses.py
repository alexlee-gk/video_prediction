import tensorflow as tf

from video_prediction.ops import sigmoid_kl_with_logits


def l1_loss(pred, target):
    return tf.reduce_mean(tf.abs(target - pred))


def l2_loss(pred, target):
    return tf.reduce_mean(tf.square(target - pred))


def gan_loss(logits, labels, gan_loss_type):
    # use 1.0 (or 1.0 - discrim_label_smooth) for real data and 0.0 for fake data
    if gan_loss_type == 'GAN':
        # discrim_loss = tf.reduce_mean(-(tf.log(predict_real + EPS) + tf.log(1 - predict_fake + EPS)))
        # gen_loss = tf.reduce_mean(-tf.log(predict_fake + EPS))
        if labels in (0.0, 1.0):
            labels = tf.constant(labels, dtype=logits.dtype, shape=logits.get_shape())
            loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))
        else:
            loss = tf.reduce_mean(sigmoid_kl_with_logits(logits, labels))
    elif gan_loss_type == 'LSGAN':
        # discrim_loss = tf.reduce_mean((tf.square(predict_real - 1) + tf.square(predict_fake)))
        # gen_loss = tf.reduce_mean(tf.square(predict_fake - 1))
        loss = tf.reduce_mean(tf.square(logits - labels))
    else:
        raise ValueError('Unknown GAN loss type %s' % gan_loss_type)
    return loss


def kl_loss(mu, log_sigma_sq, clip=False):
    if clip:
        sigma_sq = tf.exp(tf.clip_by_value(log_sigma_sq, -10, 10))
    else:
        sigma_sq = tf.exp(log_sigma_sq)
    return -0.5 * tf.reduce_mean(tf.reduce_sum(1 + log_sigma_sq - tf.square(mu) - sigma_sq, axis=-1))
