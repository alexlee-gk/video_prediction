import functools
import itertools
import os
from collections import OrderedDict

import tensorflow as tf
from tensorflow.contrib.training import HParams
from tensorflow.python.util import nest

import video_prediction as vp
from video_prediction.functional_ops import foldl
from video_prediction.ops import flatten
from video_prediction.utils import tf_utils
from video_prediction.utils.tf_utils import compute_averaged_gradients, reduce_tensors, local_device_setter, \
    replace_read_ops, print_loss_info, transpose_batch_time, add_tensor_summaries, add_scalar_summaries, \
    add_plot_summaries, add_summaries
from . import vgg_network


class BaseVideoPredictionModel:
    def __init__(self, mode='train', hparams_dict=None, hparams=None,
                 num_gpus=None, eval_num_samples=100, eval_parallel_iterations=1):
        """
        Base video prediction model.

        Trainable and non-trainable video prediction models can be derived
        from this base class.

        Args:
            hparams_dict: a dict of `name=value` pairs, where `name` must be
                defined in `self.get_default_hparams()`.
            hparams: a string of comma separated list of `name=value` pairs,
                where `name` must be defined in `self.get_default_hparams()`.
                These values overrides any values in hparams_dict (if any).
        """
        self.mode = mode
        cuda_visible_devices = os.environ['CUDA_VISIBLE_DEVICES']
        if cuda_visible_devices == '':
            max_num_gpus = 0
        else:
            max_num_gpus = len(cuda_visible_devices.split(','))
        if num_gpus is None:
            num_gpus = max_num_gpus
        elif num_gpus > max_num_gpus:
            raise ValueError('num_gpus=%d is greater than the number of visible devices %d' % (num_gpus, max_num_gpus))
        self.num_gpus = num_gpus
        self.eval_num_samples = eval_num_samples
        self.eval_parallel_iterations = eval_parallel_iterations
        self.hparams = self.parse_hparams(hparams_dict, hparams)
        if not self.hparams.context_frames:
            raise ValueError('Invalid context_frames %r. It might have to be '
                             'specified.' % self.hparams.context_frames)
        if not self.hparams.sequence_length:
            raise ValueError('Invalid sequence_length %r. It might have to be '
                             'specified.' % self.hparams.sequence_length)

        # should be overriden by descendant class if the model is stochastic
        self.deterministic = True

        # member variables that should be set by `self.build_graph`
        self.inputs = None
        self.targets = None
        self.gen_images = None
        self.outputs = None
        self.metrics = None
        self.eval_outputs = None
        self.eval_metrics = None

    def get_default_hparams_dict(self):
        """
        The keys of this dict define valid hyperparameters for instances of
        this class. A class inheriting from this one should override this
        method if it has a different set of hyperparameters.

        Returns:
            A dict with the following hyperparameters.

            context_frames: the number of ground-truth frames to pass in at
                start. Must be specified during instantiation.
            sequence_length: the number of frames in the video sequence,
                including the context frames, so this model predicts
                `sequence_length - context_frames` future frames. Must be
                specified during instantiation.
            repeat: the number of repeat actions (if applicable).
        """
        hparams = dict(
            context_frames=0,
            sequence_length=0,
            repeat=1,
        )
        return hparams

    def get_default_hparams(self):
        return HParams(**self.get_default_hparams_dict())

    def parse_hparams(self, hparams_dict, hparams):
        parsed_hparams = self.get_default_hparams().override_from_dict(hparams_dict or {})
        if hparams:
            if not isinstance(hparams, (list, tuple)):
                hparams = [hparams]
            for hparam in hparams:
                parsed_hparams.parse(hparam)
        return parsed_hparams

    def build_graph(self, inputs, targets=None):
        self.inputs = inputs
        self.targets = targets

        # call it once here to create the variables
        with tf.variable_scope('vgg'):
            vgg_network.vgg16(tf.placeholder(tf.float32, shape=[None] * 4))

    def metrics_fn(self, inputs, outputs, targets):
        metrics = OrderedDict()
        target_images = targets
        gen_images = outputs['gen_images']
        metric_fns = [
            ('psnr', vp.metrics.peak_signal_to_noise_ratio),
            ('mse', vp.metrics.mean_squared_error),
            ('ssim', vp.metrics.structural_similarity),
            ('ssim_scikit', vp.metrics.structural_similarity_scikit),
            ('ssim_finn', vp.metrics.structural_similarity_finn),
            ('vgg_csim', vp.metrics.vgg_cosine_similarity),
            ('vgg_cdist', vp.metrics.vgg_cosine_distance),
        ]
        for metric_name, metric_fn in metric_fns:
            metrics[metric_name] = metric_fn(target_images, gen_images)
        return metrics

    def eval_outputs_and_metrics_fn(self, inputs, outputs, targets, num_samples=None, parallel_iterations=None):
        num_samples = num_samples or self.eval_num_samples
        parallel_iterations = parallel_iterations or self.eval_parallel_iterations
        eval_outputs = OrderedDict()
        eval_metrics = OrderedDict()
        metric_fns = [
            ('psnr', vp.metrics.peak_signal_to_noise_ratio),
            ('mse', vp.metrics.mean_squared_error),
            ('ssim', vp.metrics.structural_similarity),
            ('ssim_scikit', vp.metrics.structural_similarity_scikit),
            ('ssim_finn', vp.metrics.structural_similarity_finn),
            ('vgg_csim', vp.metrics.vgg_cosine_similarity),
            ('vgg_cdist', vp.metrics.vgg_cosine_distance),
        ]
        eval_outputs['eval_images'] = targets
        if self.deterministic:
            target_images = targets
            gen_images = outputs['gen_images']
            for metric_name, metric_fn in metric_fns:
                metric = metric_fn(target_images, gen_images, keep_axis=(0, 1))
                eval_metrics['eval_%s/min' % metric_name] = metric
                eval_metrics['eval_%s/avg' % metric_name] = metric
                eval_metrics['eval_%s/max' % metric_name] = metric
            eval_outputs['eval_gen_images'] = gen_images
        else:
            def where_axis1(cond, x, y):
                return transpose_batch_time(tf.where(cond, transpose_batch_time(x), transpose_batch_time(y)))

            with tf.variable_scope('vgg', reuse=tf.AUTO_REUSE):
                _, target_vgg_features = vp.metrics._with_flat_batch(vgg_network.vgg16)(targets)

            def sort_criterion(x):
                return tf.reduce_mean(x, axis=0)

            def accum_gen_images_and_metrics_fn(a, unused):
                with tf.variable_scope(self.generator_scope, reuse=True):
                    gen_images, _ = self.generator_fn(inputs)
                with tf.variable_scope('vgg', reuse=tf.AUTO_REUSE):
                    _, gen_vgg_features = vp.metrics._with_flat_batch(vgg_network.vgg16)(gen_images)
                for name, metric_fn in metric_fns:
                    if name in ('vgg_csim', 'vgg_cdist'):
                        metric_fn = {'vgg_csim': vp.metrics.cosine_similarity, 'vgg_cdist': vp.metrics.cosine_distance}[name]
                        metric = 0.0
                        for feature0, feature1 in zip(target_vgg_features, gen_vgg_features):
                            metric += metric_fn(feature0, feature1, keep_axis=(0, 1))
                        metric /= len(target_vgg_features)
                    else:
                        metric = metric_fn(targets, gen_images, keep_axis=(0, 1))  # time, batch_size
                    cond_min = tf.less(sort_criterion(metric), sort_criterion(a['eval_%s/min' % name]))
                    cond_max = tf.greater(sort_criterion(metric), sort_criterion(a['eval_%s/max' % name]))
                    a['eval_%s/min' % name] = where_axis1(cond_min, metric, a['eval_%s/min' % name])
                    a['eval_%s/sum' % name] = metric + a['eval_%s/sum' % name]
                    a['eval_%s/max' % name] = where_axis1(cond_max, metric, a['eval_%s/max' % name])
                    a['eval_gen_images_%s/min' % name] = where_axis1(cond_min, gen_images, a['eval_gen_images_%s/min' % name])
                    a['eval_gen_images_%s/sum' % name] = gen_images + a['eval_gen_images_%s/sum' % name]
                    a['eval_gen_images_%s/max' % name] = where_axis1(cond_max, gen_images, a['eval_gen_images_%s/max' % name])
                return a

            initializer = {}
            for name, _ in metric_fns:
                initializer['eval_gen_images_%s/min' % name] = tf.zeros_like(targets)
                initializer['eval_gen_images_%s/sum' % name] = tf.zeros_like(targets)
                initializer['eval_gen_images_%s/max' % name] = tf.zeros_like(targets)
                initializer['eval_%s/min' % name] = tf.fill(targets.shape[:2], float('inf'))
                initializer['eval_%s/sum' % name] = tf.zeros(targets.shape[:2])
                initializer['eval_%s/max' % name] = tf.fill(targets.shape[:2], float('-inf'))

            eval_outputs_and_metrics = foldl(accum_gen_images_and_metrics_fn, tf.zeros([num_samples, 0]),
                                             initializer=initializer, back_prop=False, parallel_iterations=parallel_iterations)

            for name, _ in metric_fns:
                eval_outputs['eval_gen_images_%s/min' % name] = eval_outputs_and_metrics['eval_gen_images_%s/min' % name]
                eval_outputs['eval_gen_images_%s/avg' % name] = eval_outputs_and_metrics['eval_gen_images_%s/sum' % name] / float(num_samples)
                eval_outputs['eval_gen_images_%s/max' % name] = eval_outputs_and_metrics['eval_gen_images_%s/max' % name]
                eval_metrics['eval_%s/min' % name] = eval_outputs_and_metrics['eval_%s/min' % name]
                eval_metrics['eval_%s/avg' % name] = eval_outputs_and_metrics['eval_%s/sum' % name] / float(num_samples)
                eval_metrics['eval_%s/max' % name] = eval_outputs_and_metrics['eval_%s/max' % name]
        return eval_outputs, eval_metrics

    def restore(self, sess, checkpoints):
        vgg_network.vgg_assign_from_values_fn()(sess)

        if checkpoints:
            # possibly restore from multiple checkpoints. useful if subset of weights
            # (e.g. generator or discriminator) are on different checkpoints.
            if not isinstance(checkpoints, (list, tuple)):
                checkpoints = [checkpoints]
            # automatically skip global_step if more than one checkpoint is provided
            skip_global_step = len(checkpoints) > 1
            savers = []
            for checkpoint in checkpoints:
                print("creating restore saver from checkpoint %s" % checkpoint)
                saver, _ = tf_utils.get_checkpoint_restore_saver(checkpoint, skip_global_step=skip_global_step)
                savers.append(saver)
            restore_op = [saver.saver_def.restore_op_name for saver in savers]
            sess.run(restore_op)


class VideoPredictionModel(BaseVideoPredictionModel):
    def __init__(self,
                 generator_fn,
                 discriminator_fn=None,
                 encoder_fn=None,
                 generator_scope='generator',
                 discriminator_scope='discriminator',
                 encoder_scope='encoder',
                 discriminator_encoder_scope='discriminator_encoder',
                 mode='train',
                 hparams_dict=None,
                 hparams=None,
                 **kwargs):
        """
        Trainable video prediction model with CPU and multi-GPU support.

        If num_gpus <= 1, the devices for the ops in `self.build_graph` are
        automatically chosen by TensorFlow (i.e. `tf.device` is not specified),
        otherwise they are explicitly chosen.

        Args:
            generator_fn: callable that takes in inputs (and optionally
                what's returned by encoder_fn) and returns generated images
                and a dict of tensors.
            discriminator_fn: callable that takes in fake/real data (and
                optionally conditioned on inputs) and returns logits and a
                dict of tensors.
            encoder_fn: callable that takes in inputs and returns a dict of
                tensors.
            hparams_dict: a dict of `name=value` pairs, where `name` must be
                defined in `self.get_default_hparams()`.
            hparams: a string of comma separated list of `name=value` pairs,
                where `name` must be defined in `self.get_default_hparams()`.
                These values overrides any values in hparams_dict (if any).
        """
        super(VideoPredictionModel, self).__init__(mode, hparams_dict, hparams, **kwargs)

        self.generator_fn = functools.partial(generator_fn, hparams=self.hparams)
        self.encoder_fn = functools.partial(encoder_fn, hparams=self.hparams) if encoder_fn else None
        self.discriminator_fn = functools.partial(discriminator_fn, hparams=self.hparams) if discriminator_fn else None

        self.generator_scope = generator_scope
        self.encoder_scope = encoder_scope
        self.discriminator_scope = discriminator_scope
        self.discriminator_encoder_scope = discriminator_encoder_scope

        if any(self.hparams.decay_steps):
            lr, end_lr = self.hparams.lr, self.hparams.end_lr
            start_step, end_step = self.hparams.decay_steps
            step = tf.clip_by_value(tf.train.get_or_create_global_step(), start_step, end_step)
            self.learning_rate = lr + (end_lr - lr) * tf.to_float(step - start_step) / tf.to_float(end_step - start_step)
        else:
            self.learning_rate = self.hparams.lr
        if mode == 'train':
            self.g_optimizer = tf.train.AdamOptimizer(self.learning_rate, self.hparams.beta1, self.hparams.beta2)
            self.d_optimizer = tf.train.AdamOptimizer(self.learning_rate, self.hparams.beta1, self.hparams.beta2)
        else:
            self.g_optimizer = None
            self.d_optimizer = None

        if self.hparams.kl_weight:
            if self.hparams.kl_anneal == 'none':
                self.kl_weight = tf.constant(self.hparams.kl_weight, tf.float32)
            elif self.hparams.kl_anneal == 'sigmoid':
                k = self.hparams.kl_anneal_k
                if k == -1.0:
                    raise ValueError('Invalid kl_anneal_k %d when kl_anneal is sigmoid.' % k)
                iter_num = tf.train.get_or_create_global_step()
                self.kl_weight = self.hparams.kl_weight / (1 + k * tf.exp(-tf.to_float(iter_num) / k))
            elif self.hparams.kl_anneal == 'linear':
                start_step, end_step = self.hparams.kl_anneal_steps
                step = tf.clip_by_value(tf.train.get_or_create_global_step(), start_step, end_step)
                self.kl_weight = self.hparams.kl_weight * tf.to_float(step - start_step) / tf.to_float(end_step - start_step)
            else:
                raise NotImplementedError
        else:
            self.kl_weight = None

        # member variables that should be set by `self.build_graph`
        # (in addition to the ones in the base class)
        self.gen_images_enc = None
        self.g_losses = None
        self.d_losses = None
        self.g_loss = None
        self.d_loss = None
        self.g_vars = None
        self.d_vars = None
        self.train_op = None

    def get_default_hparams_dict(self):
        """
        The keys of this dict define valid hyperparameters for instances of
        this class. A class inheriting from this one should override this
        method if it has a different set of hyperparameters.

        Returns:
            A dict with the following hyperparameters.

            batch_size: batch size for training.
            lr: learning rate. if decay steps is non-zero, this is the
                learning rate for steps <= decay_step.
            end_lr: learning rate for steps >= end_decay_step if decay_steps
                is non-zero, ignored otherwise.
            decay_steps: (decay_step, end_decay_step) tuple.
            max_steps: number of training steps.
            beta1: momentum term of Adam.
            beta2: momentum term of Adam.
            context_frames: the number of ground-truth frames to pass in at
                start. Must be specified during instantiation.
            sequence_length: the number of frames in the video sequence,
                including the context frames, so this model predicts
                `sequence_length - context_frames` future frames. Must be
                specified during instantiation.
        """
        default_hparams = super(VideoPredictionModel, self).get_default_hparams_dict()
        hparams = dict(
            batch_size=16,
            lr=0.001,
            end_lr=0.0,
            decay_steps=(200000, 300000),
            max_steps=300000,
            beta1=0.9,
            beta2=0.999,
            context_frames=0,
            sequence_length=0,
            clip_length=10,
            l1_weight=0.0,
            l2_weight=1.0,
            vgg_cdist_weight=0.0,
            feature_l2_weight=0.0,
            ae_l2_weight=0.0,
            state_weight=0.0,
            tv_weight=0.0,
            gan_weight=0.0,
            vae_gan_weight=0.0,
            tuple_gan_weight=0.0,
            tuple_vae_gan_weight=0.0,
            image_gan_weight=0.0,
            image_vae_gan_weight=0.0,
            video_gan_weight=0.0,
            video_vae_gan_weight=0.0,
            acvideo_gan_weight=0.0,
            acvideo_vae_gan_weight=0.0,
            image_sn_gan_weight=0.0,
            image_sn_vae_gan_weight=0.0,
            video_sn_gan_weight=0.0,
            video_sn_vae_gan_weight=0.0,
            gan_feature_l2_weight=0.0,
            gan_feature_cdist_weight=0.0,
            gan_loss_type='LSGAN',
            kl_weight=0.0,
            kl_anneal='linear',
            kl_anneal_k=-1.0,
            kl_anneal_steps=(50000, 100000),
            z_l1_weight=0.0,
        )
        return dict(itertools.chain(default_hparams.items(), hparams.items()))

    def tower_fn(self, inputs, targets=None):
        """
        This method doesn't have side-effects. `inputs`, `targets`, and
        `outputs` are batch-major but internal calculations use time-major
        tensors.
        """
        # batch-major to time-major
        inputs, targets = nest.map_structure(transpose_batch_time, (inputs, targets))

        with tf.variable_scope(self.generator_scope) as gen_scope:
            gen_images, gen_outputs = self.generator_fn(inputs)

        if self.encoder_fn:
            with tf.variable_scope(gen_scope):
                with tf.variable_scope(self.encoder_scope):
                    outputs_enc = self.encoder_fn(inputs)
            with tf.variable_scope(gen_scope, reuse=True):
                with tf.name_scope(self.encoder_scope):
                    gen_images_enc, gen_outputs_enc = self.generator_fn(inputs, outputs_enc=outputs_enc)
                    gen_outputs_enc = OrderedDict([(k + '_enc', v) for k, v in gen_outputs_enc.items()])
        else:
            outputs_enc = {}
            gen_images_enc = None
            gen_outputs_enc = {}

        if self.discriminator_fn and targets is not None:
            # TODO: make sure tuple_gan is not discriminating on context frames
            discrim_inputs = OrderedDict(list(inputs.items()) + list(gen_outputs.items()))
            with tf.name_scope("real"):
                with tf.variable_scope(self.discriminator_scope) as discrim_scope:
                    # pre-update discriminator tensors (i.e. before the discriminator weights have been updated)
                    _, discrim_outputs_real = self.discriminator_fn(targets, discrim_inputs)
                    discrim_outputs_real = OrderedDict([(k + '_real', v) for k, v in discrim_outputs_real.items()])
                with tf.variable_scope(discrim_scope, reuse=True):
                    # post-update discriminator tensors (i.e. after the discriminator weights have been updated)
                    _, discrim_outputs_real_post = self.discriminator_fn(targets, discrim_inputs)
                    discrim_outputs_real_post = OrderedDict([(k + '_real', v) for k, v in discrim_outputs_real_post.items()])
            with tf.name_scope("fake"):
                with tf.variable_scope(discrim_scope, reuse=True):
                    # pre-update discriminator tensors (i.e. before the discriminator weights have been updated)
                    _, discrim_outputs_fake = self.discriminator_fn(gen_images, discrim_inputs)
                    discrim_outputs_fake = OrderedDict([(k + '_fake', v) for k, v in discrim_outputs_fake.items()])
                    # post-update discriminator tensors (i.e. after the discriminator weights have been updated)
                    _, discrim_outputs_fake_post = self.discriminator_fn(gen_images, discrim_inputs)
                    discrim_outputs_fake_post = OrderedDict([(k + '_fake', v) for k, v in discrim_outputs_fake_post.items()])
        else:
            discrim_outputs_real = {}
            discrim_outputs_real_post = {}
            discrim_outputs_fake = {}
            discrim_outputs_fake_post = {}

        if self.discriminator_fn and self.encoder_fn and targets is not None:
            discrim_inputs_enc = OrderedDict(list(inputs.items()) + list(gen_outputs_enc.items()))
            same_discriminator = self.discriminator_scope == self.discriminator_encoder_scope
            with tf.name_scope("real"), tf.name_scope(self.encoder_scope):
                with tf.variable_scope(self.discriminator_encoder_scope, reuse=same_discriminator) as discrim_enc_scope:
                    # pre-update discriminator tensors (i.e. before the discriminator weights have been updated)
                    _, discrim_outputs_enc_real = self.discriminator_fn(targets, discrim_inputs_enc)
                    discrim_outputs_enc_real = OrderedDict([(k + '_enc_real', v) for k, v in discrim_outputs_enc_real.items()])
                with tf.variable_scope(discrim_enc_scope, reuse=True):
                    # post-update discriminator tensors (i.e. after the discriminator weights have been updated)
                    _, discrim_outputs_enc_real_post = self.discriminator_fn(targets, discrim_inputs_enc)
                    discrim_outputs_enc_real_post = OrderedDict([(k + '_enc_real', v) for k, v in discrim_outputs_enc_real_post.items()])
            with tf.name_scope("fake"), tf.name_scope(self.encoder_scope):
                with tf.variable_scope(discrim_enc_scope, reuse=True):
                    # pre-update discriminator tensors (i.e. before the discriminator weights have been updated)
                    _, discrim_outputs_enc_fake = self.discriminator_fn(gen_images_enc, discrim_inputs_enc)
                    discrim_outputs_enc_fake = OrderedDict([(k + '_enc_fake', v) for k, v in discrim_outputs_enc_fake.items()])
                    # post-update discriminator tensors (i.e. after the discriminator weights have been updated)
                    _, discrim_outputs_enc_fake_post = self.discriminator_fn(gen_images_enc, discrim_inputs_enc)
                    discrim_outputs_enc_fake_post = OrderedDict([(k + '_enc_fake', v) for k, v in discrim_outputs_enc_fake_post.items()])
        else:
            discrim_outputs_enc_real = {}
            discrim_outputs_enc_real_post = {}
            discrim_outputs_enc_fake = {}
            discrim_outputs_enc_fake_post = {}

        outputs = [gen_outputs, outputs_enc, gen_outputs_enc,
                   discrim_outputs_real, discrim_outputs_fake,
                   discrim_outputs_enc_real, discrim_outputs_enc_fake]
        total_num_outputs = sum([len(output) for output in outputs])
        outputs = OrderedDict(itertools.chain(*[output.items() for output in outputs]))
        assert len(outputs) == total_num_outputs  # ensure no output is lost because of repeated keys

        if isinstance(self.learning_rate, tf.Tensor):
            outputs['learning_rate'] = self.learning_rate
        if isinstance(self.kl_weight, tf.Tensor):
            outputs['kl_weight'] = self.kl_weight

        if targets is not None:
            if self.mode != 'test':
                with tf.name_scope("discriminator_loss"):
                    d_losses = self.discriminator_loss_fn(inputs, outputs, targets)
                    print_loss_info(d_losses, inputs, outputs, targets)
                with tf.name_scope("generator_loss"):
                    g_losses = self.generator_loss_fn(inputs, outputs, targets)
                    print_loss_info(g_losses, inputs, outputs, targets)
                    if discrim_outputs_real_post or discrim_outputs_fake_post or \
                            discrim_outputs_enc_real_post or discrim_outputs_enc_fake_post:
                        outputs_post = OrderedDict(itertools.chain(outputs.items(),
                                                                   discrim_outputs_real_post.items(),
                                                                   discrim_outputs_fake_post.items(),
                                                                   discrim_outputs_enc_real_post.items(),
                                                                   discrim_outputs_enc_fake_post.items()))
                        g_losses_post = self.generator_loss_fn(inputs, outputs_post, targets)
                    else:
                        g_losses_post = g_losses
            else:
                d_losses = {}
                g_losses = {}
                g_losses_post = {}
            with tf.name_scope("metrics"):
                metrics = self.metrics_fn(inputs, outputs, targets)
            with tf.name_scope("eval_outputs_and_metrics"):
                eval_outputs, eval_metrics = self.eval_outputs_and_metrics_fn(inputs, outputs, targets)
        else:
            d_losses = {}
            g_losses = {}
            g_losses_post = {}
            metrics = {}
            eval_outputs = {}
            eval_metrics = {}

        # time-major to batch-major
        outputs_tuple = (gen_images, gen_images_enc, outputs, eval_outputs)
        outputs_tuple = nest.map_structure(transpose_batch_time, outputs_tuple)
        losses_tuple = (d_losses, g_losses, g_losses_post)
        losses_tuple = nest.map_structure(tf.convert_to_tensor, losses_tuple)
        metrics_tuple = (metrics, eval_metrics)
        metrics_tuple = nest.map_structure(transpose_batch_time, metrics_tuple)
        return outputs_tuple, losses_tuple, metrics_tuple

    def build_graph(self, inputs, targets=None):
        BaseVideoPredictionModel.build_graph(self, inputs, targets=targets)

        global_step = tf.train.get_or_create_global_step()

        if self.num_gpus <= 1:  # cpu or 1 gpu
            outputs_tuple, losses_tuple, metrics_tuple = self.tower_fn(self.inputs, self.targets)
            self.gen_images, self.gen_images_enc, self.outputs, self.eval_outputs = outputs_tuple
            self.d_losses, self.g_losses, g_losses_post = losses_tuple
            self.metrics, self.eval_metrics = metrics_tuple
            self.d_loss = sum(loss * weight for loss, weight in self.d_losses.values())
            self.g_loss = sum(loss * weight for loss, weight in self.g_losses.values())
            g_loss_post = sum(loss * weight for loss, weight in g_losses_post.values())

            d_vars = tf.trainable_variables(self.discriminator_scope)
            de_vars = tf.trainable_variables(self.discriminator_encoder_scope)
            self.d_vars = d_vars + [de_var for de_var in de_vars if de_var not in d_vars]
            self.g_vars = tf.trainable_variables(self.generator_scope)

            if self.mode == 'train' and (self.d_losses or self.g_losses):
                with tf.name_scope('optimize'):
                    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                        if self.d_losses:
                            d_gradvars = self.d_optimizer.compute_gradients(self.d_loss, var_list=self.d_vars)
                            d_train_op = self.d_optimizer.apply_gradients(d_gradvars)
                        else:
                            d_train_op = tf.no_op()
                    with tf.control_dependencies([d_train_op]):
                        if g_losses_post:
                            replace_read_ops(g_loss_post, self.d_vars)
                            g_gradvars = self.g_optimizer.compute_gradients(g_loss_post, var_list=self.g_vars)
                            g_train_op = self.g_optimizer.apply_gradients(
                                g_gradvars, global_step=global_step)  # also increments global_step
                        else:
                            g_train_op = tf.assign_add(global_step, 1)
                self.train_op = g_train_op
            else:
                self.train_op = None
        else:
            tower_inputs = [OrderedDict() for _ in range(self.num_gpus)]
            for name, input in self.inputs.items():
                input_splits = tf.split(input, self.num_gpus)  # assumes batch_size is divisible by num_gpus
                for i in range(self.num_gpus):
                    tower_inputs[i][name] = input_splits[i]
            if targets is not None:
                tower_targets = tf.split(targets, self.num_gpus)
            else:
                tower_targets = [None] * self.num_gpus

            tower_outputs_tuple = []
            tower_d_losses = []
            tower_g_losses = []
            tower_g_losses_post = []
            tower_metrics_tuple = []
            tower_d_loss = []
            tower_g_loss = []
            tower_g_loss_post = []
            for i in range(self.num_gpus):
                worker_device = '/{}:{}'.format('gpu', i)
                device_setter = local_device_setter(worker_device=worker_device)
                with tf.variable_scope('', reuse=bool(i > 0)):
                    with tf.name_scope('tower_%d' % i):
                        with tf.device(device_setter):
                            outputs_tuple, losses_tuple, metrics_tuple = self.tower_fn(tower_inputs[i],
                                                                                       tower_targets[i])
                            tower_outputs_tuple.append(outputs_tuple)
                            d_losses, g_losses, g_losses_post = losses_tuple
                            tower_d_losses.append(d_losses)
                            tower_g_losses.append(g_losses)
                            tower_g_losses_post.append(g_losses_post)
                            tower_metrics_tuple.append(metrics_tuple)
                            d_loss = sum(loss * weight for loss, weight in d_losses.values())
                            g_loss = sum(loss * weight for loss, weight in g_losses.values())
                            g_loss_post = sum(loss * weight for loss, weight in g_losses_post.values())
                            tower_d_loss.append(d_loss)
                            tower_g_loss.append(g_loss)
                            tower_g_loss_post.append(g_loss_post)

            d_vars = tf.trainable_variables(self.discriminator_scope)
            de_vars = tf.trainable_variables(self.discriminator_encoder_scope)
            self.d_vars = d_vars + [de_var for de_var in de_vars if de_var not in d_vars]
            self.g_vars = tf.trainable_variables(self.generator_scope)

            if self.mode == 'train' and (any(tower_d_losses) or any(tower_g_losses)):
                with tf.name_scope('optimize'):
                    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                        if any(tower_d_losses):
                            d_gradvars = compute_averaged_gradients(self.d_optimizer, tower_d_loss,
                                                                    var_list=self.d_vars)
                            d_train_op = self.d_optimizer.apply_gradients(d_gradvars)
                        else:
                            d_train_op = tf.no_op()
                    with tf.control_dependencies([d_train_op]):
                        if any(tower_g_losses_post):
                            replace_read_ops(tower_g_loss_post, self.d_vars)
                            g_gradvars = compute_averaged_gradients(self.g_optimizer, tower_g_loss_post,
                                                                    var_list=self.g_vars)
                            g_train_op = self.g_optimizer.apply_gradients(
                                g_gradvars, global_step=global_step)  # also increments global_step
                        else:
                            g_train_op = tf.assign_add(global_step, 1)
                self.train_op = g_train_op
            else:
                self.train_op = None

            # Device that runs the ops to apply global gradient updates.
            consolidation_device = '/cpu:0'
            with tf.device(consolidation_device):
                self.gen_images, self.gen_images_enc, self.outputs, self.eval_outputs = reduce_tensors(
                    tower_outputs_tuple)
                self.d_losses = reduce_tensors(tower_d_losses, shallow=True)
                self.g_losses = reduce_tensors(tower_g_losses, shallow=True)
                self.metrics, self.eval_metrics = reduce_tensors(tower_metrics_tuple)
                self.d_loss = reduce_tensors(tower_d_loss)
                self.g_loss = reduce_tensors(tower_g_loss)

        add_summaries({name: tensor for name, tensor in self.inputs.items()
                       if tensor.shape.ndims == 4 or (tensor.shape.ndims > 4 and
                                                      tensor.shape[4].value in (1, 3))})
        if self.targets is not None:
            add_summaries({'targets': self.targets})
        add_summaries({name: tensor for name, tensor in self.outputs.items()
                       if tensor.shape.ndims == 4 or (tensor.shape.ndims > 4 and
                                                      tensor.shape[4].value in (1, 3))})
        add_scalar_summaries({name: tensor for name, tensor in self.outputs.items() if tensor.shape.ndims == 0})
        add_scalar_summaries(self.d_losses)
        add_scalar_summaries(self.g_losses)
        add_scalar_summaries(self.metrics)
        add_tensor_summaries(self.eval_outputs, collections=[tf_utils.EVAL_SUMMARIES, tf_utils.IMAGE_SUMMARIES])
        add_plot_summaries({name: tf.reduce_mean(metric, axis=0) for name, metric in self.eval_metrics.items()},
                           x_offset=self.hparams.context_frames + 1, collections=[tf_utils.EVAL_SUMMARIES])

    def generator_loss_fn(self, inputs, outputs, targets):
        hparams = self.hparams
        gen_losses = OrderedDict()
        if hparams.l1_weight or hparams.l2_weight or hparams.vgg_cdist_weight:
            gen_images = outputs.get('gen_images_enc', outputs['gen_images'])
            target_images = targets
        if hparams.l1_weight:
            gen_l1_loss = vp.losses.l1_loss(gen_images, target_images)
            gen_losses["gen_l1_loss"] = (gen_l1_loss, hparams.l1_weight)
        if hparams.l2_weight:
            gen_l2_loss = vp.losses.l2_loss(gen_images, target_images)
            gen_losses["gen_l2_loss"] = (gen_l2_loss, hparams.l2_weight)
        if hparams.vgg_cdist_weight:
            gen_vgg_cdist_loss = vp.metrics.vgg_cosine_distance(gen_images, target_images)
            gen_losses['gen_vgg_cdist_loss'] = (gen_vgg_cdist_loss, hparams.vgg_cdist_weight)
        if hparams.feature_l2_weight:
            gen_features = outputs.get('gen_features_enc', outputs['gen_features'])
            target_features = outputs['features'][hparams.context_frames:]
            gen_feature_l2_loss = vp.losses.l2_loss(gen_features, target_features)
            gen_losses["gen_feature_l2_loss"] = (gen_feature_l2_loss, hparams.feature_l2_weight)
        if hparams.ae_l2_weight:
            gen_images_dec = outputs.get('gen_images_dec_enc', outputs['gen_images_dec'])  # they both should be the same
            target_images = inputs['images']
            gen_ae_l2_loss = vp.losses.l2_loss(gen_images_dec, target_images)
            gen_losses["gen_ae_l2_loss"] = (gen_ae_l2_loss, hparams.ae_l2_weight)
        if hparams.state_weight:
            gen_states = outputs.get('gen_states_enc', outputs['gen_states'])
            target_states = inputs['states'][hparams.context_frames:]
            gen_state_loss = vp.losses.l2_loss(gen_states, target_states)
            gen_losses["gen_state_loss"] = (gen_state_loss, hparams.state_weight)
        if hparams.tv_weight:
            gen_flows = outputs.get('gen_flows_enc', outputs['gen_flows'])
            gen_flows_reshaped = flatten(flatten(gen_flows, 0, 1), -2)
            gen_tv_loss = tf.reduce_mean(tf.image.total_variation(gen_flows_reshaped))
            gen_losses['gen_tv_loss'] = (gen_tv_loss, hparams.tv_weight)
        gan_weights = {'': hparams.gan_weight,
                       '_tuple': hparams.tuple_gan_weight,
                       '_image': hparams.image_gan_weight,
                       '_video': hparams.video_gan_weight,
                       '_acvideo': hparams.acvideo_gan_weight,
                       '_image_sn': hparams.image_sn_gan_weight,
                       '_video_sn': hparams.video_sn_gan_weight}
        for infix, gan_weight in gan_weights.items():
            if gan_weight:
                gen_gan_loss = vp.losses.gan_loss(outputs['discrim%s_logits_fake' % infix], 1.0, hparams.gan_loss_type)
                gen_losses["gen%s_gan_loss" % infix] = (gen_gan_loss, gan_weight)
            if gan_weight and (hparams.gan_feature_l2_weight or hparams.gan_feature_cdist_weight):
                i_feature = 0
                discrim_features_fake = []
                discrim_features_real = []
                while True:
                    discrim_feature_fake = outputs.get('discrim%s_feature%d_fake' % (infix, i_feature))
                    discrim_feature_real = outputs.get('discrim%s_feature%d_real' % (infix, i_feature))
                    if discrim_feature_fake is None or discrim_feature_real is None:
                        break
                    discrim_features_fake.append(discrim_feature_fake)
                    discrim_features_real.append(discrim_feature_real)
                    i_feature += 1
                if hparams.gan_feature_l2_weight:
                    gen_gan_feature_l2_loss = sum([vp.losses.l2_loss(discrim_feature_fake, discrim_feature_real)
                                                   for discrim_feature_fake, discrim_feature_real in zip(discrim_features_fake, discrim_features_real)])
                    gen_losses["gen%s_gan_feature_l2_loss" % infix] = (gen_gan_feature_l2_loss, hparams.gan_feature_l2_weight)
                if hparams.gan_feature_cdist_weight:
                    gen_gan_feature_cdist_loss = sum([vp.metrics.cosine_distance(discrim_feature_fake, discrim_feature_real)
                                                      for discrim_feature_fake, discrim_feature_real in zip(discrim_features_fake, discrim_features_real)])
                    gen_losses["gen%s_gan_feature_cdist_loss" % infix] = (gen_gan_feature_cdist_loss, hparams.gan_feature_cdist_weight)
        vae_gan_weights = {'': hparams.vae_gan_weight,
                           '_tuple': hparams.tuple_vae_gan_weight,
                           '_image': hparams.image_vae_gan_weight,
                           '_video': hparams.video_vae_gan_weight,
                           '_acvideo': hparams.acvideo_vae_gan_weight,
                           '_image_sn': hparams.image_sn_vae_gan_weight,
                           '_video_sn': hparams.video_sn_vae_gan_weight}
        for infix, vae_gan_weight in vae_gan_weights.items():
            if vae_gan_weight:
                gen_vae_gan_loss = vp.losses.gan_loss(outputs['discrim%s_logits_enc_fake' % infix], 1.0, hparams.gan_loss_type)
                gen_losses["gen%s_vae_gan_loss" % infix] = (gen_vae_gan_loss, vae_gan_weight)
            if vae_gan_weight and (hparams.gan_feature_l2_weight or hparams.gan_feature_cdist_weight):
                i_feature = 0
                discrim_features_enc_fake = []
                discrim_features_enc_real = []
                while True:
                    discrim_feature_enc_fake = outputs.get('discrim%s_feature%d_enc_fake' % (infix, i_feature))
                    discrim_feature_enc_real = outputs.get('discrim%s_feature%d_enc_real' % (infix, i_feature))
                    if discrim_feature_enc_fake is None or discrim_feature_enc_real is None:
                        break
                    discrim_features_enc_fake.append(discrim_feature_enc_fake)
                    discrim_features_enc_real.append(discrim_feature_enc_real)
                    i_feature += 1
                if hparams.gan_feature_l2_weight:
                    gen_vae_gan_feature_l2_loss = sum([vp.losses.l2_loss(discrim_feature_enc_fake, discrim_feature_enc_real)
                                                       for discrim_feature_enc_fake, discrim_feature_enc_real in zip(discrim_features_enc_fake, discrim_features_enc_real)])
                    gen_losses["gen%s_vae_gan_feature_l2_loss" % infix] = (gen_vae_gan_feature_l2_loss, hparams.gan_feature_l2_weight)
                if hparams.gan_feature_cdist_weight:
                    gen_vae_gan_feature_cdist_loss = sum([vp.metrics.cosine_distance(discrim_feature_enc_fake, discrim_feature_enc_real)
                                                          for discrim_feature_enc_fake, discrim_feature_enc_real in zip(discrim_features_enc_fake, discrim_features_enc_real)])
                    gen_losses["gen%s_vae_gan_feature_cdist_loss" % infix] = (gen_vae_gan_feature_cdist_loss, hparams.gan_feature_cdist_weight)
        if hparams.kl_weight:
            gen_kl_loss = vp.losses.kl_loss(outputs['enc_zs_mu'], outputs['enc_zs_log_sigma_sq'])
            gen_losses["gen_kl_loss"] = (gen_kl_loss, self.kl_weight)  # possibly annealed kl_weight
        if hparams.z_l1_weight:
            gen_z_l1_loss = vp.losses.l1_loss(outputs['gen_enc_zs_mu'], outputs['gen_zs_random'])
            gen_losses["gen_z_l1_loss"] = (gen_z_l1_loss, hparams.z_l1_weight)
        return gen_losses

    def discriminator_loss_fn(self, inputs, outputs, targets):
        hparams = self.hparams
        discrim_losses = OrderedDict()
        gan_weights = {'': hparams.gan_weight,
                       '_tuple': hparams.tuple_gan_weight,
                       '_image': hparams.image_gan_weight,
                       '_video': hparams.video_gan_weight,
                       '_acvideo': hparams.acvideo_gan_weight,
                       '_image_sn': hparams.image_sn_gan_weight,
                       '_video_sn': hparams.video_sn_gan_weight}
        for infix, gan_weight in gan_weights.items():
            if gan_weight:
                discrim_gan_loss_real = vp.losses.gan_loss(outputs['discrim%s_logits_real' % infix], 1.0, hparams.gan_loss_type)
                discrim_gan_loss_fake = vp.losses.gan_loss(outputs['discrim%s_logits_fake' % infix], 0.0, hparams.gan_loss_type)
                discrim_gan_loss = discrim_gan_loss_real + discrim_gan_loss_fake
                discrim_losses["discrim%s_gan_loss" % infix] = (discrim_gan_loss, gan_weight)
        vae_gan_weights = {'': hparams.vae_gan_weight,
                           '_tuple': hparams.tuple_vae_gan_weight,
                           '_image': hparams.image_vae_gan_weight,
                           '_video': hparams.video_vae_gan_weight,
                           '_acvideo': hparams.acvideo_vae_gan_weight,
                           '_image_sn': hparams.image_sn_vae_gan_weight,
                           '_video_sn': hparams.video_sn_vae_gan_weight}
        for infix, vae_gan_weight in vae_gan_weights.items():
            if vae_gan_weight:
                discrim_vae_gan_loss_real = vp.losses.gan_loss(outputs['discrim%s_logits_enc_real' % infix], 1.0, hparams.gan_loss_type)
                discrim_vae_gan_loss_fake = vp.losses.gan_loss(outputs['discrim%s_logits_enc_fake' % infix], 0.0, hparams.gan_loss_type)
                discrim_vae_gan_loss = discrim_vae_gan_loss_real + discrim_vae_gan_loss_fake
                discrim_losses["discrim%s_vae_gan_loss" % infix] = (discrim_vae_gan_loss, vae_gan_weight)
        return discrim_losses
