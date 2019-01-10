import functools
import itertools
import os
import re
from collections import OrderedDict

import numpy as np
import tensorflow as tf
from tensorflow.contrib.training import HParams
from tensorflow.python.util import nest

import video_prediction as vp
from video_prediction.utils import tf_utils
from video_prediction.utils.tf_utils import compute_averaged_gradients, reduce_tensors, local_device_setter, \
    replace_read_ops, print_loss_info, transpose_batch_time, add_gif_summaries, add_scalar_summaries, \
    add_plot_and_scalar_summaries, add_summaries


class BaseVideoPredictionModel(object):
    def __init__(self, mode='train', hparams_dict=None, hparams=None,
                 num_gpus=None, eval_num_samples=100,
                 eval_num_samples_for_diversity=10, eval_parallel_iterations=1):
        """
        Base video prediction model.

        Trainable and non-trainable video prediction models can be derived
        from this base class.

        Args:
            mode: `'train'` or `'test'`.
            hparams_dict: a dict of `name=value` pairs, where `name` must be
                defined in `self.get_default_hparams()`.
            hparams: a string of comma separated list of `name=value` pairs,
                where `name` must be defined in `self.get_default_hparams()`.
                These values overrides any values in hparams_dict (if any).
        """
        if mode not in ('train', 'test'):
            raise ValueError('mode must be train or test, but %s given' % mode)
        self.mode = mode
        cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
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
        self.eval_num_samples_for_diversity = eval_num_samples_for_diversity
        self.eval_parallel_iterations = eval_parallel_iterations
        self.hparams = self.parse_hparams(hparams_dict, hparams)
        if self.hparams.context_frames == -1:
            raise ValueError('Invalid context_frames %r. It might have to be '
                             'specified.' % self.hparams.context_frames)
        if self.hparams.sequence_length == -1:
            raise ValueError('Invalid sequence_length %r. It might have to be '
                             'specified.' % self.hparams.sequence_length)

        # should be overriden by descendant class if the model is stochastic
        self.deterministic = True

        # member variables that should be set by `self.build_graph`
        self.inputs = None
        self.gen_images = None
        self.outputs = None
        self.metrics = None
        self.eval_outputs = None
        self.eval_metrics = None
        self.accum_eval_metrics = None
        self.saveable_variables = None
        self.post_init_ops = None

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
            context_frames=-1,
            sequence_length=-1,
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

    def build_graph(self, inputs):
        self.inputs = inputs

    def metrics_fn(self, inputs, outputs):
        metrics = OrderedDict()
        sequence_length = tf.shape(inputs['images'])[0]
        context_frames = self.hparams.context_frames
        future_length = sequence_length - context_frames
        # target_images and pred_images include only the future frames
        target_images = inputs['images'][-future_length:]
        pred_images = outputs['gen_images'][-future_length:]
        metric_fns = [
            ('psnr', vp.metrics.psnr),
            ('mse', vp.metrics.mse),
            ('ssim', vp.metrics.ssim),
            ('lpips', vp.metrics.lpips),
        ]
        for metric_name, metric_fn in metric_fns:
            metrics[metric_name] = tf.reduce_mean(metric_fn(target_images, pred_images))
        return metrics

    def eval_outputs_and_metrics_fn(self, inputs, outputs, num_samples=None,
                                    num_samples_for_diversity=None, parallel_iterations=None):
        num_samples = num_samples or self.eval_num_samples
        num_samples_for_diversity = num_samples_for_diversity or self.eval_num_samples_for_diversity
        parallel_iterations = parallel_iterations or self.eval_parallel_iterations

        sequence_length, batch_size = inputs['images'].shape[:2].as_list()
        if batch_size is None:
            batch_size = tf.shape(inputs['images'])[1]
        if sequence_length is None:
            sequence_length = tf.shape(inputs['images'])[0]
        context_frames = self.hparams.context_frames
        future_length = sequence_length - context_frames
        # the outputs include all the frames, whereas the metrics include only the future frames
        eval_outputs = OrderedDict()
        eval_metrics = OrderedDict()
        metric_fns = [
            ('psnr', vp.metrics.psnr),
            ('mse', vp.metrics.mse),
            ('ssim', vp.metrics.ssim),
            ('lpips', vp.metrics.lpips),
        ]
        # images and gen_images include all the frames
        images = inputs['images']
        gen_images = outputs['gen_images']
        # target_images and pred_images include only the future frames
        target_images = inputs['images'][-future_length:]
        pred_images = outputs['gen_images'][-future_length:]
        # ground truth is the same for deterministic and stochastic models
        eval_outputs['eval_images'] = images
        if self.deterministic:
            for metric_name, metric_fn in metric_fns:
                metric = metric_fn(target_images, pred_images)
                eval_metrics['eval_%s/min' % metric_name] = metric
                eval_metrics['eval_%s/avg' % metric_name] = metric
                eval_metrics['eval_%s/max' % metric_name] = metric
            eval_outputs['eval_gen_images'] = gen_images
        else:
            def where_axis1(cond, x, y):
                return transpose_batch_time(tf.where(cond, transpose_batch_time(x), transpose_batch_time(y)))

            def sort_criterion(x):
                return tf.reduce_mean(x, axis=0)

            def accum_gen_images_and_metrics_fn(a, unused):
                with tf.variable_scope(self.generator_scope, reuse=True):
                    outputs_sample = self.generator_fn(inputs)
                    gen_images_sample = outputs_sample['gen_images']
                    pred_images_sample = gen_images_sample[-future_length:]
                    # set the posisbly static shape since it might not have been inferred correctly
                    pred_images_sample = tf.reshape(pred_images_sample, tf.shape(a['eval_pred_images_last']))
                for name, metric_fn in metric_fns:
                    metric = metric_fn(target_images, pred_images_sample)  # time, batch_size
                    cond_min = tf.less(sort_criterion(metric), sort_criterion(a['eval_%s/min' % name]))
                    cond_max = tf.greater(sort_criterion(metric), sort_criterion(a['eval_%s/max' % name]))
                    a['eval_%s/min' % name] = where_axis1(cond_min, metric, a['eval_%s/min' % name])
                    a['eval_%s/sum' % name] = metric + a['eval_%s/sum' % name]
                    a['eval_%s/max' % name] = where_axis1(cond_max, metric, a['eval_%s/max' % name])
                    a['eval_gen_images_%s/min' % name] = where_axis1(cond_min, gen_images_sample, a['eval_gen_images_%s/min' % name])
                    a['eval_gen_images_%s/sum' % name] = gen_images_sample + a['eval_gen_images_%s/sum' % name]
                    a['eval_gen_images_%s/max' % name] = where_axis1(cond_max, gen_images_sample, a['eval_gen_images_%s/max' % name])

                a['eval_diversity'] = tf.cond(
                    tf.logical_and(tf.less(0, a['eval_sample_ind']),
                                   tf.less_equal(a['eval_sample_ind'], num_samples_for_diversity)),
                    lambda: -vp.metrics.lpips(a['eval_pred_images_last'], pred_images_sample) + a['eval_diversity'],
                    lambda: a['eval_diversity'])
                a['eval_sample_ind'] = 1 + a['eval_sample_ind']
                a['eval_pred_images_last'] = pred_images_sample
                return a

            initializer = {}
            for name, _ in metric_fns:
                initializer['eval_gen_images_%s/min' % name] = tf.zeros_like(gen_images)
                initializer['eval_gen_images_%s/sum' % name] = tf.zeros_like(gen_images)
                initializer['eval_gen_images_%s/max' % name] = tf.zeros_like(gen_images)
                initializer['eval_%s/min' % name] = tf.fill([future_length, batch_size], float('inf'))
                initializer['eval_%s/sum' % name] = tf.zeros([future_length, batch_size])
                initializer['eval_%s/max' % name] = tf.fill([future_length, batch_size], float('-inf'))
            initializer['eval_diversity'] = tf.zeros([future_length, batch_size])
            initializer['eval_sample_ind'] = tf.zeros((), dtype=tf.int32)
            initializer['eval_pred_images_last'] = tf.zeros_like(pred_images)

            eval_outputs_and_metrics = tf.foldl(
                accum_gen_images_and_metrics_fn, tf.zeros([num_samples, 0]), initializer=initializer, back_prop=False,
                parallel_iterations=parallel_iterations)

            for name, _ in metric_fns:
                eval_outputs['eval_gen_images_%s/min' % name] = eval_outputs_and_metrics['eval_gen_images_%s/min' % name]
                eval_outputs['eval_gen_images_%s/avg' % name] = eval_outputs_and_metrics['eval_gen_images_%s/sum' % name] / float(num_samples)
                eval_outputs['eval_gen_images_%s/max' % name] = eval_outputs_and_metrics['eval_gen_images_%s/max' % name]
                eval_metrics['eval_%s/min' % name] = eval_outputs_and_metrics['eval_%s/min' % name]
                eval_metrics['eval_%s/avg' % name] = eval_outputs_and_metrics['eval_%s/sum' % name] / float(num_samples)
                eval_metrics['eval_%s/max' % name] = eval_outputs_and_metrics['eval_%s/max' % name]
            eval_metrics['eval_diversity'] = eval_outputs_and_metrics['eval_diversity'] / float(num_samples_for_diversity)
        return eval_outputs, eval_metrics

    def restore(self, sess, checkpoints, restore_to_checkpoint_mapping=None):
        if checkpoints:
            var_list = self.saveable_variables
            # possibly restore from multiple checkpoints. useful if subset of weights
            # (e.g. generator or discriminator) are on different checkpoints.
            if not isinstance(checkpoints, (list, tuple)):
                checkpoints = [checkpoints]
            # automatically skip global_step if more than one checkpoint is provided
            skip_global_step = len(checkpoints) > 1
            savers = []
            for checkpoint in checkpoints:
                print("creating restore saver from checkpoint %s" % checkpoint)
                saver, _ = tf_utils.get_checkpoint_restore_saver(
                    checkpoint, var_list, skip_global_step=skip_global_step,
                    restore_to_checkpoint_mapping=restore_to_checkpoint_mapping)
                savers.append(saver)
            restore_op = [saver.saver_def.restore_op_name for saver in savers]
            sess.run(restore_op)


class VideoPredictionModel(BaseVideoPredictionModel):
    def __init__(self,
                 generator_fn,
                 discriminator_fn=None,
                 generator_scope='generator',
                 discriminator_scope='discriminator',
                 aggregate_nccl=False,
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
            generator_fn: callable that takes in inputs and returns a dict of
                tensors.
            discriminator_fn: callable that takes in fake/real data (and
                optionally conditioned on inputs) and returns a dict of
                tensors.
            hparams_dict: a dict of `name=value` pairs, where `name` must be
                defined in `self.get_default_hparams()`.
            hparams: a string of comma separated list of `name=value` pairs,
                where `name` must be defined in `self.get_default_hparams()`.
                These values overrides any values in hparams_dict (if any).
        """
        super(VideoPredictionModel, self).__init__(mode, hparams_dict, hparams, **kwargs)
        self.generator_fn = functools.partial(generator_fn, mode=self.mode, hparams=self.hparams)
        self.discriminator_fn = functools.partial(discriminator_fn, mode=self.mode, hparams=self.hparams) if discriminator_fn else None
        self.generator_scope = generator_scope
        self.discriminator_scope = discriminator_scope
        self.aggregate_nccl = aggregate_nccl

        if any(self.hparams.lr_boundaries):
            global_step = tf.train.get_or_create_global_step()
            lr_values = list(self.hparams.lr * 0.1 ** np.arange(len(self.hparams.lr_boundaries) + 1))
            self.learning_rate = tf.train.piecewise_constant(global_step, self.hparams.lr_boundaries, lr_values)
        elif any(self.hparams.decay_steps):
            lr, end_lr = self.hparams.lr, self.hparams.end_lr
            start_step, end_step = self.hparams.decay_steps
            if start_step == end_step:
                schedule = tf.cond(tf.less(tf.train.get_or_create_global_step(), start_step),
                                   lambda: 0.0, lambda: 1.0)
            else:
                step = tf.clip_by_value(tf.train.get_or_create_global_step(), start_step, end_step)
                schedule = tf.to_float(step - start_step) / tf.to_float(end_step - start_step)
            self.learning_rate = lr + (end_lr - lr) * schedule
        else:
            self.learning_rate = self.hparams.lr

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
        self.summary_op = None
        self.image_summary_op = None
        self.eval_summary_op = None
        self.accum_eval_summary_op = None
        self.accum_eval_metrics_reset_op = None

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
            lr_boundaries=(0,),
            max_steps=300000,
            beta1=0.9,
            beta2=0.999,
            context_frames=-1,
            sequence_length=-1,
            clip_length=10,
            l1_weight=0.0,
            l2_weight=1.0,
            vgg_cdist_weight=0.0,
            feature_l2_weight=0.0,
            ae_l2_weight=0.0,
            state_weight=0.0,
            tv_weight=0.0,
            image_sn_gan_weight=0.0,
            image_sn_vae_gan_weight=0.0,
            images_sn_gan_weight=0.0,
            images_sn_vae_gan_weight=0.0,
            video_sn_gan_weight=0.0,
            video_sn_vae_gan_weight=0.0,
            gan_feature_l2_weight=0.0,
            gan_feature_cdist_weight=0.0,
            vae_gan_feature_l2_weight=0.0,
            vae_gan_feature_cdist_weight=0.0,
            gan_loss_type='LSGAN',
            joint_gan_optimization=False,
            kl_weight=0.0,
            kl_anneal='linear',
            kl_anneal_k=-1.0,
            kl_anneal_steps=(50000, 100000),
            z_l1_weight=0.0,
        )
        return dict(itertools.chain(default_hparams.items(), hparams.items()))

    def tower_fn(self, inputs):
        """
        This method doesn't have side-effects. `inputs`, `targets`, and
        `outputs` are batch-major but internal calculations use time-major
        tensors.
        """
        # batch-major to time-major
        inputs = nest.map_structure(transpose_batch_time, inputs)

        with tf.variable_scope(self.generator_scope):
            gen_outputs = self.generator_fn(inputs)

        if self.discriminator_fn:
            with tf.variable_scope(self.discriminator_scope) as discrim_scope:
                discrim_outputs = self.discriminator_fn(inputs, gen_outputs)
            # post-update discriminator tensors (i.e. after the discriminator weights have been updated)
            with tf.variable_scope(discrim_scope, reuse=True):
                discrim_outputs_post = self.discriminator_fn(inputs, gen_outputs)
        else:
            discrim_outputs = {}
            discrim_outputs_post = {}

        outputs = [gen_outputs, discrim_outputs]
        total_num_outputs = sum([len(output) for output in outputs])
        outputs = OrderedDict(itertools.chain(*[output.items() for output in outputs]))
        assert len(outputs) == total_num_outputs  # ensure no output is lost because of repeated keys

        if isinstance(self.learning_rate, tf.Tensor):
            outputs['learning_rate'] = self.learning_rate
        if isinstance(self.kl_weight, tf.Tensor):
            outputs['kl_weight'] = self.kl_weight

        if self.mode == 'train':
            with tf.name_scope("discriminator_loss"):
                d_losses = self.discriminator_loss_fn(inputs, outputs)
                print_loss_info(d_losses, inputs, outputs)
            with tf.name_scope("generator_loss"):
                g_losses = self.generator_loss_fn(inputs, outputs)
                print_loss_info(g_losses, inputs, outputs)
                if discrim_outputs_post:
                    outputs_post = OrderedDict(itertools.chain(gen_outputs.items(), discrim_outputs_post.items()))
                    # generator losses after the discriminator weights have been updated
                    g_losses_post = self.generator_loss_fn(inputs, outputs_post)
                else:
                    g_losses_post = g_losses
        else:
            d_losses = {}
            g_losses = {}
            g_losses_post = {}
        with tf.name_scope("metrics"):
            metrics = self.metrics_fn(inputs, outputs)
        with tf.name_scope("eval_outputs_and_metrics"):
            eval_outputs, eval_metrics = self.eval_outputs_and_metrics_fn(inputs, outputs)

        # time-major to batch-major
        outputs_tuple = (outputs, eval_outputs)
        outputs_tuple = nest.map_structure(transpose_batch_time, outputs_tuple)
        losses_tuple = (d_losses, g_losses, g_losses_post)
        losses_tuple = nest.map_structure(tf.convert_to_tensor, losses_tuple)
        loss_tuple = tuple(tf.accumulate_n([loss * weight for loss, weight in losses.values()])
                           if losses else tf.zeros(()) for losses in losses_tuple)
        metrics_tuple = (metrics, eval_metrics)
        metrics_tuple = nest.map_structure(transpose_batch_time, metrics_tuple)
        return outputs_tuple, losses_tuple, loss_tuple, metrics_tuple

    def build_graph(self, inputs):
        BaseVideoPredictionModel.build_graph(self, inputs)

        global_step = tf.train.get_or_create_global_step()
        # Capture the variables created from here until the train_op for the
        # saveable_variables. Note that if variables are being reused (e.g.
        # they were created by a previously built model), those variables won't
        # be captured here.
        original_global_variables = tf.global_variables()

        if self.num_gpus <= 1:  # cpu or 1 gpu
            outputs_tuple, losses_tuple, loss_tuple, metrics_tuple = self.tower_fn(self.inputs)
            self.outputs, self.eval_outputs = outputs_tuple
            self.d_losses, self.g_losses, g_losses_post = losses_tuple
            self.d_loss, self.g_loss, g_loss_post = loss_tuple
            self.metrics, self.eval_metrics = metrics_tuple

            self.d_vars = tf.trainable_variables(self.discriminator_scope)
            self.g_vars = tf.trainable_variables(self.generator_scope)
            g_optimizer = tf.train.AdamOptimizer(self.learning_rate, self.hparams.beta1, self.hparams.beta2)
            d_optimizer = tf.train.AdamOptimizer(self.learning_rate, self.hparams.beta1, self.hparams.beta2)

            if self.mode == 'train' and (self.d_losses or self.g_losses):
                with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                    if self.d_losses:
                        with tf.name_scope('d_compute_gradients'):
                            d_gradvars = d_optimizer.compute_gradients(self.d_loss, var_list=self.d_vars)
                        with tf.name_scope('d_apply_gradients'):
                            d_train_op = d_optimizer.apply_gradients(d_gradvars)
                    else:
                        d_train_op = tf.no_op()
                with tf.control_dependencies([d_train_op] if not self.hparams.joint_gan_optimization else []):
                    if g_losses_post:
                        if not self.hparams.joint_gan_optimization:
                            replace_read_ops(g_loss_post, self.d_vars)
                        with tf.name_scope('g_compute_gradients'):
                            g_gradvars = g_optimizer.compute_gradients(g_loss_post, var_list=self.g_vars)
                        with tf.name_scope('g_apply_gradients'):
                            g_train_op = g_optimizer.apply_gradients(g_gradvars)
                    else:
                        g_train_op = tf.no_op()
                with tf.control_dependencies([g_train_op]):
                    train_op = tf.assign_add(global_step, 1)
                self.train_op = train_op
            else:
                self.train_op = None

            global_variables = [var for var in tf.global_variables() if var not in original_global_variables]
            self.saveable_variables = [global_step] + global_variables
            self.post_init_ops = []
        else:
            if tf.get_variable_scope().name:
                # This is because how variable scope works with empty strings when it's not the root scope, causing
                # repeated forward slashes.
                raise NotImplementedError('Unable to handle multi-gpu model created within a non-root variable scope.')

            tower_inputs = [OrderedDict() for _ in range(self.num_gpus)]
            for name, input in self.inputs.items():
                input_splits = tf.split(input, self.num_gpus)  # assumes batch_size is divisible by num_gpus
                for i in range(self.num_gpus):
                    tower_inputs[i][name] = input_splits[i]

            tower_outputs_tuple = []
            tower_d_losses = []
            tower_g_losses = []
            tower_g_losses_post = []
            tower_d_loss = []
            tower_g_loss = []
            tower_g_loss_post = []
            tower_metrics_tuple = []
            for i in range(self.num_gpus):
                worker_device = '/gpu:%d' % i
                if self.aggregate_nccl:
                    scope_name = '' if i == 0 else 'v%d' % i
                    scope_reuse = False
                    device_setter = worker_device
                else:
                    scope_name = ''
                    scope_reuse = i > 0
                    device_setter = local_device_setter(worker_device=worker_device)
                with tf.variable_scope(scope_name, reuse=scope_reuse):
                    with tf.device(device_setter):
                        outputs_tuple, losses_tuple, loss_tuple, metrics_tuple = self.tower_fn(tower_inputs[i])
                        tower_outputs_tuple.append(outputs_tuple)
                        d_losses, g_losses, g_losses_post = losses_tuple
                        tower_d_losses.append(d_losses)
                        tower_g_losses.append(g_losses)
                        tower_g_losses_post.append(g_losses_post)
                        d_loss, g_loss, g_loss_post = loss_tuple
                        tower_d_loss.append(d_loss)
                        tower_g_loss.append(g_loss)
                        tower_g_loss_post.append(g_loss_post)
                        tower_metrics_tuple.append(metrics_tuple)
            self.d_vars = tf.trainable_variables(self.discriminator_scope)
            self.g_vars = tf.trainable_variables(self.generator_scope)

            if self.aggregate_nccl:
                scope_replica = lambda scope, i: ('' if i == 0 else 'v%d/' % i) + scope
                tower_d_vars = [tf.trainable_variables(
                    scope_replica(self.discriminator_scope, i)) for i in range(self.num_gpus)]
                tower_g_vars = [tf.trainable_variables(
                    scope_replica(self.generator_scope, i)) for i in range(self.num_gpus)]
                assert self.d_vars == tower_d_vars[0]
                assert self.g_vars == tower_g_vars[0]
                tower_d_optimizer = [tf.train.AdamOptimizer(
                    self.learning_rate, self.hparams.beta1, self.hparams.beta2) for _ in range(self.num_gpus)]
                tower_g_optimizer = [tf.train.AdamOptimizer(
                    self.learning_rate, self.hparams.beta1, self.hparams.beta2) for _ in range(self.num_gpus)]

                if self.mode == 'train' and (any(tower_d_losses) or any(tower_g_losses)):
                    tower_d_gradvars = []
                    tower_g_gradvars = []
                    tower_d_train_op = []
                    tower_g_train_op = []
                    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                        if any(tower_d_losses):
                            for i in range(self.num_gpus):
                                with tf.device('/gpu:%d' % i):
                                    with tf.name_scope(scope_replica('d_compute_gradients', i)):
                                        d_gradvars = tower_d_optimizer[i].compute_gradients(
                                            tower_d_loss[i], var_list=tower_d_vars[i])
                                        tower_d_gradvars.append(d_gradvars)

                            all_d_grads, all_d_vars = tf_utils.split_grad_list(tower_d_gradvars)
                            all_d_grads = tf_utils.allreduce_grads(all_d_grads, average=True)
                            tower_d_gradvars = tf_utils.merge_grad_list(all_d_grads, all_d_vars)

                            for i in range(self.num_gpus):
                                with tf.device('/gpu:%d' % i):
                                    with tf.name_scope(scope_replica('d_apply_gradients', i)):
                                        d_train_op = tower_d_optimizer[i].apply_gradients(tower_d_gradvars[i])
                                        tower_d_train_op.append(d_train_op)
                            d_train_op = tf.group(*tower_d_train_op)
                        else:
                            d_train_op = tf.no_op()
                    with tf.control_dependencies([d_train_op] if not self.hparams.joint_gan_optimization else []):
                        if any(tower_g_losses_post):
                            for i in range(self.num_gpus):
                                with tf.device('/gpu:%d' % i):
                                    if not self.hparams.joint_gan_optimization:
                                        replace_read_ops(tower_g_loss_post[i], tower_d_vars[i])

                                    with tf.name_scope(scope_replica('g_compute_gradients', i)):
                                        g_gradvars = tower_g_optimizer[i].compute_gradients(
                                            tower_g_loss_post[i], var_list=tower_g_vars[i])
                                        tower_g_gradvars.append(g_gradvars)

                            all_g_grads, all_g_vars = tf_utils.split_grad_list(tower_g_gradvars)
                            all_g_grads = tf_utils.allreduce_grads(all_g_grads, average=True)
                            tower_g_gradvars = tf_utils.merge_grad_list(all_g_grads, all_g_vars)

                            for i, g_gradvars in enumerate(tower_g_gradvars):
                                with tf.device('/gpu:%d' % i):
                                    with tf.name_scope(scope_replica('g_apply_gradients', i)):
                                        g_train_op = tower_g_optimizer[i].apply_gradients(g_gradvars)
                                        tower_g_train_op.append(g_train_op)
                            g_train_op = tf.group(*tower_g_train_op)
                        else:
                            g_train_op = tf.no_op()
                    with tf.control_dependencies([g_train_op]):
                        train_op = tf.assign_add(global_step, 1)
                    self.train_op = train_op
                else:
                    self.train_op = None

                global_variables = [var for var in tf.global_variables() if var not in original_global_variables]
                tower_saveable_vars = [[] for _ in range(self.num_gpus)]
                for var in global_variables:
                    m = re.match('v(\d+)/.*', var.name)
                    i = int(m.group(1)) if m else 0
                    tower_saveable_vars[i].append(var)
                self.saveable_variables = [global_step] + tower_saveable_vars[0]

                post_init_ops = []
                for i, saveable_vars in enumerate(tower_saveable_vars[1:], 1):
                    assert len(saveable_vars) == len(tower_saveable_vars[0])
                    for var, var0 in zip(saveable_vars, tower_saveable_vars[0]):
                        assert var.name == 'v%d/%s' % (i, var0.name)
                        post_init_ops.append(var.assign(var0.read_value()))
                self.post_init_ops = post_init_ops
            else:  # not self.aggregate_nccl (i.e. aggregation in cpu)
                g_optimizer = tf.train.AdamOptimizer(self.learning_rate, self.hparams.beta1, self.hparams.beta2)
                d_optimizer = tf.train.AdamOptimizer(self.learning_rate, self.hparams.beta1, self.hparams.beta2)

                if self.mode == 'train' and (any(tower_d_losses) or any(tower_g_losses)):
                    with tf.control_dependencies(tf.get_collection(tf.GraphKeys.UPDATE_OPS)):
                        if any(tower_d_losses):
                            with tf.name_scope('d_compute_gradients'):
                                d_gradvars = compute_averaged_gradients(
                                    d_optimizer, tower_d_loss, var_list=self.d_vars)
                            with tf.name_scope('d_apply_gradients'):
                                d_train_op = d_optimizer.apply_gradients(d_gradvars)
                        else:
                            d_train_op = tf.no_op()
                    with tf.control_dependencies([d_train_op] if not self.hparams.joint_gan_optimization else []):
                        if any(tower_g_losses_post):
                            for g_loss_post in tower_g_loss_post:
                                if not self.hparams.joint_gan_optimization:
                                    replace_read_ops(g_loss_post, self.d_vars)
                            with tf.name_scope('g_compute_gradients'):
                                g_gradvars = compute_averaged_gradients(
                                    g_optimizer, tower_g_loss_post, var_list=self.g_vars)
                            with tf.name_scope('g_apply_gradients'):
                                g_train_op = g_optimizer.apply_gradients(g_gradvars)
                        else:
                            g_train_op = tf.no_op()
                    with tf.control_dependencies([g_train_op]):
                        train_op = tf.assign_add(global_step, 1)
                    self.train_op = train_op
                else:
                    self.train_op = None

                global_variables = [var for var in tf.global_variables() if var not in original_global_variables]
                self.saveable_variables = [global_step] + global_variables
                self.post_init_ops = []

            # Device that runs the ops to apply global gradient updates.
            consolidation_device = '/cpu:0'
            with tf.device(consolidation_device):
                with tf.name_scope('consolidation'):
                    self.outputs, self.eval_outputs = reduce_tensors(tower_outputs_tuple)
                    self.d_losses = reduce_tensors(tower_d_losses, shallow=True)
                    self.g_losses = reduce_tensors(tower_g_losses, shallow=True)
                    self.metrics, self.eval_metrics = reduce_tensors(tower_metrics_tuple)
                    self.d_loss = reduce_tensors(tower_d_loss)
                    self.g_loss = reduce_tensors(tower_g_loss)

        original_local_variables = set(tf.local_variables())
        self.accum_eval_metrics = OrderedDict()
        for name, eval_metric in self.eval_metrics.items():
            _, self.accum_eval_metrics['accum_' + name] = tf.metrics.mean_tensor(eval_metric)
        local_variables = set(tf.local_variables()) - original_local_variables
        self.accum_eval_metrics_reset_op = tf.group([tf.assign(v, tf.zeros_like(v)) for v in local_variables])

        original_summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))
        add_summaries(self.inputs)
        add_summaries(self.outputs)
        add_scalar_summaries(self.d_losses)
        add_scalar_summaries(self.g_losses)
        add_scalar_summaries(self.metrics)
        if self.d_losses:
            add_scalar_summaries({'d_loss': self.d_loss})
        if self.g_losses:
            add_scalar_summaries({'g_loss': self.g_loss})
        if self.d_losses and self.g_losses:
            add_scalar_summaries({'loss': self.d_loss + self.g_loss})
        summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES)) - original_summaries
        # split summaries into non-image summaries and image summaries
        self.summary_op = tf.summary.merge(list(summaries - set(tf.get_collection(tf_utils.IMAGE_SUMMARIES))))
        self.image_summary_op = tf.summary.merge(list(summaries & set(tf.get_collection(tf_utils.IMAGE_SUMMARIES))))

        original_summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))
        add_gif_summaries(self.eval_outputs)
        add_plot_and_scalar_summaries(
            {name: tf.reduce_mean(metric, axis=0) for name, metric in self.eval_metrics.items()},
            x_offset=self.hparams.context_frames + 1)
        summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES)) - original_summaries
        self.eval_summary_op = tf.summary.merge(list(summaries))

        original_summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES))
        add_plot_and_scalar_summaries(
            {name: tf.reduce_mean(metric, axis=0) for name, metric in self.accum_eval_metrics.items()},
            x_offset=self.hparams.context_frames + 1)
        summaries = set(tf.get_collection(tf.GraphKeys.SUMMARIES)) - original_summaries
        self.accum_eval_summary_op = tf.summary.merge(list(summaries))

    def generator_loss_fn(self, inputs, outputs):
        hparams = self.hparams
        gen_losses = OrderedDict()
        if hparams.l1_weight or hparams.l2_weight or hparams.vgg_cdist_weight:
            gen_images = outputs.get('gen_images_enc', outputs['gen_images'])
            target_images = inputs['images'][1:]
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
            target_features = outputs['features'][1:]
            gen_feature_l2_loss = vp.losses.l2_loss(gen_features, target_features)
            gen_losses["gen_feature_l2_loss"] = (gen_feature_l2_loss, hparams.feature_l2_weight)
        if hparams.ae_l2_weight:
            gen_images_dec = outputs.get('gen_images_dec_enc', outputs['gen_images_dec'])  # they both should be the same
            target_images = inputs['images']
            gen_ae_l2_loss = vp.losses.l2_loss(gen_images_dec, target_images)
            gen_losses["gen_ae_l2_loss"] = (gen_ae_l2_loss, hparams.ae_l2_weight)
        if hparams.state_weight:
            gen_states = outputs.get('gen_states_enc', outputs['gen_states'])
            target_states = inputs['states'][1:]
            gen_state_loss = vp.losses.l2_loss(gen_states, target_states)
            gen_losses["gen_state_loss"] = (gen_state_loss, hparams.state_weight)
        if hparams.tv_weight:
            gen_flows = outputs.get('gen_flows_enc', outputs['gen_flows'])
            flow_diff1 = gen_flows[..., 1:, :, :, :] - gen_flows[..., :-1, :, :, :]
            flow_diff2 = gen_flows[..., :, 1:, :, :] - gen_flows[..., :, :-1, :, :]
            # sum over the multiple transformations but take the mean for the other dimensions
            gen_tv_loss = (tf.reduce_mean(tf.reduce_sum(tf.abs(flow_diff1), axis=(-2, -1))) +
                           tf.reduce_mean(tf.reduce_sum(tf.abs(flow_diff2), axis=(-2, -1))))
            gen_losses['gen_tv_loss'] = (gen_tv_loss, hparams.tv_weight)
        gan_weights = {'_image_sn': hparams.image_sn_gan_weight,
                       '_images_sn': hparams.images_sn_gan_weight,
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
                    gen_gan_feature_cdist_loss = sum([vp.losses.cosine_distance(discrim_feature_fake, discrim_feature_real)
                                                      for discrim_feature_fake, discrim_feature_real in zip(discrim_features_fake, discrim_features_real)])
                    gen_losses["gen%s_gan_feature_cdist_loss" % infix] = (gen_gan_feature_cdist_loss, hparams.gan_feature_cdist_weight)
        vae_gan_weights = {'_image_sn': hparams.image_sn_vae_gan_weight,
                           '_images_sn': hparams.images_sn_vae_gan_weight,
                           '_video_sn': hparams.video_sn_vae_gan_weight}
        for infix, vae_gan_weight in vae_gan_weights.items():
            if vae_gan_weight:
                gen_vae_gan_loss = vp.losses.gan_loss(outputs['discrim%s_logits_enc_fake' % infix], 1.0, hparams.gan_loss_type)
                gen_losses["gen%s_vae_gan_loss" % infix] = (gen_vae_gan_loss, vae_gan_weight)
            if vae_gan_weight and (hparams.vae_gan_feature_l2_weight or hparams.vae_gan_feature_cdist_weight):
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
                if hparams.vae_gan_feature_l2_weight:
                    gen_vae_gan_feature_l2_loss = sum([vp.losses.l2_loss(discrim_feature_enc_fake, discrim_feature_enc_real)
                                                       for discrim_feature_enc_fake, discrim_feature_enc_real in zip(discrim_features_enc_fake, discrim_features_enc_real)])
                    gen_losses["gen%s_vae_gan_feature_l2_loss" % infix] = (gen_vae_gan_feature_l2_loss, hparams.vae_gan_feature_l2_weight)
                if hparams.vae_gan_feature_cdist_weight:
                    gen_vae_gan_feature_cdist_loss = sum([vp.losses.cosine_distance(discrim_feature_enc_fake, discrim_feature_enc_real)
                                                          for discrim_feature_enc_fake, discrim_feature_enc_real in zip(discrim_features_enc_fake, discrim_features_enc_real)])
                    gen_losses["gen%s_vae_gan_feature_cdist_loss" % infix] = (gen_vae_gan_feature_cdist_loss, hparams.vae_gan_feature_cdist_weight)
        if hparams.kl_weight:
            gen_kl_loss = vp.losses.kl_loss(outputs['zs_mu_enc'], outputs['zs_log_sigma_sq_enc'],
                                            outputs.get('zs_mu_prior'), outputs.get('zs_log_sigma_sq_prior'))
            gen_losses["gen_kl_loss"] = (gen_kl_loss, self.kl_weight)  # possibly annealed kl_weight
        return gen_losses

    def discriminator_loss_fn(self, inputs, outputs):
        hparams = self.hparams
        discrim_losses = OrderedDict()
        gan_weights = {'_image_sn': hparams.image_sn_gan_weight,
                       '_images_sn': hparams.images_sn_gan_weight,
                       '_video_sn': hparams.video_sn_gan_weight}
        for infix, gan_weight in gan_weights.items():
            if gan_weight:
                discrim_gan_loss_real = vp.losses.gan_loss(outputs['discrim%s_logits_real' % infix], 1.0, hparams.gan_loss_type)
                discrim_gan_loss_fake = vp.losses.gan_loss(outputs['discrim%s_logits_fake' % infix], 0.0, hparams.gan_loss_type)
                discrim_gan_loss = discrim_gan_loss_real + discrim_gan_loss_fake
                discrim_losses["discrim%s_gan_loss" % infix] = (discrim_gan_loss, gan_weight)
        vae_gan_weights = {'_image_sn': hparams.image_sn_vae_gan_weight,
                           '_images_sn': hparams.images_sn_vae_gan_weight,
                           '_video_sn': hparams.video_sn_vae_gan_weight}
        for infix, vae_gan_weight in vae_gan_weights.items():
            if vae_gan_weight:
                discrim_vae_gan_loss_real = vp.losses.gan_loss(outputs['discrim%s_logits_enc_real' % infix], 1.0, hparams.gan_loss_type)
                discrim_vae_gan_loss_fake = vp.losses.gan_loss(outputs['discrim%s_logits_enc_fake' % infix], 0.0, hparams.gan_loss_type)
                discrim_vae_gan_loss = discrim_vae_gan_loss_real + discrim_vae_gan_loss_fake
                discrim_losses["discrim%s_vae_gan_loss" % infix] = (discrim_vae_gan_loss, vae_gan_weight)
        return discrim_losses
