import functools
import itertools
from collections import OrderedDict

import tensorflow as tf
from tensorflow.contrib.training import HParams
from tensorflow.python.util import nest

import video_prediction as vp
from video_prediction.tf_utils import compute_averaged_gradients, reduce_tensors, local_device_setter, \
    replace_read_ops, print_loss_info, add_image_summaries, add_scalar_summaries


class SoftPlacementVideoPredictionModel:
    def __init__(self,
                 generator_fn,
                 discriminator_fn=None,
                 encoder_fn=None,
                 generator_scope='generator',
                 discriminator_scope='discriminator',
                 encoder_scope='encoder',
                 discriminator_encoder_scope='discriminator',
                 hparams_dict=None,
                 hparams=None):
        """
        Video prediction model with automatically chosen devices.

        The devices for the ops in `self.build_graph` are automatically chosen
        by TensorFlow, i.e. `tf.device` is not specified.

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
        self.hparams = self.get_default_hparams().set_from_map(hparams_dict or {}).parse(hparams or '')
        if not self.hparams.context_frames:
            raise ValueError('Invalid context_frames %r. It might have to be '
                             'specified.' % self.hparams.context_frames)
        if not self.hparams.sequence_length:
            raise ValueError('Invalid sequence_length %r. It might have to be '
                             'specified.' % self.hparams.sequence_length)

        self.generator_fn = functools.partial(generator_fn, hparams=self.hparams)
        self.encoder_fn = functools.partial(encoder_fn, hparams=self.hparams) if encoder_fn else None
        self.discriminator_fn = functools.partial(discriminator_fn, hparams=self.hparams) if discriminator_fn else None

        self.generator_scope = generator_scope
        self.encoder_scope = encoder_scope
        self.discriminator_scope = discriminator_scope
        self.discriminator_encoder_scope = discriminator_encoder_scope

        self.g_optimizer = tf.train.AdamOptimizer(self.hparams.lr, self.hparams.beta1, self.hparams.beta2)
        self.d_optimizer = tf.train.AdamOptimizer(self.hparams.lr, self.hparams.beta1, self.hparams.beta2)

        # member variables that should be set by `self.build_graph`
        self.inputs = None
        self.targets = None
        self.gen_images = None
        self.gen_images_enc = None
        self.outputs = None
        self.g_losses = None
        self.d_losses = None
        self.metrics = None
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
        """
        hparams = dict(
            context_frames=0,  # should be specified in init
            sequence_length=0,  # should be specified in init
            l1_weight=0.0,
            l2_weight=1.0,
            state_weight=1e-4,
            gan_weight=0.0,
            vae_gan_weight=0.0,
            tuple_gan_weight=0.0,
            vae_tuple_gan_weight=0.0,
            gan_loss_type='LSGAN',
            kl_weight=0.0,
            kl_anneal_k=-1.0,
            z_l1_weight=0.0,
            lr=0.001,
            beta1=0.9,
            beta2=0.999,
        )
        return hparams

    def get_default_hparams(self):
        return HParams(**self.get_default_hparams_dict())

    def tower_fn(self, inputs, targets=None):
        """
        This method doesn't have side-effects. `inputs`, `targets`, and
        `outputs` are batch-major but internal calculations use time-major
        tensors.
        """
        # batch-major to time-major
        def transpose_batch_time(x):
            if x is None:
                return None
            return tf.transpose(x, [1, 0] + list(range(2, x.shape.ndims)))
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
                    _, discrim_outputs_real = self.discriminator_fn(targets, discrim_inputs)
                    discrim_outputs_real = OrderedDict([(k + '_real', v) for k, v in discrim_outputs_real.items()])
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
            discrim_outputs_fake = {}
            discrim_outputs_fake_post = {}

        if self.discriminator_fn and self.encoder_fn and targets is not None:
            discrim_inputs_enc = OrderedDict(list(inputs.items()) + list(gen_outputs_enc.items()))
            same_discriminator = self.discriminator_scope == self.discriminator_encoder_scope
            with tf.name_scope("real"), tf.name_scope(self.encoder_scope):
                with tf.variable_scope(self.discriminator_encoder_scope, reuse=same_discriminator) as discrim_enc_scope:
                    _, discrim_outputs_enc_real = self.discriminator_fn(targets, discrim_inputs_enc)
                    discrim_outputs_enc_real = OrderedDict([(k + '_enc_real', v) for k, v in discrim_outputs_enc_real.items()])
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
            discrim_outputs_enc_fake = {}
            discrim_outputs_enc_fake_post = {}

        outputs = [gen_outputs, outputs_enc, gen_outputs_enc,
                   discrim_outputs_real, discrim_outputs_fake,
                   discrim_outputs_enc_real, discrim_outputs_enc_fake]
        total_num_outputs = sum([len(output) for output in outputs])
        outputs = OrderedDict(itertools.chain(*[output.items() for output in outputs]))
        assert len(outputs) == total_num_outputs  # ensure no output is lost because of repeated keys

        if targets is not None:
            with tf.name_scope("discriminator_loss"):
                d_losses = self.discriminator_loss_fn(inputs, outputs, targets)
                print_loss_info(d_losses, inputs, outputs, targets)
            with tf.name_scope("generator_loss"):
                g_losses = self.generator_loss_fn(inputs, outputs, targets)
                print_loss_info(g_losses, inputs, outputs, targets)
                outputs_post = OrderedDict(itertools.chain(outputs.items(),
                                                           discrim_outputs_fake_post.items(),
                                                           discrim_outputs_enc_fake_post.items()))
                g_losses_post = self.generator_loss_fn(inputs, outputs_post, targets)
            with tf.name_scope("metrics"):
                metrics = self.metrics_fn(inputs, outputs, targets)
        else:
            d_losses = {}
            g_losses = {}
            g_losses_post = {}
            metrics = {}

        # time-major to batch-major
        outputs_tuple = (gen_images, gen_images_enc, outputs)
        outputs_tuple = nest.map_structure(transpose_batch_time, outputs_tuple)
        losses_tuple = (d_losses, g_losses, g_losses_post)
        losses_tuple = nest.map_structure(tf.convert_to_tensor, losses_tuple)
        return outputs_tuple, losses_tuple, metrics

    def build_graph(self, inputs, targets=None):
        self.inputs = inputs
        self.targets = targets
        outputs_tuple, losses_tuple, metrics = self.tower_fn(self.inputs, self.targets)
        self.gen_images, self.gen_images_enc, self.outputs = outputs_tuple
        self.d_losses, self.g_losses, g_losses_post = losses_tuple
        self.metrics = metrics
        self.d_loss = sum(loss * weight for loss, weight in self.d_losses.values())
        self.g_loss = sum(loss * weight for loss, weight in self.g_losses.values())
        g_loss_post = sum(loss * weight for loss, weight in g_losses_post.values())

        self.d_vars = tf.trainable_variables(self.discriminator_scope)
        self.g_vars = tf.trainable_variables(self.generator_scope)

        if self.d_losses or self.g_losses:
            with tf.name_scope('optimize'):
                if self.d_losses:
                    d_gradvars = self.d_optimizer.compute_gradients(self.d_loss, var_list=self.d_vars)
                    d_train_op = self.d_optimizer.apply_gradients(d_gradvars)
                else:
                    d_train_op = tf.no_op()
                if g_losses_post:
                    with tf.control_dependencies([d_train_op]):
                        replace_read_ops(g_loss_post, self.d_vars)
                        g_gradvars = self.g_optimizer.compute_gradients(g_loss_post, var_list=self.g_vars)
                        g_train_op = self.g_optimizer.apply_gradients(
                            g_gradvars, global_step=tf.train.get_or_create_global_step())  # also increments global_step
                else:
                    g_train_op = tf.assign_add(tf.train.get_or_create_global_step(), 1)
            self.train_op = g_train_op
        else:
            self.train_op = None

        add_image_summaries(self.outputs)
        add_scalar_summaries(self.d_losses)
        add_scalar_summaries(self.g_losses)
        add_scalar_summaries(self.metrics)

    def generator_loss_fn(self, inputs, outputs, targets):
        hparams = self.hparams
        gen_losses = OrderedDict()
        if hparams.l1_weight or hparams.l2_weight:
            gen_images = outputs.get('gen_images_enc', outputs['gen_images'])
            target_images = targets
        if hparams.l1_weight:
            gen_l1_loss = vp.losses.l1_loss(gen_images, target_images)
            gen_losses["gen_l1_loss"] = (gen_l1_loss, hparams.l1_weight)
        if hparams.l2_weight:
            gen_l2_loss = vp.losses.l2_loss(gen_images, target_images)
            gen_losses["gen_l2_loss"] = (gen_l2_loss, hparams.l2_weight)
        if hparams.state_weight:
            gen_states = outputs.get('gen_states_enc', outputs['gen_states'])
            target_states = inputs['states'][hparams.context_frames:]
            gen_state_loss = vp.losses.l2_loss(gen_states, target_states)
            gen_losses["gen_state_loss"] = (gen_state_loss, hparams.state_weight)
        if hparams.gan_weight:
            gen_gan_loss = vp.losses.gan_loss(outputs['discrim_logits_fake'], 1.0, hparams.gan_loss_type)
            gen_losses["gen_gan_loss"] = (gen_gan_loss, hparams.gan_weight)
        if hparams.vae_gan_weight:
            gen_vae_gan_loss = vp.losses.gan_loss(outputs['discrim_logits_enc_fake'], 1.0, hparams.gan_loss_type)
            gen_losses["gen_vae_gan_loss"] = (gen_vae_gan_loss, hparams.vae_gan_weight)
        if hparams.tuple_gan_weight:
            gen_tuple_gan_loss = vp.losses.gan_loss(outputs['discrim_tuple_logits_fake'], 1.0, hparams.gan_loss_type)
            gen_losses["gen_tuple_gan_loss"] = (gen_tuple_gan_loss, hparams.tuple_gan_weight)
        if hparams.vae_tuple_gan_weight:
            gen_vae_tuple_gan_loss = vp.losses.gan_loss(outputs['discrim_tuple_logits_enc_fake'], 1.0, hparams.gan_loss_type)
            gen_losses["gen_vae_tuple_gan_loss"] = (gen_vae_tuple_gan_loss, hparams.vae_tuple_gan_weight)
        if hparams.kl_weight:
            gen_kl_loss = vp.losses.kl_loss(outputs['zs_enc_mu'], outputs['zs_enc_log_sigma_sq'])
            if hparams.kl_anneal_k == -1:
                kl_weight = tf.constant(hparams.kl_weight, tf.float32)
            else:
                iter_num = tf.train.get_or_create_global_step()
                kl_weight = hparams.kl_weight / (1 + hparams.kl_anneal_k * tf.exp(-tf.to_float(iter_num) / hparams.kl_anneal_k))
            gen_losses["gen_kl_loss"] = (gen_kl_loss, kl_weight)
        if hparams.z_l1_weight:
            gen_z_l1_loss = vp.losses.l1_loss(outputs['gen_zs_enc_mu'], outputs['gen_zs_random'])
            gen_losses["gen_z_l1_loss"] = (gen_z_l1_loss, hparams.z_l1_weight)
        return gen_losses

    def discriminator_loss_fn(self, inputs, outputs, targets):
        hparams = self.hparams
        discrim_losses = OrderedDict()
        if hparams.gan_weight:
            discrim_gan_loss_real = vp.losses.gan_loss(outputs['discrim_logits_real'], 1.0, hparams.gan_loss_type)
            discrim_gan_loss_fake = vp.losses.gan_loss(outputs['discrim_logits_fake'], 0.0, hparams.gan_loss_type)
            discrim_gan_loss = discrim_gan_loss_real + discrim_gan_loss_fake
            discrim_losses["discrim_gan_loss"] = (discrim_gan_loss, hparams.gan_weight)
        if hparams.vae_gan_weight:
            discrim_vae_gan_loss_real = vp.losses.gan_loss(outputs['discrim_logits_enc_real'], 1.0, hparams.gan_loss_type)
            discrim_vae_gan_loss_fake = vp.losses.gan_loss(outputs['discrim_logits_enc_fake'], 0.0, hparams.gan_loss_type)
            discrim_vae_gan_loss = discrim_vae_gan_loss_real + discrim_vae_gan_loss_fake
            discrim_losses["discrim_vae_gan_loss"] = (discrim_vae_gan_loss, hparams.vae_gan_weight)
        if hparams.tuple_gan_weight:
            discrim_tuple_gan_loss_real = vp.losses.tuple_gan_loss(outputs['discrim_tuple_logits_real'], 1.0, hparams.gan_loss_type)
            discrim_tuple_gan_loss_fake = vp.losses.tuple_gan_loss(outputs['discrim_tuple_logits_fake'], 0.0, hparams.gan_loss_type)
            discrim_tuple_gan_loss = discrim_tuple_gan_loss_real + discrim_tuple_gan_loss_fake
            discrim_losses["discrim_tuple_gan_loss"] = (discrim_tuple_gan_loss, hparams.tuple_gan_weight)
        if hparams.vae_tuple_gan_weight:
            discrim_vae_tuple_gan_loss_real = vp.losses.gan_loss(outputs['discrim_tuple_logits_enc_real'], 1.0, hparams.gan_loss_type)
            discrim_vae_tuple_gan_loss_fake = vp.losses.gan_loss(outputs['discrim_tuple_logits_enc_fake'], 0.0, hparams.gan_loss_type)
            discrim_vae_tuple_gan_loss = discrim_vae_tuple_gan_loss_real + discrim_vae_tuple_gan_loss_fake
            discrim_losses["discrim_vae_tuple_gan_loss"] = (discrim_vae_tuple_gan_loss, hparams.vae_tuple_gan_weight)
        return discrim_losses

    def metrics_fn(self, inputs, outputs, targets):
        metrics = OrderedDict()
        target_images = targets
        gen_images = outputs['gen_images']
        metrics['psnr'] = vp.metrics.peak_signal_to_noise_ratio(target_images, gen_images)
        metrics['mse'] = vp.metrics.mean_squared_error(target_images, gen_images)
        metrics['ssim'] = vp.metrics.structural_similarity(target_images, gen_images)
        return metrics


class VideoPredictionModel(SoftPlacementVideoPredictionModel):
    def __init__(self, *args, **kwargs):
        """
        Video prediction model with multi-GPU support.
        """
        self.num_gpus = kwargs.pop('num_gpus', 1)
        super(VideoPredictionModel, self).__init__(*args, **kwargs)

    def build_graph(self, inputs, targets=None):
        self.inputs = inputs
        self.targets = targets
        tower_inputs = [OrderedDict() for _ in range(self.num_gpus)]
        for name, input in self.inputs.items():
            input_splits = tf.split(input, self.num_gpus)  # assumes batch_size is divisible by num_gpus
            for i in range(self.num_gpus):
                tower_inputs[i][name] = input_splits[i]
        tower_targets = tf.split(targets, self.num_gpus) if targets is not None else [None] * self.num_gpus

        tower_outputs_tuple = []
        tower_d_losses = []
        tower_g_losses = []
        tower_g_losses_post = []
        tower_metrics = []
        tower_d_loss = []
        tower_g_loss = []
        tower_g_loss_post = []
        for i in range(self.num_gpus):
            worker_device = '/{}:{}'.format('gpu', i)
            device_setter = local_device_setter(worker_device=worker_device)
            with tf.variable_scope('', reuse=bool(i > 0)):
                with tf.name_scope('tower_%d' % i):
                    with tf.device(device_setter):
                        outputs_tuple, losses_tuple, metrics = self.tower_fn(tower_inputs[i], tower_targets[i])
                        tower_outputs_tuple.append(outputs_tuple)
                        d_losses, g_losses, g_losses_post = losses_tuple
                        tower_d_losses.append(d_losses)
                        tower_g_losses.append(g_losses)
                        tower_g_losses_post.append(g_losses_post)
                        tower_metrics.append(metrics)
                        d_loss = sum(loss * weight for loss, weight in d_losses.values())
                        g_loss = sum(loss * weight for loss, weight in g_losses.values())
                        g_loss_post = sum(loss * weight for loss, weight in g_losses_post.values())
                        tower_d_loss.append(d_loss)
                        tower_g_loss.append(g_loss)
                        tower_g_loss_post.append(g_loss_post)

        self.d_vars = tf.trainable_variables(self.discriminator_scope)
        self.g_vars = tf.trainable_variables(self.generator_scope)

        if any(tower_d_losses) or any(tower_g_losses):
            with tf.name_scope('optimize'):
                if any(tower_d_losses):
                    d_gradvars = compute_averaged_gradients(self.d_optimizer, tower_d_loss, var_list=self.d_vars)
                    d_train_op = self.d_optimizer.apply_gradients(d_gradvars)
                else:
                    d_train_op = tf.no_op()
                if any(tower_g_losses_post):
                    with tf.control_dependencies([d_train_op]):
                        replace_read_ops(tower_g_loss_post, self.d_vars)
                        g_gradvars = compute_averaged_gradients(self.g_optimizer, tower_g_loss_post, var_list=self.g_vars)
                        g_train_op = self.g_optimizer.apply_gradients(
                            g_gradvars, global_step=tf.train.get_global_step())  # also increments global_step
                else:
                    g_train_op = tf.assign_add(tf.train.get_global_step(), 1)
            self.train_op = g_train_op
        else:
            self.train_op = None

        # Device that runs the ops to apply global gradient updates.
        consolidation_device = '/cpu:0'
        with tf.device(consolidation_device):
            self.gen_images, self.gen_images_enc, self.outputs = reduce_tensors(tower_outputs_tuple)
            self.d_losses = reduce_tensors(tower_d_losses, shallow=True)
            self.g_losses = reduce_tensors(tower_g_losses, shallow=True)
            self.metrics = reduce_tensors(tower_metrics)
            self.d_loss = reduce_tensors(tower_d_loss)
            self.g_loss = reduce_tensors(tower_g_loss)

        add_image_summaries(self.outputs)
        add_scalar_summaries(self.d_losses)
        add_scalar_summaries(self.g_losses)
        add_scalar_summaries(self.metrics)
