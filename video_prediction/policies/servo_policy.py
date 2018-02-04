import functools

import numpy as np
import tensorflow as tf
from tensorflow.contrib.training import HParams

from video_prediction import models
from video_prediction.cem import cem


class ServoPolicy:
    def __init__(self, model, sess, hparams_dict=None, hparams=None):
        """
        model should have already been built
        """
        self.model = model
        self.sess = sess
        self.hparams = self.parse_hparams(hparams_dict, hparams)
        if not self.hparams.cem_batch_size:
            self.hparams.cem_batch_size = self.model.inputs['actions'].shape[0].value
        if not self.hparams.plan_horizon:
            self.hparams.plan_horizon = self.model.inputs['actions'].shape[1].value
        if not self.hparams.repeat:
            self.hparams.repeat = self.model.hparams.repeat

        image_shape = self.model.inputs['images'].shape.as_list()[2:]
        self.goal_image_ph = tf.placeholder(tf.float32, shape=[self.hparams.cem_batch_size] + image_shape)
        self.costs = tf.reduce_mean(tf.square(model.outputs['gen_images'][:, -1] - self.goal_image_ph), axis=(1, 2, 3))
        if self.hparams.lambda_ != 0.0:
            self.costs += self.hparams.lambda_ * tf.reduce_mean(tf.square(model.inputs['actions']), axis=(1, 2))

    def get_default_hparams_dict(self):
        hparams = dict(
            cem_batch_size=0,
            cem_n_iters=4,
            plan_horizon=0,
            repeat=0,
            lambda_=0.0,
        )
        return hparams

    def get_default_hparams(self):
        return HParams(**self.get_default_hparams_dict())

    def parse_hparams(self, hparams_dict, hparams):
        parsed_hparams = self.get_default_hparams().set_from_map(hparams_dict or {})
        if hparams:
            if not isinstance(hparams, (list, tuple)):
                hparams = [hparams]
            for hparam in hparams:
                parsed_hparams.parse(hparam)
        return parsed_hparams

    def initial_theta_distribution(self):
        action_dim = self.model.inputs['actions'].shape[-1].value
        xy_std = .035
        grasp_std = 1.
        lift_std = 1.
        rot_std = np.pi / 8
        if action_dim == 5:
            action_std = np.array([xy_std, xy_std, lift_std, rot_std, grasp_std])
        elif action_dim == 4:
            action_std = np.array([xy_std, xy_std, grasp_std, lift_std])
        elif action_dim == 3:
            action_std = np.array([xy_std, xy_std, lift_std])
        else:
            raise NotImplementedError
        effective_horizon = int(np.ceil(self.hparams.plan_horizon / self.hparams.repeat))
        theta_mean = np.zeros(effective_horizon * action_dim)
        theta_std = np.reshape(np.repeat(action_std[None], effective_horizon, axis=0), -1)
        theta_cov = np.diag(theta_std ** 2)
        return theta_mean, theta_cov

    def project_action(self, actions, max_shift=.09, max_rot=np.pi / 4):
        action_dim = actions.shape[-1]
        if action_dim == 4:
            discrete_inds = [2, 3]
        elif action_dim == 5:
            discrete_inds = [2, 4]
        else:
            discrete_inds = []
        # discretize and clip discrete actions
        actions[..., discrete_inds] = np.clip(np.floor(actions[..., discrete_inds]), 0, 4)
        # clip the other actions
        actions[..., :2] = np.clip(actions[..., :2], -max_shift, max_shift)
        if action_dim == 5:  # if rotation is enabled
            actions[..., 3] = np.clip(actions[..., 3], -max_rot, max_rot)
        return actions

    def theta_to_action(self, theta):
        action_dim = self.model.inputs['actions'].shape[-1].value
        # reshape theta's last dimension to time and action dimensions
        action = np.reshape(theta, theta.shape[:-1] + (-1, action_dim))
        action = np.repeat(action, self.hparams.repeat, axis=-2)
        action = self.project_action(action)
        # discard the last repeated actions (if any)
        action = action[..., :self.hparams.plan_horizon, :]
        return action

    def noisy_evaluation(self, obs, thetas, fetches=None):
        context_images = obs['context_images']
        context_state = obs['context_state']
        goal_image = obs['goal_image']
        image_shape = tuple(self.model.inputs['images'].shape.as_list()[-3:])
        state_dim = self.model.inputs['states'].shape[-1].value
        if not isinstance(self.model, models.GroundTruthVideoPredictionModel):
            assert context_images.shape == ((self.model.hparams.context_frames,) + image_shape)
        assert context_state.shape == (state_dim,)
        assert goal_image.shape == image_shape

        actions = self.theta_to_action(thetas)

        feed_dict = {
            self.model.inputs['images']: np.repeat(context_images[None], self.hparams.cem_batch_size, axis=0),
            self.model.inputs['states']: np.repeat(context_state[None, None], self.hparams.cem_batch_size, axis=0),
            self.goal_image_ph: np.repeat(goal_image[None], self.hparams.cem_batch_size, axis=0),
            self.model.inputs['actions']: actions,
        }
        if fetches is None:
            fetches = self.costs
        else:
            fetches = (self.costs, fetches)
        return self.sess.run(fetches, feed_dict=feed_dict)

    def act(self, obs, fetches=None):
        noisy_evaluation = functools.partial(self.noisy_evaluation, obs, fetches=fetches)
        theta_mean, theta_cov = self.initial_theta_distribution()
        theta_best, results_best = cem(noisy_evaluation, theta_mean, theta_cov,
                                       self.hparams.cem_batch_size,
                                       self.hparams.cem_n_iters)
        action_best = self.theta_to_action(theta_best)
        if fetches is None:
            return action_best
        else:
            _, results_best = results_best
            return action_best, results_best
