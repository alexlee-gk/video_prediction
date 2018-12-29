from collections import OrderedDict
from tensorflow.python.util import nest
from video_prediction.utils.tf_utils import transpose_batch_time

import tensorflow as tf

from .base_model import BaseVideoPredictionModel


class NonTrainableVideoPredictionModel(BaseVideoPredictionModel):
    pass


class GroundTruthVideoPredictionModel(NonTrainableVideoPredictionModel):
    def build_graph(self, inputs):
        super(GroundTruthVideoPredictionModel, self).build_graph(inputs)

        self.outputs = OrderedDict()
        self.outputs['gen_images'] = self.inputs['images'][:, 1:]
        if 'pix_distribs' in self.inputs:
            self.outputs['gen_pix_distribs'] = self.inputs['pix_distribs'][:, 1:]

        inputs, outputs = nest.map_structure(transpose_batch_time, (self.inputs, self.outputs))
        with tf.name_scope("metrics"):
            metrics = self.metrics_fn(inputs, outputs)
        with tf.name_scope("eval_outputs_and_metrics"):
            eval_outputs, eval_metrics = self.eval_outputs_and_metrics_fn(inputs, outputs)
        self.metrics, self.eval_outputs, self.eval_metrics = nest.map_structure(
            transpose_batch_time, (metrics, eval_outputs, eval_metrics))


class RepeatVideoPredictionModel(NonTrainableVideoPredictionModel):
    def build_graph(self, inputs):
        super(RepeatVideoPredictionModel, self).build_graph(inputs)

        self.outputs = OrderedDict()
        tile_pattern = [1, self.hparams.sequence_length - self.hparams.context_frames, 1, 1, 1]
        last_context_images = self.inputs['images'][:, self.hparams.context_frames - 1]
        self.outputs['gen_images'] = tf.concat([
            self.inputs['images'][:, 1:self.hparams.context_frames - 1],
            tf.tile(last_context_images[:, None], tile_pattern)], axis=-1)
        if 'pix_distribs' in self.inputs:
            last_context_pix_distrib = self.inputs['pix_distribs'][:, self.hparams.context_frames - 1]
            self.outputs['gen_pix_distribs'] = tf.concat([
                self.inputs['pix_distribs'][:, 1:self.hparams.context_frames - 1],
                tf.tile(last_context_pix_distrib[:, None], tile_pattern)], axis=-1)

        inputs, outputs = nest.map_structure(transpose_batch_time, (self.inputs, self.outputs))
        with tf.name_scope("metrics"):
            metrics = self.metrics_fn(inputs, outputs)
        with tf.name_scope("eval_outputs_and_metrics"):
            eval_outputs, eval_metrics = self.eval_outputs_and_metrics_fn(inputs, outputs)
        self.metrics, self.eval_outputs, self.eval_metrics = nest.map_structure(
            transpose_batch_time, (metrics, eval_outputs, eval_metrics))
