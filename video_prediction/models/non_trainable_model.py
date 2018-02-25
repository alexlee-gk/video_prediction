from collections import OrderedDict

import tensorflow as tf

from .base_model import BaseVideoPredictionModel


class NonTrainableVideoPredictionModel(BaseVideoPredictionModel):
    pass


class GroundTruthVideoPredictionModel(NonTrainableVideoPredictionModel):
    def build_graph(self, inputs, targets=None):
        super(GroundTruthVideoPredictionModel, self).build_graph(inputs, targets=targets)

        self.outputs = OrderedDict()
        self.outputs['gen_images'] = self.inputs['images'][:, self.hparams.context_frames:]
        if 'pix_distribs' in self.inputs:
            self.outputs['gen_pix_distribs'] = self.inputs['pix_distribs'][:, self.hparams.context_frames:]
        self.gen_images = self.outputs['gen_images']

        if self.targets is not None:
            self.metrics = self.metrics_fn(self.inputs, self.outputs, self.targets)
        else:
            self.metrics = {}


class RepeatVideoPredictionModel(NonTrainableVideoPredictionModel):
    def build_graph(self, inputs, targets=None):
        super(RepeatVideoPredictionModel, self).build_graph(inputs, targets=targets)

        self.outputs = OrderedDict()
        tile_pattern = [1, self.hparams.sequence_length - self.hparams.context_frames, 1, 1, 1]
        last_context_images = self.inputs['images'][:, self.hparams.context_frames - 1]
        self.outputs['gen_images'] = tf.tile(last_context_images[:, None], tile_pattern)
        if 'pix_distribs' in self.inputs:
            initial_pix_distrib = self.inputs['pix_distribs'][:, 0]
            self.outputs['gen_pix_distribs'] = tf.tile(initial_pix_distrib[:, None], tile_pattern)
        self.gen_images = self.outputs['gen_images']

        if self.targets is not None:
            self.metrics = self.metrics_fn(self.inputs, self.outputs, self.targets)
        else:
            self.metrics = {}
