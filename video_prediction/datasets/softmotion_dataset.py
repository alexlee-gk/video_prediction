import itertools
import os
import re

import tensorflow as tf

from video_prediction.utils import tf_utils
from .base_dataset import VideoDataset


class SoftmotionVideoDataset(VideoDataset):
    """
    https://sites.google.com/view/sna-visual-mpc
    """
    def __init__(self, *args, **kwargs):
        super(SoftmotionVideoDataset, self).__init__(*args, **kwargs)
        if 'softmotion30_44k' in self.input_dir.split('/'):
            self.state_like_names_and_shapes['images'] = '%d/image_aux1/encoded', None
        else:
            self.state_like_names_and_shapes['images'] = '%d/image_view0/encoded', None
        if self.hparams.use_state:
            self.state_like_names_and_shapes['states'] = '%d/endeffector_pos', (3,)
            self.action_like_names_and_shapes['actions'] = '%d/action', (4,)
        if os.path.basename(self.input_dir).endswith('annotations'):
            self.state_like_names_and_shapes['object_pos'] = '%d/object_pos', None  # shape is (2 * num_designated_pixels)
        self._check_or_infer_shapes()

    def get_default_hparams_dict(self):
        default_hparams = super(SoftmotionVideoDataset, self).get_default_hparams_dict()
        hparams = dict(
            context_frames=2,
            sequence_length=12,
            time_shift=2,
        )
        return dict(itertools.chain(default_hparams.items(), hparams.items()))

    @property
    def jpeg_encoding(self):
        return False

    def parser(self, serialized_example):
        state_like_seqs, action_like_seqs = super(SoftmotionVideoDataset, self).parser(serialized_example)
        if 'object_pos' in state_like_seqs:
            object_pos = state_like_seqs['object_pos']
            height, width, _ = self.state_like_names_and_shapes['images'][1]
            object_pos = tf.reshape(object_pos, [object_pos.shape[0].value, -1, 2])
            pix_distribs = tf.stack([tf_utils.pixel_distribution(object_pos_, height, width)
                                     for object_pos_ in tf.unstack(object_pos, axis=1)], axis=-1)
            state_like_seqs['pix_distribs'] = pix_distribs
        return state_like_seqs, action_like_seqs

    def num_examples_per_epoch(self):
        # extract information from filename to count the number of trajectories in the dataset
        count = 0
        for filename in self.filenames:
            match = re.search('traj_(\d+)_to_(\d+).tfrecords', os.path.basename(filename))
            start_traj_iter = int(match.group(1))
            end_traj_iter = int(match.group(2))
            count += end_traj_iter - start_traj_iter + 1

        # alternatively, the dataset size can be determined like this, but it's very slow
        # count = sum(sum(1 for _ in tf.python_io.tf_record_iterator(filename)) for filename in filenames)
        return count
