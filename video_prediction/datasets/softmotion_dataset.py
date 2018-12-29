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
        # infer name of image feature and check if object_pos feature is present
        from google.protobuf.json_format import MessageToDict
        example = next(tf.python_io.tf_record_iterator(self.filenames[0]))
        dict_message = MessageToDict(tf.train.Example.FromString(example))
        feature = dict_message['features']['feature']
        image_names = set()
        for name in feature.keys():
            m = re.search('\d+/(\w+)/encoded', name)
            if m:
                image_names.add(m.group(1))
        # look for image_aux1 and image_view0 in that order of priority
        image_name = None
        for name in ['image_aux1', 'image_view0']:
            if name in image_names:
                image_name = name
                break
        if not image_name:
            if len(image_names) == 1:
                image_name = image_names.pop()
            else:
                raise ValueError('The examples have images under more than one name.')
        self.state_like_names_and_shapes['images'] = '%%d/%s/encoded' % image_name, None
        if self.hparams.use_state:
            self.state_like_names_and_shapes['states'] = '%d/endeffector_pos', (3,)
            self.action_like_names_and_shapes['actions'] = '%d/action', (4,)
        if any([re.search('\d+/object_pos', name) for name in feature.keys()]):
            self.state_like_names_and_shapes['object_pos'] = '%d/object_pos', None  # shape is (2 * num_designated_pixels)
        self._check_or_infer_shapes()

    def get_default_hparams_dict(self):
        default_hparams = super(SoftmotionVideoDataset, self).get_default_hparams_dict()
        hparams = dict(
            context_frames=2,
            sequence_length=12,
            long_sequence_length=30,
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
