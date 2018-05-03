import itertools
import re

import tensorflow as tf

from .base_dataset import VideoDataset
from .softmotion_dataset import SoftmotionVideoDataset


class CartgripperVideoDataset(SoftmotionVideoDataset):
    def __init__(self, *args, **kwargs):
        VideoDataset.__init__(self, *args, **kwargs)
        if self.hparams.image_view != -1:
            self.state_like_names_and_shapes['images'] = \
                '%%d/image_view%d/encoded' % self.hparams.image_view, (48, 64, 3)
        else:
            # infer name of image feature(s)
            from google.protobuf.json_format import MessageToDict
            example = next(tf.python_io.tf_record_iterator(self.filenames[0]))
            dict_message = MessageToDict(tf.train.Example.FromString(example))
            feature = dict_message['features']['feature']
            image_names = set()
            for name in feature.keys():
                m = re.search('\d+/(\w+)/encoded', name)
                if m:
                    image_names.add(m.group(1))
            for image_name in image_names:
                m = re.search('image_view(\d+)', image_name)
                if m:
                    i = int(m.group(1))
                    suffix = '%d' % i if i > 0 else ''
                    self.state_like_names_and_shapes['images' + suffix] = \
                        '%%d/%s/encoded' % image_name, (48, 64, 3)
        if self.hparams.use_state:
            self.state_like_names_and_shapes['states'] = '%d/endeffector_pos', None
            self.action_like_names_and_shapes['actions'] = '%d/action', None
        self._check_or_infer_shapes()

    def get_default_hparams_dict(self):
        default_hparams = super(CartgripperVideoDataset, self).get_default_hparams_dict()
        hparams = dict(
            context_frames=2,
            sequence_length=15,
            time_shift=3,
            use_state=True,
            image_view=-1,
        )
        return dict(itertools.chain(default_hparams.items(), hparams.items()))
