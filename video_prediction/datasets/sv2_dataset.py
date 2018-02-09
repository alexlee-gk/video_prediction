import itertools
import os

from .base_dataset import VideoDataset


class SV2PVideoDataset(VideoDataset):
    def __init__(self, *args, **kwargs):
        super(SV2PVideoDataset, self).__init__(*args, **kwargs)
        self.dataset_name = os.path.basename(os.path.split(self.input_dir)[0])
        self.state_like_names_and_shapes['images'] = 'image_%d', (64, 64, 3)
        if self.dataset_name == 'shape':
            if self.hparams.use_state:
                self.state_like_names_and_shapes['states'] = 'state_%d', (2,)
                self.action_like_names_and_shapes['actions'] = 'action_%d', (2,)
        elif self.dataset_name == 'humans':
            if self.hparams.use_state:
                raise ValueError('SV2PVideoDataset does not have states, use_state should be False')
        else:
            raise NotImplementedError
        self._check_or_infer_shapes()

    def get_default_hparams_dict(self):
        default_hparams = super(SV2PVideoDataset, self).get_default_hparams_dict()
        if self.dataset_name == 'shape':
            hparams = dict(
                context_frames=1,
                sequence_length=6,
                time_shift=0,
                use_state=False,
            )
        elif self.dataset_name == 'humans':
            hparams = dict(
                context_frames=10,
                sequence_length=20,
                use_state=False,
            )
        else:
            raise NotImplementedError
        return dict(itertools.chain(default_hparams.items(), hparams.items()))

    def num_examples_per_epoch(self):
        if self.dataset_name == 'shape':
            if os.path.basename(self.input_dir) == 'train':
                count = 43415
            elif os.path.basename(self.input_dir) == 'val':
                count = 2898
            else:  # shape dataset doesn't have a test set
                raise NotImplementedError
        elif self.dataset_name == 'humans':
            if os.path.basename(self.input_dir) == 'train':
                count = 23910
            elif os.path.basename(self.input_dir) == 'val':
                count = 10472
            elif os.path.basename(self.input_dir) == 'test':
                count = 7722
            else:
                raise NotImplementedError
        else:
            raise NotImplementedError
        return count

    @property
    def jpeg_encoding(self):
        return True
