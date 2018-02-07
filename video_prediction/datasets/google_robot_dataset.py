import itertools
import os

from .base_dataset import VideoDataset


class GoogleRobotVideoDataset(VideoDataset):
    """
    https://sites.google.com/site/brainrobotdata/home/push-dataset
    """
    def __init__(self, *args, **kwargs):
        super(GoogleRobotVideoDataset, self).__init__(*args, **kwargs)
        self.state_like_names_and_shapes['images'] = 'move/%d/image/encoded', (512, 640, 3)
        if self.hparams.use_state:
            self.state_like_names_and_shapes['states'] = 'move/%d/endeffector/vec_pitch_yaw', (5,)
            self.action_like_names_and_shapes['actions'] = 'move/%d/commanded_pose/vec_pitch_yaw', (5,)
        self._check_or_infer_shapes()

    def get_default_hparams_dict(self):
        default_hparams = super(GoogleRobotVideoDataset, self).get_default_hparams_dict()
        hparams = dict(
            context_frames=2,
            sequence_length=15,
        )
        return dict(itertools.chain(default_hparams.items(), hparams.items()))

    def num_examples_per_epoch(self):
        if os.path.basename(self.input_dir) == 'push_train':
            count = 51615
        elif os.path.basename(self.input_dir) == 'push_testseen':
            count = 1038
        elif os.path.basename(self.input_dir) == 'push_testnovel':
            count = 995
        else:
            raise NotImplementedError
        return count

    @property
    def jpeg_encoding(self):
        return True
