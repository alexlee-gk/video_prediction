import itertools

from .base_dataset import SequenceExampleVideoDataset


class KTHVideoDataset(SequenceExampleVideoDataset):
    def __init__(self, *args, **kwargs):
        super(KTHVideoDataset, self).__init__(*args, **kwargs)
        self.state_like_names_and_shapes['images'] = 'image/encoded', (64, 80, 3)

    def get_default_hparams_dict(self):
        default_hparams = super(KTHVideoDataset, self).get_default_hparams_dict()
        hparams = dict(
            context_frames=10,
            sequence_length=20,
        )
        return dict(itertools.chain(default_hparams.items(), hparams.items()))

    @property
    def jpeg_encoding(self):
        return False
