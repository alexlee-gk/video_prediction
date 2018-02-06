from .base_model import BaseVideoPredictionModel
from .base_model import SoftPlacementVideoPredictionModel, VideoPredictionModel
from .non_trainable_model import NonTrainableVideoPredictionModel
from .non_trainable_model import GroundTruthVideoPredictionModel, RepeatVideoPredictionModel
from .pix2pix_model import Pix2PixVideoPredictionModel
from .improved_dna_model import ImprovedDNAVideoPredictionModel
from .dna_model import DNAVideoPredictionModel
from .sna_model import SNAVideoPredictionModel


def get_model_class(model):
    model_mappings = {
        'ground_truth': 'GroundTruthVideoPredictionModel',
        'repeat': 'RepeatVideoPredictionModel',
        'pix2pix': 'Pix2PixVideoPredictionModel',
        'improved_dna': 'ImprovedDNAVideoPredictionModel',
        'dna': 'DNAVideoPredictionModel',
        'sna': 'SNAVideoPredictionModel',
    }
    model_class = model_mappings.get(model, model)
    model_class = globals().get(model_class)
    if model_class is None or not issubclass(model_class, BaseVideoPredictionModel):
        raise ValueError('Invalid model %s' % model)
    return model_class
