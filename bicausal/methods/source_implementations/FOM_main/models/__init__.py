from .anm import ANM
from .igci import IGCI
from .reci import RECI
from .fom import FOM


def build_model(model_name):
    assert model_name in ['ANM', 'IGCI', 'RECI', 'FOM']
    models = {'ANM': ANM(), 'IGCI': IGCI(), 'RECI': RECI(), 'FOM': FOM()}
    return models.get(model_name)