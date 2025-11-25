from .fom import FOM


def build_model(model_name):
    assert model_name in ['FOM']
    models = {'FOM': FOM()}
    return models.get(model_name)