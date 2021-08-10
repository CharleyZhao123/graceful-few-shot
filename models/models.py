import torch


models = {}
def register(name):
    def decorator(cls):
        models[name] = cls
        return cls
    return decorator


def make(name, **kwargs):
    if name is None:
        return None
    model = models[name](**kwargs)
    if torch.cuda.is_available():
        model.cuda()
    return model


def load(model_sv, name=None, new_model_args=None):
    if name is None:
        name = 'model'
    if new_model_args is None:
        model = make(model_sv[name], **model_sv[name + '_args'])
        print(model_sv[name + '_args'])
    else:
        print(model_sv[name])
        model = make(model_sv[name], **new_model_args)
        print(new_model_args)
    model.load_state_dict(model_sv[name + '_sd'])
    return model

