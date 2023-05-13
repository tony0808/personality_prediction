from utils import ModelNames

def iterate_model_names():
    model_names = []
    for attr in dir(ModelNames):
        if not callable(getattr(ModelNames, attr)) and not attr.startswith("__"):
            model_names.append(getattr(ModelNames, attr))
    return model_names

print(iterate_model_names())