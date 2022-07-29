from .base import BaseModel
from FCTT.utils import all_subclasses
from FCTT.utils import import_all_subclasses
import_all_subclasses(__file__, __name__, BaseModel)

MODELS = {c.code():c
          for c in all_subclasses(BaseModel)
          if c.code() is not None}


def model_factory(args,train_load):
    model = MODELS[args.model_code]
    return model(args,train_load)
