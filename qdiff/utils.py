import numpy as np
import torch
import random
import os
import logging.config
import torch.nn as nn

class StraightThrough(nn.Module):
    def __init__(self, channel_num: int = 1):
        super().__init__()

    def forward(self, input):
        return input

def apply_func_to_submodules(module, class_type, function, parent_name="", return_d=None, **kwargs):
    """
    Recursively iterates through all submodules of a PyTorch module and applies a hook function
    if the submodule matches the specified class type. The parent name is appended to the submodule name.

    Args:
        module (torch.nn.Module): The PyTorch module to iterate through.
        class_type (type): The class type to match against submodules.
        function (callable): The function to apply if a submodule matches the class type.
        parent_name (str): The name of the parent module (used for recursion).
    """

    for name, submodule in module.named_children():
        full_name = f"{parent_name}.{name}" if parent_name else name

        # INFO: pass from the parent call into func
        if 'name' in kwargs:
            kwargs['name']=name
        if 'full_name' in kwargs:
            kwargs['full_name'] = full_name
        if 'parent_module' in kwargs:
            kwargs['parent_module'] = module
        # if 'quant_param_dict' in kwargs:
            # kwargs['quant_param_dict'] = quant_param_dict
        if isinstance(submodule, class_type):
            if return_d is not None:
                return_d[full_name] = function(submodule, **kwargs)
            else:
                function(submodule, **kwargs)

        # Recursively apply the function to submodules
        apply_func_to_submodules(submodule, class_type, function, full_name, return_d, **kwargs)

    if return_d is not None:
        return return_d

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def setup_logging(log_file):
    logging_config = {
        'version': 1,
        'disable_existing_loggers': False,
        'formatters': {
            'standard': {
                'format': '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            },
        },
        'handlers': {
            'console': {
                'class': 'logging.StreamHandler',
                'level': 'DEBUG',
                'formatter': 'standard',
                'stream': 'ext://sys.stdout'
            },
            'file': {
                'class': 'logging.FileHandler',
                'level': 'DEBUG',
                'formatter': 'standard',
                'filename': log_file,
                'mode': 'a',
            }
        },
        'loggers': {
            '': {
                'handlers': ['console', 'file'],
                'level': 'DEBUG',
                'propagate': True
            }
        }
    }
    logging.config.dictConfig(logging_config)
