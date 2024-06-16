import argparse
import pickle
import logging
import datetime
import os, re
from os.path import join, split
from importlib.machinery import SourceFileLoader
from types import ModuleType
from tqdm import tqdm
from glob import glob

logger = logging.getLogger(__name__)

def read_configs():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=None)
    parser.add_argument("--data", default=None)
    parser.add_argument("--features", default=None)
    args = parser.parse_args()
    configs = {}
    for config_type in vars(args):
        path = getattr(args, config_type)
        if not path:
            raise ValueError(f"Missing {config_type} config path")
        config_name = re.match(r"configs/(model/|data/|features/)?([\w\_\-]+).py", path).group(2)
        config = import_configs_objs(path)
        config.config_name = config_name
        configs[config_type]=config
    return [configs[cfg_type] for cfg_type in ['model','data','features']]

def import_configs_objs(config_file):
    """Dynamicaly loads the configuration file"""
    if config_file is None:
        raise ValueError("No config path")
    loader = SourceFileLoader('config', config_file)
    mod = ModuleType(loader.name)
    loader.exec_module(mod)
    for var in ["__name__", "__doc__", "__package__", "__loader__", "__spec__", "__builtins__"]:
        delattr(mod, var)
    return mod

def create_result_folder(*params):
    names = [params[i].config_name for i in range(len(params))]
    base = join('results',*names)
    now = datetime.datetime.now()
    os.makedirs(base, exist_ok=True)
    # dump confgis files
    for param in params:
        configs = {}
        for setting in dir(param):
            configs[setting] = getattr(param, setting)
        with open(os.path.join(base, f'{param.config_name}_config.pkl'), 'wb') as f:
            pickle.dump(configs, f, protocol=pickle.HIGHEST_PROTOCOL)
    # dump run date
    with open(os.path.join(base, 'run_date.txt'), 'w') as f:
        f.write(str(now))
        f.close
    return base

def set_seed(seed):
    import torch
    import random
    import numpy as np
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)