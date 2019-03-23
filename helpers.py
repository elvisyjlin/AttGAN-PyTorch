# Copyright (C) 2018 Elvis Yu-Jing Lin <elvisyjlin@gmail.com>
# 
# This work is licensed under the MIT License. To view a copy of this license,
# visit https://opensource.org/licenses/MIT.

"""Helper functions for training."""

def run_from_ipython():
    try:
        __IPYTHON__
        return True
    except NameError:
        return False

def name_experiment(prefix='', suffix=''):
    import datetime
    import platform
    
    experiment_name = datetime.datetime.now().strftime('%b%d_%H-%M-%S_') + platform.node()
    if prefix is not None and prefix != '':
        experiment_name = prefix + '_' + experiment_name
    if suffix is not None and suffix != '':
        experiment_name = experiment_name + '_' + suffix
    return experiment_name

class Progressbar():
    def __init__(self):
        self.p = None
    def __call__(self, iterable):
        from tqdm import tqdm
        self.p = tqdm(iterable)
        return self.p
    def say(self, **kwargs):
        if self.p is not None:
            self.p.set_postfix(**kwargs)

def add_scalar_dict(writer, scalar_dict, iteration, directory=None):
    for key in scalar_dict:
        key_ = directory + '/' + key if directory is not None else key
        writer.add_scalar(key_, scalar_dict[key], iteration)