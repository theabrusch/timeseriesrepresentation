import dn3
from dn3 import configuratron
from dn3.configuratron import ExperimentConfig
data_path = '/Volumes/SED/training/'

config_filename = 'sleepeeg.yml'
experiment = ExperimentConfig(config_filename)
ds_config = experiment.datasets['sleepeeg']
dataset = ds_config.auto_construct_dataset()
